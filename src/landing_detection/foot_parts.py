"""Fine foot landmark estimation using MediaPipe Pose."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import ssl
from urllib.error import URLError
from urllib.request import urlopen, urlretrieve

import cv2
import numpy as np

from .models import Point, PoseDetection


LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = Path(".cache/mediapipe/pose_landmarker_lite.task")


@dataclass(frozen=True)
class FootLandmark:
    point: Point
    visibility: float


@dataclass(frozen=True)
class RejectedFootLandmark:
    side: str
    part: str
    point: Point
    reason: str
    visibility: float


@dataclass(frozen=True)
class FootPart:
    ankle: FootLandmark | None
    heel: FootLandmark | None
    toe: FootLandmark | None


@dataclass(frozen=True)
class FootPartEstimate:
    left: FootPart
    right: FootPart

    def by_side(self, side: str) -> FootPart | None:
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        return None


@dataclass(frozen=True)
class RefinedContact:
    """A foot contact point after MediaPipe sanity checks."""

    part: str
    point: Point
    status: str


@dataclass(frozen=True)
class FootPartEstimatorConfig:
    min_visibility: float = 0.55
    crop_margin_ratio: float = 0.12
    crop_upscale: float = 3.0
    y_difference_threshold_px: int = 10
    max_foot_length_ratio_of_bbox: float = 0.22
    max_foot_length_px: int = 90
    min_foot_length_px: int = 35
    lower_body_min_ratio: float = 0.42
    foot_roi_x_ratio_of_bbox: float = 0.34
    foot_roi_y_ratio_of_bbox: float = 0.34
    history_frames: int = 3
    max_foot_point_jump_px: int = 45
    max_missing_history_frames: int = 2
    use_ankle_anchor: bool = True
    min_anchor_offsets: int = 2
    model_path: Path = DEFAULT_MODEL_PATH
    model_url: str = DEFAULT_MODEL_URL


class FootPartEstimator:
    """Estimate ankle, heel, and toe/foot-index points for the locked athlete."""

    def __init__(self, config: FootPartEstimatorConfig | None = None) -> None:
        self.config = config or FootPartEstimatorConfig()
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise SystemExit(
                "错误：当前 Python 环境未安装 mediapipe。请运行：python -m pip install -r requirements.txt"
            ) from exc

        self._mp = mp
        model_path = self._ensure_model_file()
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._next_timestamp_ms = 0
        self._trusted_history: dict[tuple[str, str], deque[Point]] = {}
        self._landmark_history: dict[tuple[str, str], deque[FootLandmark]] = {}
        self._anchor_offset_history: dict[tuple[str, str], deque[Point]] = {}
        self._missing_counts: dict[tuple[str, str], int] = {}
        self.last_rejected_landmarks: list[RejectedFootLandmark] = []

    def estimate(
        self,
        frame: np.ndarray,
        detection: PoseDetection,
        timestamp_ms: int | None = None,
    ) -> FootPartEstimate | None:
        self.last_rejected_landmarks.clear()
        crop, offset = self._crop_athlete(frame, detection)
        if crop.size == 0:
            return None

        model_crop = self._upscale_crop(crop)
        rgb = cv2.cvtColor(model_crop, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if timestamp_ms is None:
            timestamp_ms = self._next_timestamp_ms
            self._next_timestamp_ms += 33
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        raw_estimate = FootPartEstimate(
            left=FootPart(
                ankle=self._landmark_to_point(landmarks, LEFT_ANKLE, crop.shape, offset),
                heel=self._landmark_to_point(landmarks, LEFT_HEEL, crop.shape, offset),
                toe=self._landmark_to_point(landmarks, LEFT_FOOT_INDEX, crop.shape, offset),
            ),
            right=FootPart(
                ankle=self._landmark_to_point(landmarks, RIGHT_ANKLE, crop.shape, offset),
                heel=self._landmark_to_point(landmarks, RIGHT_HEEL, crop.shape, offset),
                toe=self._landmark_to_point(landmarks, RIGHT_FOOT_INDEX, crop.shape, offset),
            ),
        )
        return self._stabilize_estimate(raw_estimate, detection)

    def reset_history(self) -> None:
        """Clear trusted foot-part history when the locked person changes."""
        self._trusted_history.clear()
        self._landmark_history.clear()
        self._anchor_offset_history.clear()
        self._missing_counts.clear()
        self.last_rejected_landmarks.clear()

    def classify_contact(
        self,
        estimate: FootPartEstimate | None,
        foot_side: str,
        contact_point: Point,
    ) -> str:
        if estimate is None:
            return "unknown"
        foot = estimate.by_side(foot_side)
        if foot is None or foot.heel is None or foot.toe is None:
            return "unknown"

        toe_y = foot.toe.point[1]
        heel_y = foot.heel.point[1]
        if toe_y - heel_y >= self.config.y_difference_threshold_px:
            return "toe"
        if heel_y - toe_y >= self.config.y_difference_threshold_px:
            return "heel"
        return "unknown"

    def refine_contact(
        self,
        estimate: FootPartEstimate | None,
        foot_side: str,
        fallback_point: Point,
        detection: PoseDetection | None = None,
    ) -> RefinedContact:
        """Return a trusted toe/heel contact or fall back to the YOLO ankle.

        MediaPipe foot points are useful only when the person is large enough and
        unobstructed. In landing videos they can jump to the canopy, target mat,
        or background. This method accepts toe/heel only when it stays near the
        YOLO ankle, in the lower-body foot ROI, and does not jump sharply from
        recent trusted points.
        """
        contact_part = self.classify_contact(estimate, foot_side, fallback_point)
        if estimate is None or contact_part == "unknown":
            return RefinedContact(contact_part, fallback_point, "unknown")

        foot = estimate.by_side(foot_side)
        if foot is None:
            return RefinedContact("unknown", fallback_point, "unknown")

        landmark = None
        if contact_part == "toe" and foot.toe is not None:
            landmark = foot.toe
        if contact_part == "heel" and foot.heel is not None:
            landmark = foot.heel
        if landmark is None:
            return RefinedContact("unknown", fallback_point, "unknown")

        reject_reason = self._reject_reason(
            point=landmark.point,
            fallback_point=fallback_point,
            detection=detection,
            foot_side=foot_side,
            contact_part=contact_part,
        )
        if reject_reason is not None:
            return RefinedContact("unknown", fallback_point, reject_reason)

        self._remember_trusted_point(foot_side, contact_part, landmark.point)
        return RefinedContact(contact_part, landmark.point, "trusted")

    def _ensure_model_file(self) -> Path:
        model_path = self.config.model_path
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        if model_path.exists():
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe pose landmarker model: {model_path}")
        try:
            urlretrieve(self.config.model_url, model_path)
        except URLError:
            print("SSL verification failed while downloading MediaPipe model; retrying with unverified context.")
            context = ssl._create_unverified_context()
            with urlopen(self.config.model_url, context=context) as response:
                model_path.write_bytes(response.read())
        return model_path

    def _crop_athlete(self, frame: np.ndarray, detection: PoseDetection) -> tuple[np.ndarray, Point]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        margin = int(max(box_w, box_h) * self.config.crop_margin_ratio)
        left = max(0, x1 - margin)
        top = max(0, y1 - margin)
        right = min(width, x2 + margin)
        bottom = min(height, y2 + margin)
        return frame[top:bottom, left:right], (left, top)

    def _upscale_crop(self, crop: np.ndarray) -> np.ndarray:
        scale = self.config.crop_upscale
        if scale <= 1.0:
            return crop
        width = max(1, int(round(crop.shape[1] * scale)))
        height = max(1, int(round(crop.shape[0] * scale)))
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)

    def _landmark_to_point(
        self,
        landmarks,
        index: int,
        crop_shape: tuple[int, ...],
        offset: Point,
    ) -> FootLandmark | None:
        landmark = landmarks[index]
        crop_h, crop_w = crop_shape[:2]
        x = int(round(landmark.x * crop_w + offset[0]))
        y = int(round(landmark.y * crop_h + offset[1]))
        return FootLandmark(point=(x, y), visibility=float(landmark.visibility))

    def _stabilize_estimate(
        self,
        estimate: FootPartEstimate,
        detection: PoseDetection,
    ) -> FootPartEstimate:
        return FootPartEstimate(
            left=FootPart(
                ankle=self._stabilize_landmark("left", "ankle", estimate.left.ankle, detection),
                heel=self._stabilize_landmark("left", "heel", estimate.left.heel, detection),
                toe=self._stabilize_landmark("left", "toe", estimate.left.toe, detection),
            ),
            right=FootPart(
                ankle=self._stabilize_landmark("right", "ankle", estimate.right.ankle, detection),
                heel=self._stabilize_landmark("right", "heel", estimate.right.heel, detection),
                toe=self._stabilize_landmark("right", "toe", estimate.right.toe, detection),
            ),
        )

    def _stabilize_landmark(
        self,
        foot_side: str,
        part: str,
        landmark: FootLandmark | None,
        detection: PoseDetection,
    ) -> FootLandmark | None:
        key = (foot_side, part)
        fallback_point = self._fallback_ankle_point(foot_side, detection)
        if landmark is None:
            if part in {"heel", "toe"} and fallback_point is not None:
                return self._anchored_or_recent_landmark(key, fallback_point)
            return self._recent_landmark_or_none(key)

        if landmark.visibility < self.config.min_visibility:
            self._remember_rejected_landmark(foot_side, part, landmark, "rejected_low_visibility")
            if part in {"heel", "toe"} and fallback_point is not None:
                return self._anchored_or_recent_landmark(key, fallback_point)
            return self._recent_landmark_or_none(key)

        if part in {"heel", "toe"}:
            if fallback_point is None:
                return self._recent_landmark_or_none(key)
            reject_reason = self._landmark_reject_reason(landmark.point, fallback_point, detection, key)
            if reject_reason is not None:
                self._remember_rejected_landmark(foot_side, part, landmark, reject_reason)
                return self._anchored_or_recent_landmark(key, fallback_point)
            self._remember_anchor_offset(key, landmark.point, fallback_point)
        else:
            history = self._landmark_history.get(key)
            if history and self._distance(landmark.point, history[-1].point) > self.config.max_foot_point_jump_px:
                return self._recent_landmark_or_none(key)

        self._remember_landmark(key, landmark)
        return self._smoothed_landmark(key)

    def _recent_landmark_or_none(self, key: tuple[str, str]) -> FootLandmark | None:
        count = self._missing_counts.get(key, 0) + 1
        self._missing_counts[key] = count
        if count > self.config.max_missing_history_frames:
            return None
        return self._smoothed_landmark(key)

    def _remember_landmark(self, key: tuple[str, str], landmark: FootLandmark) -> None:
        if key not in self._landmark_history:
            self._landmark_history[key] = deque(maxlen=self.config.history_frames)
        self._landmark_history[key].append(landmark)
        self._missing_counts[key] = 0

    def _remember_anchor_offset(
        self,
        key: tuple[str, str],
        point: Point,
        ankle_point: Point,
    ) -> None:
        if not self.config.use_ankle_anchor:
            return
        if key not in self._anchor_offset_history:
            self._anchor_offset_history[key] = deque(maxlen=self.config.history_frames)
        self._anchor_offset_history[key].append((point[0] - ankle_point[0], point[1] - ankle_point[1]))

    def _anchored_or_recent_landmark(
        self,
        key: tuple[str, str],
        ankle_point: Point,
    ) -> FootLandmark | None:
        anchored = self._anchored_landmark_from_history(key, ankle_point)
        if anchored is not None:
            self._remember_landmark(key, anchored)
            return self._smoothed_landmark(key)
        return self._recent_landmark_or_none(key)

    def _anchored_landmark_from_history(
        self,
        key: tuple[str, str],
        ankle_point: Point,
    ) -> FootLandmark | None:
        if not self.config.use_ankle_anchor:
            return None
        offsets = self._anchor_offset_history.get(key)
        if offsets is None or len(offsets) < self.config.min_anchor_offsets:
            return None
        dx = int(round(float(np.median(np.array([offset[0] for offset in offsets], dtype=np.float32)))))
        dy = int(round(float(np.median(np.array([offset[1] for offset in offsets], dtype=np.float32)))))
        return FootLandmark(point=(ankle_point[0] + dx, ankle_point[1] + dy), visibility=0.0)

    def _remember_rejected_landmark(
        self,
        foot_side: str,
        part: str,
        landmark: FootLandmark,
        reason: str,
    ) -> None:
        if part not in {"heel", "toe"}:
            return
        self.last_rejected_landmarks.append(
            RejectedFootLandmark(
                side=foot_side,
                part=part,
                point=landmark.point,
                reason=reason,
                visibility=landmark.visibility,
            )
        )

    def _smoothed_landmark(self, key: tuple[str, str]) -> FootLandmark | None:
        history = self._landmark_history.get(key)
        if not history:
            return None
        xs = np.array([landmark.point[0] for landmark in history], dtype=np.float32)
        ys = np.array([landmark.point[1] for landmark in history], dtype=np.float32)
        visibility = min(landmark.visibility for landmark in history)
        return FootLandmark(
            point=(int(round(float(np.median(xs)))), int(round(float(np.median(ys))))),
            visibility=visibility,
        )

    def _fallback_ankle_point(self, foot_side: str, detection: PoseDetection) -> Point | None:
        if foot_side == "left":
            return detection.foot_points.left
        if foot_side == "right":
            return detection.foot_points.right
        return None

    def _landmark_reject_reason(
        self,
        point: Point,
        fallback_point: Point,
        detection: PoseDetection,
        key: tuple[str, str],
    ) -> str | None:
        reject_reason = self._roi_reject_reason(point, fallback_point, detection)
        if reject_reason is not None:
            return reject_reason

        history = self._landmark_history.get(key)
        if history and self._distance(point, history[-1].point) > self.config.max_foot_point_jump_px:
            return "rejected_temporal_jump"
        return None

    def _distance(self, p1: Point, p2: Point) -> float:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _roi_reject_reason(
        self,
        point: Point,
        fallback_point: Point,
        detection: PoseDetection,
    ) -> str | None:
        x1, y1, x2, y2 = detection.bbox
        bbox_w = max(x2 - x1, 1)
        bbox_h = max(y2 - y1, 1)
        if point[1] < y1 + bbox_h * self.config.lower_body_min_ratio:
            return "rejected_outside_foot_roi"

        max_foot_length = max(
            self.config.min_foot_length_px,
            min(
                self.config.max_foot_length_px,
                int(round(bbox_h * self.config.max_foot_length_ratio_of_bbox)),
            ),
        )
        if self._distance(point, fallback_point) > max_foot_length:
            return "rejected_far_from_ankle"

        roi_half_w = max(
            self.config.min_foot_length_px,
            int(round(bbox_w * self.config.foot_roi_x_ratio_of_bbox)),
        )
        roi_half_h = max(
            self.config.min_foot_length_px,
            int(round(bbox_h * self.config.foot_roi_y_ratio_of_bbox)),
        )
        if abs(point[0] - fallback_point[0]) > roi_half_w or abs(point[1] - fallback_point[1]) > roi_half_h:
            return "rejected_outside_foot_roi"
        return None

    def _reject_reason(
        self,
        point: Point,
        fallback_point: Point,
        detection: PoseDetection | None,
        foot_side: str,
        contact_part: str,
    ) -> str | None:
        if detection is not None:
            roi_reject_reason = self._roi_reject_reason(point, fallback_point, detection)
            if roi_reject_reason is not None:
                return roi_reject_reason
        elif self._distance(point, fallback_point) > self.config.max_foot_length_px:
            return "rejected_far_from_ankle"

        history = self._trusted_history.get((foot_side, contact_part))
        if history and self._distance(point, history[-1]) > self.config.max_foot_point_jump_px:
            return "rejected_temporal_jump"
        return None

    def _remember_trusted_point(self, foot_side: str, contact_part: str, point: Point) -> None:
        key = (foot_side, contact_part)
        if key not in self._trusted_history:
            self._trusted_history[key] = deque(maxlen=self.config.history_frames)
        self._trusted_history[key].append(point)
