#!/usr/bin/env python3
"""Browse MediaPipe toe/heel contact candidates on a landing video.

Open this file in VSCode, edit VIDEO_PATH below, then click "Run Python File".
"""

from __future__ import annotations

# ===== 在这里切换要浏览的视频 =====
VIDEO_PATH = "training_video/video6.mp4"
MODEL_NAME = "yolo11n-pose.pt"
CONFIDENCE_THRESHOLD = 0.25
KEYPOINT_CONFIDENCE_THRESHOLD = 0.35
CONTACT_THRESHOLD_PX = 8.0
CONTACT_FRAMES = 3
SHOW_HORIZON_LINE = True
SHOW_REJECTED_FOOT_PARTS = True
HORIZON_MAX_Y_RATIO = 0.70
GROUND_MIN_GAP_BELOW_HORIZON_RATIO = 0.12
GROUND_MIN_Y_RATIO = 0.70
MIN_BODY_BOTTOM_RATIO_FOR_CONTACT = 0.64
USE_MANUAL_GROUND_LINE = False
MANUAL_GROUND_Y_RATIO = 0.88
WRITE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = "output/mediapipe_foot_contact_demo.mp4"
DISPLAY_SCALE = 0.75
# ==============================

import logging
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from landing_detection.athlete import AthleteSelector
from landing_detection.foot_parts import FootLandmark, FootPartEstimate, FootPartEstimator
from landing_detection.ground import GroundLineConfig, GroundLineEstimator, HorizonLineConfig, HorizonLineEstimator
from landing_detection.models import GroundLine, Point, PoseDetection
from landing_detection.pose import PoseDetector, PoseDetectorConfig
from landing_detection.visualizer import COCO_SKELETON


WINDOW_NAME = "MediaPipe Foot Contact Demo"
LEFT_ANKLE_INDEX = 15
RIGHT_ANKLE_INDEX = 16


@dataclass(frozen=True)
class FootContactCandidate:
    frame_index: int
    timestamp: float
    foot: str
    part: str
    point: Point
    distance_px: float


@dataclass(frozen=True)
class FootDistance:
    foot: str
    part: str
    point: Point
    distance_px: float


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class MediaPipeFootContactBrowser:
    """Small OpenCV browser for inspecting MediaPipe toe/heel contact candidates."""

    def __init__(self) -> None:
        self.video_path = self._resolve_path(VIDEO_PATH)
        self.output_video_path = self._resolve_path(OUTPUT_VIDEO_PATH)
        self.detector = PoseDetector(
            PoseDetectorConfig(
                model_name=MODEL_NAME,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                keypoint_confidence_threshold=KEYPOINT_CONFIDENCE_THRESHOLD,
            )
        )
        self.athlete_selector = AthleteSelector()
        self.foot_part_estimator = FootPartEstimator()
        self.horizon_estimator = HorizonLineEstimator(
            HorizonLineConfig(max_candidate_y_ratio=HORIZON_MAX_Y_RATIO)
        )
        self.ground_estimator = GroundLineEstimator(
            GroundLineConfig(
                min_candidate_y_ratio=GROUND_MIN_Y_RATIO,
                min_gap_below_horizon_ratio=GROUND_MIN_GAP_BELOW_HORIZON_RATIO,
            )
        )
        self.left_trail: deque[Point] = deque(maxlen=180)
        self.right_trail: deque[Point] = deque(maxlen=180)
        self.current_track_id: int | None = None
        self.contact_streak: list[FootContactCandidate] = []
        self.first_contact: FootContactCandidate | None = None
        self.first_contact_printed = False
        self.latest_distances: list[FootDistance] = []
        self.latest_candidate: FootContactCandidate | None = None
        self.latest_detection_mode = "NO PERSON"
        self.paused = False
        self.step_once = False
        self.last_inference_seconds = 0.0

    def run(self) -> int:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise SystemExit(f"错误：无法打开视频：{self.video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        delay_ms = max(1, int(1000 / max(fps, 1.0)))
        writer = self._open_writer(fps, width, height) if WRITE_OUTPUT_VIDEO else None

        print(f"视频：{self.video_path}")
        print(f"尺寸：{width}x{height}，FPS：{fps:.2f}，总帧数：{frame_count}")
        print("按键：Space 暂停/继续，n 暂停时前进一帧，m 暂停时后退一帧，r 回到开头，q 或 Esc 退出。")
        if writer is not None:
            print(f"标注视频写入：{self.output_video_path}")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        annotated_cache: dict[int, np.ndarray] = {}
        annotated: np.ndarray | None = None
        current_frame_index = -1

        try:
            while True:
                if not self.paused or self.step_once:
                    target_frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                    if self.paused and target_frame_index in annotated_cache:
                        annotated = annotated_cache[target_frame_index]
                        current_frame_index = target_frame_index
                        capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index + 1)
                    else:
                        ok, frame = capture.read()
                        if not ok:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            annotated_cache.clear()
                            current_frame_index = -1
                            self._reset_runtime_state()
                            continue

                        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                        annotated = self._annotate_frame(frame, frame_index, fps, frame_count)
                        annotated_cache[frame_index] = annotated
                        current_frame_index = frame_index
                        if writer is not None and not self.paused:
                            writer.write(annotated)
                    self.step_once = False

                if annotated is None:
                    continue
                cv2.imshow(WINDOW_NAME, self._resize_for_display(annotated))

                key = cv2.waitKey(delay_ms if not self.paused else 30)
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    self.paused = not self.paused
                elif key == ord("n") and self.paused:
                    self.step_once = True
                elif key == ord("m") and self.paused:
                    previous_frame_index = max(0, current_frame_index - 1)
                    if previous_frame_index in annotated_cache:
                        annotated = annotated_cache[previous_frame_index]
                        current_frame_index = previous_frame_index
                        capture.set(cv2.CAP_PROP_POS_FRAMES, previous_frame_index + 1)
                    else:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, previous_frame_index)
                        self._reset_runtime_state()
                        self.step_once = True
                elif key == ord("r"):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    annotated_cache.clear()
                    current_frame_index = -1
                    self._reset_runtime_state()
                    self.paused = False
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

        return 0

    def _annotate_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
        frame_count: int,
    ) -> np.ndarray:
        started_at = time.perf_counter()
        detections = self.detector.detect_all(frame)
        horizon_line = self.horizon_estimator.estimate(frame) if SHOW_HORIZON_LINE else None
        pre_ground_line = self._estimate_ground_line(frame, horizon_line, athlete_bbox=None)
        detection = self.athlete_selector.update(
            detections,
            frame_index,
            ground_line=pre_ground_line,
            frame_shape=frame.shape,
        )
        selected_track_id = self.athlete_selector.track_id_for_detection(detection)
        if detection is not None and self.athlete_selector.locked_track_id is not None:
            self.latest_detection_mode = "LOCKED"
        elif detection is not None:
            self.latest_detection_mode = "CANDIDATE"
        else:
            self.latest_detection_mode = "NO PERSON"
        if detection is None and detections:
            detection = detections[0]
            self.latest_detection_mode = "FALLBACK"
            selected_track_id = self.athlete_selector.track_id_for_detection(detection)
        if selected_track_id is None and detection is not None:
            selected_track_id = 0
        if selected_track_id != self.current_track_id:
            self.left_trail.clear()
            self.right_trail.clear()
            self.contact_streak.clear()
            self.foot_part_estimator.reset_history()
            self.current_track_id = selected_track_id

        ground_line = self._estimate_ground_line(
            frame,
            horizon_line,
            athlete_bbox=detection.bbox if detection is not None else None,
        )
        foot_part_estimate = None
        self.latest_distances = []
        self.latest_candidate = None

        output = frame.copy()
        self._draw_other_detections(output, detections, detection)
        if detection is not None:
            foot_part_estimate = self.foot_part_estimator.estimate(
                frame,
                detection,
                timestamp_ms=int(round(frame_index / max(fps, 0.001) * 1000)),
            )
            self._update_trails(detection)
            if self._contact_detection_is_armed(frame, detection):
                self.latest_candidate = self._update_contact_candidate(
                    foot_part_estimate,
                    ground_line,
                    frame_index,
                    fps,
                )
            else:
                self.contact_streak.clear()
            self._draw_pose(output, detection)
            self._draw_foot_parts(output, foot_part_estimate, ground_line)
            if SHOW_REJECTED_FOOT_PARTS:
                self._draw_rejected_foot_parts(output)
        else:
            self.contact_streak.clear()

        self.last_inference_seconds = time.perf_counter() - started_at
        if SHOW_HORIZON_LINE and horizon_line is not None:
            self._draw_horizon_line(output, horizon_line)
        self._draw_ground_line(output, ground_line)
        self._draw_trails(output)
        self._draw_contact_marker(output)
        self._draw_overlay(output, detection, len(detections), ground_line, horizon_line, frame_index, fps, frame_count)
        return output

    def _update_contact_candidate(
        self,
        estimate: FootPartEstimate | None,
        ground_line: GroundLine,
        frame_index: int,
        fps: float,
    ) -> FootContactCandidate | None:
        distances = self._foot_distances(estimate, ground_line)
        self.latest_distances = distances
        close_points = [
            item for item in distances if abs(item.distance_px) <= CONTACT_THRESHOLD_PX
        ]
        if not close_points:
            self.contact_streak.clear()
            return None

        closest = min(close_points, key=lambda item: abs(item.distance_px))
        candidate = FootContactCandidate(
            frame_index=frame_index,
            timestamp=frame_index / max(fps, 0.001),
            foot=closest.foot,
            part=closest.part,
            point=closest.point,
            distance_px=closest.distance_px,
        )
        if self.contact_streak and (
            self.contact_streak[-1].foot != candidate.foot
            or self.contact_streak[-1].part != candidate.part
        ):
            self.contact_streak.clear()
        self.contact_streak.append(candidate)
        if len(self.contact_streak) > CONTACT_FRAMES:
            self.contact_streak = self.contact_streak[-CONTACT_FRAMES:]

        if (
            self.first_contact is None
            and len(self.contact_streak) >= CONTACT_FRAMES
        ):
            self.first_contact = self.contact_streak[0]
            self._print_first_contact()
        return candidate

    def _foot_distances(
        self,
        estimate: FootPartEstimate | None,
        ground_line: GroundLine,
    ) -> list[FootDistance]:
        if estimate is None:
            return []
        distances: list[FootDistance] = []
        for foot_name, foot in (("left", estimate.left), ("right", estimate.right)):
            for part_name, landmark in (("heel", foot.heel), ("toe", foot.toe)):
                if landmark is None:
                    continue
                distances.append(
                    FootDistance(
                        foot=foot_name,
                        part=part_name,
                        point=landmark.point,
                        distance_px=ground_line.distance_to_point(landmark.point),
                    )
                )
        return distances

    def _print_first_contact(self) -> None:
        if self.first_contact is None or self.first_contact_printed:
            return
        contact = self.first_contact
        print(
            "MediaPipe 首次触地候选："
            f"frame={contact.frame_index}, "
            f"time={contact.timestamp:.3f}s, "
            f"foot={contact.foot}, "
            f"part={contact.part}, "
            f"point={contact.point}, "
            f"distance={contact.distance_px:.1f}px"
        )
        self.first_contact_printed = True

    def _draw_other_detections(
        self,
        frame: np.ndarray,
        detections: list[PoseDetection],
        selected: PoseDetection | None,
    ) -> None:
        selected_bbox = selected.bbox if selected is not None else None
        for detection in detections:
            if detection.bbox == selected_bbox:
                continue
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
            cv2.putText(
                frame,
                f"{self.athlete_selector.label_for_detection(detection)} {detection.confidence:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )

    def _draw_pose(self, frame: np.ndarray, detection: PoseDetection) -> None:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 3)
        cv2.putText(
            frame,
            f"{self.athlete_selector.label_for_detection(detection)} {detection.confidence:.2f}",
            (x1, max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (60, 220, 60),
            2,
            cv2.LINE_AA,
        )
        for start, end in COCO_SKELETON:
            p1 = self._visible_point(detection, start)
            p2 = self._visible_point(detection, end)
            if p1 is not None and p2 is not None:
                cv2.line(frame, p1, p2, (255, 180, 80), 2)
        for index in range(len(detection.keypoints)):
            point = self._visible_point(detection, index)
            if point is None:
                continue
            color = (80, 220, 255)
            radius = 4
            if index == LEFT_ANKLE_INDEX:
                color = (255, 40, 40)
                radius = 8
            elif index == RIGHT_ANKLE_INDEX:
                color = (40, 40, 255)
                radius = 8
            cv2.circle(frame, point, radius, color, -1)

    def _draw_foot_parts(
        self,
        frame: np.ndarray,
        estimate: FootPartEstimate | None,
        ground_line: GroundLine,
    ) -> None:
        if estimate is None:
            return
        for side_label, foot_name, foot in (
            ("L", "left", estimate.left),
            ("R", "right", estimate.right),
        ):
            self._draw_foot_landmark(frame, foot.ankle, f"{side_label} mp ankle", (190, 190, 190))
            self._draw_foot_landmark(frame, foot.heel, f"{side_label} heel", (0, 170, 255))
            self._draw_foot_landmark(frame, foot.toe, f"{side_label} toe", (0, 255, 255))
            nearest = self._nearest_part_for_foot(foot_name)
            if nearest is not None:
                self._draw_distance_label(frame, nearest, ground_line)

    def _draw_foot_landmark(
        self,
        frame: np.ndarray,
        landmark: FootLandmark | None,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        if landmark is None:
            return
        cv2.circle(frame, landmark.point, 7, color, -1)
        cv2.putText(
            frame,
            label,
            (landmark.point[0] + 8, landmark.point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    def _nearest_part_for_foot(self, foot_name: str) -> FootDistance | None:
        distances = [item for item in self.latest_distances if item.foot == foot_name]
        if not distances:
            return None
        return min(distances, key=lambda item: abs(item.distance_px))

    def _draw_distance_label(
        self,
        frame: np.ndarray,
        distance: FootDistance,
        ground_line: GroundLine,
    ) -> None:
        color = (0, 255, 0) if abs(distance.distance_px) <= CONTACT_THRESHOLD_PX else (0, 220, 255)
        ground_point = (
            distance.point[0],
            int(round(ground_line.y_at(distance.point[0]))),
        )
        cv2.line(frame, distance.point, ground_point, color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{distance.part} {distance.distance_px:+.1f}px",
            (distance.point[0] + 8, distance.point[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_rejected_foot_parts(self, frame: np.ndarray) -> None:
        for landmark in self.foot_part_estimator.last_rejected_landmarks:
            cv2.circle(frame, landmark.point, 6, (150, 150, 150), 1)
            cv2.line(
                frame,
                (landmark.point[0] - 5, landmark.point[1] - 5),
                (landmark.point[0] + 5, landmark.point[1] + 5),
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                (landmark.point[0] - 5, landmark.point[1] + 5),
                (landmark.point[0] + 5, landmark.point[1] - 5),
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )

    def _visible_point(self, detection: PoseDetection, index: int) -> Point | None:
        if index >= len(detection.keypoints):
            return None
        if (
            detection.keypoint_confidences is not None
            and detection.keypoint_confidences[index] < KEYPOINT_CONFIDENCE_THRESHOLD
        ):
            return None
        x, y = detection.keypoints[index][:2]
        if x <= 0 and y <= 0:
            return None
        return int(round(float(x))), int(round(float(y)))

    def _update_trails(self, detection: PoseDetection) -> None:
        if detection.foot_points.left is not None:
            self.left_trail.append(detection.foot_points.left)
        if detection.foot_points.right is not None:
            self.right_trail.append(detection.foot_points.right)

    def _draw_trails(self, frame: np.ndarray) -> None:
        self._draw_trail(frame, list(self.left_trail), (255, 40, 40))
        self._draw_trail(frame, list(self.right_trail), (40, 40, 255))

    def _draw_trail(self, frame: np.ndarray, points: list[Point], color: tuple[int, int, int]) -> None:
        for p1, p2 in zip(points, points[1:]):
            cv2.line(frame, p1, p2, color, 2)
        if points:
            cv2.circle(frame, points[-1], 6, color, -1)

    def _draw_ground_line(self, frame: np.ndarray, ground_line: GroundLine) -> None:
        height, width = frame.shape[:2]
        y_left = int(round(ground_line.y_at(0)))
        y_right = int(round(ground_line.y_at(width - 1)))
        color = (0, 180, 255) if not ground_line.fallback else (0, 120, 255)
        cv2.line(frame, (0, y_left), (width - 1, y_right), color, 2)
        cv2.putText(
            frame,
            self._ground_label(ground_line),
            (12, max(24, min(height - 12, y_left - 10))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_horizon_line(self, frame: np.ndarray, horizon_line: GroundLine) -> None:
        height, width = frame.shape[:2]
        y_left = int(round(horizon_line.y_at(0)))
        y_right = int(round(horizon_line.y_at(width - 1)))
        color = (255, 160, 0)
        cv2.line(frame, (0, y_left), (width - 1, y_right), color, 2)
        cv2.putText(
            frame,
            "HORIZON",
            (12, max(24, min(height - 12, y_left - 10))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_contact_marker(self, frame: np.ndarray) -> None:
        contact = self.first_contact or self.latest_candidate
        if contact is None:
            return
        color = (0, 0, 255) if self.first_contact is not None else (0, 255, 0)
        x, y = contact.point
        cv2.circle(frame, (x, y), 12, color, -1)
        cv2.circle(frame, (x, y), 18, (255, 255, 255), 2)
        prefix = "FIRST" if self.first_contact is not None else "candidate"
        cv2.putText(
            frame,
            f"{prefix} {contact.foot} {contact.part} {contact.distance_px:+.1f}px",
            (max(10, x - 150), max(32, y - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            3,
            cv2.LINE_AA,
        )

    def _draw_overlay(
        self,
        frame: np.ndarray,
        detection: PoseDetection | None,
        detection_count: int,
        ground_line: GroundLine,
        horizon_line: GroundLine | None,
        frame_index: int,
        fps: float,
        frame_count: int,
    ) -> None:
        timestamp = frame_index / max(fps, 0.001)
        infer_fps = 1.0 / max(self.last_inference_seconds, 0.001)
        status = "PAUSED" if self.paused else "PLAY"
        found = self.latest_detection_mode
        armed = detection is not None and self._contact_detection_is_armed(frame, detection)
        mp_status = "MP_FOOT=YES" if self.latest_distances else "MP_FOOT=NO"
        candidate_text = self._candidate_text()
        lines = [
            f"{status} | {found} | people={detection_count} | frame {frame_index + 1}/{frame_count} | t={timestamp:.2f}s",
            f"inference={infer_fps:.1f} fps | model={MODEL_NAME} | {mp_status}",
            self.athlete_selector.status_text(),
            f"contact={'ARMED' if armed else 'WAIT'} | threshold={CONTACT_THRESHOLD_PX:.1f}px x {CONTACT_FRAMES}",
            self._distance_summary("left"),
            self._distance_summary("right"),
            f"candidate={candidate_text}",
            f"ground={ground_line.status} | horizon={'on' if horizon_line is not None else 'off'} | video={self.video_path}",
            "Space pause/play | n next frame | m prev frame | r restart | q/Esc quit",
        ]
        y = 28
        for line in lines:
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            y += 27

    def _distance_summary(self, foot_name: str) -> str:
        items = [item for item in self.latest_distances if item.foot == foot_name]
        if not items:
            return f"{foot_name}: heel=- toe=-"
        by_part = {item.part: item.distance_px for item in items}
        heel = by_part.get("heel")
        toe = by_part.get("toe")
        heel_text = "-" if heel is None else f"{heel:+.1f}px"
        toe_text = "-" if toe is None else f"{toe:+.1f}px"
        return f"{foot_name}: heel={heel_text} toe={toe_text}"

    def _candidate_text(self) -> str:
        candidate = self.first_contact or self.latest_candidate
        if candidate is None:
            return "-"
        suffix = "confirmed" if self.first_contact is not None else "candidate"
        return f"{candidate.foot} {candidate.part} contact {suffix} ({candidate.distance_px:+.1f}px)"

    def _estimate_ground_line(
        self,
        frame: np.ndarray,
        horizon_line: GroundLine | None,
        athlete_bbox: tuple[int, int, int, int] | None,
    ) -> GroundLine:
        if USE_MANUAL_GROUND_LINE:
            return GroundLine(
                slope=0.0,
                intercept=frame.shape[0] * MANUAL_GROUND_Y_RATIO,
                confidence=1.0,
                fallback=True,
                status="manual",
            )
        return self.ground_estimator.estimate(
            frame,
            horizon_line=horizon_line,
            athlete_bbox=athlete_bbox,
        )

    def _contact_detection_is_armed(self, frame: np.ndarray, detection: PoseDetection) -> bool:
        return detection.bbox[3] / max(frame.shape[0], 1) >= MIN_BODY_BOTTOM_RATIO_FOR_CONTACT

    def _ground_label(self, ground_line: GroundLine) -> str:
        if USE_MANUAL_GROUND_LINE:
            return "GROUND MANUAL"
        if ground_line.fallback:
            return "GROUND FALLBACK"
        if ground_line.status not in ("detected", "fallback"):
            return f"GROUND {ground_line.status.upper()}"
        return "GROUND"

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        if DISPLAY_SCALE == 1.0:
            return frame
        width = max(1, int(frame.shape[1] * DISPLAY_SCALE))
        height = max(1, int(frame.shape[0] * DISPLAY_SCALE))
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _open_writer(self, fps: float, width: int, height: int) -> cv2.VideoWriter:
        self.output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video writer: {self.output_video_path}")
        return writer

    def _resolve_path(self, path_text: str) -> Path:
        path = Path(path_text)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    def _reset_runtime_state(self) -> None:
        self.athlete_selector.reset()
        self.foot_part_estimator = FootPartEstimator()
        self.horizon_estimator = HorizonLineEstimator(
            HorizonLineConfig(max_candidate_y_ratio=HORIZON_MAX_Y_RATIO)
        )
        self.ground_estimator = GroundLineEstimator(
            GroundLineConfig(
                min_candidate_y_ratio=GROUND_MIN_Y_RATIO,
                min_gap_below_horizon_ratio=GROUND_MIN_GAP_BELOW_HORIZON_RATIO,
            )
        )
        self.left_trail.clear()
        self.right_trail.clear()
        self.current_track_id = None
        self.contact_streak.clear()
        self.first_contact = None
        self.first_contact_printed = False
        self.latest_distances = []
        self.latest_candidate = None
        self.latest_detection_mode = "NO PERSON"


def main() -> int:
    configure_logging()
    return MediaPipeFootContactBrowser().run()


if __name__ == "__main__":
    raise SystemExit(main())
