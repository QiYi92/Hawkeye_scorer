#!/usr/bin/env python3
"""OAK-D Lite 单目标完整姿态与左右脚双目测距演示。

YOLO Pose + ByteTrack 锁定主要人体；MediaPipe Pose Landmarker 在目标 ROI 内
输出包含 heel/foot-index 的 33 点姿态；对齐到 RGB 的双目深度用于计算相机
光心到左右脚测距点的三维直线距离。
"""

from __future__ import annotations

import argparse
import logging
import math
import ssl
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen, urlretrieve

import cv2
import numpy as np

try:
    import depthai as dai
except ImportError as exc:
    raise SystemExit("错误：未安装 depthai，请运行 python -m pip install -r requirements.txt") from exc


LOGGER = logging.getLogger("oak_live_full_pose")
WINDOW_NAME = "OAK-D Lite Foot Distance Tracking"
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_TOE, RIGHT_TOE = 31, 32
YOLO_BODY_LAYERS = ((5, 6), (11, 12), (13, 14), (15, 16))
MEDIAPIPE_BODY_LAYERS = (
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP),
    (LEFT_KNEE, RIGHT_KNEE), (LEFT_ANKLE, RIGHT_ANKLE),
)
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(x2 - x1, 0) * max(y2 - y1, 0)


@dataclass(frozen=True)
class MediaPipePose:
    points: tuple[tuple[int, int], ...]
    visibility: tuple[float, ...]


@dataclass(frozen=True)
class DepthSample:
    median_mm: float
    valid_ratio: float


@dataclass(frozen=True)
class FootMeasurement:
    side: str
    part: str
    point: tuple[int, int]
    depth_mm: float
    distance_m: float
    visibility: float
    valid_ratio: float


@dataclass(frozen=True)
class JumpLandingEvent:
    """第一次跳跃落地事件（位置使用相机坐标系，单位为米）。"""

    timestamp_s: float
    point: tuple[int, int]
    side: str
    part: str
    camera_xyz_m: tuple[float, float, float]


class FirstJumpLandingDetector:
    """从脚部纵向轨迹识别第一次完整的起跳和落地。"""

    def __init__(
        self,
        takeoff_ratio: float = 0.05,
        landing_ratio: float = 0.025,
        takeoff_frames: int = 2,
        landing_frames: int = 2,
        baseline_frames: int = 12,
        max_missing_frames: int = 3,
    ) -> None:
        self.takeoff_ratio = takeoff_ratio
        self.landing_ratio = landing_ratio
        self.takeoff_frames = takeoff_frames
        self.landing_frames = landing_frames
        self.baseline_frames = baseline_frames
        self.max_missing_frames = max_missing_frames
        self.reset()

    def reset(self, target_id: int | None = None) -> None:
        self.target_id = target_id
        self.state = "calibrating"
        self.event: JumpLandingEvent | None = None
        self._baseline: deque[float] = deque(maxlen=self.baseline_frames)
        self._airborne_count = 0
        self._landing_count = 0
        self._previous_y: float | None = None
        self._candidate: tuple[float, FootMeasurement] | None = None
        self._missing_frames = 0
        self._recent_contact: FootMeasurement | None = None
        self._recent_contact_age = self.max_missing_frames + 1

    @property
    def baseline_y(self) -> float | None:
        return float(np.median(self._baseline)) if self._baseline else None

    def update(
        self,
        measurements: dict[str, FootMeasurement | None],
        bbox_height: int,
        timestamp_s: float,
        intrinsics: CameraIntrinsics,
        target_id: int | None,
        pose: MediaPipePose | None = None,
        min_visibility: float = 0.45,
    ) -> JumpLandingEvent | None:
        if self.target_id is None:
            self.reset(target_id)
        elif target_id is not None:
            # ByteTrack 在快速动作中可能换 ID；保留已学习的基线和跳跃状态。
            self.target_id = target_id
        if self.event is not None:
            return self.event

        feet = [item for item in measurements.values() if item is not None]
        if feet:
            self._recent_contact = max(feet, key=lambda item: item.point[1])
            self._recent_contact_age = 0
        else:
            self._recent_contact_age += 1

        pose_feet = self._pose_feet(pose, min_visibility)
        if pose_feet:
            foot_y = float(np.mean([point[1] for _, _, point in pose_feet]))
        else:
            measured_feet = [item for item in measurements.values() if item is not None]
            if not measured_feet:
                self._missing_frames += 1
                if self._missing_frames > self.max_missing_frames and self.state == "airborne":
                    self._landing_count = 0
                    self._candidate = None
                return None
            foot_y = float(np.mean([item.point[1] for item in measured_feet]))
        self._missing_frames = 0

        # 深度只用于记录最终三维落点，不再阻断起跳和落地动作判断。
        lowest = max(feet, key=lambda item: item.point[1]) if feet else None
        if foot_y < 0:
            return None
        height = max(bbox_height, 1)
        takeoff_px = max(6.0, height * self.takeoff_ratio)
        landing_px = max(4.0, height * self.landing_ratio)

        if len(self._baseline) < self.baseline_frames:
            self._baseline.append(foot_y)
            self._previous_y = foot_y
            return None
        baseline = self.baseline_y
        assert baseline is not None

        if self.state in ("calibrating", "ready"):
            self.state = "ready"
            if foot_y <= baseline - takeoff_px:
                self._airborne_count += 1
                if self._airborne_count >= self.takeoff_frames:
                    self.state = "airborne"
            else:
                self._airborne_count = 0
                # 起跳前缓慢更新站立基线，适应小幅姿态与检测抖动。
                self._baseline.append(foot_y)
        elif self.state == "airborne":
            descending = self._previous_y is None or foot_y >= self._previous_y - 1.0
            near_ground = foot_y >= baseline - landing_px
            if descending and near_ground:
                self._landing_count += 1
                if self._candidate is None:
                    contact = lowest
                    if contact is None and self._recent_contact_age <= self.max_missing_frames:
                        contact = self._recent_contact
                    if contact is not None:
                        self._candidate = (timestamp_s, contact)
                if self._landing_count >= self.landing_frames and self._candidate is not None:
                    event_time, contact = self._candidate
                    xyz = pixel_to_camera_point(contact.point, contact.depth_mm, intrinsics)
                    self.event = JumpLandingEvent(
                        event_time, contact.point, contact.side, contact.part,
                        tuple(float(value) for value in xyz),
                    )
                    self.state = "landed"
            else:
                self._landing_count = 0
                self._candidate = None

        self._previous_y = foot_y
        return self.event

    @staticmethod
    def _pose_feet(
        pose: MediaPipePose | None, min_visibility: float
    ) -> list[tuple[str, str, tuple[int, int]]]:
        if pose is None:
            return []
        result: list[tuple[str, str, tuple[int, int]]] = []
        candidates = {
            "left": (("toe", LEFT_TOE), ("heel", LEFT_HEEL), ("ankle", LEFT_ANKLE)),
            "right": (("toe", RIGHT_TOE), ("heel", RIGHT_HEEL), ("ankle", RIGHT_ANKLE)),
        }
        for side, parts in candidates.items():
            # 固定按 toe -> heel -> ankle 回退，避免不同关键点之间来回切换造成纵坐标跳变。
            for part, index in parts:
                if pose.visibility[index] >= min_visibility:
                    result.append((side, part, pose.points[index]))
                    break
        return result


class LandingClipRecorder:
    """缓存落地前画面，并在事件发生后保存一段带标记的 MP4。"""

    def __init__(
        self,
        enabled: bool,
        output_dir: Path,
        fps: float,
        frame_size: tuple[int, int],
        before_frames: int = 300,
        after_frames: int = 60,
        jpeg_quality: int = 90,
    ) -> None:
        self.enabled = enabled
        self.output_dir = output_dir
        self.fps = fps
        self.frame_size = frame_size
        self.before_frames = before_frames
        self.after_frames = after_frames
        self.jpeg_quality = jpeg_quality
        self.buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=max(1, before_frames))
        self.writer: cv2.VideoWriter | None = None
        self.event_time_s: float | None = None
        self.output_path: Path | None = None
        self.completed = False
        self.frames_after_event = 0

    def add_frame(self, frame: np.ndarray, timestamp_s: float) -> Path | None:
        if not self.enabled or self.completed:
            return None
        if self.writer is not None:
            self.writer.write(frame)
            self.frames_after_event += 1
            if self.frames_after_event >= self.after_frames:
                return self.finish()
            return None
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if ok:
            self.buffer.append((timestamp_s, encoded))
        return None

    def trigger(self, event_time_s: float) -> None:
        if not self.enabled or self.writer is not None or self.completed:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_path = self.output_dir / f"demo10_landing_{stamp}.mp4"
        output_fps = self.fps
        if len(self.buffer) >= 2:
            elapsed = self.buffer[-1][0] - self.buffer[0][0]
            if elapsed > 0:
                # 使用实际姿态处理帧率，避免低于相机请求帧率时回放加速。
                output_fps = min(self.fps, max(1.0, (len(self.buffer) - 1) / elapsed))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, output_fps, self.frame_size
        )
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise RuntimeError(f"无法创建落地录像：{self.output_path}")
        for _, encoded in self.buffer:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                self.writer.write(decoded)
        self.buffer.clear()
        self.event_time_s = event_time_s
        self.frames_after_event = 0

    def finish(self) -> Path | None:
        if self.writer is None:
            return None
        self.writer.release()
        self.writer = None
        self.completed = True
        return self.output_path

    def close(self) -> Path | None:
        """退出时保存已经触发但尚未录满后段的录像。"""
        return self.finish()


def _socket(*names: str):
    for name in names:
        if hasattr(dai.CameraBoardSocket, name):
            return getattr(dai.CameraBoardSocket, name)
    raise RuntimeError(f"当前 DepthAI 版本不支持相机接口：{names}")


def build_rgb_depth_pipeline(width: int, height: int, fps: float, sync_threshold_ms: int) -> dai.Pipeline:
    """建立 RGB、对齐深度和设备端同步输出管线。"""
    pipeline = dai.Pipeline()
    color = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    sync = pipeline.create(dai.node.Sync)
    output = pipeline.create(dai.node.XLinkOut)
    output.setStreamName("rgb_depth")

    color.setBoardSocket(_socket("CAM_A", "RGB"))
    color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color.setPreviewSize(width, height)
    color.setInterleaved(False)
    color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    color.setFps(fps)

    left.setBoardSocket(_socket("CAM_B", "LEFT"))
    right.setBoardSocket(_socket("CAM_C", "RIGHT"))
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setFps(fps)
    right.setFps(fps)

    if hasattr(dai.node.StereoDepth.PresetMode, "DEFAULT"):
        preset = dai.node.StereoDepth.PresetMode.DEFAULT
    else:
        preset = dai.node.StereoDepth.PresetMode.HIGH_DENSITY
    stereo.setDefaultProfilePreset(preset)
    stereo.setDepthAlign(_socket("CAM_A", "RGB"))
    stereo.setOutputSize(width, height)
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)

    color.preview.link(sync.inputs["rgb"])
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(sync.inputs["depth"])
    sync.setSyncThreshold(timedelta(milliseconds=sync_threshold_ms))
    sync.setSyncAttempts(-1)
    sync.out.link(output.input)
    return pipeline


def read_rgb_intrinsics(device: dai.Device, width: int, height: int) -> CameraIntrinsics:
    matrix = device.readCalibration().getCameraIntrinsics(_socket("CAM_A", "RGB"), width, height)
    return CameraIntrinsics(
        fx=float(matrix[0][0]), fy=float(matrix[1][1]),
        cx=float(matrix[0][2]), cy=float(matrix[1][2]),
    )


def ensure_mediapipe_model(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载 MediaPipe 姿态模型：{path}")
    try:
        urlretrieve(MEDIAPIPE_MODEL_URL, path)
    except URLError:
        context = ssl._create_unverified_context()
        with urlopen(MEDIAPIPE_MODEL_URL, context=context) as response:
            path.write_bytes(response.read())
    return path


class PrimaryTargetSelector:
    """锁定面积最大的完整人体，并容忍短时漏检。"""

    def __init__(self, lost_frames: int) -> None:
        self.lost_frames = lost_frames
        self.track_id: int | None = None
        self.missing_frames = 0

    def select(self, people: list[TrackedPerson]) -> TrackedPerson | None:
        if self.track_id is not None:
            current = next((person for person in people if person.track_id == self.track_id), None)
            if current is not None:
                self.missing_frames = 0
                return current
            self.missing_frames += 1
            if self.missing_frames <= self.lost_frames:
                return None
            self.track_id = None
            self.missing_frames = 0

        if not people:
            return None
        selected = max(people, key=lambda person: (person.area, person.confidence))
        self.track_id = selected.track_id
        return selected


class FullPoseTracker:
    """YOLO 跟踪人体，MediaPipe 仅估计锁定目标 ROI 的完整姿态。"""

    def __init__(
        self,
        yolo_model_path: Path,
        mediapipe_model_path: Path,
        confidence: float,
        landmark_visibility: float,
        min_person_aspect: float,
        target_lost_frames: int,
        crop_margin_ratio: float = 0.12,
        pose_hold_frames: int = 6,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit("错误：缺少 ultralytics 或 mediapipe，请安装 requirements.txt") from exc

        self._mp = mp
        self._yolo = YOLO(str(yolo_model_path))
        self._confidence = confidence
        self._landmark_visibility = landmark_visibility
        self._min_person_aspect = min_person_aspect
        self._crop_margin_ratio = crop_margin_ratio
        self._pose_hold_frames = pose_hold_frames
        self._last_pose: MediaPipePose | None = None
        self._pose_missing_frames = 0
        self._last_pose_target_id: int | None = None
        self.target_selector = PrimaryTargetSelector(target_lost_frames)
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(ensure_mediapipe_model(mediapipe_model_path)),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=confidence,
            min_pose_presence_confidence=confidence,
            min_tracking_confidence=confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def _is_complete_body(
        points: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        bbox: tuple[int, int, int, int],
        min_aspect: float,
        body_layers: tuple[tuple[int, int], ...],
        nose_index: int,
    ) -> bool:
        x1, y1, x2, y2 = bbox
        box_width, box_height = max(x2 - x1, 1), max(y2 - y1, 1)
        if box_height / box_width < min_aspect:
            return False
        visible_count = 0
        layer_y: list[float] = []
        for pair in body_layers:
            visible = [index for index in pair if scores[index] >= threshold]
            if not visible:
                return False
            visible_count += len(visible)
            layer_y.append(float(np.median(points[visible, 1])))
        if visible_count < 7:
            return False
        tolerance = box_height * 0.06
        if not all(upper + tolerance < lower for upper, lower in zip(layer_y, layer_y[1:])):
            return False
        top_candidates = [nose_index, *body_layers[0]]
        visible_top = [float(points[index, 1]) for index in top_candidates if scores[index] >= threshold]
        return bool(visible_top) and layer_y[-1] - min(visible_top) >= box_height * 0.58

    def process(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> tuple[list[TrackedPerson], TrackedPerson | None, MediaPipePose | None]:
        results = self._yolo.track(
            source=frame, persist=True, tracker="bytetrack.yaml",
            conf=self._confidence, verbose=False,
        )
        people = self._tracked_people(results)
        target = self.target_selector.select(people)
        if target is None:
            return people, None, None
        pose = self._estimate_target_pose(frame, target, timestamp_ms)
        if target.track_id != self._last_pose_target_id:
            self._last_pose = None
            self._pose_missing_frames = 0
            self._last_pose_target_id = target.track_id
        if pose is not None:
            self._last_pose = pose
            self._pose_missing_frames = 0
        elif self._last_pose is not None and self._pose_missing_frames < self._pose_hold_frames:
            self._pose_missing_frames += 1
            pose = self._last_pose
        return people, target, pose

    def _tracked_people(self, results) -> list[TrackedPerson]:
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes, keypoints = results[0].boxes, results[0].keypoints
        if keypoints is None or keypoints.xy is None or keypoints.conf is None:
            return []
        ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(boxes))
        coords = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        keypoint_xy = keypoints.xy.detach().cpu().numpy()
        keypoint_conf = keypoints.conf.detach().cpu().numpy()
        people: list[TrackedPerson] = []
        for index in range(len(boxes)):
            bbox = tuple(int(round(value)) for value in coords[index])
            is_complete = self._is_complete_body(
                keypoint_xy[index], keypoint_conf[index], self._landmark_visibility,
                bbox, self._min_person_aspect, YOLO_BODY_LAYERS, 0,
            )
            # 锁定后允许起跳、蜷腿和快速落地等非站立姿态继续进入跟踪。
            if is_complete or int(ids[index]) == self.target_selector.track_id:
                people.append(TrackedPerson(int(ids[index]), bbox, float(confidences[index])))
        return people

    def _estimate_target_pose(
        self, frame: np.ndarray, target: TrackedPerson, timestamp_ms: int
    ) -> MediaPipePose | None:
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = target.bbox
        margin = int(max(x2 - x1, y2 - y1) * self._crop_margin_ratio)
        left, top = max(0, x1 - margin), max(0, y1 - margin)
        right, bottom = min(frame_width, x2 + margin), min(frame_height, y2 + margin)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        crop_height, crop_width = crop.shape[:2]
        points = tuple(
            (
                int(round(point.x * crop_width + left)),
                int(round(point.y * crop_height + top)),
            )
            for point in landmarks
        )
        visibility = tuple(float(point.visibility) for point in landmarks)
        score_array = np.asarray(visibility)
        # 动作过程中不再套用“完整站立人体”的层级/高宽比过滤，只要求有足够关键点。
        if int(np.count_nonzero(score_array >= self._landmark_visibility)) < 8:
            return None
        return MediaPipePose(points, visibility)

    def close(self) -> None:
        self._landmarker.close()


def map_point_between_frames(
    point: tuple[int, int], source_size: tuple[int, int], target_size: tuple[int, int]
) -> tuple[int, int]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    x = int(round(point[0] * target_width / max(source_width, 1)))
    y = int(round(point[1] * target_height / max(source_height, 1)))
    return min(max(x, 0), target_width - 1), min(max(y, 0), target_height - 1)


def sample_depth(
    depth_frame: np.ndarray,
    point: tuple[int, int],
    box_size: int,
    min_depth_mm: int,
    max_depth_mm: int,
) -> DepthSample | None:
    height, width = depth_frame.shape[:2]
    x, y = point
    half = max(box_size // 2, 1)
    sample = depth_frame[
        max(y - half, 0):min(y + half + 1, height),
        max(x - half, 0):min(x + half + 1, width),
    ]
    if sample.size == 0:
        return None
    valid = sample[(sample >= min_depth_mm) & (sample <= max_depth_mm)]
    if valid.size == 0:
        return None
    return DepthSample(float(np.median(valid)), float(valid.size / sample.size))


def pixel_to_camera_point(
    point: tuple[int, int], depth_mm: float, intrinsics: CameraIntrinsics
) -> np.ndarray:
    z = depth_mm / 1000.0
    x = (point[0] - intrinsics.cx) * z / intrinsics.fx
    y = (point[1] - intrinsics.cy) * z / intrinsics.fy
    return np.asarray((x, y, z), dtype=np.float64)


def camera_distance_m(
    point: tuple[int, int], depth_mm: float, intrinsics: CameraIntrinsics
) -> float:
    return float(np.linalg.norm(pixel_to_camera_point(point, depth_mm, intrinsics)))


class FootDistanceEstimator:
    """为左右脚自动选择 heel/toe，并平滑三维直线距离。"""

    CANDIDATES = {
        "left": (("toe", LEFT_TOE), ("heel", LEFT_HEEL)),
        "right": (("toe", RIGHT_TOE), ("heel", RIGHT_HEEL)),
    }

    def __init__(
        self,
        box_size: int,
        min_depth_mm: int,
        max_depth_mm: int,
        min_valid_ratio: float,
        smoothing_frames: int,
        max_distance_jump_m: float,
        max_missing_frames: int = 3,
    ) -> None:
        self.box_size = box_size
        self.min_depth_mm = min_depth_mm
        self.max_depth_mm = max_depth_mm
        self.min_valid_ratio = min_valid_ratio
        self.smoothing_frames = smoothing_frames
        self.max_distance_jump_m = max_distance_jump_m
        self.max_missing_frames = max_missing_frames
        self.histories = {
            "left": deque(maxlen=smoothing_frames),
            "right": deque(maxlen=smoothing_frames),
        }
        self.missing = {"left": 0, "right": 0}
        self.target_id: int | None = None

    def reset_for_target(self, target_id: int | None) -> None:
        if target_id == self.target_id:
            return
        self.target_id = target_id
        for history in self.histories.values():
            history.clear()
        self.missing = {"left": 0, "right": 0}

    def estimate(
        self,
        pose: MediaPipePose | None,
        rgb_shape: tuple[int, ...],
        depth_frame: np.ndarray,
        intrinsics: CameraIntrinsics,
        min_visibility: float,
    ) -> dict[str, FootMeasurement | None]:
        results: dict[str, FootMeasurement | None] = {}
        for side in ("left", "right"):
            measurement = self._estimate_side(
                side, pose, rgb_shape, depth_frame, intrinsics, min_visibility
            )
            if measurement is None:
                self.missing[side] += 1
                if self.missing[side] > self.max_missing_frames:
                    self.histories[side].clear()
                results[side] = None
            else:
                self.missing[side] = 0
                self.histories[side].append(measurement.distance_m)
                smoothed = float(np.median(self.histories[side]))
                results[side] = FootMeasurement(
                    measurement.side, measurement.part, measurement.point,
                    measurement.depth_mm, smoothed, measurement.visibility,
                    measurement.valid_ratio,
                )
        return results

    def _estimate_side(
        self,
        side: str,
        pose: MediaPipePose | None,
        rgb_shape: tuple[int, ...],
        depth_frame: np.ndarray,
        intrinsics: CameraIntrinsics,
        min_visibility: float,
    ) -> FootMeasurement | None:
        if pose is None:
            return None
        rgb_height, rgb_width = rgb_shape[:2]
        depth_height, depth_width = depth_frame.shape[:2]
        previous = float(np.median(self.histories[side])) if self.histories[side] else None
        candidates: list[tuple[float, FootMeasurement]] = []
        for part, index in self.CANDIDATES[side]:
            visibility = pose.visibility[index]
            if visibility < min_visibility:
                continue
            rgb_point = pose.points[index]
            depth_point = map_point_between_frames(
                rgb_point, (rgb_width, rgb_height), (depth_width, depth_height)
            )
            sample = sample_depth(
                depth_frame, depth_point, self.box_size,
                self.min_depth_mm, self.max_depth_mm,
            )
            if sample is None or sample.valid_ratio < self.min_valid_ratio:
                continue
            distance = camera_distance_m(rgb_point, sample.median_mm, intrinsics)
            if previous is not None and abs(distance - previous) > self.max_distance_jump_m:
                continue
            continuity_bonus = 0.0 if previous is None else max(
                0.0, 1.0 - abs(distance - previous) / max(self.max_distance_jump_m, 1e-6)
            )
            score = visibility * 0.55 + sample.valid_ratio * 0.35 + continuity_bonus * 0.10
            candidates.append((score, FootMeasurement(
                side, part, rgb_point, sample.median_mm, distance,
                visibility, sample.valid_ratio,
            )))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None


def draw_pose(
    frame: np.ndarray,
    target: TrackedPerson,
    pose: MediaPipePose | None,
    measurements: dict[str, FootMeasurement | None],
    min_visibility: float,
) -> None:
    x1, y1, x2, y2 = target.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
    cv2.putText(frame, f"PRIMARY ID {target.track_id} {target.confidence:.2f}",
                (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 220, 40), 2)
    if pose is None:
        return
    for start, end in POSE_CONNECTIONS:
        if pose.visibility[start] >= min_visibility and pose.visibility[end] >= min_visibility:
            cv2.line(frame, pose.points[start], pose.points[end], (90, 220, 90), 2, cv2.LINE_AA)
    for index, point in enumerate(pose.points):
        if pose.visibility[index] >= min_visibility:
            cv2.circle(frame, point, 3, (255, 180, 40), -1, cv2.LINE_AA)

    colors = {"left": (255, 80, 255), "right": (0, 230, 255)}
    for side in ("left", "right"):
        measurement = measurements[side]
        if measurement is None:
            continue
        color = colors[side]
        point = measurement.point
        label = f"{side[0].upper()} {measurement.part} {measurement.distance_m:.3f}m"
        cv2.circle(frame, point, 8, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (point[0] + 9, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def draw_landing_event(frame: np.ndarray, event: JumpLandingEvent | None, state: str) -> None:
    """持续标记第一次落点，并显示检测状态、时间和三维位置。"""
    if event is None:
        labels = {
            "calibrating": "JUMP: CALIBRATING",
            "ready": "JUMP: READY",
            "airborne": "JUMP: AIRBORNE",
        }
        cv2.putText(frame, labels.get(state, f"JUMP: {state.upper()}"), (12, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 255), 2, cv2.LINE_AA)
        return
    x, y = event.point
    color = (0, 0, 255)
    cv2.drawMarker(frame, (x, y), color, cv2.MARKER_TILTED_CROSS, 28, 4, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 16, color, 2, cv2.LINE_AA)
    px, py, pz = event.camera_xyz_m
    cv2.putText(frame, f"FIRST LANDING {event.timestamp_s:.3f}s",
                (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{event.side} {event.part} XYZ=({px:.2f},{py:.2f},{pz:.2f})m",
                (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def require_oak_device() -> None:
    """在加载推理模型前确认 OAK 已被 DepthAI 枚举到。"""
    devices = dai.Device.getAllAvailableDevices()
    if devices:
        return
    raise SystemExit(
        "未发现 OAK/DepthAI 设备，Demo 10 需要已连接的 OAK-D Lite。\n"
        "请按以下顺序检查：\n"
        "  1. 拔插 OAK-D Lite，并直接连接 Mac（不要经过无供电扩展坞）；\n"
        "  2. 更换支持数据传输的 USB-C 线缆和 USB-C 接口；\n"
        "  3. 关闭可能占用相机的其他 Python 程序、DepthAI Viewer 或 Luxonis 工具；\n"
        "  4. 运行 01_demo_OAK设备检测.py，确认“可用设备数量”大于 0。"
    )


def run(args: argparse.Namespace) -> int:
    if not args.yolo_model.is_file():
        raise SystemExit(f"找不到 YOLO 姿态模型：{args.yolo_model}")
    require_oak_device()
    tracker = FullPoseTracker(
        args.yolo_model, args.mediapipe_model, args.confidence,
        args.landmark_visibility, args.min_person_aspect, args.target_lost_frames,
    )
    distance_estimator = FootDistanceEstimator(
        args.depth_box_size, args.min_depth_mm, args.max_depth_mm,
        args.min_depth_valid_ratio, args.smoothing_frames, args.max_distance_jump,
    )
    landing_detector = FirstJumpLandingDetector(
        args.jump_takeoff_ratio, args.jump_landing_ratio,
        args.jump_takeoff_frames, args.jump_landing_frames, args.jump_baseline_frames,
    )
    clip_recorder = LandingClipRecorder(
        args.record_landing, args.record_output_dir, args.fps,
        (args.width, args.height), args.record_before_frames, args.record_after_frames,
    )
    pipeline = build_rgb_depth_pipeline(args.width, args.height, args.fps, args.sync_threshold_ms)
    frame_count = 0
    started_at = time.monotonic()
    display_fps = 0.0
    max_people = 0
    valid_left = 0
    valid_right = 0

    try:
        with dai.Device(pipeline) as device:
            intrinsics = read_rgb_intrinsics(device, args.width, args.height)
            print(f"已连接 OAK 设备：{device.getMxId()}")
            print(f"USB 速度：{device.getUsbSpeed()}")
            print(f"RGB内参：fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}")
            print("左右脚各显示一个自动选择的 heel/toe 三维直线距离；按 r 重置落地检测，按 q 或 Esc 退出。")
            queue = device.getOutputQueue(name="rgb_depth", maxSize=1, blocking=False)
            first_device_timestamp_s: float | None = None
            while True:
                group = queue.tryGet()
                if group is None:
                    time.sleep(0.001)
                    continue
                rgb_message = group["rgb"]
                depth_message = group["depth"]
                frame = rgb_message.getCvFrame()
                depth_frame = depth_message.getFrame()
                frame_count += 1
                timestamp_ms = int(rgb_message.getTimestampDevice().total_seconds() * 1000)
                device_timestamp_s = timestamp_ms / 1000.0
                if first_device_timestamp_s is None:
                    first_device_timestamp_s = device_timestamp_s
                video_time_s = device_timestamp_s - first_device_timestamp_s
                people, target, pose = tracker.process(frame, timestamp_ms)
                max_people = max(max_people, len(people))
                target_id = target.track_id if target is not None else tracker.target_selector.track_id
                distance_estimator.reset_for_target(target_id)
                measurements = distance_estimator.estimate(
                    pose, frame.shape, depth_frame, intrinsics, args.landmark_visibility
                )
                valid_left += int(measurements["left"] is not None)
                valid_right += int(measurements["right"] is not None)
                if target is not None:
                    draw_pose(frame, target, pose, measurements, args.landmark_visibility)
                    bbox_height = max(target.bbox[3] - target.bbox[1], 1)
                    event_before = landing_detector.event
                    event = landing_detector.update(
                        measurements, bbox_height, video_time_s, intrinsics, target_id,
                        pose, args.landmark_visibility,
                    )
                    if event is not None and event_before is None:
                        x, y, z = event.camera_xyz_m
                        print(
                            f"首次跳跃落地：t={event.timestamp_s:.3f}s，"
                            f"{event.side} {event.part}，像素={event.point}，"
                            f"相机坐标=({x:.3f}, {y:.3f}, {z:.3f})m"
                        )
                draw_landing_event(frame, landing_detector.event, landing_detector.state)

                if landing_detector.event is not None and clip_recorder.event_time_s is None:
                    clip_recorder.trigger(landing_detector.event.timestamp_s)
                    print(f"开始保存落地录像：{clip_recorder.output_path}")

                elapsed = max(time.monotonic() - started_at, 0.001)
                display_fps = frame_count / elapsed
                target_text = "-" if target_id is None else str(target_id)
                depth_status = "L:{} R:{}".format(
                    "OK" if measurements["left"] else "--",
                    "OK" if measurements["right"] else "--",
                )
                cv2.putText(frame, f"target={target_text} fps={display_fps:.1f} depth {depth_status}",
                            (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
                cv2.putText(frame, f"camera height={args.camera_height_m:.2f}m (reserved)",
                            (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                saved_path = clip_recorder.add_frame(frame, video_time_s)
                if saved_path is not None:
                    print(f"落地录像已保存：{saved_path}")
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1)
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    landing_detector.reset(target_id)
                    print("已重置第一次跳跃落地检测。")
                if args.duration is not None and elapsed >= args.duration:
                    break
    finally:
        partial_path = clip_recorder.close()
        if partial_path is not None:
            print(f"落地录像已保存（提前退出，落地后帧数可能不足 {args.record_after_frames} 帧）：{partial_path}")
        tracker.close()
        cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started_at, 0.001)
    print(
        f"运行汇总：同步帧={frame_count}，平均处理={frame_count / elapsed:.1f} fps，"
        f"最大完整人体={max_people}，左脚有效帧={valid_left}，右脚有效帧={valid_right}"
    )
    return 0 if frame_count > 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OAK-D Lite 单目标完整姿态与左右脚双目测距。")
    parser.add_argument("--yolo-model", type=Path, default=Path("yolo11n-pose.pt"))
    parser.add_argument("--mediapipe-model", type=Path, default=Path(".cache/mediapipe/pose_landmarker_lite.task"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--landmark-visibility", type=float, default=0.45)
    parser.add_argument("--min-person-aspect", type=float, default=0.85)
    parser.add_argument("--target-lost-frames", type=int, default=15)
    parser.add_argument("--camera-height-m", type=float, default=0.80)
    parser.add_argument("--depth-box-size", type=int, default=9)
    parser.add_argument("--min-depth-mm", type=int, default=200)
    parser.add_argument("--max-depth-mm", type=int, default=8000)
    parser.add_argument("--min-depth-valid-ratio", type=float, default=0.30)
    parser.add_argument("--smoothing-frames", type=int, default=5)
    parser.add_argument("--max-distance-jump", type=float, default=0.60)
    parser.add_argument("--sync-threshold-ms", type=int, default=25)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--jump-takeoff-ratio", type=float, default=0.05)
    parser.add_argument("--jump-landing-ratio", type=float, default=0.025)
    parser.add_argument("--jump-takeoff-frames", type=int, default=2)
    parser.add_argument("--jump-landing-frames", type=int, default=2)
    parser.add_argument("--jump-baseline-frames", type=int, default=12)
    parser.add_argument("--record-landing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-output-dir", type=Path, default=Path("output/demo10_landing_clips"))
    parser.add_argument("--record-before-frames", type=int, default=300)
    parser.add_argument("--record-after-frames", type=int, default=60)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.width < 1 or args.height < 1 or args.fps <= 0:
        raise SystemExit("画面宽高和帧率必须大于 0。")
    if args.camera_height_m <= 0:
        raise SystemExit("摄像头高度必须大于 0。")
    if args.depth_box_size < 1 or args.depth_box_size % 2 == 0:
        raise SystemExit("深度采样框必须是大于 0 的奇数。")
    if args.min_depth_mm < 0 or args.max_depth_mm <= args.min_depth_mm:
        raise SystemExit("有效深度范围不正确。")
    if not 0 < args.min_depth_valid_ratio <= 1:
        raise SystemExit("深度有效比例必须在 (0, 1] 范围内。")
    if args.smoothing_frames < 1 or args.target_lost_frames < 0:
        raise SystemExit("平滑帧数必须大于 0，目标丢失帧数不能小于 0。")
    if not 0 < args.jump_landing_ratio < args.jump_takeoff_ratio < 1:
        raise SystemExit("跳跃阈值需满足 0 < 落地比例 < 起跳比例 < 1。")
    if min(args.jump_takeoff_frames, args.jump_landing_frames, args.jump_baseline_frames) < 1:
        raise SystemExit("跳跃检测所需帧数必须大于 0。")
    if args.record_before_frames < 1 or args.record_after_frames < 1:
        raise SystemExit("落地录像前后帧数必须大于 0。")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    validate_args(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
