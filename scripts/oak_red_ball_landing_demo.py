#!/usr/bin/env python3
"""OAK-D Lite 红色小球实时双目测距与首次落地检测。"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

try:
    import depthai as dai
except ImportError as exc:
    raise SystemExit("错误：未安装 depthai，请运行 python -m pip install -r requirements.txt") from exc


WINDOW_NAME = "OAK-D Lite Red Ball Landing"


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class BallDetection:
    center: tuple[int, int]
    radius_px: float
    area: float
    circularity: float
    contour: np.ndarray


@dataclass(frozen=True)
class DepthSample:
    median_mm: float
    valid_ratio: float


@dataclass(frozen=True)
class GroundPlane:
    normal: np.ndarray
    offset: float
    inlier_ratio: float
    rms_error_m: float

    @property
    def camera_height_m(self) -> float:
        return abs(self.offset)


@dataclass(frozen=True)
class BallMeasurement:
    frame: int
    elapsed_s: float
    device_timestamp_s: float
    wall_time: str
    pixel_x: int
    pixel_y: int
    depth_m: float
    radius_m: float
    camera_x_m: float
    camera_y_m: float
    camera_z_m: float
    bottom_height_m: float | None
    ground_x_m: float | None
    ground_forward_m: float | None
    contact_x_m: float | None
    contact_y_m: float | None
    contact_z_m: float | None
    camera_distance_m: float
    vertical_velocity_mps: float | None
    state: str


def _socket(*names: str):
    for name in names:
        if hasattr(dai.CameraBoardSocket, name):
            return getattr(dai.CameraBoardSocket, name)
    raise RuntimeError(f"当前 DepthAI 版本不支持相机接口：{names}")


def build_pipeline(width: int, height: int, fps: float, sync_threshold_ms: int) -> dai.Pipeline:
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


def read_intrinsics(device: dai.Device, width: int, height: int) -> CameraIntrinsics:
    matrix = device.readCalibration().getCameraIntrinsics(_socket("CAM_A", "RGB"), width, height)
    return CameraIntrinsics(float(matrix[0][0]), float(matrix[1][1]), float(matrix[0][2]), float(matrix[1][2]))


def pixel_to_camera(point: tuple[int, int], depth_m: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    return np.asarray(((point[0] - intrinsics.cx) * depth_m / intrinsics.fx,
                       (point[1] - intrinsics.cy) * depth_m / intrinsics.fy, depth_m), dtype=np.float64)


def detect_red_candidates(
    frame: np.ndarray, red1_low: tuple[int, int, int], red1_high: tuple[int, int, int],
    red2_low: tuple[int, int, int], red2_high: tuple[int, int, int], min_area: float,
    min_circularity: float,
) -> tuple[list[BallDetection], np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray(red1_low, np.uint8), np.asarray(red1_high, np.uint8))
    mask |= cv2.inRange(hsv, np.asarray(red2_low, np.uint8), np.asarray(red2_high, np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[BallDetection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < min_area or perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        fill_ratio = area / max(math.pi * radius * radius, 1.0)
        if circularity >= min_circularity and fill_ratio >= 0.55 and radius >= 3:
            found.append(BallDetection((int(round(x)), int(round(y))), float(radius), area,
                                       circularity, contour))
    return found, mask


class BallTracker:
    def __init__(self, max_missing_frames: int) -> None:
        self.max_missing_frames = max_missing_frames
        self.previous: BallDetection | None = None
        self.velocity = np.zeros(2, dtype=np.float64)
        self.missing = 0

    def reset(self) -> None:
        self.previous = None
        self.velocity[:] = 0
        self.missing = 0

    def select(self, candidates: list[BallDetection]) -> BallDetection | None:
        if not candidates:
            self.missing += 1
            if self.missing > self.max_missing_frames:
                self.reset()
            return None
        if self.previous is None:
            chosen = max(candidates, key=lambda item: (item.circularity, item.area))
        else:
            predicted = np.asarray(self.previous.center) + self.velocity
            diagonal = max(self.previous.radius_px * 8.0, 60.0)
            scored = [(np.linalg.norm(np.asarray(item.center) - predicted) / diagonal
                       + abs(math.log(max(item.radius_px, 1) / max(self.previous.radius_px, 1))) * 0.35
                       - item.circularity * 0.15, item) for item in candidates]
            score, chosen = min(scored, key=lambda pair: pair[0])
            if score > 2.0:
                self.missing += 1
                return None
            delta = np.asarray(chosen.center) - np.asarray(self.previous.center)
            self.velocity = self.velocity * 0.55 + delta * 0.45
        self.previous = chosen
        self.missing = 0
        return chosen


def skin_contact_ratio(frame: np.ndarray, detection: BallDetection, ring_scale: float) -> float:
    """估计紧贴球边缘的肤色比例。

    只检查球轮廓外的环形区域，避免红球本身被肤色阈值计入。
    """
    height, width = frame.shape[:2]
    x, y = detection.center
    outer_radius = max(int(round(detection.radius_px * ring_scale)), 2)
    inner_radius = max(int(round(detection.radius_px * 1.02)), 1)
    x1, x2 = max(0, x - outer_radius), min(width, x + outer_radius + 1)
    y1, y2 = max(0, y - outer_radius), min(height, y + outer_radius + 1)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    yy, xx = np.ogrid[y1:y2, x1:x2]
    distance_sq = (xx - x) ** 2 + (yy - y) ** 2
    ring = (distance_sq <= outer_radius ** 2) & (distance_sq >= inner_radius ** 2)
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    # 宽松的通用肤色范围；只用于判断球周围是否有手，不做人体识别。
    skin = cv2.inRange(ycrcb, np.asarray((40, 133, 77), np.uint8),
                      np.asarray((255, 180, 135), np.uint8)) > 0
    pixels = int(ring.sum())
    return 0.0 if pixels == 0 else float(np.count_nonzero(skin & ring) / pixels)


class HandReleaseGate:
    """只有观察到“被手握住→连续脱离手部”后才解锁落地判定。"""

    def __init__(self, required: bool, release_frames: int, skin_threshold: float) -> None:
        self.required = required
        self.release_frames = release_frames
        self.skin_threshold = skin_threshold
        self.reset()

    def reset(self) -> None:
        self.seen_held = False
        self.free_frames = 0
        self.released = not self.required

    def update(self, skin_ratio: float) -> bool:
        if self.released:
            return True
        touching = skin_ratio >= self.skin_threshold
        if touching:
            self.seen_held = True
            self.free_frames = 0
        elif self.seen_held:
            self.free_frames += 1
            if self.free_frames >= self.release_frames:
                self.released = True
        return self.released

    @property
    def status(self) -> str:
        if self.released:
            return "RELEASED"
        return "HELD" if self.seen_held else "WAITING FOR HAND"


def sample_ball_depth(depth: np.ndarray, detection: BallDetection, inner_ratio: float,
                      min_mm: int, max_mm: int) -> DepthSample | None:
    x, y = detection.center
    radius = max(2, int(round(detection.radius_px * inner_ratio)))
    yy, xx = np.ogrid[:depth.shape[0], :depth.shape[1]]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    sample = depth[inside]
    if sample.size == 0:
        return None
    valid = sample[(sample >= min_mm) & (sample <= max_mm)]
    if valid.size == 0:
        return None
    return DepthSample(float(np.median(valid)), float(valid.size / sample.size))


def estimate_ball_geometry(detection: BallDetection, front_depth_mm: float,
                           intrinsics: CameraIntrinsics) -> tuple[np.ndarray, float]:
    front_depth_m = front_depth_mm / 1000.0
    focal = (intrinsics.fx + intrinsics.fy) * 0.5
    radius_m = detection.radius_px * front_depth_m / max(focal - detection.radius_px, 1.0)
    center_depth_m = front_depth_m + radius_m
    return pixel_to_camera(detection.center, center_depth_m, intrinsics), radius_m


def depth_to_points(depth: np.ndarray, intrinsics: CameraIntrinsics, start_ratio: float,
                    min_mm: int, max_mm: int, stride: int = 6) -> np.ndarray:
    start_y = int(depth.shape[0] * start_ratio)
    ys, xs = np.mgrid[start_y:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[start_y:depth.shape[0]:stride, 0:depth.shape[1]:stride].astype(np.float64) / 1000.0
    valid = (z >= min_mm / 1000.0) & (z <= max_mm / 1000.0)
    z, xs, ys = z[valid], xs[valid], ys[valid]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack(((xs - intrinsics.cx) * z / intrinsics.fx,
                            (ys - intrinsics.cy) * z / intrinsics.fy, z))


def fit_ground_plane(points: np.ndarray, camera_height_m: float, distance_threshold_m: float,
                     min_inlier_ratio: float, height_tolerance_m: float,
                     iterations: int = 180, seed: int = 7) -> GroundPlane | None:
    if len(points) < 80:
        return None
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-8:
            continue
        normal /= length
        offset = -float(np.dot(normal, sample[0]))
        distances = np.abs(points @ normal + offset)
        inliers = distances <= distance_threshold_m
        if best is None or int(inliers.sum()) > int(best.sum()):
            best = inliers
    if best is None or float(best.mean()) < min_inlier_ratio:
        return None
    selected = points[best]
    centroid = selected.mean(axis=0)
    _, _, vh = np.linalg.svd(selected - centroid, full_matrices=False)
    normal = vh[-1]
    offset = -float(np.dot(normal, centroid))
    if offset < 0:
        normal, offset = -normal, -offset
    distances = points @ normal + offset
    inliers = np.abs(distances) <= distance_threshold_m
    ratio = float(inliers.mean())
    rms = float(np.sqrt(np.mean(np.square(distances[inliers]))))
    # 地面法线必须大致朝向相机上方（相机 Y 轴向下）。
    if normal[1] > -0.20 or abs(offset - camera_height_m) > height_tolerance_m:
        return None
    return GroundPlane(normal, offset, ratio, rms)


def signed_plane_distance(point: np.ndarray, plane: GroundPlane) -> float:
    return float(np.dot(plane.normal, point) + plane.offset)


def project_to_plane(point: np.ndarray, plane: GroundPlane) -> np.ndarray:
    return point - signed_plane_distance(point, plane) * plane.normal


def ground_coordinates(point: np.ndarray, plane: GroundPlane) -> tuple[float, float]:
    origin = project_to_plane(np.zeros(3), plane)
    x_axis = np.asarray((1.0, 0.0, 0.0))
    x_axis -= np.dot(x_axis, plane.normal) * plane.normal
    x_axis /= np.linalg.norm(x_axis)
    forward = np.asarray((0.0, 0.0, 1.0))
    forward -= np.dot(forward, plane.normal) * plane.normal
    forward -= np.dot(forward, x_axis) * x_axis
    forward /= np.linalg.norm(forward)
    relative = point - origin
    return float(np.dot(relative, x_axis)), float(np.dot(relative, forward))


class LandingDetector:
    STATES = {"waiting": "WAITING", "tracked": "TRACKED", "airborne": "AIRBORNE",
              "contact_candidate": "CONTACT?", "landed": "LANDED"}

    def __init__(self, airborne_height_m: float, contact_height_m: float, contact_frames: int,
                 min_down_speed_mps: float, max_contact_down_speed_mps: float) -> None:
        self.airborne_height_m = airborne_height_m
        self.contact_height_m = contact_height_m
        self.contact_frames = contact_frames
        self.min_down_speed_mps = min_down_speed_mps
        self.max_contact_down_speed_mps = max_contact_down_speed_mps
        self.reset()

    def reset(self) -> None:
        self.state = "waiting"
        self.previous_height: float | None = None
        self.previous_time: float | None = None
        self.was_airborne = False
        self.strongest_down_speed = 0.0
        self.candidates: list[BallMeasurement] = []
        self.event: BallMeasurement | None = None
        self.confirmation_frame: int | None = None
        self.last_velocity: float | None = None

    def update(self, measurement: BallMeasurement) -> tuple[str, BallMeasurement | None]:
        height, now = measurement.bottom_height_m, measurement.elapsed_s
        if height is None:
            return self.state, self.event
        velocity = None
        if self.previous_height is not None and self.previous_time is not None and now > self.previous_time:
            velocity = (height - self.previous_height) / (now - self.previous_time)
            self.strongest_down_speed = min(self.strongest_down_speed, velocity)
        self.last_velocity = velocity
        self.previous_height, self.previous_time = height, now
        measurement = BallMeasurement(**{**asdict(measurement), "vertical_velocity_mps": velocity})
        if self.event is not None:
            self.state = "landed"
            return self.state, self.event
        if height > self.airborne_height_m:
            self.was_airborne = True
            self.state = "airborne" if self.strongest_down_speed <= -self.min_down_speed_mps else "tracked"
            self.candidates.clear()
            return self.state, None
        descending_before = self.strongest_down_speed <= -self.min_down_speed_mps
        if self.was_airborne and descending_before and height <= self.contact_height_m:
            self.candidates.append(measurement)
            self.state = "contact_candidate"
            if len(self.candidates) >= self.contact_frames:
                recent_velocity = velocity if velocity is not None else -math.inf
                if recent_velocity >= -self.max_contact_down_speed_mps - 1e-9:
                    self.event = self.candidates[0]
                    self.confirmation_frame = measurement.frame
                    self.state = "landed"
            return self.state, self.event
        self.state = "tracked"
        self.candidates.clear()
        return self.state, None


def measurement_dict(item: BallMeasurement | None) -> dict | None:
    return None if item is None else asdict(item)


def write_result(path: Path, event: BallMeasurement | None, trajectory: list[dict],
                 plane: GroundPlane | None, args: argparse.Namespace, usb_speed: str,
                 confirmation_frame: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "demo": 11, "camera": "OAK-D Lite", "usb_speed": usb_speed,
        "camera_height_m": args.camera_height_m,
        "ground_plane": None if plane is None else {
            "normal": plane.normal.tolist(), "offset": plane.offset,
            "inlier_ratio": plane.inlier_ratio, "rms_error_m": plane.rms_error_m,
        },
        "first_landing": measurement_dict(event),
        "trajectory": trajectory if args.save_trajectory else [],
        "note": "Engineering demo only; not an official judging result.",
    }
    if data["first_landing"] is not None:
        data["first_landing"]["status"] = "confirmed"
        data["first_landing"]["confirmation_frame"] = confirmation_frame
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_triplet(values: list[str]) -> tuple[int, int, int]:
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def run(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args.width, args.height, args.fps, args.sync_threshold_ms)
    tracker = BallTracker(args.max_missing_frames)
    release_gate = HandReleaseGate(args.require_hand_release, args.release_frames,
                                   args.skin_ratio_threshold)
    landing = LandingDetector(args.airborne_height_m, args.contact_height_m, args.contact_frames,
                              args.min_down_speed_mps, args.max_contact_down_speed_mps)
    histories: deque[np.ndarray] = deque(maxlen=args.smoothing_frames)
    trajectory: list[dict] = []
    plane: GroundPlane | None = None
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    start_monotonic = time.monotonic()
    first_device_timestamp: float | None = None
    last_depth_m: float | None = None
    usb_speed = "UNKNOWN"
    red_ranges = (parse_triplet(args.red1_low), parse_triplet(args.red1_high),
                  parse_triplet(args.red2_low), parse_triplet(args.red2_high))
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dai.Device(pipeline) as device:
            intrinsics = read_intrinsics(device, args.width, args.height)
            usb_speed = str(device.getUsbSpeed())
            print(f"已连接 OAK 设备：{device.getMxId()}  USB 速度：{usb_speed}")
            if "SUPER" not in usb_speed:
                print("警告：当前未检测到 USB 3.x SUPER 速度，帧率可能不足。")
            print("请保持画面下半部地面可见；按 r 重置抛球，g 重标地面，q/Esc 退出。")
            queue = device.getOutputQueue(name="rgb_depth", maxSize=1, blocking=False)
            while True:
                group = queue.tryGet()
                if group is None:
                    time.sleep(0.001)
                    continue
                rgb_message, depth_message = group["rgb"], group["depth"]
                frame, depth = rgb_message.getCvFrame(), depth_message.getFrame()
                frame_count += 1
                device_s = rgb_message.getTimestampDevice().total_seconds()
                if first_device_timestamp is None:
                    first_device_timestamp = device_s
                elapsed_s = device_s - first_device_timestamp
                detections, _ = detect_red_candidates(frame, *red_ranges, args.min_area, args.min_circularity)
                detection = tracker.select(detections)
                if plane is None:
                    points = depth_to_points(depth, intrinsics, args.ground_start_ratio,
                                             args.min_depth_mm, args.max_depth_mm)
                    plane = fit_ground_plane(points, args.camera_height_m, args.ground_distance_m,
                                             args.ground_min_inlier_ratio, args.height_tolerance_m)
                current: BallMeasurement | None = None
                skin_ratio: float | None = None
                if detection is not None:
                    skin_ratio = skin_contact_ratio(frame, detection, args.hand_ring_scale)
                    released = release_gate.update(skin_ratio)
                    sample = sample_ball_depth(depth, detection, args.depth_inner_ratio,
                                               args.min_depth_mm, args.max_depth_mm)
                    if sample is not None and sample.valid_ratio >= args.min_depth_valid_ratio:
                        depth_m = sample.median_mm / 1000.0
                        if last_depth_m is None or abs(depth_m - last_depth_m) <= args.max_depth_jump_m:
                            center, radius_m = estimate_ball_geometry(detection, sample.median_mm, intrinsics)
                            histories.append(center)
                            center = np.median(np.stack(histories), axis=0)
                            last_depth_m = depth_m
                            bottom_height = contact_point = gx = gf = None
                            if plane is not None:
                                bottom_height = signed_plane_distance(center, plane) - radius_m
                                contact_point = project_to_plane(center, plane)
                                gx, gf = ground_coordinates(contact_point, plane)
                            current = BallMeasurement(
                                frame_count, elapsed_s, device_s, datetime.now().astimezone().isoformat(timespec="milliseconds"),
                                detection.center[0], detection.center[1], depth_m, radius_m,
                                float(center[0]), float(center[1]), float(center[2]), bottom_height, gx, gf,
                                None if contact_point is None else float(contact_point[0]),
                                None if contact_point is None else float(contact_point[1]),
                                None if contact_point is None else float(contact_point[2]),
                                float(np.linalg.norm(contact_point if contact_point is not None else center)), None,
                                landing.state,
                            )
                            if released:
                                state, event = landing.update(current)
                            else:
                                state, event = "waiting", None
                            current = BallMeasurement(**{
                                **asdict(current), "vertical_velocity_mps": landing.last_velocity,
                                "state": state,
                            })
                            trajectory.append(asdict(current))
                            cv2.circle(frame, detection.center, int(round(detection.radius_px)), (0, 255, 255), 2)
                            label = f"ball Z={depth_m:.2f}m R={radius_m * 100:.1f}cm"
                            cv2.putText(frame, label, (max(5, detection.center[0] - 90), max(20, detection.center[1] - 14)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                status = LandingDetector.STATES[landing.state]
                ground_text = "GROUND OK" if plane else "GROUND CALIBRATING/FAILED"
                fps = frame_count / max(time.monotonic() - start_monotonic, 0.001)
                cv2.putText(frame, f"{ground_text}  release={release_gate.status}  state={status}  fps={fps:.1f}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 230, 40) if plane else (0, 100, 255), 2)
                if current is not None and current.bottom_height_m is not None:
                    skin_text = "-" if skin_ratio is None else f"{skin_ratio:.2f}"
                    cv2.putText(frame, f"bottom height={current.bottom_height_m:.3f}m  skin={skin_text}", (10, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
                if landing.event is not None:
                    event = landing.event
                    cv2.putText(frame, f"FIRST LANDING t={event.elapsed_s:.3f}s  ground=({event.ground_x_m:.3f}, {event.ground_forward_m:.3f})m",
                                (10, args.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"无法创建输出视频：{args.output_video}")
                writer.write(frame)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    landing.reset(); release_gate.reset(); tracker.reset(); histories.clear(); trajectory.clear(); last_depth_m = None
                if key == ord("g"):
                    plane = None; landing.reset(); release_gate.reset(); tracker.reset(); histories.clear(); trajectory.clear(); last_depth_m = None
                if args.duration is not None and elapsed_s >= args.duration:
                    break
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        write_result(args.output_json, landing.event, trajectory, plane, args, usb_speed,
                     landing.confirmation_frame)
    print(f"已保存：{args.output_video}\n已保存：{args.output_json}")
    return 0 if frame_count else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OAK-D Lite 红球首次落地追踪。")
    p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=360)
    p.add_argument("--fps", type=float, default=30); p.add_argument("--camera-height-m", type=float, default=.90)
    p.add_argument("--sync-threshold-ms", type=int, default=25)
    p.add_argument("--red1-low", nargs=3, default=["0", "90", "60"]); p.add_argument("--red1-high", nargs=3, default=["12", "255", "255"])
    p.add_argument("--red2-low", nargs=3, default=["168", "90", "60"]); p.add_argument("--red2-high", nargs=3, default=["179", "255", "255"])
    p.add_argument("--min-area", type=float, default=100); p.add_argument("--min-circularity", type=float, default=.58)
    p.add_argument("--max-missing-frames", type=int, default=8); p.add_argument("--min-depth-mm", type=int, default=250)
    p.add_argument("--require-hand-release", action="store_true"); p.add_argument("--release-frames", type=int, default=3)
    p.add_argument("--skin-ratio-threshold", type=float, default=.06); p.add_argument("--hand-ring-scale", type=float, default=1.75)
    p.add_argument("--max-depth-mm", type=int, default=8000); p.add_argument("--depth-inner-ratio", type=float, default=.45)
    p.add_argument("--min-depth-valid-ratio", type=float, default=.25); p.add_argument("--max-depth-jump-m", type=float, default=.8)
    p.add_argument("--smoothing-frames", type=int, default=3); p.add_argument("--ground-start-ratio", type=float, default=.48)
    p.add_argument("--ground-distance-m", type=float, default=.025); p.add_argument("--ground-min-inlier-ratio", type=float, default=.28)
    p.add_argument("--height-tolerance-m", type=float, default=.25); p.add_argument("--airborne-height-m", type=float, default=.08)
    p.add_argument("--contact-height-m", type=float, default=.025); p.add_argument("--contact-frames", type=int, default=2)
    p.add_argument("--min-down-speed-mps", type=float, default=.12); p.add_argument("--max-contact-down-speed-mps", type=float, default=.10)
    p.add_argument("--output-video", type=Path, default=Path("output/demo_11_red_ball_landing.mp4"))
    p.add_argument("--output-json", type=Path, default=Path("output/demo_11_red_ball_landing.json"))
    p.add_argument("--save-trajectory", action="store_true"); p.add_argument("--duration", type=float)
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.camera_height_m <= 0:
        raise SystemExit("画面、帧率和相机高度必须大于 0。")
    if not 0 < args.min_circularity <= 1 or not 0 < args.min_depth_valid_ratio <= 1:
        raise SystemExit("圆度和深度有效比例必须在 (0, 1] 内。")
    if args.min_depth_mm >= args.max_depth_mm or args.contact_frames < 1 or args.smoothing_frames < 1:
        raise SystemExit("深度范围或帧数参数不正确。")
    if args.release_frames < 1 or not 0 <= args.skin_ratio_threshold <= 1 or args.hand_ring_scale <= 1:
        raise SystemExit("脱手确认参数不正确。")


def main() -> int:
    args = parse_args(); validate_args(args); return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
