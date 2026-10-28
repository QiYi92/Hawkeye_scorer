#!/usr/bin/env python3
"""Demo 12: independent multi-scan landing review.

Three inexpensive pose passes propose a stable athlete/time/foot region.  A
separate full-resolution pass then confirms first contact.  Demo 08 is not
imported: this module talks only to the lower-level detector components.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from browse_target_center_video import SmoothedTarget, TargetCenterDetector, TargetDetection
from target_overlay import TargetOverlayMixin
from target_plan_view import draw_target_plan, project_target_point
from landing_detection.athlete import AthleteSelector
from landing_detection.foot_parts import FootPartEstimator
from landing_detection.ground import GroundLineConfig, GroundLineEstimator, HorizonLineConfig, HorizonLineEstimator
from landing_detection.models import GroundLine, Point, PoseDetection
from landing_detection.pose import PoseDetector, PoseDetectorConfig


WINDOW_NAME = "Demo12 - Multi-scan Landing Review"
SCAN_CONFIGS = (
    ("A", 0.50, 4),
    ("B", 0.75, 2),
    ("C", 1.00, 3),
)
MIN_DESCENT_PX = 18.0
TOUCH_CONFIRMATION_FRAMES = 2
TARGET_LOCK_FRAMES = 5
# A video ground line is useful visual context, but it is not reliable enough to
# be a hard gate: the white target is often detected as the ground and a
# crouching athlete can move the estimate by tens of pixels.  Contact is
# therefore confirmed from the foot trajectory, as in Demo 08.
MAX_FOOT_POINT_JUMP_PX = 55.0
TOUCH_STABILITY_PX = 12.0
FALLBACK_GROUND_MARGIN_PX = 30.0
# MediaPipe 脚点在鞋底受遮挡时会有约 10px 的后续下移；该值仍远小于
# 真正尚未接触时的持续下降幅度（本视频误判轨迹为 34px）。
POST_CONTACT_MAX_DESCENT_PX = 12.0
# 仅为已经可信的鞋底关键点做很小的地面贴合；不能把 YOLO 脚踝或空中脚
# 强行投到地面，否则会再次造成靶盘边缘的假触地。
MAX_CONTACT_PROJECTION_PX = 12.0
FONT_PATHS = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
)


@lru_cache(maxsize=16)
def _load_chinese_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_chinese_text(
    frame: np.ndarray,
    text: str,
    origin: Point,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.62,
    thickness: int = 2,
) -> None:
    """Draw UTF-8 labels reliably on macOS OpenCV windows."""
    size = max(12, int(round(scale * 26)))
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text(
        (origin[0], origin[1] - size), text, font=_load_chinese_font(size),
        fill=(color[2], color[1], color[0]), stroke_width=max(1, thickness), stroke_fill=(0, 0, 0),
    )
    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@dataclass(frozen=True)
class ScanCandidate:
    scan: str
    frame_index: int
    bbox: tuple[int, int, int, int]
    foot_point: Point
    confidence: float


@dataclass(frozen=True)
class FusedCandidate:
    frame_index: int
    bbox: tuple[int, int, int, int]
    foot_point: Point
    support: int
    confidence: float


@dataclass(frozen=True)
class ContactResult:
    frame_index: int
    timestamp: float
    foot: str
    part: str
    point: Point
    zone: str
    source: str
    bbox: tuple[int, int, int, int]
    target: SmoothedTarget | None


@dataclass(frozen=True)
class FootTrackSample:
    """同一只脚的连续观测；保留原始脚点和对应地面高度。"""

    frame_index: int
    point: Point
    part: str = "unknown"
    ground_y: float | None = None


class PrescanProgressWindow:
    """显示预扫描阶段；左上角参数与进度始终可见。"""

    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def show(
        self,
        frame: np.ndarray,
        scan_number: int,
        scan_name: str,
        scale: float,
        stride: int,
        frame_index: int,
        current_candidates: int,
        total_candidates: int,
        selected_bbox: tuple[int, int, int, int] | None,
    ) -> None:
        output = frame.copy()
        if selected_bbox is not None:
            x1, y1, x2, y2 = selected_bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (60, 220, 60), 2)
            draw_chinese_text(output, "预扫描中的候选运动员", (x1, max(28, y1 - 8)), (60, 220, 60), 0.52, 1)
        progress = (frame_index + 1) / max(self.frame_count, 1) * 100
        lines = (
            f"预扫描进度｜第 {scan_number}/3 轮｜策略 {scan_name}",
            f"处理帧：{frame_index + 1}/{self.frame_count}（{progress:.1f}%）",
            f"参数：缩放={scale:.2f}｜抽帧间隔={stride}｜本轮候选={current_candidates}｜累计候选={total_candidates}",
            "正在建立运动员轨迹与首次触地候选，请稍候……",
        )
        for row, line in enumerate(lines):
            draw_chinese_text(output, line, (12, 30 + row * 29), (255, 255, 255), 0.60, 2)
        cv2.imshow(WINDOW_NAME, output)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyWindow(WINDOW_NAME)


class CameraMotionMonitor:
    """Detect sustained background motion so a locked target is not kept stale."""

    def __init__(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.recent_shifts: deque[tuple[float, float]] = deque(maxlen=5)

    def update(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return False
        previous = self.previous_gray
        self.previous_gray = gray
        points = cv2.goodFeaturesToTrack(previous, maxCorners=120, qualityLevel=0.02, minDistance=18, blockSize=7)
        if points is None or len(points) < 12:
            return False
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
        if next_points is None or status is None:
            return False
        valid = status.reshape(-1).astype(bool)
        if int(valid.sum()) < 12:
            return False
        dx, dy = np.median((next_points.reshape(-1, 2) - points.reshape(-1, 2))[valid], axis=0)
        self.recent_shifts.append((float(dx), float(dy)))
        total_x, total_y = (sum(item[index] for item in self.recent_shifts) for index in (0, 1))
        return len(self.recent_shifts) == self.recent_shifts.maxlen and math.hypot(total_x, total_y) >= 10.0


def scale_bbox(bbox: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    return tuple(int(round(value / scale)) for value in bbox)  # type: ignore[return-value]


def scale_point(point: Point, scale: float) -> Point:
    return int(round(point[0] / scale)), int(round(point[1] / scale))


def roi_from_bbox(bbox: tuple[int, int, int, int], frame_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Return a lower-body ROI with enough margin for both feet."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    box_width, box_height = max(x2 - x1, 1), max(y2 - y1, 1)
    return (
        max(0, int(round(x1 - box_width * 0.25))),
        max(0, int(round(y1 + box_height * 0.35))),
        min(width, int(round(x2 + box_width * 0.25))),
        min(height, int(round(y2 + box_height * 0.60))),
    )


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def fuse_scan_candidates(candidates: list[ScanCandidate], fps: float, frame_diagonal: float) -> FusedCandidate | None:
    """Cluster scan candidates; a valid fusion needs agreement from two scans."""
    if not candidates:
        return None
    clusters: list[list[ScanCandidate]] = []
    max_frame_gap = max(int(round(fps)), 1)
    max_point_gap = max(frame_diagonal * 0.18, 40.0)
    for candidate in sorted(candidates, key=lambda item: item.frame_index):
        for cluster in clusters:
            anchor = cluster[0]
            if abs(candidate.frame_index - anchor.frame_index) <= max_frame_gap and _distance(candidate.foot_point, anchor.foot_point) <= max_point_gap:
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    best = max(clusters, key=lambda cluster: (len({item.scan for item in cluster}), sum(item.confidence for item in cluster)))
    support = len({item.scan for item in best})
    if support < 2:
        return None
    ordered_frames = sorted(item.frame_index for item in best)
    middle = len(best) // 2
    points_x = sorted(item.foot_point[0] for item in best)
    points_y = sorted(item.foot_point[1] for item in best)
    boxes = [item.bbox for item in best]
    return FusedCandidate(
        frame_index=ordered_frames[middle],
        bbox=tuple(int(round(sum(box[index] for box in boxes) / len(boxes))) for index in range(4)),  # type: ignore[arg-type]
        foot_point=(points_x[middle], points_y[middle]),
        support=support,
        confidence=sum(item.confidence for item in best) / len(best),
    )


def point_in_ellipse(point: Point, ellipse, margin_px: float = 0.0) -> bool:
    (cx, cy), (axis_a, axis_b), angle_deg = ellipse
    radius_a, radius_b = max(axis_a / 2 + margin_px, 1.0), max(axis_b / 2 + margin_px, 1.0)
    angle = math.radians(angle_deg)
    dx, dy = point[0] - cx, point[1] - cy
    x = dx * math.cos(angle) + dy * math.sin(angle)
    y = -dx * math.sin(angle) + dy * math.cos(angle)
    return x * x / (radius_a * radius_a) + y * y / (radius_b * radius_b) <= 1.0


def classify_contact_zone(point: Point, target: SmoothedTarget | None) -> str:
    if target is None:
        return "UNKNOWN_TARGET"
    if target.ellipse is not None and point_in_ellipse(point, target.ellipse, 4.0):
        return "BULLSEYE"
    if _distance(point, (int(target.center[0]), int(target.center[1]))) <= target.radius_px + 4.0:
        return "BULLSEYE"
    return "TARGET_DISK" if point_in_ellipse(point, target.disk_ellipse, 6.0) else "OUTSIDE_TARGET"


def 中文脚部(value: str) -> str:
    return {"left": "左脚", "right": "右脚"}.get(value, value)


def 中文部位(value: str) -> str:
    return {"heel": "脚跟", "toe": "脚尖"}.get(value, value)


def 中文区域(value: str) -> str:
    return {
        "BULLSEYE": "黑色靶心",
        "TARGET_DISK": "靶盘内",
        "OUTSIDE_TARGET": "靶盘外",
        "UNKNOWN_TARGET": "靶盘未知",
    }.get(value, value)


def should_show_contact_marker(frame_index: int, result: ContactResult | None) -> bool:
    """在回放真正到达确认帧前，不能提前泄露最终触地点。"""
    return result is not None and frame_index >= result.frame_index


class LandingAnalyzer:
    def __init__(self, video_path: Path, model_path: str) -> None:
        self.video_path = video_path
        self.model_path = model_path
        self.last_diagnostic = "尚未开始精细复核"

    def _pose_detector(self) -> PoseDetector:
        return PoseDetector(PoseDetectorConfig(model_name=self.model_path))

    def prescan(self) -> tuple[FusedCandidate | None, float, int, int]:
        metadata = cv2.VideoCapture(str(self.video_path))
        if not metadata.isOpened():
            raise SystemExit(f"无法打开视频：{self.video_path}")
        fps = float(metadata.get(cv2.CAP_PROP_FPS) or 30.0)
        width, height = int(metadata.get(cv2.CAP_PROP_FRAME_WIDTH)), int(metadata.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(metadata.get(cv2.CAP_PROP_FRAME_COUNT))
        metadata.release()
        all_candidates: list[ScanCandidate] = []
        preview = PrescanProgressWindow(frame_count)
        try:
            for scan_number, (name, scale, stride) in enumerate(SCAN_CONFIGS, start=1):
                print(f"预扫描 {name}: scale={scale:.2f}, stride={stride}")
                all_candidates.extend(self._run_prescan(name, scan_number, scale, stride, preview, all_candidates))
        finally:
            preview.close()
        fused = fuse_scan_candidates(all_candidates, fps, math.hypot(width, height))
        return fused, fps, width, height

    def _run_prescan(
        self,
        name: str,
        scan_number: int,
        scale: float,
        stride: int,
        preview: PrescanProgressWindow,
        previous_candidates: list[ScanCandidate],
    ) -> list[ScanCandidate]:
        detector, selector = self._pose_detector(), AthleteSelector()
        capture = cv2.VideoCapture(str(self.video_path))
        histories: dict[int, deque[tuple[int, PoseDetection]]] = defaultdict(lambda: deque(maxlen=4))
        found: list[ScanCandidate] = []
        index = -1
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
                if index % stride:
                    continue
                reduced = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
                detections = detector.detect_all(reduced)
                selected = selector.update(detections, index, ground_line=None, frame_shape=reduced.shape)
                if selected is None and detections:
                    selected = detections[0]
                preview_bbox = scale_bbox(selected.bbox, scale) if selected is not None else None
                preview.show(frame, scan_number, name, scale, stride, index, len(found), len(previous_candidates) + len(found), preview_bbox)
                if selected is None:
                    continue
                track_id = selector.track_id_for_detection(selected) or 0
                history = histories[track_id]
                history.append((index, selected))
                if len(history) < 3:
                    continue
                earlier, previous, latest = history[-3], history[-2], history[-1]
                foot = max((point for point in (previous[1].foot_points.left, previous[1].foot_points.right) if point is not None), key=lambda p: p[1], default=None)
                if foot is None:
                    continue
                descent = max((point[1] for point in (latest[1].foot_points.left, latest[1].foot_points.right) if point is not None), default=foot[1]) - max((point[1] for point in (earlier[1].foot_points.left, earlier[1].foot_points.right) if point is not None), default=foot[1])
                if descent >= max(8.0, MIN_DESCENT_PX * scale * 0.45) and foot[1] >= reduced.shape[0] * 0.55:
                    found.append(ScanCandidate(name, previous[0], scale_bbox(previous[1].bbox, scale), scale_point(foot, scale), previous[1].confidence))
        finally:
            capture.release()
        return found

    def analyze(self, fused: FusedCandidate | None, fps: float, width: int, height: int) -> ContactResult | None:
        frame_count = self._frame_count()
        if fused is None:
            print("预扫描未形成共识，自动进行全片精细扫描。")
            return self._analyze_range(0, frame_count - 1, fps, "fallback", None)
        before, after = int(round(fps)), int(round(fps * 1.5))
        start, end = max(0, fused.frame_index - before), min(frame_count - 1, fused.frame_index + after)
        print(f"预扫描共识：{fused.support}/3，复核帧窗口={start}-{end}")
        result = self._analyze_range(start, end, fps, "fine", fused)
        if result is None:
            print("候选窗口未确认触地，自动进行全片精细扫描。")
            return self._analyze_range(0, frame_count - 1, fps, "fallback", fused)
        return result

    def _frame_count(self) -> int:
        capture = cv2.VideoCapture(str(self.video_path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return count

    def _analyze_range(self, start: int, end: int, fps: float, source: str, fused: FusedCandidate | None) -> ContactResult | None:
        detector = self._pose_detector()
        foot_estimator = FootPartEstimator()
        selector = AthleteSelector()
        target_detector = TargetCenterDetector()
        camera_motion = CameraMotionMonitor()
        horizon = HorizonLineEstimator(HorizonLineConfig(max_candidate_y_ratio=0.70))
        ground = GroundLineEstimator(GroundLineConfig(min_candidate_y_ratio=0.70, min_gap_below_horizon_ratio=0.12))
        capture = cv2.VideoCapture(str(self.video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        tracks: dict[str, deque[FootTrackSample]] = defaultdict(lambda: deque(maxlen=6))
        athlete_bottoms: deque[tuple[int, int]] = deque(maxlen=6)
        diagnostic_counts: dict[str, int] = defaultdict(int)
        locked_target: SmoothedTarget | None = None
        target_frames = 0
        result: ContactResult | None = None
        try:
            for index in range(start, end + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                if locked_target is not None and camera_motion.update(frame):
                    print("检测到相机明显移动，解除靶盘锁定并重新识别。")
                    locked_target = None
                    target_frames = 0
                    target_detector.reset()
                detected_target = target_detector.update(target_detector.detect(frame))
                if locked_target is None:
                    if detected_target is not None and detected_target.disk_locked and detected_target.bullseye_detected:
                        target_frames += 1
                        if target_frames >= TARGET_LOCK_FRAMES:
                            locked_target = detected_target
                    else:
                        target_frames = 0
                target = locked_target or detected_target
                detections = detector.detect_all(frame)
                horizon_line = horizon.estimate(frame)
                ground_line = ground.estimate(frame, horizon_line=horizon_line, athlete_bbox=None)
                selected = selector.update(detections, index, ground_line=ground_line, frame_shape=frame.shape)
                if selected is None and detections:
                    selected = self._nearest_detection(detections, fused) or detections[0]
                if selected is None:
                    diagnostic_counts["未锁定运动员"] += 1
                    continue
                athlete_bottoms.append((index, selected.bbox[3]))
                ground_line = ground.estimate(frame, horizon_line=horizon_line, athlete_bbox=selected.bbox)
                estimate = foot_estimator.estimate(frame, selected, timestamp_ms=int(round(index / fps * 1000)))
                if estimate is None:
                    diagnostic_counts["脚部姿态不可用"] += 1
                    continue
                confirmed: list[tuple[str, FootTrackSample, Point]] = []
                visible_feet = 0
                for foot_name, fallback_point in (("left", selected.foot_points.left), ("right", selected.foot_points.right)):
                    if fallback_point is None:
                        continue
                    # 以 YOLO 脚踝作为锚点选择受稳定性检查的脚尖/脚跟。此前让
                    # heel/toe 各自竞争，会把空中脚的错误关键点抢成“首次触地”。
                    refined = foot_estimator.refine_contact(estimate, foot_name, fallback_point, selected)
                    if refined.status != "trusted":
                        diagnostic_counts["脚部关键点未通过可信度检查"] += 1
                        continue
                    visible_feet += 1
                    history = tracks[foot_name]
                    sample = FootTrackSample(
                        index,
                        refined.point,
                        refined.part,
                        ground_line.y_at(refined.point[0]),
                    )
                    if history and _distance(history[-1].point, sample.point) > MAX_FOOT_POINT_JUMP_PX:
                        history.clear()
                        diagnostic_counts["脚部关键点跳变"] += 1
                    history.append(sample)
                    candidate = self._confirmed_contact(history)
                    if candidate is None:
                        candidate = self._first_visible_target_contact(
                            history, athlete_bottoms, target, frame.shape[0],
                        )
                    if candidate is None:
                        continue
                    contact_point = self._project_contact_point(candidate)
                    if not self._contact_point_is_plausible(candidate, ground_line):
                        diagnostic_counts["脚部轨迹未接近地面"] += 1
                        continue
                    confirmed.append((foot_name, candidate, contact_point))
                if confirmed:
                    # 先选择实际更早的候选帧；同一时刻优先更低、更接近鞋底的脚点。
                    foot_name, candidate, contact_point = min(
                        confirmed,
                        key=lambda item: (item[1].frame_index, -item[1].point[1]),
                    )
                    result = ContactResult(
                        candidate.frame_index,
                        candidate.frame_index / fps,
                        foot_name,
                        candidate.part,
                        contact_point,
                        classify_contact_zone(contact_point, target),
                        source,
                        selected.bbox,
                        target,
                    )
                    return result
                if not visible_feet:
                    diagnostic_counts["脚尖/脚跟关键点缺失"] += 1
                elif foot_estimator.last_rejected_landmarks:
                    diagnostic_counts[foot_estimator.last_rejected_landmarks[-1].reason] += 1
                else:
                    diagnostic_counts["脚部轨迹尚未满足触地条件"] += 1
        finally:
            capture.release()
            self.last_diagnostic = (
                "已基于脚尖/脚跟轨迹确认首次触地"
                if result is not None
                else self._format_diagnostic(diagnostic_counts)
            )
        return result

    @staticmethod
    def _nearest_detection(detections: list[PoseDetection], fused: FusedCandidate | None) -> PoseDetection | None:
        if fused is None:
            return None
        center = ((fused.bbox[0] + fused.bbox[2]) // 2, (fused.bbox[1] + fused.bbox[3]) // 2)
        return min(detections, key=lambda detection: _distance(((detection.bbox[0] + detection.bbox[2]) // 2, (detection.bbox[1] + detection.bbox[3]) // 2), center))

    @staticmethod
    def _confirmed_contact(history: deque[FootTrackSample]) -> FootTrackSample | None:
        """Confirm a descending foot that subsequently settles or rebounds.

        Samples are deliberately recorded before any ground-line comparison.
        The previous implementation only kept points near an automatically
        estimated line, so a biased line made confirmation impossible.
        """
        if len(history) < 5:
            return None
        first, candidate, first_after, second_after = history[-5], history[-3], history[-2], history[-1]
        if candidate.part == "unknown":
            return None
        descent = candidate.point[1] - first.point[1]
        # 旧逻辑只看候选后的一个点，并把继续向下 4px 当“停稳”。必须再等待
        # 一个帧，且两个后续点都不能继续明显下落，才能避免在靶盘边缘提前触发。
        max_post_descent = max(first_after.point[1], second_after.point[1]) - candidate.point[1]
        if descent >= MIN_DESCENT_PX and max_post_descent <= POST_CONTACT_MAX_DESCENT_PX:
            return candidate
        return None

    @staticmethod
    def _first_visible_target_contact(
        history: deque[FootTrackSample],
        athlete_bottoms: deque[tuple[int, int]],
        target: SmoothedTarget | None,
        frame_height: int,
    ) -> FootTrackSample | None:
        """Handle a foot which first becomes visible only at the target.

        This is a constrained fallback for occlusion at touchdown, not a
        substitute for the normal descent-and-settle confirmation.
        """
        if len(history) < TOUCH_CONFIRMATION_FRAMES + 1 or len(athlete_bottoms) < 2:
            return None
        first, latest = history[0], history[-1]
        zone = classify_contact_zone(first.point, target)
        body_descent = athlete_bottoms[-1][1] - athlete_bottoms[0][1]
        stable = max(sample.point[1] for sample in history) - min(sample.point[1] for sample in history) <= TOUCH_STABILITY_PX
        athlete_is_low = athlete_bottoms[-1][1] / max(frame_height, 1) >= 0.55
        if zone in {"BULLSEYE", "TARGET_DISK"} and stable and (body_descent >= MIN_DESCENT_PX or athlete_is_low):
            return first
        return None

    @staticmethod
    def _project_contact_point(sample: FootTrackSample) -> Point:
        """Project a trusted toe/heel landmark to the local ground surface.

        Pose models commonly locate a visible shoe feature above the sole.  The
        projected point is only used when that feature is already close enough
        to the estimated local ground, so it improves score location without
        turning a clearly airborne foot into a contact.
        """
        if sample.ground_y is None or abs(sample.ground_y - sample.point[1]) > MAX_CONTACT_PROJECTION_PX:
            return sample.point
        return sample.point[0], int(round(sample.ground_y))

    @staticmethod
    def _contact_point_is_plausible(sample: FootTrackSample, ground_line: GroundLine) -> bool:
        """Reject an airborne trajectory without using the target as a gate.

        The target must classify a confirmed landing, not decide whether a
        landing exists; otherwise real landings outside the white disk vanish.
        """
        ground_y = sample.ground_y if sample.ground_y is not None else ground_line.y_at(sample.point[0])
        return abs(ground_y - sample.point[1]) <= MAX_CONTACT_PROJECTION_PX

    @staticmethod
    def _format_diagnostic(counts: dict[str, int]) -> str:
        if not counts:
            return "已完成精细复核，但没有形成可确认的脚部触地轨迹"
        reason, count = max(counts.items(), key=lambda item: item[1])
        return f"精细复核未确认：{reason}（{count} 帧）；按 r 仅重播，不会重新分析"


class ReviewWindow(TargetOverlayMixin):
    _draw_text = staticmethod(draw_chinese_text)

    def __init__(self, video_path: Path, result: ContactResult | None, fused: FusedCandidate | None, scale: float, diagnostic: str) -> None:
        self.video_path, self.result, self.fused, self.scale, self.diagnostic = video_path, result, fused, scale, diagnostic
        self.target_detector = TargetCenterDetector()
        self.plan_contact_point = project_target_point(result.point, result.target) if result is not None else None
        self._reset_target_tracking()

    def _reset_target_tracking(self) -> None:
        self.target_detector.reset()
        self.camera_motion = CameraMotionMonitor()
        self.locked_target: SmoothedTarget | None = None
        self.target_frames = 0
        self.last_target_frame: int | None = None

    def _update_target(self, frame: np.ndarray, index: int) -> tuple[SmoothedTarget | None, TargetDetection | None]:
        # Seeking or looping must not carry geometry from another video time.
        if self.last_target_frame is not None and index != self.last_target_frame + 1:
            self._reset_target_tracking()
        moved = self.camera_motion.update(frame)
        if moved:
            self._reset_target_tracking()
            self.camera_motion.update(frame)
        detection = self.target_detector.detect(frame)
        target = self.target_detector.update(detection)
        if self.locked_target is None:
            if target is not None and target.disk_locked and target.bullseye_detected:
                self.target_frames += 1
                if self.target_frames >= TARGET_LOCK_FRAMES:
                    self.locked_target = target
            else:
                self.target_frames = 0
        self.last_target_frame = index
        return self.locked_target or target, detection

    def run(self) -> int:
        capture = cv2.VideoCapture(str(self.video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        focus = self.result.frame_index if self.result else (self.fused.frame_index if self.fused else 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, focus - int(fps)))
        paused, step = False, False
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        try:
            while True:
                if not paused or step:
                    ok, frame = capture.read()
                    if not ok:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, focus - int(fps)))
                        continue
                    index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    annotated = self._draw(frame, index, fps, frame_count, paused)
                    step = False
                cv2.imshow(WINDOW_NAME, cv2.resize(annotated, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA))
                key = cv2.waitKey(max(1, int(1000 / fps)) if not paused else 30)
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key == ord("n") and paused:
                    step = True
                elif key == ord("r"):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, focus - int(fps)))
                    step = True
        finally:
            capture.release()
            cv2.destroyAllWindows()
        return 0

    def _draw_plan(self, output: np.ndarray, index: int) -> None:
        point = None
        detail = ""
        if self.result is None:
            status = "未确认首次触地"
        elif not should_show_contact_marker(index, self.result):
            status = "等待首次触地"
        else:
            point = self.plan_contact_point
            status = "已确认首次触地" if point is not None else "已确认触地，暂无靶盘定位"
            detail = f"{中文脚部(self.result.foot)} {中文部位(self.result.part)}｜{中文区域(self.result.zone)}"
        draw_target_plan(output, draw_chinese_text, point, status, detail)

    def _draw(self, frame: np.ndarray, index: int, fps: float, count: int, paused: bool) -> np.ndarray:
        output = frame.copy()
        target, detection = self._update_target(frame, index)
        self._draw_target(output, target, detection)
        show_contact = should_show_contact_marker(index, self.result)
        if self.fused is not None:
            x1, y1, x2, y2 = roi_from_bbox(self.fused.bbox, frame.shape)
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 180, 0), 2)
            draw_chinese_text(output, f"预扫描脚部复核区｜一致性={self.fused.support}/3", (x1, max(28, y1 - 8)), (255, 180, 0), 0.55, 2)
        if self.result is not None:
            if show_contact:
                x, y = self.result.point
                cv2.circle(output, (x, y), 14, (0, 0, 255), -1)
                cv2.circle(output, (x, y), 21, (255, 255, 255), 2)
                draw_chinese_text(
                    output,
                    f"首次触地｜{中文脚部(self.result.foot)} {中文部位(self.result.part)}｜{中文区域(self.result.zone)}",
                    (max(8, x - 185), max(32, y - 28)), (0, 0, 255), 0.68, 2,
                )
        stage = "已暂停" if paused else "复核播放中"
        outcome = "未确认首次触地" if self.result is None else ("候选窗口精细确认" if self.result.source == "fine" else "全片精细扫描确认")
        lines = (
            f"Demo 12｜{stage}｜{outcome}",
            f"帧：{index + 1}/{count}｜时间：{index / fps:.2f} 秒｜原始分辨率复核",
            self.diagnostic if self.result is None else ("等待脚部落至确认帧" if not show_contact else "已基于脚尖/脚跟轨迹确认首次触地"),
            "空格：暂停/播放｜n：下一帧｜r：重新开始｜q/Esc：退出",
        )
        for row, line in enumerate(lines):
            draw_chinese_text(output, line, (12, 30 + row * 28), (255, 255, 255), 0.60, 2)
        self._draw_plan(output, index)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo 12：多轮预扫描与首次落地复核")
    parser.add_argument("--video", default="training_video/video7.mp4", help="输入视频路径")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="YOLO Pose 模型路径")
    parser.add_argument("--display-scale", type=float, default=0.75, help="预览窗口缩放比例")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = Path(args.video)
    if not video.is_absolute():
        video = REPO_ROOT / video
    analyzer = LandingAnalyzer(video, args.model)
    fused, fps, width, height = analyzer.prescan()
    result = analyzer.analyze(fused, fps, width, height)
    if result is None:
        print("Demo 12：未确认首次触地。")
    else:
        source = "候选窗口精细确认" if result.source == "fine" else "全片精细扫描确认"
        print(
            f"Demo 12：首次触地｜帧={result.frame_index}，时间={result.timestamp:.3f}秒，"
            f"脚={中文脚部(result.foot)}，部位={中文部位(result.part)}，坐标={result.point}，"
            f"区域={中文区域(result.zone)}，来源={source}"
        )
    return ReviewWindow(video, result, fused, max(args.display_scale, 0.05), analyzer.last_diagnostic).run()


if __name__ == "__main__":
    raise SystemExit(main())
