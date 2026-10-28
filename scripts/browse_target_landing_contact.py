#!/usr/bin/env python3
"""Browse landing foot contact and classify it against the target disk."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
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

import browse_mediapipe_foot_contact as foot_demo
from browse_mediapipe_foot_contact import FootContactCandidate
from browse_target_center_video import (
    BULLSEYE_RING_RADIUS_M,
    RING_INTERVAL_M,
    TARGET_REAL_RADIUS_M,
    TARGET_RING_COUNT,
    SmoothedTarget,
    TargetCenterDetector,
    TargetDetection,
)
from target_overlay import TargetOverlayMixin
from landing_detection.models import GroundLine, Point, PoseDetection


VIDEO_PATH = "training_video/video8.mp4"
# macOS OpenCV 的窗口标题不可靠地支持中文，使用短 ASCII 标题避免出现乱码；画面内文字均为中文。
WINDOW_NAME = "Demo08 - Landing Target"
TARGET_MARGIN_PX = 6.0
BULLSEYE_MARGIN_PX = 4.0
TRACK_HISTORY_FRAMES = 8
MIN_DESCENT_PX = 18.0
MAX_POINT_JUMP_PX = 80.0
TOUCH_SLOWDOWN_RATIO = 0.35
TOUCH_MAX_POST_SPEED_PX = 5.0
TOUCH_CONFIRMATION_FRAMES = 2
TARGET_LOCK_CONFIRMATION_FRAMES = 5
CAMERA_MOTION_WINDOW_FRAMES = 5
CAMERA_MOTION_UNLOCK_PX = 10.0
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


class CameraMotionMonitor:
    """用背景特征的中位位移判断相机是否明显移动。"""

    def __init__(self) -> None:
        self.previous_gray: np.ndarray | None = None
        self.recent_motion: deque[tuple[float, float]] = deque(maxlen=CAMERA_MOTION_WINDOW_FRAMES)

    def reset(self) -> None:
        self.previous_gray = None
        self.recent_motion.clear()

    def update(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return False
        previous_gray = self.previous_gray
        points = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=180, qualityLevel=0.02, minDistance=18, blockSize=7,
        )
        self.previous_gray = gray
        if points is None or len(points) < 12:
            return False
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, gray, points, None,
        )
        if next_points is None or status is None:
            return False
        valid = status.reshape(-1).astype(bool)
        if int(valid.sum()) < 12:
            return False
        shifts = (next_points.reshape(-1, 2) - points.reshape(-1, 2))[valid]
        dx, dy = np.median(shifts, axis=0)
        self.recent_motion.append((float(dx), float(dy)))
        total_dx = sum(item[0] for item in self.recent_motion)
        total_dy = sum(item[1] for item in self.recent_motion)
        return len(self.recent_motion) == CAMERA_MOTION_WINDOW_FRAMES and (
            total_dx * total_dx + total_dy * total_dy
        ) ** 0.5 >= CAMERA_MOTION_UNLOCK_PX


@dataclass(frozen=True)
class LandingTargetResult:
    frame_index: int
    timestamp: float
    foot: str
    part: str
    point: Point
    zone: str


@dataclass(frozen=True)
class FootPartTrackSample:
    frame_index: int
    timestamp: float
    point: Point


@dataclass
class FootPartTrack:
    foot: str
    part: str
    samples: deque[FootPartTrackSample]
    pending: "PhysicalContactCandidate | None" = None
    missing_frames: int = 0


@dataclass(frozen=True)
class PhysicalContactCandidate:
    frame_index: int
    timestamp: float
    foot: str
    part: str
    point: Point
    zone: str
    descent_px: float
    pre_speed_px: float
    post_speed_px: float


class TargetLandingContactBrowser(TargetOverlayMixin, foot_demo.MediaPipeFootContactBrowser):
    """Realtime demo combining target tracking and toe/heel first contact."""

    def __init__(self, video_path: Path, display_scale: float) -> None:
        foot_demo.VIDEO_PATH = str(video_path)
        foot_demo.DISPLAY_SCALE = display_scale
        foot_demo.WRITE_OUTPUT_VIDEO = False
        foot_demo.WINDOW_NAME = WINDOW_NAME
        super().__init__()
        self.video_path = video_path
        self.target_detector = TargetCenterDetector()
        self.camera_motion_monitor = CameraMotionMonitor()
        self.latest_target: SmoothedTarget | None = None
        self.latest_target_detection: TargetDetection | None = None
        self.locked_target: SmoothedTarget | None = None
        self.target_lock_confirmation_frames = 0
        self.camera_moved = False
        self.latest_landing_candidate: LandingTargetResult | None = None
        self.first_landing_result: LandingTargetResult | None = None
        self.foot_part_tracks: dict[tuple[str, str], FootPartTrack] = {}
        self.latest_physical_candidate: PhysicalContactCandidate | None = None
        self.contact_status = "WAIT"
        self.athlete_bottom_history: deque[tuple[int, int]] = deque(maxlen=10)
        self.current_frame_height = 1

    def _annotate_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
        frame_count: int,
    ) -> np.ndarray:
        started_at = time.perf_counter()
        self.current_frame_height = max(frame.shape[0], 1)
        self.camera_moved = self.camera_motion_monitor.update(frame)
        if self.locked_target is not None and self.camera_moved:
            self._unlock_target_after_camera_move()
        self.latest_target_detection = self.target_detector.detect(frame)
        detected_target = self.target_detector.update(self.latest_target_detection)
        if self.locked_target is not None:
            # 运动员遮挡、降落伞或阴影都不能再改变已锁定的计分靶盘几何位置。
            self.latest_target = self.locked_target
        else:
            self.latest_target = detected_target
            if detected_target is not None and detected_target.disk_locked and detected_target.bullseye_detected:
                self.target_lock_confirmation_frames += 1
                if self.target_lock_confirmation_frames >= TARGET_LOCK_CONFIRMATION_FRAMES:
                    self.locked_target = detected_target
                    self.latest_target = self.locked_target
                    print("靶盘位置已锁定；检测到相机明显移动时将自动重新识别。")
            else:
                self.target_lock_confirmation_frames = 0

        detections = self.detector.detect_all(frame)
        horizon_line = self.horizon_estimator.estimate(frame) if foot_demo.SHOW_HORIZON_LINE else None
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
            self.athlete_bottom_history.clear()
            self.current_track_id = selected_track_id

        ground_line = self._estimate_ground_line(
            frame,
            horizon_line,
            athlete_bbox=detection.bbox if detection is not None else None,
        )
        foot_part_estimate = None
        self.latest_distances = []
        self.latest_candidate = None
        self.latest_landing_candidate = None
        self.latest_physical_candidate = None

        output = frame.copy()
        self._draw_target(output, self.latest_target, self.latest_target_detection)
        self._draw_other_detections(output, detections, detection)
        if detection is not None:
            self.athlete_bottom_history.append((frame_index, detection.bbox[3]))
            foot_part_estimate = self.foot_part_estimator.estimate(
                frame,
                detection,
                timestamp_ms=int(round(frame_index / max(fps, 0.001) * 1000)),
            )
            self._update_trails(detection)
            if self._contact_detection_is_armed(frame, detection):
                self.latest_candidate = self._update_landing_contact_candidate(
                    foot_part_estimate,
                    ground_line,
                    self.latest_target,
                    frame_index,
                    fps,
                )
            else:
                self.contact_streak.clear()
                self.contact_status = "WAIT_PERSON_LOW"
            self._draw_pose(output, detection)
            self._draw_foot_parts(output, foot_part_estimate, ground_line)
            if foot_demo.SHOW_REJECTED_FOOT_PARTS:
                self._draw_rejected_foot_parts(output)
        else:
            self.contact_streak.clear()
            self.contact_status = "WAIT_PERSON"

        self.last_inference_seconds = time.perf_counter() - started_at
        if foot_demo.SHOW_HORIZON_LINE and horizon_line is not None:
            self._draw_horizon_line(output, horizon_line)
        self._draw_ground_line(output, ground_line)
        self._draw_trails(output)
        self._draw_landing_marker(output)
        self._draw_combined_overlay(output, detection, len(detections), ground_line, horizon_line, frame_index, fps, frame_count)
        return output

    def _update_landing_contact_candidate(
        self,
        estimate,
        ground_line: GroundLine,
        target: SmoothedTarget | None,
        frame_index: int,
        fps: float,
    ) -> FootContactCandidate | None:
        distances = self._foot_distances(estimate, ground_line)
        self.latest_distances = distances
        candidate = self._update_physical_contact_candidates(estimate, target, frame_index, fps)
        if candidate is None:
            if self.first_landing_result is None:
                self.contact_status = self._motion_status_text()
            return None

        self.latest_physical_candidate = candidate
        self.latest_landing_candidate = LandingTargetResult(
            frame_index=candidate.frame_index,
            timestamp=candidate.timestamp,
            foot=candidate.foot,
            part=candidate.part,
            point=candidate.point,
            zone=candidate.zone,
        )
        contact_candidate = FootContactCandidate(
            frame_index=candidate.frame_index,
            timestamp=candidate.timestamp,
            foot=candidate.foot,
            part=candidate.part,
            point=candidate.point,
            distance_px=0.0,
        )
        if self.first_landing_result is None:
            self.first_contact = contact_candidate
            self.first_landing_result = self.latest_landing_candidate
            self.contact_status = "FIRST_CONFIRMED"
            self._print_first_landing_result()
        return contact_candidate

    def _update_physical_contact_candidates(
        self,
        estimate,
        target: SmoothedTarget | None,
        frame_index: int,
        fps: float,
    ) -> PhysicalContactCandidate | None:
        timestamp = frame_index / max(fps, 0.001)
        active_keys: set[tuple[str, str]] = set()
        confirmed: list[PhysicalContactCandidate] = []
        if estimate is None:
            self._mark_missing_tracks(active_keys)
            return None

        for foot_name, foot in (("left", estimate.left), ("right", estimate.right)):
            for part_name, landmark in (("heel", foot.heel), ("toe", foot.toe)):
                if landmark is None:
                    continue
                key = (foot_name, part_name)
                active_keys.add(key)
                track = self._track_for(key)
                sample = FootPartTrackSample(frame_index=frame_index, timestamp=timestamp, point=landmark.point)
                if track.samples and point_distance(track.samples[-1].point, sample.point) > MAX_POINT_JUMP_PX:
                    track.samples.clear()
                    track.pending = None
                track.samples.append(sample)
                track.missing_frames = 0
                pending = self._candidate_from_track(track, target)
                if pending is not None:
                    track.pending = pending
                if track.pending is not None and frame_index - track.pending.frame_index >= TOUCH_CONFIRMATION_FRAMES:
                    if self._track_confirms_contact(track):
                        confirmed.append(track.pending)
                        track.pending = None

        self._mark_missing_tracks(active_keys)
        if not confirmed:
            return None
        return self._best_physical_candidate(confirmed)

    def _track_for(self, key: tuple[str, str]) -> FootPartTrack:
        if key not in self.foot_part_tracks:
            self.foot_part_tracks[key] = FootPartTrack(
                foot=key[0],
                part=key[1],
                samples=deque(maxlen=TRACK_HISTORY_FRAMES),
            )
        return self.foot_part_tracks[key]

    def _candidate_from_track(
        self,
        track: FootPartTrack,
        target: SmoothedTarget | None,
    ) -> PhysicalContactCandidate | None:
        samples = list(track.samples)
        if len(samples) < 4:
            return self._first_visible_touch_candidate(track, target)
        previous = samples[-4]
        candidate_sample = samples[-2]
        latest = samples[-1]
        descent_px = candidate_sample.point[1] - previous.point[1]
        pre_speed = candidate_sample.point[1] - previous.point[1]
        post_speed = latest.point[1] - candidate_sample.point[1]
        if descent_px < MIN_DESCENT_PX:
            return None
        if candidate_sample.point[1] < max(sample.point[1] for sample in samples[:-1]) - 3:
            return None
        slowed = abs(post_speed) <= max(TOUCH_MAX_POST_SPEED_PX, abs(pre_speed) * TOUCH_SLOWDOWN_RATIO)
        rebounded = post_speed < -2.0
        if not (slowed or rebounded):
            return None
        return PhysicalContactCandidate(
            frame_index=candidate_sample.frame_index,
            timestamp=candidate_sample.timestamp,
            foot=track.foot,
            part=track.part,
            point=candidate_sample.point,
            zone=classify_contact_zone(candidate_sample.point, target),
            descent_px=float(descent_px),
            pre_speed_px=float(pre_speed),
            post_speed_px=float(post_speed),
        )

    def _first_visible_touch_candidate(
        self,
        track: FootPartTrack,
        target: SmoothedTarget | None,
    ) -> PhysicalContactCandidate | None:
        samples = list(track.samples)
        if len(samples) < 4:
            if not samples or not (self._recent_athlete_descending() or self._athlete_is_low()):
                return None
            sample = samples[0]
            zone = classify_contact_zone(sample.point, target)
            if zone == "UNKNOWN_TARGET":
                return None
            if zone == "OUTSIDE_TARGET" and target is not None:
                return None
            return PhysicalContactCandidate(
                frame_index=sample.frame_index,
                timestamp=sample.timestamp,
                foot=track.foot,
                part=track.part,
                point=sample.point,
                zone=zone,
                descent_px=self._recent_athlete_descent_px(),
                pre_speed_px=self._recent_athlete_descent_px(),
                post_speed_px=0.0,
            )
        return None

    def _track_confirms_contact(self, track: FootPartTrack) -> bool:
        if track.pending is None:
            return False
        samples = [sample for sample in track.samples if sample.frame_index >= track.pending.frame_index]
        if len(samples) >= 2 and self._recent_athlete_descending():
            if classify_contact_zone(samples[0].point, self.latest_target) in {"BULLSEYE", "TARGET_DISK"}:
                return True
        if len(samples) < TOUCH_CONFIRMATION_FRAMES + 1:
            return False
        ys = [sample.point[1] for sample in samples]
        return max(ys) - min(ys) <= 12 or ys[-1] <= ys[0] + 4

    def _recent_athlete_descending(self) -> bool:
        return self._recent_athlete_descent_px() >= MIN_DESCENT_PX

    def _recent_athlete_descent_px(self) -> float:
        if len(self.athlete_bottom_history) < 2:
            return 0.0
        return float(self.athlete_bottom_history[-1][1] - self.athlete_bottom_history[0][1])

    def _athlete_is_low(self) -> bool:
        if not self.athlete_bottom_history:
            return False
        return self.athlete_bottom_history[-1][1] / max(self.current_frame_height, 1) >= foot_demo.MIN_BODY_BOTTOM_RATIO_FOR_CONTACT

    def _best_physical_candidate(self, candidates: list[PhysicalContactCandidate]) -> PhysicalContactCandidate:
        zone_rank = {"BULLSEYE": 3, "TARGET_DISK": 2, "OUTSIDE_TARGET": 1, "UNKNOWN_TARGET": 0}
        return max(
            candidates,
            key=lambda item: (
                zone_rank.get(item.zone, 0),
                item.point[1],
                item.descent_px,
            ),
        )

    def _mark_missing_tracks(self, active_keys: set[tuple[str, str]]) -> None:
        for key, track in self.foot_part_tracks.items():
            if key in active_keys:
                continue
            track.missing_frames += 1
            if track.missing_frames > 3:
                track.samples.clear()
                track.pending = None

    def _motion_status_text(self) -> str:
        if not self.foot_part_tracks:
            return "WAIT_FOOT_TRACK"
        if any(track.pending is not None for track in self.foot_part_tracks.values()):
            return "TOUCH_CANDIDATE"
        if any(len(track.samples) >= 2 and track.samples[-1].point[1] > track.samples[0].point[1] for track in self.foot_part_tracks.values()):
            return "DESCENDING"
        return "WAIT_TOUCH"

    @staticmethod
    def _中文状态(status: str) -> str:
        labels = {
            "LOCKED": "已锁定运动员", "CANDIDATE": "候选运动员", "NO PERSON": "未检测到人体",
            "FALLBACK": "备用人体", "WAIT": "等待", "WAIT_PERSON": "等待人体",
            "WAIT_PERSON_LOW": "等待人体靠近地面", "WAIT_FOOT_TRACK": "等待脚部轨迹",
            "TOUCH_CANDIDATE": "触地候选", "DESCENDING": "正在下降", "WAIT_TOUCH": "等待触地",
            "FIRST_CONFIRMED": "首次触地已确认", "PAUSED": "已暂停", "PLAY": "播放中",
            "detected": "已检测", "fallback": "备用估计", "manual": "手动设定",
            "airborne": "空中", "grounded": "已落地", "other": "其他",
            "short_track": "轨迹过短", "static": "静止", "not_airborne": "非空中状态",
        }
        return labels.get(status, status)

    @staticmethod
    def _中文脚部(foot: str) -> str:
        return {"left": "左脚", "right": "右脚"}.get(foot, foot)

    @staticmethod
    def _中文部位(part: str) -> str:
        return {"heel": "脚跟", "toe": "脚尖", "ankle": "脚踝"}.get(part, part)

    @staticmethod
    def _中文区域(zone: str) -> str:
        return {
            "BULLSEYE": "黑色靶心", "TARGET_DISK": "靶盘内",
            "OUTSIDE_TARGET": "靶盘外", "UNKNOWN_TARGET": "靶盘未知",
        }.get(zone, zone)

    def _运动员状态文字(self) -> str:
        selector = self.athlete_selector
        locked = selector.locked_track_id if selector.locked_track_id is not None else "-"
        tracks = selector._tracks
        best = max((track.last_score for track in tracks), default=0.0)
        track = next((item for item in tracks if item.track_id == selector.locked_track_id), None)
        if track is None and tracks:
            track = max(tracks, key=lambda item: item.last_score)
        state = "-" if track is None else self._中文状态(track.state)
        rejected = "-" if track is None or track.rejected_reason is None else self._中文状态(track.rejected_reason)
        return f"运动员轨迹={locked}｜状态={state}｜轨迹数={len(tracks)}｜评分={best:.1f}｜排除原因={rejected}"

    def _人员标签(self, detection: PoseDetection) -> str:
        raw = self.athlete_selector.label_for_detection(detection).split()
        role = "运动员" if raw and raw[0] == "athlete" else "其他人员"
        state = self._中文状态(raw[1]) if len(raw) > 1 else ""
        return f"{role} {state}".strip()

    def _draw_other_detections(self, frame: np.ndarray, detections: list[PoseDetection], selected: PoseDetection | None) -> None:
        selected_bbox = selected.bbox if selected is not None else None
        for detection in detections:
            if detection.bbox == selected_bbox:
                continue
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
            self._draw_text(frame, f"{self._人员标签(detection)} {detection.confidence:.2f}", (x1, max(18, y1 - 6)), (160, 160, 160), 0.5, 1)

    def _draw_pose(self, frame: np.ndarray, detection: PoseDetection) -> None:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 3)
        self._draw_text(frame, f"{self._人员标签(detection)} {detection.confidence:.2f}", (x1, max(28, y1 - 10)), (60, 220, 60), 0.7, 2)
        for start, end in foot_demo.COCO_SKELETON:
            p1, p2 = self._visible_point(detection, start), self._visible_point(detection, end)
            if p1 is not None and p2 is not None:
                cv2.line(frame, p1, p2, (255, 180, 80), 2)
        for index in range(len(detection.keypoints)):
            point = self._visible_point(detection, index)
            if point is None:
                continue
            color, radius = ((255, 40, 40), 8) if index == foot_demo.LEFT_ANKLE_INDEX else ((40, 40, 255), 8) if index == foot_demo.RIGHT_ANKLE_INDEX else ((80, 220, 255), 4)
            cv2.circle(frame, point, radius, color, -1)

    def _draw_foot_parts(self, frame: np.ndarray, estimate, ground_line: GroundLine) -> None:
        if estimate is None:
            return
        for foot_name, foot in (("left", estimate.left), ("right", estimate.right)):
            for part, landmark, color in (("ankle", foot.ankle, (190, 190, 190)), ("heel", foot.heel, (0, 170, 255)), ("toe", foot.toe, (0, 255, 255))):
                if landmark is not None:
                    cv2.circle(frame, landmark.point, 7, color, -1)
                    self._draw_text(frame, f"{self._中文脚部(foot_name)}{self._中文部位(part)}", (landmark.point[0] + 8, landmark.point[1] - 8), color, 0.45, 1)
            nearest = self._nearest_part_for_foot(foot_name)
            if nearest is not None:
                color = (0, 255, 0) if abs(nearest.distance_px) <= foot_demo.CONTACT_THRESHOLD_PX else (0, 220, 255)
                ground_point = (nearest.point[0], int(round(ground_line.y_at(nearest.point[0]))))
                cv2.line(frame, nearest.point, ground_point, color, 1, cv2.LINE_AA)
                self._draw_text(frame, f"{self._中文部位(nearest.part)} 距地面 {nearest.distance_px:+.1f}px", (nearest.point[0] + 8, nearest.point[1] + 18), color, 0.45, 1)

    def _draw_ground_line(self, frame: np.ndarray, ground_line: GroundLine) -> None:
        height, width = frame.shape[:2]
        y_left, y_right = int(round(ground_line.y_at(0))), int(round(ground_line.y_at(width - 1)))
        color = (0, 180, 255) if not ground_line.fallback else (0, 120, 255)
        cv2.line(frame, (0, y_left), (width - 1, y_right), color, 2)
        label = "地面线（手动）" if foot_demo.USE_MANUAL_GROUND_LINE else "地面线（备用估计）" if ground_line.fallback else "地面线"
        self._draw_text(frame, label, (12, max(24, min(height - 12, y_left - 10))), color, 0.6, 2)

    def _draw_horizon_line(self, frame: np.ndarray, horizon_line: GroundLine) -> None:
        height, width = frame.shape[:2]
        y_left, y_right = int(round(horizon_line.y_at(0))), int(round(horizon_line.y_at(width - 1)))
        color = (255, 160, 0)
        cv2.line(frame, (0, y_left), (width - 1, y_right), color, 2)
        self._draw_text(frame, "地平线", (12, max(24, min(height - 12, y_left - 10))), color, 0.6, 2)

    def _draw_landing_marker(self, frame: np.ndarray) -> None:
        result = self.first_landing_result or self.latest_landing_candidate
        if result is None:
            return
        confirmed = self.first_landing_result is not None
        color = zone_color(result.zone) if confirmed else (0, 255, 0)
        x, y = result.point
        cv2.circle(frame, (x, y), 13, color, -1)
        cv2.circle(frame, (x, y), 20, (255, 255, 255), 2)
        prefix = "首次触地" if confirmed else "触地候选"
        label = f"{prefix}｜{self._中文脚部(result.foot)} {self._中文部位(result.part)}｜{self._中文区域(result.zone)}"
        self._draw_text(frame, label, (max(10, x - 190), max(32, y - 30)), color, 0.72, 2)

    def _draw_combined_overlay(
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
        status = "已暂停" if self.paused else "播放中"
        armed = detection is not None and self._contact_detection_is_armed(frame, detection)
        target_text = self._target_status_text()
        result_text = self._landing_result_text()
        contact_status = "首次触地已确认" if self.first_landing_result is not None else self._中文状态(self.contact_status)
        lines = [
            f"{status}｜{self._中文状态(self.latest_detection_mode)}｜{target_text}｜帧 {frame_index + 1}/{frame_count}｜时间={timestamp:.2f}秒",
            f"推理速度={infer_fps:.1f} FPS｜人数={detection_count}｜触地={contact_status if armed else '等待'}｜结果={result_text}",
            self._运动员状态文字(),
            self._distance_summary("left"),
            self._distance_summary("right"),
            f"地面线={self._中文状态(ground_line.status)}｜地平线={'开启' if horizon_line is not None else '关闭'}｜视频={self.video_path.name}",
            "空格：暂停/播放｜n：下一帧｜m：上一帧｜r：重新开始｜q/Esc：退出",
        ]
        y = 28
        for line in lines:
            self._draw_text(frame, line, (12, y), (255, 255, 255), 0.62, 2)
            y += 27

    def _target_status_text(self) -> str:
        target = self.latest_target
        if target is None:
            return "靶盘：未检测到"
        if self.locked_target is not None:
            return "靶盘：位置已锁定"
        if target.bullseye_detected:
            return "靶盘：已检测到"
        return f"靶盘：暂时遮挡（{target.occluded_frames}帧）"

    def _landing_result_text(self) -> str:
        result = self.first_landing_result or self.latest_landing_candidate
        if result is None:
            return "-"
        prefix = "首次触地" if self.first_landing_result is not None else "触地候选"
        return f"{prefix} {self._中文脚部(result.foot)} {self._中文部位(result.part)} 坐标={result.point} {self._中文区域(result.zone)}"

    def _distance_summary(self, foot_name: str) -> str:
        items = [item for item in self.latest_distances if item.foot == foot_name]
        label = self._中文脚部(foot_name)
        if not items:
            return f"{label}：脚跟=－｜脚尖=－"
        by_part = {item.part: item.distance_px for item in items}
        heel = by_part.get("heel")
        toe = by_part.get("toe")
        heel_text = "－" if heel is None else f"{heel:+.1f}px"
        toe_text = "－" if toe is None else f"{toe:+.1f}px"
        return f"{label}：脚跟={heel_text}｜脚尖={toe_text}"

    def _print_first_landing_result(self) -> None:
        if self.first_landing_result is None or self.first_contact_printed:
            return
        result = self.first_landing_result
        print(
            "靶盘首次触地："
            f"帧={result.frame_index}，"
            f"时间={result.timestamp:.3f}秒，"
            f"脚={self._中文脚部(result.foot)}，"
            f"部位={self._中文部位(result.part)}，"
            f"坐标={result.point}，"
            f"区域={self._中文区域(result.zone)}"
        )
        self.first_contact_printed = True

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        origin: Point,
        color: tuple[int, int, int],
        scale: float,
        thickness: int,
    ) -> None:
        font_size = max(12, int(round(scale * 26)))
        font = _load_chinese_font(font_size)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        rgb_color = (color[2], color[1], color[0])
        draw.text((origin[0], origin[1] - font_size), text, font=font, fill=rgb_color,
                  stroke_width=max(1, thickness), stroke_fill=(0, 0, 0))
        frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def _round_point(self, point: tuple[float, float]) -> Point:
        return int(round(point[0])), int(round(point[1]))

    def _reset_runtime_state(self) -> None:
        super()._reset_runtime_state()
        self.target_detector.reset()
        self.camera_motion_monitor.reset()
        self.latest_target = None
        self.latest_target_detection = None
        self.locked_target = None
        self.target_lock_confirmation_frames = 0
        self.camera_moved = False
        self.latest_landing_candidate = None
        self.first_landing_result = None
        self.foot_part_tracks.clear()
        self.latest_physical_candidate = None
        self.contact_status = "WAIT"
        self.athlete_bottom_history.clear()

    def _unlock_target_after_camera_move(self) -> None:
        """相机平移/抖动超过阈值后，丢弃旧靶盘并重新建立稳定锁定。"""
        self.locked_target = None
        self.latest_target = None
        self.latest_target_detection = None
        self.target_lock_confirmation_frames = 0
        self.target_detector.reset()
        self.camera_motion_monitor.recent_motion.clear()
        print("检测到相机明显移动，已解除靶盘锁定并开始重新识别。")


def point_in_ellipse(
    point: Point,
    ellipse: tuple[tuple[float, float], tuple[float, float], float],
    margin_px: float = 0.0,
) -> bool:
    (center_x, center_y), (axis_a, axis_b), angle_deg = ellipse
    radius_a = max(axis_a / 2.0 + margin_px, 1.0)
    radius_b = max(axis_b / 2.0 + margin_px, 1.0)
    angle = math.radians(angle_deg)
    dx = point[0] - center_x
    dy = point[1] - center_y
    rotated_x = dx * math.cos(angle) + dy * math.sin(angle)
    rotated_y = -dx * math.sin(angle) + dy * math.cos(angle)
    return (rotated_x * rotated_x) / (radius_a * radius_a) + (rotated_y * rotated_y) / (radius_b * radius_b) <= 1.0


def classify_contact_zone(point: Point, target: SmoothedTarget | None) -> str:
    if target is None:
        return "UNKNOWN_TARGET"
    if target.ellipse is not None and point_in_ellipse(point, target.ellipse, BULLSEYE_MARGIN_PX):
        return "BULLSEYE"
    if _point_in_bullseye_circle(point, target, BULLSEYE_MARGIN_PX):
        return "BULLSEYE"
    if point_in_ellipse(point, target.disk_ellipse, TARGET_MARGIN_PX):
        return "TARGET_DISK"
    return "OUTSIDE_TARGET"


def _point_in_bullseye_circle(point: Point, target: SmoothedTarget, margin_px: float) -> bool:
    dx = point[0] - target.center[0]
    dy = point[1] - target.center[1]
    return dx * dx + dy * dy <= (target.radius_px + margin_px) ** 2


def point_distance(first: Point, second: Point) -> float:
    return float(((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5)


def zone_color(zone: str) -> tuple[int, int, int]:
    if zone == "BULLSEYE":
        return (0, 0, 255)
    if zone == "TARGET_DISK":
        return (0, 220, 255)
    if zone == "OUTSIDE_TARGET":
        return (255, 80, 80)
    return (180, 180, 180)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时识别滑翔伞首次触地位置并判定靶盘区域。")
    parser.add_argument("--video", default=VIDEO_PATH, help="输入视频路径。")
    parser.add_argument("--display-scale", type=float, default=0.75, help="预览窗口缩放比例。")
    return parser.parse_args()


def main() -> int:
    foot_demo.configure_logging()
    args = parse_args()
    browser = TargetLandingContactBrowser(
        video_path=resolve_path(args.video),
        display_scale=max(args.display_scale, 0.05),
    )
    return browser.run()


if __name__ == "__main__":
    raise SystemExit(main())
