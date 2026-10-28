#!/usr/bin/env python3
"""Browse a video with live YOLO Pose annotations.

Open this file in VSCode, edit VIDEO_PATH below, then click "Run Python File".
"""

from __future__ import annotations

# ===== 在这里切换要浏览的视频 =====
VIDEO_PATH = "training_video/video8.mp4"
MODEL_NAME = "yolo11n-pose.pt"
CONFIDENCE_THRESHOLD = 0.25
KEYPOINT_CONFIDENCE_THRESHOLD = 0.35
CONTACT_THRESHOLD_PX = 5.0
CONTACT_FRAMES = 3
SHOW_HORIZON_LINE = True
SHOW_REJECTED_FOOT_PARTS = True
HORIZON_MAX_Y_RATIO = 0.70
GROUND_MIN_GAP_BELOW_HORIZON_RATIO = 0.12
GROUND_MIN_Y_RATIO = 0.70
MIN_BODY_BOTTOM_RATIO_FOR_CONTACT = 0.64
ENABLE_MOTION_CONTACT = True
MOTION_HISTORY_FRAMES = 8
MOTION_MIN_DESCENT_PX = 35
FOOT_TO_BBOX_BOTTOM_MAX_GAP_PX = 75
USE_MANUAL_GROUND_LINE = False
MANUAL_GROUND_Y_RATIO = 0.88
DISPLAY_SCALE = 0.75
# ==============================

import logging
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from landing_detection.contact import FirstContactConfig, FirstContactDetector
from landing_detection.athlete import AthleteSelector
from landing_detection.foot_parts import FootPartEstimate, FootPartEstimator
from landing_detection.ground import GroundLineConfig, GroundLineEstimator, HorizonLineConfig, HorizonLineEstimator
from landing_detection.models import FirstContact, FootPoints, GroundLine, Point, PoseDetection
from landing_detection.pose import PoseDetector, PoseDetectorConfig
from landing_detection.visualizer import COCO_SKELETON


WINDOW_NAME = "YOLO Pose Video Browser"
LEFT_ANKLE_INDEX = 15
RIGHT_ANKLE_INDEX = 16


@dataclass(frozen=True)
class MotionContact:
    frame_index: int
    timestamp: float
    point: Point
    raw_point: Point
    descent_px: int
    foot_to_bbox_bottom_gap_px: int
    foot: str
    contact_part: str = "unknown"
    contact_part_status: str = "unknown"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class YoloVideoBrowser:
    """Small OpenCV video browser for inspecting YOLO Pose behavior."""

    def __init__(self) -> None:
        self.video_path = self._resolve_video_path(VIDEO_PATH)
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
        self.left_trail: deque[Point] = deque(maxlen=180)
        self.right_trail: deque[Point] = deque(maxlen=180)
        self.ground_estimator = GroundLineEstimator(
            GroundLineConfig(
                min_candidate_y_ratio=GROUND_MIN_Y_RATIO,
                min_gap_below_horizon_ratio=GROUND_MIN_GAP_BELOW_HORIZON_RATIO,
            )
        )
        self.contact_detector = FirstContactDetector(
            FirstContactConfig(
                contact_threshold_px=CONTACT_THRESHOLD_PX,
                consecutive_frames=CONTACT_FRAMES,
            )
        )
        self.lowest_foot_history: deque[tuple[int, Point, int]] = deque(maxlen=MOTION_HISTORY_FRAMES)
        self.motion_contact_streak: list[MotionContact] = []
        self.motion_contact: MotionContact | None = None
        self.current_track_id: int | None = None
        self.first_contact_printed = False
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

        print(f"视频：{self.video_path}")
        print(f"尺寸：{width}x{height}，FPS：{fps:.2f}，总帧数：{frame_count}")
        print("按键：Space 暂停/继续，n 暂停时前进一帧，m 暂停时后退一帧，r 回到开头，q 或 Esc 退出。")

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
                            self.left_trail.clear()
                            self.right_trail.clear()
                            self._reset_contact_detection()
                            continue

                        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                        annotated = self._annotate_frame(frame, frame_index, fps, frame_count)
                        annotated_cache[frame_index] = annotated
                        current_frame_index = frame_index
                    self.step_once = False

                if annotated is None:
                    continue
                display = self._resize_for_display(annotated)
                cv2.imshow(WINDOW_NAME, display)

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
                        self.left_trail.clear()
                        self.right_trail.clear()
                        self._reset_contact_detection()
                        self.step_once = True
                elif key == ord("r"):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    annotated_cache.clear()
                    current_frame_index = -1
                    self.left_trail.clear()
                    self.right_trail.clear()
                    self._reset_contact_detection()
                    self.paused = False
        finally:
            capture.release()
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
        pre_ground_line = self._estimate_ground_line(
            frame,
            horizon_line=horizon_line,
            athlete_bbox=None,
        )
        detection = self.athlete_selector.update(
            detections,
            frame_index,
            ground_line=pre_ground_line,
            frame_shape=frame.shape,
        )
        selected_track_id = self.athlete_selector.track_id_for_detection(detection)
        if selected_track_id != self.current_track_id:
            self.left_trail.clear()
            self.right_trail.clear()
            self.lowest_foot_history.clear()
            self.motion_contact_streak.clear()
            self.foot_part_estimator.reset_history()
            self.current_track_id = selected_track_id
        self.last_inference_seconds = time.perf_counter() - started_at

        output = frame.copy()
        ground_line = self._estimate_ground_line(
            frame,
            horizon_line=horizon_line,
            athlete_bbox=detection.bbox if detection is not None else None,
        )
        foot_part_estimate = None
        self._draw_other_detections(output, detections, detection)
        if detection is not None:
            foot_part_estimate = self.foot_part_estimator.estimate(
                frame,
                detection,
                timestamp_ms=int(round(frame_index / max(fps, 0.001) * 1000)),
            )
            self._update_trails(detection)
            self._update_motion_history(frame_index, detection)
            if self._contact_detection_is_armed(frame, detection):
                refined_foot_points = self._refined_foot_points(detection, foot_part_estimate)
                first_contact = self.contact_detector.update(
                    frame_index=frame_index,
                    fps=fps,
                    foot_points=refined_foot_points,
                    ground_line=ground_line,
                )
                if first_contact is not None and not self.first_contact_printed:
                    refined_contact = self.foot_part_estimator.refine_contact(
                        foot_part_estimate,
                        first_contact.foot,
                        first_contact.point,
                        detection,
                    )
                    first_contact = FirstContact(
                        frame_index=first_contact.frame_index,
                        timestamp=first_contact.timestamp,
                        point=refined_contact.point,
                        foot=first_contact.foot,
                        contact_part=refined_contact.part,
                        contact_part_status=refined_contact.status,
                    )
                    self.contact_detector.first_contact = first_contact
                    print(
                        "首次触地："
                        f"frame={first_contact.frame_index}, "
                        f"time={first_contact.timestamp:.3f}s, "
                        f"point={first_contact.point}, "
                        f"foot={first_contact.foot}, "
                        f"contact_part={first_contact.contact_part}, "
                        f"contact_part_status={first_contact.contact_part_status}"
                    )
                    self.first_contact_printed = True
                motion_contact = self._update_motion_contact(
                    frame_index,
                    fps,
                    detection,
                    foot_part_estimate,
                )
                if (
                    motion_contact is not None
                    and self.contact_detector.first_contact is None
                    and not self.first_contact_printed
                ):
                    print(
                        "首次触地（脚部运动判定）："
                        f"frame={motion_contact.frame_index}, "
                        f"time={motion_contact.timestamp:.3f}s, "
                        f"point={motion_contact.point}, "
                        f"raw_ankle_point={motion_contact.raw_point}, "
                        f"descent={motion_contact.descent_px}px, "
                        f"foot={motion_contact.foot}, "
                        f"contact_part={motion_contact.contact_part}, "
                        f"contact_part_status={motion_contact.contact_part_status}"
                    )
                    self.first_contact_printed = True
            elif self.contact_detector.first_contact is None:
                self.contact_detector.reset_pending()
                self.motion_contact_streak.clear()
            self._draw_pose(output, detection)
            self._draw_foot_parts(output, foot_part_estimate)
            if SHOW_REJECTED_FOOT_PARTS:
                self._draw_rejected_foot_parts(output)
        if SHOW_HORIZON_LINE and horizon_line is not None:
            self._draw_horizon_line(output, horizon_line)
        self._draw_ground_line(output, ground_line)
        self._draw_trails(output)
        if self.contact_detector.first_contact is not None or self.motion_contact is not None:
            self._draw_first_contact(output)
        self._draw_overlay(output, detection, len(detections), ground_line, horizon_line, frame_index, fps, frame_count)
        return output

    def _update_trails(self, detection: PoseDetection) -> None:
        if detection.foot_points.left is not None:
            self.left_trail.append(detection.foot_points.left)
        if detection.foot_points.right is not None:
            self.right_trail.append(detection.foot_points.right)

    def _update_motion_history(self, frame_index: int, detection: PoseDetection) -> None:
        point = detection.foot_points.lowest()
        if point is None:
            return
        self.lowest_foot_history.append((frame_index, point, detection.bbox[3]))

    def _update_motion_contact(
        self,
        frame_index: int,
        fps: float,
        detection: PoseDetection,
        foot_part_estimate: FootPartEstimate | None,
    ) -> MotionContact | None:
        if not ENABLE_MOTION_CONTACT or self.motion_contact is not None:
            return self.motion_contact
        if len(self.lowest_foot_history) < min(4, MOTION_HISTORY_FRAMES):
            return None

        foot, point = self._lowest_foot_with_side(detection)
        if point is None:
            self.motion_contact_streak.clear()
            return None

        _, earliest_point, _ = self.lowest_foot_history[0]
        descent_px = point[1] - earliest_point[1]
        foot_to_bbox_bottom_gap_px = detection.bbox[3] - point[1]
        is_candidate = (
            descent_px >= MOTION_MIN_DESCENT_PX
            and 0 <= foot_to_bbox_bottom_gap_px <= FOOT_TO_BBOX_BOTTOM_MAX_GAP_PX
        )

        if not is_candidate:
            self.motion_contact_streak.clear()
            return None

        refined_contact = self.foot_part_estimator.refine_contact(
            foot_part_estimate,
            foot,
            point,
            detection,
        )
        self.motion_contact_streak.append(
            MotionContact(
                frame_index=frame_index,
                timestamp=frame_index / max(fps, 0.001),
                point=refined_contact.point,
                raw_point=point,
                descent_px=descent_px,
                foot_to_bbox_bottom_gap_px=foot_to_bbox_bottom_gap_px,
                foot=foot,
                contact_part=refined_contact.part,
                contact_part_status=refined_contact.status,
            )
        )
        if len(self.motion_contact_streak) > CONTACT_FRAMES:
            self.motion_contact_streak = self.motion_contact_streak[-CONTACT_FRAMES:]

        if len(self.motion_contact_streak) >= CONTACT_FRAMES:
            self.motion_contact = self._best_motion_contact(self.motion_contact_streak)
        return self.motion_contact

    def _best_motion_contact(self, contacts: list[MotionContact]) -> MotionContact:
        trusted_contacts = [contact for contact in contacts if contact.contact_part_status == "trusted"]
        if trusted_contacts:
            return max(trusted_contacts, key=lambda contact: contact.point[1])
        known_contacts = [contact for contact in contacts if contact.contact_part != "unknown"]
        candidates = known_contacts or contacts
        return max(candidates, key=lambda contact: contact.point[1])

    def _lowest_foot_with_side(self, detection: PoseDetection) -> tuple[str, Point | None]:
        left = detection.foot_points.left
        right = detection.foot_points.right
        if left is None and right is None:
            return "unknown", None
        if left is None:
            return "right", right
        if right is None:
            return "left", left
        if left[1] >= right[1]:
            return "left", left
        return "right", right

    def _refined_foot_points(
        self,
        detection: PoseDetection,
        foot_part_estimate: FootPartEstimate | None,
    ) -> FootPoints:
        left = detection.foot_points.left
        right = detection.foot_points.right
        if left is not None:
            left = self.foot_part_estimator.refine_contact(
                foot_part_estimate,
                "left",
                left,
                detection,
            ).point
        if right is not None:
            right = self.foot_part_estimator.refine_contact(
                foot_part_estimate,
                "right",
                right,
                detection,
            ).point
        return FootPoints(left=left, right=right)

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
            if index == LEFT_ANKLE_INDEX:
                cv2.circle(frame, point, 9, (255, 40, 40), -1)
                cv2.putText(frame, "L ankle", (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 40, 40), 2)
            elif index == RIGHT_ANKLE_INDEX:
                cv2.circle(frame, point, 9, (40, 40, 255), -1)
                cv2.putText(frame, "R ankle", (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 255), 2)
            else:
                cv2.circle(frame, point, 4, (80, 220, 255), -1)

    def _draw_foot_parts(self, frame: np.ndarray, estimate: FootPartEstimate | None) -> None:
        if estimate is None:
            return
        for side, foot in (("L", estimate.left), ("R", estimate.right)):
            self._draw_foot_landmark(frame, foot.ankle, f"{side} mp ankle", (180, 180, 180))
            self._draw_foot_landmark(frame, foot.heel, f"{side} heel", (0, 180, 255))
            self._draw_foot_landmark(frame, foot.toe, f"{side} toe", (0, 255, 255))

    def _draw_foot_landmark(
        self,
        frame: np.ndarray,
        landmark,
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
            cv2.putText(
                frame,
                f"{landmark.side[0].upper()} raw {landmark.part}",
                (landmark.point[0] + 8, landmark.point[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
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

    def _draw_first_contact(self, frame: np.ndarray) -> None:
        first_contact = self.contact_detector.first_contact or self.motion_contact
        if first_contact is None:
            return
        x, y = first_contact.point
        cv2.circle(frame, (x, y), 12, (0, 0, 255), -1)
        cv2.circle(frame, (x, y), 18, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"FIRST CONTACT frame {first_contact.frame_index} {first_contact.contact_part} {first_contact.contact_part_status}",
            (max(10, x - 160), max(32, y - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
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
        found = "LOCKED" if detection is not None else "NO PERSON"
        armed = detection is not None and self._contact_detection_is_armed(frame, detection)
        method = "motion+line" if ENABLE_MOTION_CONTACT else "line"
        lines = [
            f"{status} | {found} | people={detection_count} | frame {frame_index + 1}/{frame_count} | t={timestamp:.2f}s",
            f"inference={infer_fps:.1f} fps | model={MODEL_NAME}",
            self.athlete_selector.status_text(),
            f"contact={'ARMED' if armed else 'WAIT'} | method={method} | line={CONTACT_THRESHOLD_PX:.1f}px x {CONTACT_FRAMES}",
            f"ground={ground_line.status} | horizon={'on' if horizon_line is not None else 'off'} | gap={GROUND_MIN_GAP_BELOW_HORIZON_RATIO:.2f}",
            f"ground_min_y={GROUND_MIN_Y_RATIO:.2f} | horizon_max_y={HORIZON_MAX_Y_RATIO:.2f} | body_min={MIN_BODY_BOTTOM_RATIO_FOR_CONTACT:.2f}",
            f"video={self.video_path}",
            "Space pause/play | n next frame | m prev frame | r restart | q/Esc quit",
        ]

        y = 28
        for line in lines:
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        if DISPLAY_SCALE == 1.0:
            return frame
        width = max(1, int(frame.shape[1] * DISPLAY_SCALE))
        height = max(1, int(frame.shape[0] * DISPLAY_SCALE))
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _resolve_video_path(self, path_text: str) -> Path:
        path = Path(path_text)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

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
        return "GROUND"

    def _reset_contact_detection(self) -> None:
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
        self.contact_detector = FirstContactDetector(
            FirstContactConfig(
                contact_threshold_px=CONTACT_THRESHOLD_PX,
                consecutive_frames=CONTACT_FRAMES,
            )
        )
        self.lowest_foot_history.clear()
        self.motion_contact_streak.clear()
        self.motion_contact = None
        self.current_track_id = None
        self.first_contact_printed = False


def main() -> int:
    configure_logging()
    return YoloVideoBrowser().run()


if __name__ == "__main__":
    raise SystemExit(main())
