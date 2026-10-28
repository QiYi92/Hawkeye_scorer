"""Visualization utilities for landing analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .models import FirstContact, GroundLine, Point, PoseDetection


COCO_SKELETON = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass
class LandingVisualizer:
    """Draw pose, foot trails, ground line, and first-contact marker."""

    max_trail_length: int = 180
    left_trail: deque[Point] = field(default_factory=deque)
    right_trail: deque[Point] = field(default_factory=deque)

    def draw(
        self,
        frame: np.ndarray,
        detection: PoseDetection | None,
        ground_line: GroundLine,
        first_contact: FirstContact | None,
    ) -> np.ndarray:
        output = frame.copy()
        self._draw_ground_line(output, ground_line)
        if detection is not None:
            self._update_trails(detection)
            self._draw_pose(output, detection)
        self._draw_trails(output)
        if first_contact is not None:
            self._draw_first_contact(output, first_contact)
        return output

    def _update_trails(self, detection: PoseDetection) -> None:
        if detection.foot_points.left is not None:
            self.left_trail.append(detection.foot_points.left)
        if detection.foot_points.right is not None:
            self.right_trail.append(detection.foot_points.right)
        while len(self.left_trail) > self.max_trail_length:
            self.left_trail.popleft()
        while len(self.right_trail) > self.max_trail_length:
            self.right_trail.popleft()

    def _draw_ground_line(self, frame: np.ndarray, ground_line: GroundLine) -> None:
        height, width = frame.shape[:2]
        y_left = int(round(ground_line.y_at(0)))
        y_right = int(round(ground_line.y_at(width - 1)))
        color = (0, 180, 255) if not ground_line.fallback else (0, 120, 255)
        cv2.line(frame, (0, y_left), (width - 1, y_right), color, 2)
        label = "GROUND" if not ground_line.fallback else "GROUND FALLBACK"
        if ground_line.status not in ("detected", "fallback"):
            label = f"GROUND {ground_line.status.upper()}"
        cv2.putText(
            frame,
            label,
            (12, max(24, min(height - 12, y_left - 10))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_pose(self, frame: np.ndarray, detection: PoseDetection) -> None:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
        cv2.putText(
            frame,
            f"person {detection.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (60, 220, 60),
            2,
            cv2.LINE_AA,
        )

        keypoints = detection.keypoints
        for start, end in COCO_SKELETON:
            if start < len(keypoints) and end < len(keypoints):
                p1 = self._visible_point(detection, start)
                p2 = self._visible_point(detection, end)
                if p1 is not None and p2 is not None:
                    cv2.line(frame, p1, p2, (255, 180, 80), 2)

        for index in range(len(keypoints)):
            point = self._visible_point(detection, index)
            if point is not None:
                color = (255, 80, 80) if index in (15, 16) else (80, 220, 255)
                cv2.circle(frame, point, 4, color, -1)

    def _visible_point(self, detection: PoseDetection, index: int) -> Point | None:
        if detection.keypoint_confidences is not None and detection.keypoint_confidences[index] < 0.20:
            return None
        x, y = detection.keypoints[index][:2]
        if x <= 0 and y <= 0:
            return None
        return int(round(float(x))), int(round(float(y)))

    def _draw_trails(self, frame: np.ndarray) -> None:
        self._draw_trail(frame, list(self.left_trail), (255, 80, 80))
        self._draw_trail(frame, list(self.right_trail), (80, 80, 255))

    def _draw_trail(self, frame: np.ndarray, points: list[Point], color: tuple[int, int, int]) -> None:
        for p1, p2 in zip(points, points[1:]):
            cv2.line(frame, p1, p2, color, 2)
        if points:
            cv2.circle(frame, points[-1], 5, color, -1)

    def _draw_first_contact(self, frame: np.ndarray, first_contact: FirstContact) -> None:
        x, y = first_contact.point
        cv2.circle(frame, (x, y), 12, (0, 0, 255), -1)
        cv2.circle(frame, (x, y), 18, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"FIRST CONTACT {first_contact.contact_part} {first_contact.contact_part_status}",
            (max(10, x - 120), max(32, y - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
