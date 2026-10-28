"""Shared data models for landing first-contact detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


Point = tuple[int, int]


@dataclass(frozen=True)
class FootPoints:
    """Visible foot keypoints for a single frame."""

    left: Point | None
    right: Point | None

    def lowest(self) -> Point | None:
        """Return the visible foot point closest to the bottom of the image."""
        points = [point for point in (self.left, self.right) if point is not None]
        if not points:
            return None
        return max(points, key=lambda point: point[1])


@dataclass(frozen=True)
class PoseDetection:
    """Highest-confidence person pose from one frame."""

    bbox: tuple[int, int, int, int]
    confidence: float
    keypoints: np.ndarray
    keypoint_confidences: np.ndarray | None
    foot_points: FootPoints


@dataclass(frozen=True)
class GroundLine:
    """Image-space ground line represented as y = slope * x + intercept."""

    slope: float
    intercept: float
    confidence: float
    fallback: bool = False
    status: str = "detected"

    def y_at(self, x: float) -> float:
        return self.slope * x + self.intercept

    def distance_to_point(self, point: Point) -> float:
        """Signed vertical distance in pixels; positive means point is above ground."""
        x, y = point
        return self.y_at(x) - y


@dataclass(frozen=True)
class ContactCandidate:
    """Per-frame contact candidate before consecutive-frame confirmation."""

    frame_index: int
    point: Point
    foot: str
    distance_px: float


@dataclass(frozen=True)
class FirstContact:
    """Confirmed first ground contact."""

    frame_index: int
    timestamp: float
    point: Point
    foot: str
    contact_part: str = "unknown"
    contact_part_status: str = "unknown"


@dataclass(frozen=True)
class LandingResult:
    """Serializable landing analysis result."""

    video_name: str
    first_contact: FirstContact | None

    def to_json_dict(self) -> dict[str, int | float | str | None]:
        if self.first_contact is None:
            return {
                "video_name": self.video_name,
                "first_contact_frame": None,
                "timestamp": None,
                "contact_x": None,
                "contact_y": None,
                "contact_foot": None,
                "contact_part": None,
                "contact_part_status": None,
            }

        x, y = self.first_contact.point
        return {
            "video_name": self.video_name,
            "first_contact_frame": self.first_contact.frame_index,
            "timestamp": round(self.first_contact.timestamp, 3),
            "contact_x": x,
            "contact_y": y,
            "contact_foot": self.first_contact.foot,
            "contact_part": self.first_contact.contact_part,
            "contact_part_status": self.first_contact.contact_part_status,
        }


@dataclass(frozen=True)
class VideoInfo:
    """Basic OpenCV video metadata."""

    path: Path
    fps: float
    width: int
    height: int
    frame_count: int
