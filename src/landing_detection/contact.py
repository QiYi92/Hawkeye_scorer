"""First-contact state machine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .models import ContactCandidate, FirstContact, FootPoints, GroundLine, Point, PoseDetection


@dataclass(frozen=True)
class FirstContactConfig:
    contact_threshold_px: float = 5.0
    consecutive_frames: int = 3


class FirstContactDetector:
    """Confirm first foot contact only after N consecutive contact frames."""

    def __init__(self, config: FirstContactConfig) -> None:
        self.config = config
        self._streak: list[ContactCandidate] = []
        self.first_contact: FirstContact | None = None

    def reset_pending(self) -> None:
        """Clear unconfirmed contact frames without removing a confirmed contact."""
        self._streak.clear()

    def update(
        self,
        frame_index: int,
        fps: float,
        foot_points: FootPoints,
        ground_line: GroundLine,
    ) -> FirstContact | None:
        if self.first_contact is not None:
            return self.first_contact

        candidate = self._candidate_for_frame(frame_index, foot_points, ground_line)
        if candidate is None:
            self._streak.clear()
            return None

        self._streak.append(candidate)
        if len(self._streak) > self.config.consecutive_frames:
            self._streak = self._streak[-self.config.consecutive_frames :]

        if len(self._streak) >= self.config.consecutive_frames:
            confirmed = self._streak[0]
            self.first_contact = FirstContact(
                frame_index=confirmed.frame_index,
                timestamp=confirmed.frame_index / max(fps, 0.001),
                point=confirmed.point,
                foot=confirmed.foot,
            )
        return self.first_contact

    def _candidate_for_frame(
        self,
        frame_index: int,
        foot_points: FootPoints,
        ground_line: GroundLine,
    ) -> ContactCandidate | None:
        candidates: list[ContactCandidate] = []
        for foot_name, point in (("left", foot_points.left), ("right", foot_points.right)):
            if point is None:
                continue
            distance_px = ground_line.distance_to_point(point)
            if abs(distance_px) <= self.config.contact_threshold_px:
                candidates.append(
                    ContactCandidate(
                        frame_index=frame_index,
                        point=point,
                        foot=foot_name,
                        distance_px=distance_px,
                    )
                )

        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(item.distance_px))


@dataclass(frozen=True)
class MotionContactConfig:
    consecutive_frames: int = 3
    history_frames: int = 8
    min_descent_px: int = 35
    foot_to_bbox_bottom_max_gap_px: int = 75
    min_body_bottom_ratio: float = 0.64


class MotionContactDetector:
    """Detect first contact from a locked athlete's descending foot motion."""

    def __init__(self, config: MotionContactConfig) -> None:
        self.config = config
        self._history: deque[tuple[int, Point, int, str]] = deque(maxlen=config.history_frames)
        self._streak: list[FirstContact] = []
        self.first_contact: FirstContact | None = None

    def reset_pending(self) -> None:
        self._streak.clear()

    def update(
        self,
        frame_index: int,
        fps: float,
        detection: PoseDetection,
        frame_height: int,
        contact_point: Point | None = None,
        contact_part: str = "unknown",
        contact_part_status: str = "unknown",
    ) -> FirstContact | None:
        if self.first_contact is not None:
            return self.first_contact

        foot, point = self._lowest_foot_with_side(detection.foot_points)
        if point is None:
            self._streak.clear()
            return None

        self._history.append((frame_index, point, detection.bbox[3], foot))
        if detection.bbox[3] / max(frame_height, 1) < self.config.min_body_bottom_ratio:
            self._streak.clear()
            return None
        if len(self._history) < min(4, self.config.history_frames):
            return None

        _, earliest_point, _, _ = self._history[0]
        descent_px = point[1] - earliest_point[1]
        foot_to_bbox_bottom_gap_px = detection.bbox[3] - point[1]
        if not (
            descent_px >= self.config.min_descent_px
            and 0 <= foot_to_bbox_bottom_gap_px <= self.config.foot_to_bbox_bottom_max_gap_px
        ):
            self._streak.clear()
            return None

        candidate = FirstContact(
            frame_index=frame_index,
            timestamp=frame_index / max(fps, 0.001),
            point=contact_point or point,
            foot=foot,
            contact_part=contact_part,
            contact_part_status=contact_part_status,
        )
        self._streak.append(candidate)
        if len(self._streak) > self.config.consecutive_frames:
            self._streak = self._streak[-self.config.consecutive_frames :]

        if len(self._streak) >= self.config.consecutive_frames:
            self.first_contact = self._best_contact(self._streak)
        return self.first_contact

    def _best_contact(self, contacts: list[FirstContact]) -> FirstContact:
        trusted_contacts = [contact for contact in contacts if contact.contact_part_status == "trusted"]
        if trusted_contacts:
            return max(trusted_contacts, key=lambda contact: contact.point[1])
        known_contacts = [contact for contact in contacts if contact.contact_part != "unknown"]
        candidates = known_contacts or contacts
        return max(candidates, key=lambda contact: contact.point[1])

    def _lowest_foot_with_side(self, foot_points: FootPoints) -> tuple[str, Point | None]:
        left = foot_points.left
        right = foot_points.right
        if left is None and right is None:
            return "unknown", None
        if left is None:
            return "right", right
        if right is None:
            return "left", left
        if left[1] >= right[1]:
            return "left", left
        return "right", right
