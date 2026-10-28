"""Select the descending paraglider athlete from multiple detected people."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .models import GroundLine, Point, PoseDetection


@dataclass(frozen=True)
class AthleteSelectorConfig:
    max_track_distance_px: float = 180.0
    max_missing_frames: int = 12
    history_frames: int = 12
    lock_score_threshold: float = 35.0
    lock_frames: int = 3
    min_descent_px: float = 18.0
    grounded_distance_px: float = 35.0
    airborne_distance_px: float = 45.0
    static_motion_px: float = 10.0


@dataclass(frozen=True)
class TrackObservation:
    frame_index: int
    detection: PoseDetection
    center: tuple[float, float]
    area: float
    lowest_foot: Point | None
    ground_distance_px: float | None = None


@dataclass
class AthleteTrack:
    track_id: int
    observations: deque[TrackObservation]
    missing_frames: int = 0
    last_score: float = 0.0
    best_score_streak: int = 0
    state: str = "other"
    rejected_reason: str | None = None

    @property
    def latest(self) -> TrackObservation:
        return self.observations[-1]


class AthleteSelector:
    """Track people briefly and lock the one with a descending landing trajectory."""

    def __init__(self, config: AthleteSelectorConfig | None = None) -> None:
        self.config = config or AthleteSelectorConfig()
        self._tracks: list[AthleteTrack] = []
        self._next_track_id = 1
        self.locked_track_id: int | None = None

    def update(
        self,
        detections: list[PoseDetection],
        frame_index: int,
        ground_line: GroundLine | None = None,
        frame_shape: tuple[int, ...] | None = None,
    ) -> PoseDetection | None:
        self._associate_detections(detections, frame_index, ground_line)
        self._drop_stale_tracks()
        if not self._tracks:
            return None

        scored_tracks = [(self._score_track(track), track) for track in self._tracks]
        for score, track in scored_tracks:
            track.last_score = score
            track.state, track.rejected_reason = self._classify_track(track)

        if self.locked_track_id is not None:
            locked = self._track_by_id(self.locked_track_id)
            if locked is not None and locked.state == "airborne":
                return locked.latest.detection
            self.locked_track_id = None

        selectable_tracks = [
            (score, track) for score, track in scored_tracks if track.state == "airborne"
        ]
        if not selectable_tracks and len(detections) == 1:
            selectable_tracks = scored_tracks
        if not selectable_tracks:
            return None

        best_score, best_track = max(selectable_tracks, key=lambda item: item[0])
        for _, track in scored_tracks:
            if (
                track.track_id == best_track.track_id
                and best_score >= self.config.lock_score_threshold
                and best_track.state == "airborne"
            ):
                track.best_score_streak += 1
            else:
                track.best_score_streak = 0

        if best_track.best_score_streak >= self.config.lock_frames:
            self.locked_track_id = best_track.track_id
            return best_track.latest.detection

        # Before lock, show the current best moving candidate if it is plausible.
        if best_score >= self.config.min_descent_px:
            return best_track.latest.detection
        return None

    def status_text(self) -> str:
        locked = self.locked_track_id if self.locked_track_id is not None else "-"
        best = max((track.last_score for track in self._tracks), default=0.0)
        state = "-"
        rejected = "-"
        locked_track = self._track_by_id(self.locked_track_id) if self.locked_track_id is not None else None
        if locked_track is not None:
            state = locked_track.state
            rejected = locked_track.rejected_reason or "-"
        elif self._tracks:
            best_track = max(self._tracks, key=lambda track: track.last_score)
            state = best_track.state
            rejected = best_track.rejected_reason or "-"
        return f"athlete_track={locked} state={state} tracks={len(self._tracks)} score={best:.1f} rejected={rejected}"

    def track_id_for_detection(self, detection: PoseDetection | None) -> int | None:
        if detection is None:
            return None
        for track in self._tracks:
            if track.latest.detection.bbox == detection.bbox:
                return track.track_id
        return None

    def label_for_detection(self, detection: PoseDetection) -> str:
        for track in self._tracks:
            if track.latest.detection.bbox == detection.bbox:
                label = "athlete" if track.track_id == self.locked_track_id else "other"
                return f"{label} {track.state}"
        return "other"

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self.locked_track_id = None

    def _associate_detections(
        self,
        detections: list[PoseDetection],
        frame_index: int,
        ground_line: GroundLine | None,
    ) -> None:
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        while unmatched_tracks and unmatched_detections:
            best_pair: tuple[int, int] | None = None
            best_distance = self.config.max_track_distance_px
            for track_index in unmatched_tracks:
                track_center = self._center(self._tracks[track_index].latest.detection)
                for detection_index in unmatched_detections:
                    distance = self._distance(track_center, self._center(detections[detection_index]))
                    if distance < best_distance:
                        best_distance = distance
                        best_pair = (track_index, detection_index)
            if best_pair is None:
                break
            track_index, detection_index = best_pair
            matches.append(best_pair)
            unmatched_tracks.remove(track_index)
            unmatched_detections.remove(detection_index)

        for track_index, detection_index in matches:
            self._append_observation(self._tracks[track_index], detections[detection_index], frame_index, ground_line)

        for track_index in unmatched_tracks:
            self._tracks[track_index].missing_frames += 1

        for detection_index in unmatched_detections:
            self._tracks.append(self._new_track(detections[detection_index], frame_index, ground_line))

    def _append_observation(
        self,
        track: AthleteTrack,
        detection: PoseDetection,
        frame_index: int,
        ground_line: GroundLine | None,
    ) -> None:
        track.observations.append(self._observation(detection, frame_index, ground_line))
        track.missing_frames = 0

    def _new_track(
        self,
        detection: PoseDetection,
        frame_index: int,
        ground_line: GroundLine | None,
    ) -> AthleteTrack:
        track = AthleteTrack(
            track_id=self._next_track_id,
            observations=deque(maxlen=self.config.history_frames),
        )
        self._next_track_id += 1
        self._append_observation(track, detection, frame_index, ground_line)
        return track

    def _drop_stale_tracks(self) -> None:
        self._tracks = [
            track for track in self._tracks if track.missing_frames <= self.config.max_missing_frames
        ]

    def _score_track(self, track: AthleteTrack) -> float:
        if len(track.observations) < 2:
            return 0.0
        first = track.observations[0]
        latest = track.latest
        center_descent = latest.center[1] - first.center[1]
        area_growth = (latest.area - first.area) / max(first.area, 1.0)

        foot_descent = 0.0
        if first.lowest_foot is not None and latest.lowest_foot is not None:
            foot_descent = latest.lowest_foot[1] - first.lowest_foot[1]

        still_penalty = 25.0 if abs(center_descent) < 8 and abs(foot_descent) < 8 else 0.0
        ground_bonus = self._ground_approach_bonus(track)
        grounded_penalty = 80.0 if self._is_grounded(track) else 0.0
        confidence_bonus = latest.detection.confidence * 8.0
        return (
            center_descent * 0.9
            + foot_descent * 1.2
            + max(area_growth, 0.0) * 35.0
            + ground_bonus
            + confidence_bonus
            - still_penalty
            - grounded_penalty
        )

    def _classify_track(self, track: AthleteTrack) -> tuple[str, str | None]:
        if self._is_grounded(track):
            return "grounded", "grounded"
        if len(track.observations) < 2:
            return "other", "short_track"
        first = track.observations[0]
        latest = track.latest
        center_descent = latest.center[1] - first.center[1]
        foot_descent = 0.0
        if first.lowest_foot is not None and latest.lowest_foot is not None:
            foot_descent = latest.lowest_foot[1] - first.lowest_foot[1]
        latest_distance = latest.ground_distance_px
        is_descending = center_descent > self.config.static_motion_px or foot_descent > self.config.min_descent_px
        is_above_ground = latest_distance is None or latest_distance > self.config.airborne_distance_px
        if is_descending and is_above_ground:
            return "airborne", None
        if abs(center_descent) < self.config.static_motion_px and abs(foot_descent) < self.config.static_motion_px:
            return "other", "static"
        return "other", "not_airborne"

    def _is_grounded(self, track: AthleteTrack) -> bool:
        distances = [
            observation.ground_distance_px
            for observation in track.observations
            if observation.ground_distance_px is not None
        ]
        if len(distances) < 3:
            return False
        recent = distances[-min(5, len(distances)) :]
        grounded_frames = sum(distance <= self.config.grounded_distance_px for distance in recent)
        return grounded_frames >= min(3, len(recent))

    def _ground_approach_bonus(self, track: AthleteTrack) -> float:
        distances = [
            observation.ground_distance_px
            for observation in track.observations
            if observation.ground_distance_px is not None
        ]
        if len(distances) < 2:
            return 0.0
        approach = distances[0] - distances[-1]
        if distances[-1] <= self.config.grounded_distance_px:
            return -40.0
        return max(min(approach, 60.0), -30.0)

    def _observation(
        self,
        detection: PoseDetection,
        frame_index: int,
        ground_line: GroundLine | None,
    ) -> TrackObservation:
        lowest_foot = detection.foot_points.lowest()
        return TrackObservation(
            frame_index=frame_index,
            detection=detection,
            center=self._center(detection),
            area=self._area(detection),
            lowest_foot=lowest_foot,
            ground_distance_px=self._ground_distance(lowest_foot, ground_line),
        )

    def _track_by_id(self, track_id: int) -> AthleteTrack | None:
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None

    def _center(self, detection: PoseDetection) -> tuple[float, float]:
        x1, y1, x2, y2 = detection.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _area(self, detection: PoseDetection) -> float:
        x1, y1, x2, y2 = detection.bbox
        return max(x2 - x1, 1) * max(y2 - y1, 1)

    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _ground_distance(self, point: Point | None, ground_line: GroundLine | None) -> float | None:
        if point is None or ground_line is None:
            return None
        return ground_line.distance_to_point(point)
