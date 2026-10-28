"""Ground line estimation from the lower part of a video frame."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .models import GroundLine

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundLineConfig:
    roi_start_ratio: float = 0.45
    min_candidate_y_ratio: float = 0.70
    fallback_y_ratio: float = 0.88
    max_abs_slope: float = 0.25
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 40
    min_line_length_ratio: float = 0.20
    max_line_gap: int = 30
    min_gap_below_horizon_ratio: float = 0.12
    max_upward_jump_ratio: float = 0.08


@dataclass(frozen=True)
class HorizonLineConfig:
    roi_start_ratio: float = 0.25
    roi_end_ratio: float = 0.72
    max_candidate_y_ratio: float = 0.70
    max_abs_slope: float = 0.20
    canny_low: int = 40
    canny_high: int = 120
    hough_threshold: int = 45
    min_line_length_ratio: float = 0.24
    max_line_gap: int = 35


class HorizonLineEstimator:
    """Estimate the far horizontal horizon/background boundary."""

    def __init__(self, config: HorizonLineConfig) -> None:
        self.config = config
        self._last_line: GroundLine | None = None

    def estimate(self, frame: np.ndarray) -> GroundLine | None:
        height, width = frame.shape[:2]
        roi_top = int(height * self.config.roi_start_ratio)
        roi_bottom = int(height * self.config.roi_end_ratio)
        roi = frame[roi_top:roi_bottom]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=max(20, int(width * self.config.min_line_length_ratio)),
            maxLineGap=self.config.max_line_gap,
        )

        candidates: list[tuple[float, GroundLine]] = []
        if lines is not None:
            for raw_line in lines[:, 0, :]:
                line = self._line_from_hough(raw_line, roi_top)
                if line is None:
                    continue
                x1, y1, x2, y2, slope, intercept, length = line
                midpoint_y = (y1 + y2) / 2
                if midpoint_y > height * self.config.max_candidate_y_ratio:
                    continue
                if abs(slope) > self.config.max_abs_slope:
                    continue
                # Prefer long, upper/middle background boundaries.
                upper_score = 1.0 - abs((midpoint_y / max(height, 1)) - 0.55)
                score = length * max(upper_score, 0.1)
                candidates.append(
                    (
                        score,
                        GroundLine(
                            slope=slope,
                            intercept=intercept,
                            confidence=score,
                            status="horizon",
                        ),
                    )
                )

        if candidates:
            _, line = max(candidates, key=lambda item: item[0])
            self._last_line = line
            return line
        return self._last_line

    def _line_from_hough(self, raw_line: np.ndarray, y_offset: int):
        x1, y1_roi, x2, y2_roi = [int(value) for value in raw_line]
        dx = x2 - x1
        if dx == 0:
            return None
        y1 = y1_roi + y_offset
        y2 = y2_roi + y_offset
        slope = (y2 - y1) / dx
        intercept = y1 - slope * x1
        length = float(np.hypot(dx, y2 - y1))
        return x1, y1, x2, y2, slope, intercept, length


class GroundLineEstimator:
    """Estimate the main nearly-horizontal ground line in image coordinates."""

    def __init__(self, config: GroundLineConfig) -> None:
        self.config = config
        self._last_line: GroundLine | None = None
        self._fallback_reported = False
        self.last_status = "unknown"

    def estimate(
        self,
        frame: np.ndarray,
        horizon_line: GroundLine | None = None,
        athlete_bbox: tuple[int, int, int, int] | None = None,
    ) -> GroundLine:
        height, width = frame.shape[:2]
        roi_top = int(height * self.config.roi_start_ratio)
        roi = frame[roi_top:height]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
        min_line_length = max(20, int(width * self.config.min_line_length_ratio))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=self.config.max_line_gap,
        )

        candidates: list[tuple[float, GroundLine]] = []
        rejected_by_horizon = 0
        rejected_by_athlete = 0
        if lines is not None:
            for raw_line in lines[:, 0, :]:
                x1, y1_roi, x2, y2_roi = [int(value) for value in raw_line]
                dx = x2 - x1
                if dx == 0:
                    continue
                y1 = y1_roi + roi_top
                y2 = y2_roi + roi_top
                slope = (y2 - y1) / dx
                if abs(slope) > self.config.max_abs_slope:
                    continue
                midpoint_y = (y1 + y2) / 2
                if midpoint_y < height * self.config.min_candidate_y_ratio:
                    continue
                if horizon_line is not None:
                    horizon_y = horizon_line.y_at((x1 + x2) / 2)
                    min_ground_y = horizon_y + height * self.config.min_gap_below_horizon_ratio
                    if midpoint_y < min_ground_y:
                        rejected_by_horizon += 1
                        continue
                intercept = y1 - slope * x1
                if athlete_bbox is not None and self._crosses_athlete_upper_body(
                    slope,
                    intercept,
                    athlete_bbox,
                ):
                    rejected_by_athlete += 1
                    continue
                length = float(np.hypot(dx, y2 - y1))
                if athlete_bbox is not None:
                    foot_x = (athlete_bbox[0] + athlete_bbox[2]) / 2
                    foot_y = athlete_bbox[3]
                else:
                    foot_x = width / 2
                    foot_y = height * self.config.fallback_y_ratio
                line_y_at_foot = slope * foot_x + intercept
                foot_proximity = 1.0 - min(abs(line_y_at_foot - foot_y) / max(height * 0.35, 1), 1.0)
                slope_score = 1.0 - min(abs(slope) / max(self.config.max_abs_slope, 0.001), 1.0)
                stability_score = self._stability_score(slope, intercept, height)
                vertical_score = midpoint_y / max(height, 1)
                score = length * (
                    0.45 * vertical_score
                    + 0.25 * foot_proximity
                    + 0.20 * slope_score
                    + 0.10 * stability_score
                    + 0.1
                )
                candidates.append(
                    (
                        score,
                        GroundLine(
                            slope=slope,
                            intercept=intercept,
                            confidence=score,
                            status="detected",
                        ),
                    )
                )

        if candidates:
            _, line = max(candidates, key=lambda item: item[0])
            if self._last_line is not None:
                last_y = self._last_line.y_at(width / 2)
                new_y = line.y_at(width / 2)
                if new_y < last_y - height * self.config.max_upward_jump_ratio:
                    self.last_status = "stable"
                    return GroundLine(
                        slope=self._last_line.slope,
                        intercept=self._last_line.intercept,
                        confidence=self._last_line.confidence,
                        fallback=self._last_line.fallback,
                        status="stable",
                    )
            self._last_line = line
            self.last_status = line.status
            return line

        if self._last_line is not None:
            self.last_status = "stable"
            return GroundLine(
                slope=self._last_line.slope,
                intercept=self._last_line.intercept,
                confidence=self._last_line.confidence,
                fallback=self._last_line.fallback,
                status="stable",
            )

        fallback_y = height * self.config.fallback_y_ratio
        if not self._fallback_reported:
            LOGGER.warning("Ground line not found; using fallback y=%.1f", fallback_y)
            self._fallback_reported = True
        if rejected_by_horizon:
            self.last_status = "rejected_by_horizon"
        elif rejected_by_athlete:
            self.last_status = "rejected_by_athlete"
        else:
            self.last_status = "fallback"
        return GroundLine(slope=0.0, intercept=fallback_y, confidence=0.0, fallback=True, status=self.last_status)

    def _crosses_athlete_upper_body(
        self,
        slope: float,
        intercept: float,
        athlete_bbox: tuple[int, int, int, int],
    ) -> bool:
        x1, y1, x2, y2 = athlete_bbox
        bbox_h = max(y2 - y1, 1)
        sample_x = (x1 + x2) / 2
        line_y = slope * sample_x + intercept
        return y1 <= line_y <= y1 + bbox_h * 0.65

    def _stability_score(self, slope: float, intercept: float, height: int) -> float:
        if self._last_line is None:
            return 0.5
        y_delta = abs((slope * 0 + intercept) - self._last_line.y_at(0))
        return 1.0 - min(y_delta / max(height * 0.20, 1), 1.0)
