"""Shared target rings and metric labels for Demo 08 and Demo 12.

The host supplies _draw_text so each browser keeps its own text renderer.
"""

from __future__ import annotations

import cv2
import numpy as np

from browse_target_center_video import (
    BULLSEYE_RING_RADIUS_M, RING_INTERVAL_M, TARGET_REAL_RADIUS_M,
    TARGET_RING_COUNT, SmoothedTarget, TargetDetection,
)


class TargetOverlayMixin:
    def _draw_target(
        self,
        frame: np.ndarray,
        target: SmoothedTarget | None,
        detection: TargetDetection | None,
    ) -> None:
        if target is None:
            return
        disk_color = (0, 255, 0) if detection is not None and target.bullseye_detected else (0, 180, 255)
        cv2.ellipse(frame, target.disk_ellipse, disk_color, 3)
        self._draw_target_rings(frame, target.disk_ellipse, target.ellipse)
        (disk_center_x, disk_center_y), (_, disk_height), _ = target.disk_ellipse
        label = "已锁定靶盘" if target.disk_locked else "靶盘"
        label_origin = (
            int(round(disk_center_x - 90)),
            max(int(round(disk_center_y - disk_height / 2 - 8)), 18),
        )
        self._draw_text(frame, label, label_origin, disk_color, 0.58, 2)

        center = tuple(int(round(value)) for value in target.center)
        radius = int(round(target.radius_px))
        if target.ellipse is not None and target.bullseye_detected:
            cv2.ellipse(frame, target.ellipse, (0, 220, 255), 2)
        cv2.circle(frame, center, radius, (0, 220, 255), 2)
        cv2.line(frame, (center[0] - 22, center[1]), (center[0] + 22, center[1]), (0, 0, 255), 2)
        cv2.line(frame, (center[0], center[1] - 22), (center[0], center[1] + 22), (0, 0, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    def _draw_target_rings(
        self,
        frame: np.ndarray,
        disk_ellipse: tuple[tuple[float, float], tuple[float, float], float],
        bullseye_ellipse: tuple[tuple[float, float], tuple[float, float], float] | None,
    ) -> None:
        outer = self._normalized_ellipse(disk_ellipse)
        inner = self._normalized_ellipse(bullseye_ellipse) if bullseye_ellipse is not None else None
        for ring_index in range(TARGET_RING_COUNT, 0, -1):
            radius_m = ring_index * RING_INTERVAL_M
            ring = self._interpolated_ring_ellipse(outer, inner, radius_m)
            is_inner_ring = ring_index == 1
            color = (0, 220, 255) if is_inner_ring else (255, 255, 0)
            thickness = 2 if is_inner_ring else 1
            cv2.ellipse(frame, ring, color, thickness, cv2.LINE_AA)
            self._draw_ring_label(frame, ring, radius_m, color)

    def _interpolated_ring_ellipse(
        self,
        outer: tuple[tuple[float, float], tuple[float, float], float],
        inner: tuple[tuple[float, float], tuple[float, float], float] | None,
        radius_m: float,
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        if inner is None:
            center, axes, angle = outer
            scale = radius_m / TARGET_REAL_RADIUS_M
            return center, (axes[0] * scale, axes[1] * scale), angle

        t = (radius_m - BULLSEYE_RING_RADIUS_M) / max(
            TARGET_REAL_RADIUS_M - BULLSEYE_RING_RADIUS_M,
            0.001,
        )
        t = min(max(t, 0.0), 1.0)
        (inner_x, inner_y), (inner_a, inner_b), inner_angle = inner
        (outer_x, outer_y), (outer_a, outer_b), outer_angle = outer
        angle_delta = (outer_angle - inner_angle + 90.0) % 180.0 - 90.0
        return (
            (
                inner_x * (1.0 - t) + outer_x * t,
                inner_y * (1.0 - t) + outer_y * t,
            ),
            (
                inner_a * (1.0 - t) + outer_a * t,
                inner_b * (1.0 - t) + outer_b * t,
            ),
            (inner_angle + angle_delta * t) % 180.0,
        )

    def _normalized_ellipse(
        self,
        ellipse: tuple[tuple[float, float], tuple[float, float], float],
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        center, (axis_a, axis_b), angle = ellipse
        if axis_a >= axis_b:
            return center, (axis_a, axis_b), angle % 180.0
        return center, (axis_b, axis_a), (angle + 90.0) % 180.0

    def _draw_ring_label(
        self,
        frame: np.ndarray,
        ring: tuple[tuple[float, float], tuple[float, float], float],
        radius_m: float,
        color: tuple[int, int, int],
    ) -> None:
        (center_x, center_y), (axis_a, axis_b), angle = ring
        label = f"{radius_m:.1f}m"
        if radius_m <= 0.4:
            label_x = center_x + axis_a / 2.0 + 10.0
            label_y = center_y + (0.4 - radius_m) * 40.0
        else:
            ring_index = int(round(radius_m / RING_INTERVAL_M))
            local_angle_deg = -32.0 if ring_index % 2 == 0 else 148.0
            local_angle = np.deg2rad(local_angle_deg)
            ellipse_angle = np.deg2rad(angle)
            local_x = (axis_a / 2.0) * 0.93 * np.cos(local_angle)
            local_y = (axis_b / 2.0) * 0.93 * np.sin(local_angle)
            label_x = center_x + local_x * np.cos(ellipse_angle) - local_y * np.sin(ellipse_angle)
            label_y = center_y + local_x * np.sin(ellipse_angle) + local_y * np.cos(ellipse_angle)
            label_y += ((ring_index % 4) - 1.5) * 5.0

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        max_x = max(frame.shape[1] - text_size[0] - 4, 4)
        max_y = max(frame.shape[0] - 4, 16)
        origin = (
            min(max(int(round(label_x + 4)), 4), max_x),
            min(max(int(round(label_y)), 16), max_y),
        )
        self._draw_text(frame, label, origin, color, 0.34, 1)
