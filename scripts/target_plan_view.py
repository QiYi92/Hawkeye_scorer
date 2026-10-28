"""Camera-oriented, approximate top-down target geometry and review inset."""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np

from browse_target_center_video import (
    BULLSEYE_RING_RADIUS_M,
    RING_INTERVAL_M,
    TARGET_REAL_RADIUS_M,
    TARGET_RING_COUNT,
    SmoothedTarget,
)


def project_target_point(
    point: tuple[float, float], target: SmoothedTarget | None,
) -> tuple[float, float] | None:
    """Return camera-oriented coordinates in outer-radius units, without clipping.

    Symmetric ellipse whitening R diag(2/a, 2/b) R^T removes foreshortening
    while retaining image orientation. This is an affine estimate, not a
    calibrated perspective reconstruction. The detected bullseye is the origin.
    """
    if target is None:
        return None
    _, (axis_a, axis_b), angle = target.disk_ellipse
    values = (*point, *target.center, axis_a, axis_b, angle)
    if not all(math.isfinite(value) for value in values) or min(axis_a, axis_b) <= 1e-6:
        return None
    theta = math.radians(angle % 180.0)
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = point[0] - target.center[0], point[1] - target.center[1]
    u = (c * dx + s * dy) * 2.0 / axis_a
    v = (-s * dx + c * dy) * 2.0 / axis_b
    x, y = c * u - s * v, s * u + c * v
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def plan_marker_position(
    point: tuple[float, float],
) -> tuple[tuple[int, int], bool]:
    """Fit a marker to the inset's drawing bounds; retain out-of-bounds direction."""
    dx, dy = point[0] * 128.0, point[1] * 128.0
    factor = min(1.0, 158.0 / max(abs(dx), 1e-9), 140.0 / max(abs(dy), 1e-9))
    return (round(180 + dx * factor), round(198 + dy * factor)), factor < 1.0


def draw_target_plan(
    frame: np.ndarray,
    draw_text: Callable,
    point: tuple[float, float] | None,
    status: str,
    detail: str = "",
) -> None:
    """Composite an always-visible inset, using fixed design coordinates."""
    height, width = frame.shape[:2]
    margin = max(2, round(min(width, height) * 0.017))
    scale = min(width * 0.25 / 360, height * 0.62 / 430)
    panel_w, panel_h = max(1, round(360 * scale)), max(1, round(430 * scale))
    x, y = margin, height - margin - panel_h
    background = frame[y:y + panel_h, x:x + panel_w]
    panel = cv2.resize(background, (360, 430)).copy()
    cv2.addWeighted(panel, 0.22, np.full_like(panel, (28, 24, 22)), 0.78, 0, panel)
    cv2.rectangle(panel, (0, 0), (359, 429), (130, 130, 130), 1)
    draw_text(panel, "靶盘平面图", (16, 32), (255, 255, 255), 0.78, 1)
    center, radius = (180, 198), 128
    cv2.circle(panel, center, radius, (205, 210, 210), -1, cv2.LINE_AA)
    cv2.circle(panel, center, round(radius * BULLSEYE_RING_RADIUS_M / TARGET_REAL_RADIUS_M), (15, 15, 15), -1, cv2.LINE_AA)
    for index in range(TARGET_RING_COUNT, 0, -1):
        ring_radius = round(radius * index * RING_INTERVAL_M / TARGET_REAL_RADIUS_M)
        color = (0, 220, 255) if index == 1 else (150, 125, 70)
        cv2.circle(panel, center, ring_radius, color, 1, cv2.LINE_AA)
        direction = 1 if index % 2 == 0 else -1
        offset = ring_radius / math.sqrt(2)
        label_x = round(center[0] + direction * offset) + (4 if direction > 0 else -39)
        label_y = round(center[1] - direction * offset)
        draw_text(panel, f"{index * RING_INTERVAL_M:.1f}m", (label_x, label_y), (255, 255, 255), 0.46, 1)
    cv2.drawMarker(panel, center, (230, 230, 230), cv2.MARKER_CROSS, 8, 1)
    if point is not None:
        position, overflow = plan_marker_position(point)
        if overflow:
            dx, dy = position[0] - center[0], position[1] - center[1]
            length = math.hypot(dx, dy)
            start = (round(position[0] - dx / length * 24), round(position[1] - dy / length * 24))
            cv2.arrowedLine(panel, start, position, (255, 255, 255), 5, cv2.LINE_AA, tipLength=0.5)
            cv2.arrowedLine(panel, start, position, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.5)
            status = "落点超出示意范围"
        else:
            cv2.circle(panel, position, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel, position, 5, (0, 0, 255), -1, cv2.LINE_AA)
    draw_text(panel, status, (16, 365), (255, 255, 255), 0.65, 1)
    draw_text(panel, detail, (16, 391), (220, 220, 220), 0.58, 1)
    draw_text(panel, "位置为几何估算", (16, 417), (180, 180, 180), 0.50, 1)
    frame[y:y + panel_h, x:x + panel_w] = cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
