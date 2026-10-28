#!/usr/bin/env python3
"""Open a target video window and track the target disk and bullseye live.

Run this file directly to open the standard prototype target video with live
on-screen annotations. The supported target is a white disk with a black
bullseye, projected as an ellipse by the camera angle.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = "training_video/video6.mp4"
WINDOW_NAME = "Target Disk and Center Browser"
TARGET_REAL_RADIUS_M = 2.0
TARGET_RING_COUNT = 10
RING_INTERVAL_M = TARGET_REAL_RADIUS_M / TARGET_RING_COUNT
BULLSEYE_RING_RADIUS_M = RING_INTERVAL_M


@dataclass(frozen=True)
class TargetDetection:
    center: tuple[float, float]
    area_px: float
    radius_px: float
    ellipse: tuple[tuple[float, float], tuple[float, float], float] | None
    disk_ellipse: tuple[tuple[float, float], tuple[float, float], float]
    disk_area_px: float
    score: float
    circularity: float
    bullseye_detected: bool


@dataclass(frozen=True)
class SmoothedTarget:
    center: tuple[float, float]
    area_px: float
    radius_px: float
    ellipse: tuple[tuple[float, float], tuple[float, float], float] | None
    disk_ellipse: tuple[tuple[float, float], tuple[float, float], float]
    disk_area_px: float
    circularity: float
    bullseye_detected: bool
    disk_locked: bool
    occluded_frames: int


class TargetCenterDetector:
    """Detect the target disk first, then locate the bullseye inside it."""

    def __init__(self) -> None:
        self.smoothed: SmoothedTarget | None = None
        self.stable_disk: tuple[tuple[float, float], tuple[float, float], float] | None = None
        self.stable_disk_area: float | None = None
        self.stable_disk_frames = 0
        self.stable_bullseye_center: tuple[float, float] | None = None
        self.stable_bullseye_radius: float | None = None
        self.stable_bullseye_area: float | None = None
        self.stable_bullseye_frames = 0

    def reset(self) -> None:
        self.smoothed = None
        self.stable_disk = None
        self.stable_disk_area = None
        self.stable_disk_frames = 0
        self.stable_bullseye_center = None
        self.stable_bullseye_radius = None
        self.stable_bullseye_area = None
        self.stable_bullseye_frames = 0

    def detect(self, frame: np.ndarray) -> TargetDetection | None:
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        disk = self._detect_white_disk(hsv, width, height)
        if disk is None:
            return None
        disk = self._stabilize_disk(disk)
        bullseye = self._detect_dark_bullseye(frame, disk, width, height)
        if bullseye is not None:
            return self._stabilize_bullseye(bullseye)
        return self._disk_only_detection(disk)

    def _stabilize_disk(
        self,
        disk: tuple[tuple[tuple[float, float], tuple[float, float], float], float],
    ) -> tuple[tuple[tuple[float, float], tuple[float, float], float], float]:
        ellipse, area = disk
        if self.stable_disk is None:
            self.stable_disk = ellipse
            self.stable_disk_area = area
            self.stable_disk_frames = 1
            return disk

        stable_ellipse = self.stable_disk
        stable_area = self.stable_disk_area if self.stable_disk_area is not None else area
        if self._ellipse_change_ratio(stable_ellipse, ellipse) < 0.035:
            self.stable_disk = self._blend_ellipse(stable_ellipse, ellipse, 0.08)
            self.stable_disk_area = stable_area * 0.92 + area * 0.08
            self.stable_disk_frames += 1
        elif self.stable_disk_frames < 8:
            self.stable_disk = self._blend_ellipse(stable_ellipse, ellipse, 0.20)
            self.stable_disk_area = stable_area * 0.80 + area * 0.20
            self.stable_disk_frames += 1

        return self.stable_disk, float(self.stable_disk_area if self.stable_disk_area is not None else area)

    def _disk_only_detection(
        self,
        disk: tuple[tuple[tuple[float, float], tuple[float, float], float], float],
    ) -> TargetDetection:
        disk_ellipse, disk_area = disk
        if self.smoothed is not None:
            center = self.smoothed.center
            radius = self.smoothed.radius_px
            area = self.smoothed.area_px
            ellipse = self.smoothed.ellipse
            circularity = self.smoothed.circularity
        else:
            (center_x, center_y), (_, disk_height), _ = disk_ellipse
            center = (float(center_x), float(center_y))
            radius = max(float(disk_height) * 0.22, 8.0)
            area = 0.0
            ellipse = None
            circularity = 0.0

        return TargetDetection(
            center=center,
            area_px=area,
            radius_px=radius,
            ellipse=ellipse,
            disk_ellipse=disk_ellipse,
            disk_area_px=disk_area,
            score=float(disk_area),
            circularity=circularity,
            bullseye_detected=False,
        )

    def _stabilize_bullseye(self, detection: TargetDetection) -> TargetDetection:
        if self.stable_bullseye_center is None:
            self.stable_bullseye_center = detection.center
            self.stable_bullseye_radius = detection.radius_px
            self.stable_bullseye_area = detection.area_px
            self.stable_bullseye_frames = 1
            return detection

        dx = detection.center[0] - self.stable_bullseye_center[0]
        dy = detection.center[1] - self.stable_bullseye_center[1]
        distance = float((dx * dx + dy * dy) ** 0.5)
        max_distance = 45.0 if self.stable_bullseye_frames >= 8 else 75.0
        stable_radius = float(self.stable_bullseye_radius or detection.radius_px)
        stable_area = float(self.stable_bullseye_area or detection.area_px)
        size_is_unstable = (
            self.stable_bullseye_frames >= 8
            and (
                detection.area_px < stable_area * 0.35
                or detection.area_px > stable_area * 2.00
                or detection.radius_px < stable_radius * 0.55
                or detection.radius_px > stable_radius * 1.55
            )
        )
        if distance > max_distance or size_is_unstable:
            return TargetDetection(
                center=self.stable_bullseye_center,
                area_px=detection.area_px,
                radius_px=stable_radius,
                ellipse=detection.ellipse,
                disk_ellipse=detection.disk_ellipse,
                disk_area_px=detection.disk_area_px,
                score=detection.score,
                circularity=detection.circularity,
                bullseye_detected=False,
            )

        alpha = 0.10 if self.stable_bullseye_frames >= 8 else 0.25
        self.stable_bullseye_center = (
            self.stable_bullseye_center[0] * (1.0 - alpha) + detection.center[0] * alpha,
            self.stable_bullseye_center[1] * (1.0 - alpha) + detection.center[1] * alpha,
        )
        self.stable_bullseye_radius = (
            stable_radius * (1.0 - alpha)
            + detection.radius_px * alpha
        )
        self.stable_bullseye_area = stable_area * (1.0 - alpha) + detection.area_px * alpha
        self.stable_bullseye_frames += 1
        return TargetDetection(
            center=self.stable_bullseye_center,
            area_px=detection.area_px,
            radius_px=self.stable_bullseye_radius,
            ellipse=detection.ellipse,
            disk_ellipse=detection.disk_ellipse,
            disk_area_px=detection.disk_area_px,
            score=detection.score,
            circularity=detection.circularity,
            bullseye_detected=True,
        )

    def _detect_white_disk(
        self,
        hsv: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[tuple[tuple[float, float], tuple[float, float], float], float] | None:
        mask = cv2.inRange(hsv, (0, 0, 160), (179, 65, 255))
        mask[: int(height * 0.35), :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, tuple[tuple[float, float], tuple[float, float], float], float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 35000.0 or area > width * height * 0.45:
                continue

            x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
            if (
                y < height * 0.50
                or y + bbox_height < height * 0.62
                or bbox_width < width * 0.20
                or bbox_height < height * 0.08
            ):
                continue

            aspect = bbox_width / max(bbox_height, 1)
            if aspect < 2.00 or aspect > 7.50:
                continue

            if len(contour) < 5:
                continue
            ellipse = cv2.fitEllipse(contour)
            (center_x, center_y), (axis_a, axis_b), _ = ellipse
            ellipse_aspect = max(axis_a, axis_b) / max(min(axis_a, axis_b), 1.0)
            if ellipse_aspect < 2.00 or ellipse_aspect > 8.00:
                continue
            if center_y < height * 0.48:
                continue

            distance_penalty = 1.0
            if self.smoothed is not None:
                previous_center = self.smoothed.disk_ellipse[0]
                dx = (center_x - previous_center[0]) / max(width, 1)
                dy = (center_y - previous_center[1]) / max(height, 1)
                distance_penalty += 20.0 * float(dx * dx + dy * dy)
            else:
                dx = (center_x - width * 0.52) / max(width, 1)
                dy = (center_y - height * 0.76) / max(height, 1)
                distance_penalty += 4.0 * float(dx * dx + dy * dy)

            score = area * min(ellipse_aspect, 4.0) / distance_penalty
            candidates.append((score, ellipse, area))

        if not candidates:
            return None
        _, ellipse, area = max(candidates, key=lambda candidate: candidate[0])
        return ellipse, area

    def _detect_dark_bullseye(
        self,
        frame: np.ndarray,
        disk: tuple[tuple[tuple[float, float], tuple[float, float], float], float],
        width: int,
        height: int,
    ) -> TargetDetection | None:
        disk_ellipse, disk_area = disk
        (disk_center_x, disk_center_y), (axis_a, axis_b), _ = disk_ellipse
        disk_width = max(axis_a, axis_b)
        disk_height = min(axis_a, axis_b)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_mask = cv2.inRange(gray, 0, 165)
        dark_mask[: int(height * 0.45), :] = 0

        ellipse_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(ellipse_mask, disk_ellipse, 255, -1)
        dark_mask = cv2.bitwise_and(dark_mask, ellipse_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[TargetDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 220.0 or area > max(16000.0, disk_area * 0.18):
                continue

            x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
            aspect = bbox_width / max(bbox_height, 1)
            if aspect < 0.60 or aspect > 5.50:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center_x = float(moments["m10"] / moments["m00"])
            center_y = float(moments["m01"] / moments["m00"])

            normalized_dx = (center_x - disk_center_x) / max(disk_width * 0.5, 1.0)
            normalized_dy = (center_y - disk_center_y) / max(disk_height * 0.5, 1.0)
            normalized_distance = float(normalized_dx * normalized_dx + normalized_dy * normalized_dy)
            if normalized_distance > 0.22:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0 if perimeter <= 0 else float(4.0 * np.pi * area / (perimeter * perimeter))
            (circle_center, radius) = cv2.minEnclosingCircle(contour)
            ellipse = cv2.fitEllipse(contour) if len(contour) >= 5 else None
            score = area * max(circularity, 0.25) / max(1.0 + normalized_distance * 8.0, 0.1)
            candidates.append(
                TargetDetection(
                    center=(float(circle_center[0]), float(circle_center[1])),
                    area_px=area,
                    radius_px=float(radius),
                    ellipse=ellipse,
                    disk_ellipse=disk_ellipse,
                    disk_area_px=disk_area,
                    score=float(score),
                    circularity=circularity,
                    bullseye_detected=True,
                )
            )

        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate.score)

    def _ellipse_change_ratio(
        self,
        previous: tuple[tuple[float, float], tuple[float, float], float],
        current: tuple[tuple[float, float], tuple[float, float], float],
    ) -> float:
        (previous_x, previous_y), (previous_a, previous_b), previous_angle = previous
        (current_x, current_y), (current_a, current_b), current_angle = current
        previous_major = max(previous_a, previous_b, 1.0)
        previous_minor = max(min(previous_a, previous_b), 1.0)
        center_change = abs(current_x - previous_x) / previous_major + abs(current_y - previous_y) / previous_minor
        axis_change = abs(current_a - previous_a) / previous_major + abs(current_b - previous_b) / previous_minor
        angle_delta = abs((current_angle - previous_angle + 90.0) % 180.0 - 90.0) / 90.0
        return float(center_change + axis_change + angle_delta * 0.2)

    def _blend_ellipse(
        self,
        previous: tuple[tuple[float, float], tuple[float, float], float],
        current: tuple[tuple[float, float], tuple[float, float], float],
        alpha: float,
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        (previous_x, previous_y), (previous_a, previous_b), previous_angle = previous
        (current_x, current_y), (current_a, current_b), current_angle = current
        angle_delta = (current_angle - previous_angle + 90.0) % 180.0 - 90.0
        return (
            (
                previous_x * (1.0 - alpha) + current_x * alpha,
                previous_y * (1.0 - alpha) + current_y * alpha,
            ),
            (
                previous_a * (1.0 - alpha) + current_a * alpha,
                previous_b * (1.0 - alpha) + current_b * alpha,
            ),
            (previous_angle + angle_delta * alpha) % 180.0,
        )

    def update(self, detection: TargetDetection | None, alpha: float = 0.30) -> SmoothedTarget | None:
        if detection is None:
            if self.smoothed is None:
                return None
            previous = self.smoothed
            self.smoothed = SmoothedTarget(
                center=previous.center,
                area_px=previous.area_px,
                radius_px=previous.radius_px,
                ellipse=previous.ellipse,
                disk_ellipse=previous.disk_ellipse,
                disk_area_px=previous.disk_area_px,
                circularity=previous.circularity,
                bullseye_detected=False,
                disk_locked=previous.disk_locked,
                occluded_frames=previous.occluded_frames + 1,
            )
            return self.smoothed

        if self.smoothed is None:
            self.smoothed = SmoothedTarget(
                center=detection.center,
                area_px=detection.area_px,
                radius_px=detection.radius_px,
                ellipse=detection.ellipse,
                disk_ellipse=detection.disk_ellipse,
                disk_area_px=detection.disk_area_px,
                circularity=detection.circularity,
                bullseye_detected=detection.bullseye_detected,
                disk_locked=self.stable_disk_frames >= 8,
                occluded_frames=0 if detection.bullseye_detected else 1,
            )
            return self.smoothed

        previous = self.smoothed
        if detection.bullseye_detected:
            center = (
                previous.center[0] * (1.0 - alpha) + detection.center[0] * alpha,
                previous.center[1] * (1.0 - alpha) + detection.center[1] * alpha,
            )
            area_px = previous.area_px * (1.0 - alpha) + detection.area_px * alpha
            radius_px = previous.radius_px * (1.0 - alpha) + detection.radius_px * alpha
            ellipse = detection.ellipse
            circularity = detection.circularity
            occluded_frames = 0
        else:
            center = previous.center
            area_px = previous.area_px
            radius_px = previous.radius_px
            ellipse = previous.ellipse
            circularity = previous.circularity
            occluded_frames = previous.occluded_frames + 1

        self.smoothed = SmoothedTarget(
            center=center,
            area_px=area_px,
            radius_px=radius_px,
            ellipse=ellipse,
            disk_ellipse=detection.disk_ellipse,
            disk_area_px=(
                previous.disk_area_px * (1.0 - alpha) + detection.disk_area_px * alpha
            ),
            circularity=circularity,
            bullseye_detected=detection.bullseye_detected,
            disk_locked=self.stable_disk_frames >= 8,
            occluded_frames=occluded_frames,
        )
        return self.smoothed


class TargetCenterVideoBrowser:
    def __init__(
        self,
        video_path: Path,
        display_scale: float,
        output_video: Path | None,
        no_preview: bool,
    ) -> None:
        self.video_path = video_path
        self.display_scale = display_scale
        self.output_video = output_video
        self.no_preview = no_preview
        self.detector = TargetCenterDetector()
        self.paused = False
        self.step_once = False
        self.last_process_seconds = 0.0
        self.detected_frames = 0
        self.tracked_frames = 0
        self.occluded_frames = 0
        self.processed_frames = 0

    def run(self) -> int:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise SystemExit(f"错误：无法打开视频：{self.video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        delay_ms = max(1, int(1000 / max(fps, 1.0)))

        writer = self._open_writer(fps, width, height)

        print(f"视频：{self.video_path}")
        print(f"尺寸：{width}x{height}，FPS：{fps:.2f}，总帧数：{frame_count}")
        if self.no_preview:
            print("模式：无预览，处理完整视频并输出统计。")
        else:
            print("模式：实时追踪窗口，靶盘范围和靶心标注会直接显示在打开的视频画面里。")
            print("按键：Space 暂停/继续，n 暂停时前进一帧，r 回到开头，q 或 Esc 退出。")
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        annotated: np.ndarray | None = None
        try:
            while True:
                if self.no_preview or not self.paused or self.step_once:
                    ok, frame = capture.read()
                    if not ok:
                        if self.no_preview:
                            break
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._reset()
                        continue

                    frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    annotated = self._annotate_frame(frame, frame_index, fps, frame_count)
                    self.step_once = False
                    if writer is not None:
                        writer.write(annotated)

                if self.no_preview:
                    continue
                if annotated is None:
                    continue

                cv2.imshow(WINDOW_NAME, self._resize_for_display(annotated))
                key = cv2.waitKey(delay_ms if not self.paused else 30)
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    self.paused = not self.paused
                elif key == ord("n") and self.paused:
                    self.step_once = True
                elif key == ord("r"):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._reset()
                    self.paused = False
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if not self.no_preview:
                cv2.destroyAllWindows()

        self._print_summary()
        return 0

    def _annotate_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
        frame_count: int,
    ) -> np.ndarray:
        started_at = time.perf_counter()
        detection = self.detector.detect(frame)
        smoothed = self.detector.update(detection)
        self.last_process_seconds = time.perf_counter() - started_at
        self.processed_frames += 1

        if detection is not None and detection.bullseye_detected and smoothed is not None:
            self.detected_frames += 1
        if smoothed is not None:
            self.tracked_frames += 1
            if not smoothed.bullseye_detected:
                self.occluded_frames += 1

        output = frame.copy()
        self._draw_detection(output, detection, smoothed)
        self._draw_overlay(output, detection, smoothed, frame_index, fps, frame_count)
        return output

    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: TargetDetection | None,
        smoothed: SmoothedTarget | None,
    ) -> None:
        if smoothed is None:
            return

        disk_color = (0, 255, 0) if smoothed.bullseye_detected else (0, 180, 255)
        cv2.ellipse(frame, smoothed.disk_ellipse, disk_color, 3)
        self._draw_target_rings(frame, smoothed.disk_ellipse, smoothed.ellipse)
        (disk_center_x, disk_center_y), (_, disk_height), _ = smoothed.disk_ellipse
        disk_label_origin = (
            int(round(disk_center_x - 90)),
            max(int(round(disk_center_y - disk_height / 2 - 8)), 18),
        )
        cv2.putText(
            frame,
            "locked white target disk" if smoothed.disk_locked else "white target disk",
            disk_label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "locked white target disk" if smoothed.disk_locked else "white target disk",
            disk_label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            disk_color,
            2,
            cv2.LINE_AA,
        )

        if smoothed.ellipse is not None and smoothed.bullseye_detected:
            cv2.ellipse(frame, smoothed.ellipse, (0, 220, 255), 2)

        center = self._round_point(smoothed.center)
        radius = int(round(smoothed.radius_px))
        cv2.circle(frame, center, radius, (0, 220, 255), 2)
        cv2.line(frame, (center[0] - 22, center[1]), (center[0] + 22, center[1]), (0, 0, 255), 2)
        cv2.line(frame, (center[0], center[1] - 22), (center[0], center[1] + 22), (0, 0, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            self._bullseye_label(smoothed, center),
            (center[0] + 12, max(center[1] - 12, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self._bullseye_label(smoothed, center),
            (center[0] + 12, max(center[1] - 12, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_target_rings(
        self,
        frame: np.ndarray,
        disk_ellipse: tuple[tuple[float, float], tuple[float, float], float],
        bullseye_ellipse: tuple[tuple[float, float], tuple[float, float], float] | None,
    ) -> None:
        outer = self._normalized_ellipse(disk_ellipse)
        inner = self._normalized_ellipse(bullseye_ellipse) if bullseye_ellipse is not None else None
        label_every = 2
        for ring_index in range(TARGET_RING_COUNT, 0, -1):
            radius_m = ring_index * RING_INTERVAL_M
            ring = self._interpolated_ring_ellipse(outer, inner, radius_m)
            is_inner_ring = ring_index == 1
            color = (0, 220, 255) if is_inner_ring else (255, 255, 0)
            thickness = 2 if is_inner_ring else 1
            cv2.ellipse(frame, ring, color, thickness, cv2.LINE_AA)
            if ring_index % label_every == 0 or is_inner_ring or ring_index == TARGET_RING_COUNT:
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
            local_angle = np.deg2rad(145.0)
            ellipse_angle = np.deg2rad(angle)
            local_x = (axis_a / 2.0) * 0.88 * np.cos(local_angle)
            local_y = (axis_b / 2.0) * 0.88 * np.sin(local_angle)
            label_x = center_x + local_x * np.cos(ellipse_angle) - local_y * np.sin(ellipse_angle)
            label_y = center_y + local_x * np.sin(ellipse_angle) + local_y * np.cos(ellipse_angle)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        max_x = max(frame.shape[1] - text_size[0] - 4, 4)
        max_y = max(frame.shape[0] - 4, 16)
        origin = (
            min(max(int(round(label_x + 4)), 4), max_x),
            min(max(int(round(label_y)), 16), max_y),
        )
        cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        detection: TargetDetection | None,
        smoothed: SmoothedTarget | None,
        frame_index: int,
        fps: float,
        frame_count: int,
    ) -> None:
        timestamp = frame_index / max(fps, 0.001)
        process_fps = 1.0 / max(self.last_process_seconds, 0.001)
        if smoothed is not None and smoothed.bullseye_detected:
            status = "LIVE TARGET"
        elif smoothed is not None:
            status = "BULLSEYE OCCLUDED"
        else:
            status = "NO TARGET"
        lines = [
            f"{status} | frame {frame_index + 1}/{frame_count} | t={timestamp:.2f}s | process={process_fps:.1f} fps",
            f"video={self.video_path}",
        ]
        if smoothed is not None:
            x, y = smoothed.center
            lines.append(
                f"center=({x:.1f}, {y:.1f}) | bullseye_area={smoothed.area_px:.0f}px | "
                f"disk_area={smoothed.disk_area_px:.0f}px | "
                f"radius={smoothed.radius_px:.1f}px | circularity={smoothed.circularity:.2f} | "
                f"occluded={smoothed.occluded_frames}"
            )
        else:
            lines.append("center=(-, -) | area_px=- | radius=- | circularity=-")
        if not self.no_preview:
            lines.append("Space pause/play | n next frame | r restart | q/Esc quit")

        y = 28
        for line in lines:
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28

    def _open_writer(self, fps: float, width: int, height: int) -> cv2.VideoWriter | None:
        if self.output_video is None:
            return None
        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"错误：无法写入输出视频：{self.output_video}")
        return writer

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        if self.display_scale == 1.0:
            return frame
        width = max(1, int(frame.shape[1] * self.display_scale))
        height = max(1, int(frame.shape[0] * self.display_scale))
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _reset(self) -> None:
        self.detector.reset()
        self.detected_frames = 0
        self.tracked_frames = 0
        self.occluded_frames = 0
        self.processed_frames = 0

    def _print_summary(self) -> None:
        detected_ratio = self.detected_frames / max(self.processed_frames, 1)
        tracked_ratio = self.tracked_frames / max(self.processed_frames, 1)
        print(
            "检测统计："
            f"processed={self.processed_frames}, "
            f"detected={self.detected_frames}, "
            f"tracked={self.tracked_frames}, "
            f"occluded={self.occluded_frames}, "
            f"detect_ratio={detected_ratio:.1%}, "
            f"track_ratio={tracked_ratio:.1%}"
        )
        if self.detector.smoothed is not None:
            target = self.detector.smoothed
            print(
                "最后靶心："
                f"center=({target.center[0]:.1f}, {target.center[1]:.1f}), "
                f"area_px={target.area_px:.0f}, "
                f"radius={target.radius_px:.1f}px"
            )
        if self.output_video is not None:
            print(f"标注视频：{self.output_video}")

    def _round_point(self, point: tuple[float, float]) -> tuple[int, int]:
        return int(round(point[0])), int(round(point[1]))

    def _bullseye_label(self, smoothed: SmoothedTarget, center: tuple[int, int]) -> str:
        if smoothed.bullseye_detected:
            return f"black bullseye ({center[0]}, {center[1]})"
        return f"bullseye occluded ({center[0]}, {center[1]})"


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时追踪视频里的靶盘范围和靶心位置。")
    parser.add_argument("--video", default=VIDEO_PATH, help="输入视频路径。")
    parser.add_argument("--display-scale", type=float, default=0.75, help="预览窗口缩放比例。")
    parser.add_argument("--output-video", default=None, help="可选调试：保存实时标注视频路径；默认不输出文件。")
    parser.add_argument("--no-preview", action="store_true", help="不打开窗口，处理完整视频并打印统计。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    browser = TargetCenterVideoBrowser(
        video_path=resolve_path(args.video),
        display_scale=max(args.display_scale, 0.05),
        output_video=resolve_path(args.output_video) if args.output_video else None,
        no_preview=args.no_preview,
    )
    return browser.run()


if __name__ == "__main__":
    raise SystemExit(main())
