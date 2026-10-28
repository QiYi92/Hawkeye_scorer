"""Ultralytics YOLO Pose wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .models import FootPoints, Point, PoseDetection

LOGGER = logging.getLogger(__name__)


COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16


@dataclass(frozen=True)
class PoseDetectorConfig:
    model_name: str = "yolo11n-pose.pt"
    confidence_threshold: float = 0.25
    keypoint_confidence_threshold: float = 0.35


class PoseDetector:
    """Detect person poses in a frame."""

    def __init__(self, config: PoseDetectorConfig) -> None:
        self.config = config
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                "错误：当前 Python 环境未安装 ultralytics。请运行：python -m pip install -r requirements.txt"
            ) from exc

        LOGGER.info("Loading YOLO pose model: %s", config.model_name)
        self.model = YOLO(config.model_name)

    def detect(self, frame: np.ndarray) -> PoseDetection | None:
        """Run pose inference and return the highest-confidence detected person."""
        detections = self.detect_all(frame)
        if not detections:
            return None
        return max(detections, key=lambda detection: detection.confidence)

    def detect_all(self, frame: np.ndarray) -> list[PoseDetection]:
        """Run pose inference and return all detected people sorted by confidence."""
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None or len(boxes) == 0:
            return []

        confidences = boxes.conf.detach().cpu().numpy()
        detections: list[PoseDetection] = []
        for index in range(len(boxes)):
            confidence = float(confidences[index])
            xyxy = boxes.xyxy[index].detach().cpu().numpy()
            x1, y1, x2, y2 = [int(round(value)) for value in xyxy.tolist()]

            keypoint_xy = keypoints.xy[index].detach().cpu().numpy()
            keypoint_conf = None
            if getattr(keypoints, "conf", None) is not None:
                keypoint_conf = keypoints.conf[index].detach().cpu().numpy()

            foot_points = self._extract_foot_points(keypoint_xy, keypoint_conf)
            detections.append(
                PoseDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    keypoints=keypoint_xy,
                    keypoint_confidences=keypoint_conf,
                    foot_points=foot_points,
                )
            )
        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)

    def _extract_foot_points(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray | None,
    ) -> FootPoints:
        """Extract left/right ankle points from COCO pose output."""
        return FootPoints(
            left=self._keypoint_to_point(keypoints, confidences, COCO_LEFT_ANKLE),
            right=self._keypoint_to_point(keypoints, confidences, COCO_RIGHT_ANKLE),
        )

    def _keypoint_to_point(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray | None,
        index: int,
    ) -> Point | None:
        if index >= len(keypoints):
            return None
        if confidences is not None and float(confidences[index]) < self.config.keypoint_confidence_threshold:
            return None

        x, y = keypoints[index][:2]
        if x <= 0 and y <= 0:
            return None
        return int(round(float(x))), int(round(float(y)))
