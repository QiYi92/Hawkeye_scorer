"""End-to-end video landing first-contact estimator."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from .athlete import AthleteSelector
from .contact import FirstContactConfig, FirstContactDetector, MotionContactConfig, MotionContactDetector
from .foot_parts import FootPartEstimator
from .ground import GroundLineConfig, GroundLineEstimator, HorizonLineConfig, HorizonLineEstimator
from .models import FirstContact, FootPoints, LandingResult, PoseDetection, VideoInfo
from .pose import PoseDetector, PoseDetectorConfig
from .visualizer import LandingVisualizer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LandingPointEstimatorConfig:
    model_name: str = "yolo11n-pose.pt"
    confidence_threshold: float = 0.25
    keypoint_confidence_threshold: float = 0.35
    contact_threshold_px: float = 5.0
    consecutive_contact_frames: int = 3
    preview: bool = False


class LandingPointEstimator:
    """Analyze a single video and estimate first foot-ground contact."""

    def __init__(self, config: LandingPointEstimatorConfig) -> None:
        self.config = config
        self.pose_detector = PoseDetector(
            PoseDetectorConfig(
                model_name=config.model_name,
                confidence_threshold=config.confidence_threshold,
                keypoint_confidence_threshold=config.keypoint_confidence_threshold,
            )
        )
        self.athlete_selector = AthleteSelector()
        self.foot_part_estimator = FootPartEstimator()
        self.horizon_estimator = HorizonLineEstimator(HorizonLineConfig())
        self.ground_estimator = GroundLineEstimator(GroundLineConfig())
        self.contact_detector = FirstContactDetector(
            FirstContactConfig(
                contact_threshold_px=config.contact_threshold_px,
                consecutive_frames=config.consecutive_contact_frames,
            )
        )
        self.motion_contact_detector = MotionContactDetector(
            MotionContactConfig(consecutive_frames=config.consecutive_contact_frames)
        )
        self.visualizer = LandingVisualizer()

    def analyze_video(
        self,
        video_path: Path,
        output_video_path: Path,
        output_json_path: Path,
    ) -> LandingResult:
        video_info, capture = self._open_video(video_path)
        LOGGER.info(
            "Video opened: %s, %dx%d, fps=%.3f, frames=%d",
            video_info.path.name,
            video_info.width,
            video_info.height,
            video_info.fps,
            video_info.frame_count,
        )

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        writer = self._open_writer(output_video_path, video_info)

        frame_index = 0
        first_contact_logged = False
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                detections = self.pose_detector.detect_all(frame)
                horizon_line = self.horizon_estimator.estimate(frame)
                ground_line = self.ground_estimator.estimate(
                    frame,
                    horizon_line=horizon_line,
                    athlete_bbox=None,
                )
                detection = self.athlete_selector.update(
                    detections,
                    frame_index,
                    ground_line=ground_line,
                    frame_shape=frame.shape,
                )
                ground_line = self.ground_estimator.estimate(
                    frame,
                    horizon_line=horizon_line,
                    athlete_bbox=detection.bbox if detection is not None else None,
                )
                first_contact = None
                foot_part_estimate = None
                if detection is not None:
                    timestamp_ms = int(round(frame_index / max(video_info.fps, 0.001) * 1000))
                    foot_part_estimate = self.foot_part_estimator.estimate(
                        frame,
                        detection,
                        timestamp_ms=timestamp_ms,
                    )
                    motion_foot = detection.foot_points.lowest()
                    motion_contact_part = "unknown"
                    motion_contact_status = "unknown"
                    motion_contact_point = None
                    if motion_foot is not None:
                        foot_side = self._lowest_foot_side(detection)
                        refined_contact = self.foot_part_estimator.refine_contact(
                            foot_part_estimate,
                            foot_side,
                            motion_foot,
                            detection,
                        )
                        motion_contact_part = refined_contact.part
                        motion_contact_point = refined_contact.point
                        motion_contact_status = refined_contact.status
                    first_contact = self.motion_contact_detector.update(
                        frame_index=frame_index,
                        fps=video_info.fps,
                        detection=detection,
                        frame_height=video_info.height,
                        contact_point=motion_contact_point,
                        contact_part=motion_contact_part,
                        contact_part_status=motion_contact_status,
                    )
                    if first_contact is None:
                        refined_foot_points = self._refined_foot_points(detection, foot_part_estimate)
                        first_contact = self.contact_detector.update(
                            frame_index=frame_index,
                            fps=video_info.fps,
                            foot_points=refined_foot_points,
                            ground_line=ground_line,
                        )
                    if first_contact is not None and not first_contact_logged:
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
                        if self.motion_contact_detector.first_contact is not None:
                            self.motion_contact_detector.first_contact = first_contact
                        LOGGER.info(
                            "First contact: frame=%d, timestamp=%.3fs, point=%s, foot=%s, part=%s, status=%s",
                            first_contact.frame_index,
                            first_contact.timestamp,
                            first_contact.point,
                            first_contact.foot,
                            first_contact.contact_part,
                            first_contact.contact_part_status,
                        )
                        first_contact_logged = True

                annotated = self.visualizer.draw(
                    frame=frame,
                    detection=detection,
                    ground_line=ground_line,
                    first_contact=self.contact_detector.first_contact
                    or self.motion_contact_detector.first_contact,
                )
                writer.write(annotated)

                if self.config.preview:
                    cv2.imshow("Landing First Contact", annotated)
                    key = cv2.waitKey(1)
                    if key in (ord("q"), 27):
                        LOGGER.info("Preview interrupted by user.")
                        break

                frame_index += 1
                if frame_index % 100 == 0:
                    LOGGER.info("Processed %d frames", frame_index)
        finally:
            capture.release()
            writer.release()
            if self.config.preview:
                cv2.destroyAllWindows()

        result = LandingResult(
            video_name=video_info.path.name,
            first_contact=self.contact_detector.first_contact
            or self.motion_contact_detector.first_contact,
        )
        self._write_json(output_json_path, result)
        if result.first_contact is None:
            LOGGER.warning("No first contact detected in %s", video_info.path.name)
        LOGGER.info("Saved annotated video: %s", output_video_path)
        LOGGER.info("Saved JSON result: %s", output_json_path)
        return result

    def _open_video(self, video_path: Path) -> tuple[VideoInfo, cv2.VideoCapture]:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
            LOGGER.warning("Video FPS unavailable; using %.1f", fps)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoInfo(video_path, fps, width, height, frame_count), capture

    def _open_writer(self, output_video_path: Path, video_info: VideoInfo) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            video_info.fps,
            (video_info.width, video_info.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video writer: {output_video_path}")
        return writer

    def _write_json(self, output_json_path: Path, result: LandingResult) -> None:
        with output_json_path.open("w", encoding="utf-8") as file:
            json.dump(result.to_json_dict(), file, ensure_ascii=False, indent=2)
            file.write("\n")

    def _lowest_foot_side(self, detection) -> str:
        left = detection.foot_points.left
        right = detection.foot_points.right
        if left is None and right is None:
            return "unknown"
        if left is None:
            return "right"
        if right is None:
            return "left"
        return "left" if left[1] >= right[1] else "right"

    def _refined_foot_points(
        self,
        detection: PoseDetection,
        foot_part_estimate,
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
