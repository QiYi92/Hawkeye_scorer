#!/usr/bin/env python3
"""Analyze a landing video and mark first foot-ground contact."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from landing_detection.estimator import LandingPointEstimator, LandingPointEstimatorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="滑翔伞运动员首次脚部落地识别 MVP。")
    parser.add_argument("--video", type=Path, required=True, help="输入视频路径。")
    parser.add_argument("--output-video", type=Path, default=Path("output/result.mp4"), help="输出标注视频路径。")
    parser.add_argument("--output-json", type=Path, default=Path("output/result.json"), help="输出 JSON 路径。")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose 模型名或本地模型路径。")
    parser.add_argument("--conf", type=float, default=0.25, help="人体检测置信度阈值。")
    parser.add_argument("--keypoint-conf", type=float, default=0.20, help="脚踝关键点置信度阈值。")
    parser.add_argument("--contact-threshold", type=float, default=5.0, help="触地像素阈值。")
    parser.add_argument("--contact-frames", type=int, default=3, help="连续触地帧数。")
    parser.add_argument("--preview", action="store_true", help="处理时显示预览窗口，按 q 或 Esc 退出。")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="日志级别。")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    if args.contact_frames < 1:
        raise SystemExit("错误：--contact-frames 必须大于等于 1。")
    if args.contact_threshold < 0:
        raise SystemExit("错误：--contact-threshold 必须大于等于 0。")

    estimator = LandingPointEstimator(
        LandingPointEstimatorConfig(
            model_name=args.model,
            confidence_threshold=args.conf,
            keypoint_confidence_threshold=args.keypoint_conf,
            contact_threshold_px=args.contact_threshold,
            consecutive_contact_frames=args.contact_frames,
            preview=args.preview,
        )
    )
    result = estimator.analyze_video(
        video_path=args.video,
        output_video_path=args.output_video,
        output_json_path=args.output_json,
    )
    print(result.to_json_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
