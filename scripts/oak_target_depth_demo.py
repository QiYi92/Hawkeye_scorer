#!/usr/bin/env python3
"""OAK-D Lite 中心靶心深度演示。

画面中心作为临时靶心。脚本会在中心附近的小方框内采样深度，并输出
深度中位数、均值和有效像素比例。

在预览窗口中按 q 或 Esc 退出。
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import depthai as dai
except ImportError as exc:
    raise SystemExit(
        "错误：当前 Python 环境未安装 depthai。请运行：python -m pip install -r requirements.txt"
    ) from exc


FONT_PATHS = [
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
]


@lru_cache(maxsize=8)
def load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_chinese_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int,
    color: tuple[int, int, int],
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text(
        position,
        text,
        font=load_font(font_size),
        fill=color,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )
    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@dataclass(frozen=True)
class DepthStats:
    median_mm: float
    mean_mm: float
    min_mm: int
    max_mm: int
    valid_pixels: int
    sample_pixels: int

    @property
    def valid_ratio(self) -> float:
        return self.valid_pixels / max(self.sample_pixels, 1)


def _socket(*names: str):
    for name in names:
        if hasattr(dai.CameraBoardSocket, name):
            return getattr(dai.CameraBoardSocket, name)
    raise RuntimeError(f"当前 DepthAI 版本不支持这些相机接口：{names}")


def build_pipeline(width: int, height: int, fps: float) -> dai.Pipeline:
    pipeline = dai.Pipeline()

    color = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)

    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")

    color.setBoardSocket(_socket("CAM_A", "RGB"))
    color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color.setPreviewSize(width, height)
    color.setInterleaved(False)
    color.setFps(fps)

    left.setBoardSocket(_socket("CAM_B", "LEFT"))
    right.setBoardSocket(_socket("CAM_C", "RIGHT"))
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setFps(fps)
    right.setFps(fps)

    if hasattr(dai.node.StereoDepth.PresetMode, "DEFAULT"):
        preset = dai.node.StereoDepth.PresetMode.DEFAULT
    else:
        preset = dai.node.StereoDepth.PresetMode.HIGH_DENSITY
    stereo.setDefaultProfilePreset(preset)
    stereo.setDepthAlign(_socket("CAM_A", "RGB"))
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)

    color.preview.link(xout_rgb.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(xout_depth.input)

    return pipeline


def center_depth_stats(depth_frame: np.ndarray, box_size: int, min_mm: int, max_mm: int) -> DepthStats | None:
    height, width = depth_frame.shape[:2]
    center_x = width // 2
    center_y = height // 2
    half = max(box_size // 2, 1)
    x1 = max(center_x - half, 0)
    x2 = min(center_x + half + 1, width)
    y1 = max(center_y - half, 0)
    y2 = min(center_y + half + 1, height)

    sample = depth_frame[y1:y2, x1:x2]
    valid = sample[(sample >= min_mm) & (sample <= max_mm)]
    if valid.size == 0:
        return None

    return DepthStats(
        median_mm=float(np.median(valid)),
        mean_mm=float(np.mean(valid)),
        min_mm=int(np.min(valid)),
        max_mm=int(np.max(valid)),
        valid_pixels=int(valid.size),
        sample_pixels=int(sample.size),
    )


def draw_overlay(frame: np.ndarray, stats: DepthStats | None, box_size: int) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    center_x = width // 2
    center_y = height // 2
    half = max(box_size // 2, 1)

    cv2.line(output, (center_x - 24, center_y), (center_x + 24, center_y), (0, 255, 255), 2)
    cv2.line(output, (center_x, center_y - 24), (center_x, center_y + 24), (0, 255, 255), 2)
    cv2.rectangle(
        output,
        (center_x - half, center_y - half),
        (center_x + half, center_y + half),
        (0, 255, 0),
        2,
    )
    cv2.circle(output, (center_x, center_y), 4, (0, 0, 255), -1)

    if stats is None:
        lines = ["靶心：画面中心", "深度：无有效采样"]
    else:
        lines = [
            "靶心：画面中心",
            f"中位数：{stats.median_mm / 1000:.3f} m",
            f"均值：{stats.mean_mm / 1000:.3f} m",
            f"有效像素：{stats.valid_ratio * 100:.1f}%",
        ]

    y = 16
    for line in lines:
        draw_chinese_text(output, line, (16, y), 18, (255, 255, 255), stroke_width=2)
        y += 24

    return output


def run(
    width: int,
    height: int,
    fps: float,
    box_size: int,
    min_mm: int,
    max_mm: int,
    duration: float | None,
    preview: bool,
) -> int:
    pipeline = build_pipeline(width, height, fps)
    started_at = time.monotonic()
    last_report = started_at
    rgb_frame = None
    latest_stats = None
    medians: list[float] = []
    rgb_count = 0
    depth_count = 0

    with dai.Device(pipeline) as device:
        print(f"已连接 OAK 设备：{device.getMxId()}")
        print(f"USB 速度：{device.getUsbSpeed()}")
        print(f"中心靶心采样框：{box_size}x{box_size}px")
        if preview:
            print("请将画面中心对准目标。在预览窗口中按 q 或 Esc 退出。")

        rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        while True:
            rgb_msg = rgb_queue.tryGet()
            depth_msg = depth_queue.tryGet()

            if rgb_msg is not None:
                rgb_frame = rgb_msg.getCvFrame()
                rgb_count += 1

            if depth_msg is not None:
                depth_frame = depth_msg.getFrame()
                latest_stats = center_depth_stats(depth_frame, box_size, min_mm, max_mm)
                depth_count += 1
                if latest_stats is not None:
                    medians.append(latest_stats.median_mm)

            if preview and rgb_frame is not None:
                cv2.imshow("OAK-D Lite 中心靶心深度", draw_overlay(rgb_frame, latest_stats, box_size))

            now = time.monotonic()
            if now - last_report >= 2.0:
                if latest_stats is None:
                    print(
                        f"运行中：{now - started_at:.1f}s，RGB帧={rgb_count}，深度帧={depth_count}，"
                        "中心深度=无有效采样"
                    )
                else:
                    print(
                        f"运行中：{now - started_at:.1f}s，RGB帧={rgb_count}，深度帧={depth_count}，"
                        f"中心中位数={latest_stats.median_mm / 1000:.3f}m，"
                        f"有效像素={latest_stats.valid_ratio * 100:.1f}%"
                    )
                last_report = now

            if preview:
                key = cv2.waitKey(1)
                if key in (ord("q"), 27):
                    break
            else:
                time.sleep(0.001)

            if duration is not None and now - started_at >= duration:
                break

    if preview:
        cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started_at, 0.001)
    print(f"采集汇总：{elapsed:.1f}s，RGB帧={rgb_count}，深度帧={depth_count}")
    if medians:
        print(
            "中心深度汇总："
            f"样本数={len(medians)}，"
            f"中位数={statistics.median(medians) / 1000:.3f}m，"
            f"最小值={min(medians) / 1000:.3f}m，"
            f"最大值={max(medians) / 1000:.3f}m"
        )
        return 0

    print("错误：没有产生有效的中心深度样本。")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测量画面中心靶心处的深度。")
    parser.add_argument("--width", type=int, default=320, help="RGB 预览宽度。")
    parser.add_argument("--height", type=int, default=180, help="RGB 预览高度。")
    parser.add_argument("--fps", type=float, default=10.0, help="相机帧率。")
    parser.add_argument("--box-size", type=int, default=21, help="中心深度采样框边长，单位像素。")
    parser.add_argument("--min-mm", type=int, default=200, help="最小有效深度，单位毫米。")
    parser.add_argument("--max-mm", type=int, default=8000, help="最大有效深度，单位毫米。")
    parser.add_argument("--duration", type=float, default=None, help="运行指定秒数后退出。")
    parser.add_argument("--no-preview", action="store_true", help="不打开窗口，仅读取数据流。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run(
            width=args.width,
            height=args.height,
            fps=args.fps,
            box_size=args.box_size,
            min_mm=args.min_mm,
            max_mm=args.max_mm,
            duration=args.duration,
            preview=not args.no_preview,
        )
    )
