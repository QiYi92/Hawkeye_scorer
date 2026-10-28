#!/usr/bin/env python3
"""OAK-D Lite RGB + 深度最小采集预览演示。

在预览窗口中按 q 或 Esc 退出。
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

try:
    import depthai as dai
except ImportError as exc:
    raise SystemExit(
        "错误：当前 Python 环境未安装 depthai。请运行：python -m pip install -r requirements.txt"
    ) from exc


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


def colorize_depth(depth_frame: np.ndarray, max_depth_mm: int) -> np.ndarray:
    clipped = np.clip(depth_frame, 0, max_depth_mm)
    depth_8bit = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)


def run(
    width: int,
    height: int,
    fps: float,
    max_depth_mm: int,
    duration: float | None,
    preview: bool,
) -> int:
    pipeline = build_pipeline(width, height, fps)

    rgb_count = 0
    depth_count = 0
    started_at = time.monotonic()
    last_report = started_at

    with dai.Device(pipeline) as device:
        print(f"已连接 OAK 设备：{device.getMxId()}")
        print(f"USB 速度：{device.getUsbSpeed()}")
        if preview:
            print("在预览窗口中按 q 或 Esc 退出。")
        elif duration is None:
            print("未开启预览且未设置运行时长；按 Ctrl+C 退出。")

        rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        while True:
            rgb_msg = rgb_queue.tryGet()
            depth_msg = depth_queue.tryGet()

            if rgb_msg is not None:
                rgb_count += 1
                if preview:
                    cv2.imshow("OAK-D Lite RGB 画面", rgb_msg.getCvFrame())

            if depth_msg is not None:
                depth_count += 1
                depth_frame = depth_msg.getFrame()
                if preview:
                    cv2.imshow("OAK-D Lite 深度画面", colorize_depth(depth_frame, max_depth_mm))

            now = time.monotonic()
            if now - last_report >= 2.0:
                elapsed = now - started_at
                print(f"运行中：{elapsed:.1f}s，RGB帧={rgb_count}，深度帧={depth_count}")
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
    print(
        "采集汇总："
        f"{elapsed:.1f}s，RGB={rgb_count}（{rgb_count / elapsed:.1f} fps），"
        f"深度={depth_count}（{depth_count / elapsed:.1f} fps）"
    )
    if rgb_count == 0 or depth_count == 0:
        print("错误：RGB 或深度流没有输出帧。")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预览 OAK-D Lite 的 RGB 和深度流。")
    parser.add_argument("--width", type=int, default=640, help="RGB 预览宽度。")
    parser.add_argument("--height", type=int, default=360, help="RGB 预览高度。")
    parser.add_argument("--fps", type=float, default=30.0, help="相机帧率。")
    parser.add_argument(
        "--max-depth-mm",
        type=int,
        default=5000,
        help="深度伪彩色预览的最大映射深度，单位毫米。",
    )
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
            max_depth_mm=args.max_depth_mm,
            duration=args.duration,
            preview=not args.no_preview,
        )
    )
