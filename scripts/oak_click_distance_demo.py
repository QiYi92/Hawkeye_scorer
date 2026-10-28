#!/usr/bin/env python3
"""OAK-D Lite 点击测距演示。

画面中心作为临时靶心。鼠标左键点击一个测试点或落点后，界面会显示
像素距离、两点深度，以及两点深度都有效时的估算三维直线距离。

该结果不是正式比赛所需的地面平面距离。
"""

from __future__ import annotations

import argparse
import math
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


# macOS 的 OpenCV 窗口标题对中文支持不稳定，标题保留 ASCII，界面正文使用中文。
WINDOW_NAME = "OAK-D Lite Click Distance"
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


@dataclass(frozen=True)
class Button:
    label: str
    action: str
    rect: tuple[int, int, int, int]

    def contains(self, x: int, y: int) -> bool:
        left, top, right, bottom = self.rect
        return left <= x <= right and top <= y <= bottom


@dataclass(frozen=True)
class TextBox:
    label: str
    field: str
    rect: tuple[int, int, int, int]

    def contains(self, x: int, y: int) -> bool:
        left, top, right, bottom = self.rect
        return left <= x <= right and top <= y <= bottom


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class AppState:
    # 点击点保存为采集画面坐标，而不是放大后的显示窗口坐标。
    clicked_point: tuple[int, int] | None = None
    should_exit: bool = False
    last_action: str = "点击画面设置一个测试点。"
    last_mouse_time: float = 0.0
    last_mouse_position: tuple[int, int] | None = None
    camera_height_m: float = 1.5
    camera_pitch_deg: float = 15.0
    height_text: str = "1.5"
    pitch_text: str = "15"
    active_field: str | None = None

    def clear_point(self) -> None:
        self.clicked_point = None
        self.last_action = "已清除测试点。"

    def reset(self) -> None:
        self.clicked_point = None
        self.last_action = "已重置。"

    def apply_camera_params(self) -> None:
        try:
            height = float(self.height_text)
            pitch = float(self.pitch_text)
        except ValueError:
            self.last_action = "参数格式错误：请输入数字。"
            return
        if height <= 0:
            self.last_action = "相机高度必须大于 0。"
            return
        if pitch <= 0 or pitch >= 89:
            self.last_action = "俯角建议输入 0 到 89 度之间。"
            return
        self.camera_height_m = height
        self.camera_pitch_deg = pitch
        self.active_field = None
        self.last_action = f"已确认参数：高度 {height:.2f}m，俯角 {pitch:.1f}度。"

    def append_input_char(self, char: str) -> None:
        if self.active_field is None or char not in "0123456789.-":
            return
        current = self.height_text if self.active_field == "height" else self.pitch_text
        if char == "." and "." in current:
            return
        if char == "-" and current:
            return
        current += char
        if self.active_field == "height":
            self.height_text = current
        else:
            self.pitch_text = current

    def backspace_input(self) -> None:
        if self.active_field == "height":
            self.height_text = self.height_text[:-1]
        elif self.active_field == "pitch":
            self.pitch_text = self.pitch_text[:-1]


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
    if hasattr(stereo, "setOutputSize"):
        stereo.setOutputSize(width, height)
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)

    color.preview.link(xout_rgb.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(xout_depth.input)

    return pipeline


def read_rgb_intrinsics(device: dai.Device, width: int, height: int) -> CameraIntrinsics:
    calib = device.readCalibration()
    matrix = calib.getCameraIntrinsics(_socket("CAM_A", "RGB"), width, height)
    return CameraIntrinsics(
        fx=float(matrix[0][0]),
        fy=float(matrix[1][1]),
        cx=float(matrix[0][2]),
        cy=float(matrix[1][2]),
    )


def point_depth_stats(
    depth_frame: np.ndarray,
    point: tuple[int, int],
    box_size: int,
    min_mm: int,
    max_mm: int,
) -> DepthStats | None:
    height, width = depth_frame.shape[:2]
    x, y = point
    half = max(box_size // 2, 1)
    x1 = max(x - half, 0)
    x2 = min(x + half + 1, width)
    y1 = max(y - half, 0)
    y2 = min(y + half + 1, height)

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


def pixel_distance(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def pixel_to_camera_point(
    point: tuple[int, int],
    depth_mm: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    z_m = depth_mm / 1000.0
    x_m = (point[0] - intrinsics.cx) * z_m / intrinsics.fx
    y_m = (point[1] - intrinsics.cy) * z_m / intrinsics.fy
    return np.array([x_m, y_m, z_m], dtype=np.float32)


def map_point_between_frames(
    point: tuple[int, int],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    if source_width == target_width and source_height == target_height:
        return point
    x_scale = target_width / max(source_width, 1)
    y_scale = target_height / max(source_height, 1)
    mapped_x = int(round(point[0] * x_scale))
    mapped_y = int(round(point[1] * y_scale))
    return (
        min(max(mapped_x, 0), target_width - 1),
        min(max(mapped_y, 0), target_height - 1),
    )


def estimated_3d_distance_m(
    target: tuple[int, int],
    point: tuple[int, int],
    target_depth_mm: float,
    point_depth_mm: float,
    intrinsics: CameraIntrinsics,
) -> float:
    target_3d = pixel_to_camera_point(target, target_depth_mm, intrinsics)
    point_3d = pixel_to_camera_point(point, point_depth_mm, intrinsics)
    return float(np.linalg.norm(target_3d - point_3d))


def pixel_to_camera_ray(point: tuple[int, int], intrinsics: CameraIntrinsics) -> np.ndarray:
    ray = np.array(
        [
            (point[0] - intrinsics.cx) / intrinsics.fx,
            (point[1] - intrinsics.cy) / intrinsics.fy,
            1.0,
        ],
        dtype=np.float64,
    )
    return ray / np.linalg.norm(ray)


def ground_intersection_m(
    point: tuple[int, int],
    intrinsics: CameraIntrinsics,
    camera_height_m: float,
    pitch_deg: float,
) -> np.ndarray | None:
    ray = pixel_to_camera_ray(point, intrinsics)
    pitch = math.radians(pitch_deg)
    ground_normal = np.array([0.0, -math.cos(pitch), math.sin(pitch)], dtype=np.float64)
    denominator = float(np.dot(ground_normal, ray))
    if abs(denominator) < 1e-6:
        return None
    t = camera_height_m / denominator
    if t <= 0:
        return None
    return ray * t


def estimated_ground_distance_m(
    target: tuple[int, int],
    point: tuple[int, int],
    intrinsics: CameraIntrinsics,
    camera_height_m: float,
    pitch_deg: float,
) -> float | None:
    target_ground = ground_intersection_m(target, intrinsics, camera_height_m, pitch_deg)
    point_ground = ground_intersection_m(point, intrinsics, camera_height_m, pitch_deg)
    if target_ground is None or point_ground is None:
        return None
    return float(np.linalg.norm(point_ground - target_ground))


def layout_buttons(canvas_width: int, canvas_height: int) -> list[Button]:
    button_height = 42
    gap = 12
    margin = 14
    available_width = canvas_width - margin * 2 - gap * 2
    button_width = max(92, available_width // 3)
    bottom = canvas_height - margin
    top = bottom - button_height
    left = margin
    specs = [("重置", "reset"), ("清除点", "clear"), ("退出", "exit")]
    buttons: list[Button] = []
    for label, action in specs:
        right = min(left + button_width, canvas_width - margin)
        buttons.append(Button(label=label, action=action, rect=(left, top, right, bottom)))
        left = right + gap
    return buttons


def control_panel_top(video_height: int) -> int:
    return video_height + 8


def layout_text_boxes(canvas_width: int, video_height: int) -> list[TextBox]:
    top = control_panel_top(video_height) + 72
    box_width = max((canvas_width - 56) // 2, 120)
    box_height = 38
    left_1 = 16
    left_2 = left_1 + box_width + 24
    return [
        TextBox("相机高度(m)", "height", (left_1, top, left_1 + box_width, top + box_height)),
        TextBox("俯角(度)", "pitch", (left_2, top, min(left_2 + box_width, canvas_width - 16), top + box_height)),
    ]


def layout_confirm_button(canvas_width: int, video_height: int) -> Button:
    top = control_panel_top(video_height) + 124
    return Button("确认参数", "apply_params", (16, top, canvas_width - 16, top + 38))


def draw_button(frame: np.ndarray, button: Button) -> None:
    left, top, right, bottom = button.rect
    cv2.rectangle(frame, (left, top), (right, bottom), (42, 42, 42), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), (210, 210, 210), 1)
    font = load_font(22)
    bbox = font.getbbox(button.label)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = left + max((right - left - text_width) // 2, 4)
    text_y = top + max((bottom - top - text_height) // 2 - 2, 2)
    draw_chinese_text(frame, button.label, (text_x, text_y), 22, (255, 255, 255))


def draw_text_box(frame: np.ndarray, text_box: TextBox, value: str, is_active: bool) -> None:
    left, top, right, bottom = text_box.rect
    border_color = (0, 220, 255) if is_active else (210, 210, 210)
    cv2.rectangle(frame, (left, top), (right, bottom), (28, 28, 28), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), border_color, 2)
    draw_chinese_text(frame, text_box.label, (left, top - 25), 20, (255, 255, 255), stroke_width=1)
    display_value = value if value else " "
    if is_active and int(time.monotonic() * 2) % 2 == 0:
        display_value += "|"
    draw_chinese_text(frame, display_value, (left + 10, top + 6), 22, (255, 255, 255))


def draw_control_panel(frame: np.ndarray, state: AppState, video_height: int) -> None:
    height, width = frame.shape[:2]
    panel_top = video_height
    cv2.rectangle(frame, (0, panel_top), (width, height), (18, 18, 18), -1)
    cv2.rectangle(frame, (0, panel_top), (width - 1, height - 1), (90, 90, 90), 1)

    current = f"当前参数：高度 {state.camera_height_m:.2f}m，俯角 {state.camera_pitch_deg:.1f}度"
    draw_chinese_text(frame, current, (16, panel_top + 10), 20, (255, 255, 255), stroke_width=1)

    values = {"height": state.height_text, "pitch": state.pitch_text}
    for text_box in layout_text_boxes(width, video_height):
        draw_text_box(frame, text_box, values[text_box.field], state.active_field == text_box.field)
    draw_button(frame, layout_confirm_button(width, video_height))


def draw_text_lines(frame: np.ndarray, lines: list[str]) -> None:
    y = 18
    for line in lines:
        draw_chinese_text(frame, line, (18, y), 22, (255, 255, 255), stroke_width=2)
        y += 29


def draw_overlay(
    frame: np.ndarray,
    state: AppState,
    target_stats: DepthStats | None,
    point_stats: DepthStats | None,
    estimated_distance_m: float | None,
    estimated_ground_distance: float | None,
    box_size: int,
    display_width: int,
    display_height: int,
    canvas_height: int,
) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    video = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
    output = np.zeros((canvas_height, display_width, 3), dtype=np.uint8)
    output[:display_height, :] = video
    target_frame = (frame_width // 2, frame_height // 2)
    target = map_point_between_frames(target_frame, frame_width, frame_height, display_width, display_height)
    half = max(
        int(round((box_size // 2) * min(display_width / frame_width, display_height / frame_height))),
        2,
    )

    cv2.line(output, (target[0] - 32, target[1]), (target[0] + 32, target[1]), (0, 255, 255), 2)
    cv2.line(output, (target[0], target[1] - 32), (target[0], target[1] + 32), (0, 255, 255), 2)
    cv2.rectangle(
        output,
        (target[0] - half, target[1] - half),
        (target[0] + half, target[1] + half),
        (0, 255, 0),
        1,
    )
    cv2.circle(output, target, 5, (0, 0, 255), -1)

    lines = ["靶心：画面中心"]
    if target_stats is None:
        lines.append("靶心深度：无有效采样")
    else:
        lines.append(f"靶心深度：{target_stats.median_mm / 1000:.3f} m")

    if state.clicked_point is None:
        lines.append("测试点：点击画面设置")
    else:
        point_frame = state.clicked_point
        point = map_point_between_frames(point_frame, frame_width, frame_height, display_width, display_height)
        dist_px = pixel_distance(target_frame, point_frame)
        cv2.line(output, target, point, (255, 255, 0), 2)
        cv2.circle(output, point, 7, (255, 0, 255), -1)
        cv2.rectangle(
            output,
            (point[0] - half, point[1] - half),
            (point[0] + half, point[1] + half),
            (255, 0, 255),
            1,
        )
        lines.append(f"像素距离：{dist_px:.1f} px")
        if point_stats is None:
            lines.append("测试点深度：无有效采样")
        else:
            lines.append(f"测试点深度：{point_stats.median_mm / 1000:.3f} m")

        if estimated_distance_m is not None:
            lines.append(f"估算三维直线距离：{estimated_distance_m:.3f} m")
        else:
            lines.append("估算三维直线距离：不可用")

        if estimated_ground_distance is not None:
            lines.append(f"估算地面距离：{estimated_ground_distance:.3f} m")
        else:
            lines.append("估算地面距离：不可用")

    lines.append(state.last_action)
    draw_text_lines(output, lines)

    for button in layout_buttons(display_width, display_height):
        draw_button(output, button)
    draw_control_panel(output, state, display_height)

    return output


def handle_mouse(event: int, x: int, y: int, _flags: int, param) -> None:
    if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_LBUTTONDBLCLK):
        return

    state, frame_width, frame_height, display_width, display_height, canvas_height = param
    now = time.monotonic()
    if state.last_mouse_position == (x, y) and now - state.last_mouse_time < 0.15:
        return
    state.last_mouse_position = (x, y)
    state.last_mouse_time = now

    for button in layout_buttons(display_width, display_height):
        if not button.contains(x, y):
            continue
        if button.action == "reset":
            state.reset()
            print("鼠标操作：重置")
        elif button.action == "clear":
            state.clear_point()
            print("鼠标操作：清除测试点")
        elif button.action == "exit":
            state.should_exit = True
            print("鼠标操作：退出")
        return

    confirm_button = layout_confirm_button(display_width, display_height)
    if confirm_button.contains(x, y):
        state.apply_camera_params()
        print(state.last_action)
        return

    for text_box in layout_text_boxes(display_width, display_height):
        if text_box.contains(x, y):
            state.active_field = text_box.field
            state.last_action = f"正在编辑：{text_box.label}"
            print(f"鼠标操作：编辑{text_box.label}")
            return

    if y >= display_height or y >= canvas_height:
        state.active_field = None
        return

    frame_point = map_point_between_frames((x, y), display_width, display_height, frame_width, frame_height)
    state.clicked_point = frame_point
    state.last_action = f"已设置测试点：({frame_point[0]}, {frame_point[1]})"
    print(f"鼠标操作：设置测试点 display=({x}, {y}) capture=({frame_point[0]}, {frame_point[1]})")


def run(
    width: int,
    height: int,
    fps: float,
    box_size: int,
    min_mm: int,
    max_mm: int,
    display_scale: float,
) -> int:
    pipeline = build_pipeline(width, height, fps)
    state = AppState()
    rgb_frame = None
    depth_frame = None
    target_stats = None
    point_stats = None
    estimated_distance = None
    estimated_ground_distance = None
    intrinsics = None
    rgb_count = 0
    depth_count = 0
    started_at = time.monotonic()
    last_report = started_at

    display_width = max(int(round(width * display_scale)), width)
    display_height = max(int(round(height * display_scale)), height)
    control_height = 190
    canvas_height = display_height + control_height

    # 使用固定图像尺寸窗口，避免 macOS 上可缩放窗口导致鼠标坐标和画面坐标不一致。
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(
        WINDOW_NAME,
        handle_mouse,
        (state, width, height, display_width, display_height, canvas_height),
    )

    try:
        with dai.Device(pipeline) as device:
            print(f"已连接 OAK 设备：{device.getMxId()}")
            print(f"USB 速度：{device.getUsbSpeed()}")
            intrinsics = read_rgb_intrinsics(device, width, height)
            print(
                "RGB真实内参："
                f"fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
                f"cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}"
            )
            print("请在画面中点击一个测试点。可按 c/r/q/Esc，或使用画面底部按钮。")

            rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            while not state.should_exit:
                rgb_msg = rgb_queue.tryGet()
                depth_msg = depth_queue.tryGet()

                if rgb_msg is not None:
                    rgb_frame = rgb_msg.getCvFrame()
                    rgb_count += 1

                if depth_msg is not None:
                    depth_frame = depth_msg.getFrame()
                    depth_count += 1

                if depth_frame is not None:
                    depth_height, depth_width = depth_frame.shape[:2]
                    target_display = (width // 2, height // 2)
                    target_depth = map_point_between_frames(
                        target_display,
                        width,
                        height,
                        depth_width,
                        depth_height,
                    )
                    target_stats = point_depth_stats(depth_frame, target_depth, box_size, min_mm, max_mm)
                    if state.clicked_point is None:
                        point_stats = None
                        estimated_distance = None
                        estimated_ground_distance = None
                    else:
                        point_depth = map_point_between_frames(
                            state.clicked_point,
                            width,
                            height,
                            depth_width,
                            depth_height,
                        )
                        point_stats = point_depth_stats(depth_frame, point_depth, box_size, min_mm, max_mm)
                        if target_stats is not None and point_stats is not None:
                            estimated_distance = estimated_3d_distance_m(
                                target=target_depth,
                                point=point_depth,
                                target_depth_mm=target_stats.median_mm,
                                point_depth_mm=point_stats.median_mm,
                                intrinsics=intrinsics,
                            )
                        else:
                            estimated_distance = None
                        estimated_ground_distance = estimated_ground_distance_m(
                            target=target_depth,
                            point=point_depth,
                            intrinsics=intrinsics,
                            camera_height_m=state.camera_height_m,
                            pitch_deg=state.camera_pitch_deg,
                        )

                if rgb_frame is not None:
                    cv2.imshow(
                        WINDOW_NAME,
                        draw_overlay(
                            rgb_frame,
                            state,
                            target_stats,
                            point_stats,
                            estimated_distance,
                            estimated_ground_distance,
                            box_size,
                            display_width,
                            display_height,
                            canvas_height,
                        ),
                    )

                now = time.monotonic()
                if now - last_report >= 2.0:
                    if state.clicked_point is None:
                        print(f"运行中：{now - started_at:.1f}s，RGB帧={rgb_count}，深度帧={depth_count}，未设置测试点")
                    else:
                        target_depth = "无" if target_stats is None else f"{target_stats.median_mm / 1000:.3f}m"
                        point_depth = "无" if point_stats is None else f"{point_stats.median_mm / 1000:.3f}m"
                        print(
                            f"运行中：{now - started_at:.1f}s，RGB帧={rgb_count}，深度帧={depth_count}，"
                            f"靶心深度={target_depth}，测试点深度={point_depth}"
                        )
                    last_report = now

                key = cv2.waitKey(1)
                if key in (ord("q"), 27):
                    break
                if key in (10, 13):
                    state.apply_camera_params()
                if key == ord("c"):
                    state.clear_point()
                if key == ord("r"):
                    state.reset()
                if key in (8, 127):
                    state.backspace_input()
                elif 32 <= key <= 126:
                    state.append_input_char(chr(key))

    finally:
        cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started_at, 0.001)
    print(f"采集汇总：{elapsed:.1f}s，RGB帧={rgb_count}，深度帧={depth_count}")
    return 0 if rgb_count > 0 and depth_count > 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="点击一个测试点，并估算其到画面中心靶心的距离。")
    parser.add_argument("--width", type=int, default=320, help="RGB 采集宽度。")
    parser.add_argument("--height", type=int, default=180, help="RGB 采集高度。")
    parser.add_argument("--fps", type=float, default=10.0, help="相机帧率。")
    parser.add_argument("--box-size", type=int, default=21, help="深度采样框边长，单位像素。")
    parser.add_argument("--min-mm", type=int, default=200, help="最小有效深度，单位毫米。")
    parser.add_argument("--max-mm", type=int, default=8000, help="最大有效深度，单位毫米。")
    parser.add_argument("--display-scale", type=float, default=2.0, help="显示窗口相对采集尺寸的放大倍数。")
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
            display_scale=args.display_scale,
        )
    )
