"""Chinese UI labels and cached CJK fonts for OpenCV video frames."""
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WINDOW_NAME = "演示13：运动员轮廓追踪｜空格暂停或继续｜Q / Esc 退出"

TRACK_STATUS = {"searching": "等待运动员下降入场", "tracked": "正在跟踪", "lost": "目标暂失或已离场"}
CONTACT_STATUS = {"unconfirmed": "尚未确认", "candidate": "候选点复核中", "confirmed_estimate": "已确认触地估计", "disabled": "已关闭"}
CONTACT_REASON = {
    "waiting_for_target": "等待主目标",
    "target_missing": "当前未检测到主目标",
    "camera_moved": "镜头移动，无法确认固定点位",
    "camera_motion_unreliable": "背景运动估计不可靠",
    "fresh_feet_mask_missing": "缺少当前帧的有效脚部分割",
    "earlier_contact_candidate_obscured": "较早候选触地过程被遮挡，无法确定首次接触",
    "observing_descent": "正在观察脚部下降轨迹",
    "target_changed": "主目标已变化，重新观察",
    "nonconsecutive_support_frames": "支撑证据中断，继续观察",
    "foot_not_stable_at_candidate": "脚部未在候选点附近保持稳定",
    "descent_followed_by_supported_stance": "下降后出现连续支撑证据",
    "waiting_for_followup_support": "等待后续帧确认支撑",
    "ambiguous_simultaneous_foot_regions": "双脚接触时序不明确",
    "insufficient_prior_descent": "此前下降证据不足",
    "foot_outside_ground_reference": "脚部位于有效地面参考区域外",
    "ground_reference_unavailable": "尚无可靠地面参考",
}


@lru_cache(maxsize=8)
def chinese_font(size=22):
    override = os.environ.get("DEMO13_FONT")
    candidates = [override] if override else [
        "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc", "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidates:
        if path and Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    raise RuntimeError("未找到可用中文字体，请将 DEMO13_FONT 设置为中文 .ttf/.ttc 字体的完整路径")


def draw_panel(canvas, lines, bottom=False, max_width=760, size=22):
    """Render only the panel ROI; wrap long Chinese strings on portrait videos."""
    h, w = canvas.shape[:2]
    width = min(w, max_width)
    padding = min(12, max(2, width//20))
    font = chinese_font(size)
    wrapped = []
    for line in lines:
        current = ""
        for character in line:
            if current and font.getlength(current+character) > max(1, width-padding*2):
                wrapped.append(current)
                current = ""
            current += character
        wrapped.append(current)
    spacing = size + 7
    height = min(h, len(wrapped)*spacing+padding*2)
    top = h-height if bottom else 0
    image = Image.new("RGB", (width, height), (25, 25, 25))
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(wrapped):
        draw.text((padding, padding+index*spacing), line, font=font, fill=(240, 240, 240), anchor="lt")
    canvas[top:top+height, :width] = np.asarray(image)[:, :, ::-1]


def draw_label(canvas, text, x, y, bgr_color, size=20):
    font = chinese_font(size)
    h, w = canvas.shape[:2]
    width = min(w, int(font.getlength(text))+6)
    height = min(h, size+10)
    x = max(0, min(int(x), w-width))
    y = max(0, min(int(y)-height, h-height))
    image = Image.fromarray(canvas[y:y+height, x:x+width, ::-1].copy())
    ImageDraw.Draw(image).text((2, 2), text, font=font, fill=tuple(reversed(bgr_color)),
                              stroke_width=1, stroke_fill=(20, 20, 20), anchor="lt")
    canvas[y:y+height, x:x+width] = np.asarray(image)[:, :, ::-1]
