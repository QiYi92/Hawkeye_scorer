"""Streaming exports: bounded RAM, original-resolution masks, explicit missing rows."""
import csv
import json

import cv2
import numpy as np

from .geometry import encode_rle
from .display_text import TRACK_STATUS, draw_label, draw_panel


CSV_FIELDS = ["frame_id", "timestamp", "track_id", "source_track_id", "class", "is_primary", "status", "confidence",
              "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "centroid_x", "centroid_y",
              "mask_area", "mask_width", "mask_height", "aspect_ratio", "orientation",
              "velocity_x_px_s", "velocity_y_px_s", "leg_area", "foot_area", "foot_bottom_y",
              "parts_source", "parts_age_frames", "contact_status", "contact_frame",
              "contact_x", "contact_y"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class Exporter:
    def __init__(self, output, fps, size, mask_format="png"):
        self.output, self.mask_format = output, mask_format
        self.video = cv2.VideoWriter(str(output / "video_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.video.isOpened():
            self.video.release()
            raise RuntimeError("无法创建 MP4 输出，请检查 OpenCV 编码器与输出目录")
        self.csv_file = self.json_file = None
        try:
            self.csv_file = (output / "tracks.csv").open("w", newline="", encoding="utf-8-sig")
            self.json_file = (output / "frames.jsonl").open("w", encoding="utf-8")
            self.csv = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
            self.csv.writeheader()
        except Exception:
            self.close()
            raise

    def mask(self, mask, category, frame_id, track_id):
        if self.mask_format == "rle":
            return {"rle": encode_rle(mask)}
        relative = f"{category}/{frame_id:06d}_id{track_id}.png"
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
            raise OSError(f"写入 Mask 失败：{path}")
        return {"path": relative}

    def frame(self, payload, overlay):
        self.json_file.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        base = {"frame_id": payload["frame_id"], "timestamp": payload["timestamp"]}
        contact = payload.get("first_contact")
        if contact:
            event = contact.get("event")
            base.update(contact_status=contact["status"], contact_frame=event["frame_id"] if event else None,
                        contact_x=event["point"][0] if event else None,
                        contact_y=event["point"][1] if event else None)
        for item in payload["tracks"]:
            g = item["geometry"]
            legs, feet = item["legs"], item["feet"]
            self.csv.writerow({**base, "track_id": item["track_id"], "source_track_id": item["source_track_id"], "class": item["class"],
                "is_primary": item["is_primary"], "status": "detected", "confidence": item["confidence"],
                **dict(zip(["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"], item["bbox"])),
                "centroid_x": g["centroid"][0], "centroid_y": g["centroid"][1],
                "mask_area": g["area"], "mask_width": g["width"], "mask_height": g["height"],
                "aspect_ratio": g["aspect_ratio"], "orientation": g["orientation"],
                "velocity_x_px_s": item["velocity_px_s"][0], "velocity_y_px_s": item["velocity_px_s"][1],
                "leg_area": legs["area"] if legs else (0 if item["parts_source"] in ("inferred", "propagated") else None),
                "foot_area": feet["area"] if feet else (0 if item["parts_source"] in ("inferred", "propagated") else None),
                "foot_bottom_y": feet["bottom_y"] if feet else None,
                "parts_source": item["parts_source"], "parts_age_frames": item["parts_age_frames"]})
        if payload["primary_status"] != "tracked":
            self.csv.writerow({**base, "track_id": payload["primary_track_id"], "is_primary": True,
                               "status": payload["primary_status"]})
        self.video.write(overlay)

    def close(self):
        self.video.release()
        for file in (self.csv_file, self.json_file):
            if file:
                file.close()


def draw_overlay(frame, instances, primary_id, status, frame_id, fps, mode, labels=None, show_bbox=True,
                 contact_enabled=False):
    canvas = frame.copy()
    for item in instances:
        color = (80, 215, 80) if item.track_id == primary_id else (200, 150, 40)
        canvas[item.mask] = (.65 * canvas[item.mask] + .35 * np.array(color)).astype(np.uint8)
        curves, _ = cv2.findContours(item.mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, curves, -1, color, 2)
        x1, y1, x2, y2 = map(int, item.bbox)
        if show_bbox:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        draw_label(canvas, f"编号：{item.track_id}  置信度：{item.confidence:.2f}", x1, y1-6, color)
    if labels is not None:
        for k, color in ((1, (0, 180, 255)), (2, (255, 80, 200))):
            selected = labels == k
            canvas[selected] = (.35 * canvas[selected] + .65 * np.array(color)).astype(np.uint8)
    lines = [f"演示13  |  帧号：{frame_id}  |  处理速度：{fps:.1f} 帧/秒",
             f"主目标编号：{primary_id if primary_id is not None else '未锁定'}  |  {TRACK_STATUS.get(status, '状态未知')}",
             "通用人体分割：装备完整性待验证" if mode == "person" else "专用分割：运动员及随身装备",
             "绿色：运动员轮廓  |  橙色：腿部  |  粉色：脚部",
             f"首次触地判断：{'已开启' if contact_enabled else '已关闭'}  |  空格暂停/继续，Q 退出"]
    draw_panel(canvas, lines)
    return canvas
