"""Automatic video segmentation/tracking entry point; see docs/demo13.md."""
import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np

from demo13.geometry import mask_geometry, TemporalFeatures
from demo13.output import Exporter, draw_overlay, write_json
from demo13.parsing import PartMasks, SCHPParser, TorchScriptParser
from demo13.tracking import YoloSegmenter, PrimaryTrack, DescendingAthleteTrack
from demo13.contact import FirstFootContact, draw_contact_overlay
from demo13.display_text import WINDOW_NAME, TRACK_STATUS, chinese_font


ROOT = Path(__file__).resolve().parents[1]


def arguments(argv=None):
    p = argparse.ArgumentParser(description="Demo13：YOLO 实例 Mask + 稳定跟踪 + 可选腿脚分割（无需框选、无关键点）")
    p.add_argument("--video", type=Path, default=ROOT / "training_video/video6.mp4")
    p.add_argument("--model", default="yolo26n-seg.pt")
    p.add_argument("--mode", choices=["person", "athlete"], default="person")
    p.add_argument("--tracker", choices=["botsort", "bytetrack"], default="botsort")
    p.add_argument("--device", default="", help="留空自动选择；Mac 可测试 mps，或指定 cpu / 0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=.25)
    p.add_argument("--output", type=Path, help="新的或空的目录；默认自动创建时间戳目录")
    p.add_argument("--mask-format", choices=["png", "rle"], default="png")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--display-scale", type=float, default=.75)
    p.add_argument("--no-bbox", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="0 为全片；短测可设为 60")
    p.add_argument("--ema-alpha", type=float, default=.65)
    p.add_argument("--confirm-frames", type=int, default=3)
    p.add_argument("--selection", choices=["descending", "largest"], default="descending",
                   help="descending 等待下降入场；largest 仅用于旧逻辑对比")
    p.add_argument("--entry-top", type=float, default=.45, help="候选初始框顶位于图像高度的此比例以上")
    p.add_argument("--descent-distance", type=float, default=.04, help="下降确认所需的最小图像高度比例")
    p.add_argument("--lost-buffer", type=int, default=30, help="主目标选择保持帧数，独立于底层 tracker buffer")
    p.add_argument("--parts", choices=["none", "schp", "torchscript"], default="none")
    p.add_argument("--parts-weights", type=Path)
    p.add_argument("--schp-repo", type=Path)
    p.add_argument("--parts-device", default="auto", help="auto / cpu / mps / cuda:0；auto 优先 GPU")
    p.add_argument("--parts-size", type=int, default=256)
    p.add_argument("--parts-interval", type=int, default=3)
    p.add_argument("--first-contact", action=argparse.BooleanOptionalAction, default=False,
                   help="启用首次脚部触地估计；确认前强制每帧腿脚分割")
    p.add_argument("--contact-confirm-frames", type=int, default=4)
    p.add_argument("--contact-tolerance", type=float, default=6., help="720p 基准像素容差")
    p.add_argument("--contact-ground-line", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                   help="可选局部地面线段，原图像素；省略则用自动检测的靶盘区域")
    a = p.parse_args(argv)
    for name in ("imgsz", "confirm_frames", "parts_size", "parts_interval"):
        if getattr(a, name) < 1:
            p.error(f"--{name.replace('_', '-')} 必须为正整数")
    if a.max_frames < 0 or a.lost_buffer < 0 or not 0 < a.conf <= 1 or not 0 < a.ema_alpha <= 1:
        p.error("帧数不能为负，conf / ema-alpha 必须在 (0,1]")
    if not np.isfinite(a.display_scale) or a.display_scale <= 0:
        p.error("display-scale 必须为有限正数")
    if not 0 < a.entry_top <= 1 or not 0 < a.descent_distance < 1:
        p.error("entry-top 必须在 (0,1]，descent-distance 必须在 (0,1)")
    if not a.video.is_file():
        p.error(f"找不到视频：{a.video}")
    if a.first_contact and a.parts == "none":
        p.error("首次触地判断依赖真实脚部 Mask，请启用 schp / torchscript，或将首次触地判断设为 False")
    if a.contact_confirm_frames < 3 or not np.isfinite(a.contact_tolerance) or a.contact_tolerance <= 0:
        p.error("contact-confirm-frames 至少 3；contact-tolerance 必须为有限正数")
    if a.contact_ground_line is not None and (not np.isfinite(a.contact_ground_line).all() or
            np.linalg.norm(np.array(a.contact_ground_line[:2])-a.contact_ground_line[2:]) < 1):
        p.error("contact-ground-line 需要两个不同的有限像素坐标")
    missing = []
    if a.parts != "none" and (a.parts_weights is None or not a.parts_weights.is_file()):
        expected = "SCHP LIP 的 exp-schp-201908261155-lip.pth" if a.parts == "schp" else "3 类腿脚 TorchScript 模型"
        missing.append(f"腿脚权重路径 / --parts-weights：需要 {expected}，当前值 {a.parts_weights}")
    if a.parts == "schp" and (a.schp_repo is None or not all(
            (a.schp_repo / name).is_file() for name in ("networks/AugmentCE2P.py", "utils/transforms.py", "networks/__init__.py"))):
        missing.append(f"SCHP仓库路径 / --schp-repo：需要完整 SCHP 源码目录，当前值 {a.schp_repo}")
    if missing:
        p.error("腿脚模式配置不完整：\n  " + "\n  ".join(missing) +
                "\n仅修改腿脚模式开关不会安装模型；yolo26n-seg.pt 不能用作腿脚权重。详见 docs/demo13.md。")
    return a


def run(a):
    chinese_font()  # Fail before inference if the configured CJK font is unavailable.
    segmenter = YoloSegmenter(a.model, a.mode, a.tracker, a.conf, a.imgsz, a.device)
    parser = None
    if a.parts == "torchscript":
        parser = TorchScriptParser(a.parts_weights, a.parts_device, a.parts_size)
    elif a.parts == "schp":
        print("正在加载 SCHP LIP 腿脚模型（首次设备预热可能较慢）…", flush=True)
        parser = SCHPParser(a.schp_repo, a.parts_weights, a.parts_device)
    if parser is not None:
        print(f"腿脚模型已就绪：{a.parts} | 推理设备：{parser.device} | 每 {a.parts_interval} 帧推理一次", flush=True)
    parts = PartMasks(parser, a.parts_interval)
    primary = None
    contact = None
    contact_snapshot = None
    contact_image_saved = False
    temporal = TemporalFeatures(a.ema_alpha)
    output = a.output or ROOT / "output" / f"demo13_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"输出目录非空，拒绝覆盖已有结果：{output}")
    capture = cv2.VideoCapture(str(a.video))
    exporter = None
    count = detected = primary_frames = switches = lost_frames = 0
    last_primary = None
    seen_ids = set()
    stage_time = {"segmentation_tracking": 0., "parts": 0., "features_export": 0., "contact": 0.}
    processing_seconds = 0.
    stop_reason, failure = "end_of_video", None
    try:
        if not capture.isOpened():
            raise ValueError(f"无法解码视频：{a.video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("视频 FPS 无效，无法确定时间戳与输出帧率")
        width, height = int(capture.get(3)), int(capture.get(4))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if a.first_contact:
            if a.contact_ground_line is not None:
                x1, y1, x2, y2 = a.contact_ground_line
                if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
                    raise ValueError("触地地面参考线必须在原图范围内")
            contact = FirstFootContact(fps, a.contact_confirm_frames, a.contact_tolerance, a.contact_ground_line)
        primary = (DescendingAthleteTrack(a.confirm_frames, a.lost_buffer, fps, a.entry_top, a.descent_distance)
                   if a.selection == "descending" else PrimaryTrack(a.confirm_frames, a.lost_buffer))
        output.mkdir(parents=True, exist_ok=True)
        exporter = Exporter(output, fps, (width, height), a.mask_format)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        write_json(output / "config.json", config)
        print(f"输出目录：{output}", flush=True)
        if contact:
            print("首次触地估计已开启：确认前逐帧分割腿脚；无可靠地面参考时只保留未确认状态。", flush=True)
        if a.selection == "descending":
            print("等待下降入场的运动员；仅显示和导出主目标，离场后不自动改选其他人。", flush=True)
        if a.mode == "person":
            print("通用人体分割不保证包含伞包和安全带；完整目标识别需专用运动员分割权重。", flush=True)
        while True:
            if a.max_frames and count >= a.max_frames:
                stop_reason = "max_frames"
                break
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                if total_frames > 0 and count < total_frames:
                    stop_reason = "decode_ended_early"
                break
            if frame.shape[:2] != (height, width):
                raise ValueError("视频中途改变了分辨率")
            # POS_MSEC preserves source timestamps when the backend provides monotonic values.
            source_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            valid_source_time = np.isfinite(source_time) and (
                (count == 0 and source_time >= 0) or (count > 0 and source_time > previous_time))
            timestamp = source_time if valid_source_time else (0. if count == 0 else previous_time + 1 / fps)
            previous_time = timestamp
            t = time.perf_counter()
            instances = segmenter.infer(frame)
            stage_time["segmentation_tracking"] += time.perf_counter() - t
            primary_id, status = (primary.update(instances, count, frame) if a.selection == "descending"
                                  else primary.update(instances, count))
            candidate_count = len(instances)
            main_instance = (next((i for i in instances if i.track_id == primary.source_track_id), None)
                             if status == "tracked" else None)
            if main_instance is not None:
                main_instance = replace(main_instance, track_id=primary_id, source_track_id=main_instance.track_id)
            # Background candidates are used only for selection, never drawn or exported.
            instances = [main_instance] if main_instance is not None else []
            if primary_id is not None:
                if last_primary is not None and primary_id != last_primary:
                    switches += 1
                last_primary = primary_id
            detected += bool(instances)
            primary_frames += status == "tracked"
            lost_frames += status == "lost"
            t = time.perf_counter()
            labels, parts_source, parts_age = parts.update(
                frame, main_instance, count, force_inference=bool(contact and contact.needs_fresh_parts))
            stage_time["parts"] += time.perf_counter() - t
            if contact:
                t = time.perf_counter()
                contact_snapshot = contact.update(frame, count, timestamp, main_instance, labels, parts_source)
                if contact.event is not None and not contact_image_saved:
                    if contact.event_image is not None:
                        if not cv2.imwrite(str(output / "first_contact_frame.jpg"),
                                           draw_contact_overlay(contact.event_image.copy(), contact_snapshot)):
                            raise OSError("无法写入首次触地回溯截图")
                    contact_image_saved = True
                    print(f"首次触地估计：帧 {contact.event['frame_id']}，点 {contact.event['point']}；"
                          f"在帧 {contact.event['confirmation_frame']} 完成复核。", flush=True)
                stage_time["contact"] += time.perf_counter() - t
            t = time.perf_counter()
            records = []
            for instance in instances:
                seen_ids.add(instance.track_id)
                geometry = mask_geometry(instance.mask, contours=True)
                smooth, velocity = temporal.update(instance.track_id, count, timestamp, geometry)
                is_primary = instance.track_id == primary_id
                record = {"track_id": instance.track_id, "source_track_id": instance.source_track_id, "class_id": instance.class_id,
                          "class": segmenter.target_name, "is_primary": is_primary,
                          "confidence": instance.confidence, "bbox": instance.bbox,
                          "geometry": geometry, "smoothed": smooth, "velocity_px_s": velocity,
                          "mask": exporter.mask(instance.mask, "masks", count, instance.track_id),
                          "parts_source": parts_source if is_primary else "not_primary",
                          "parts_age_frames": parts_age if is_primary else None,
                          "legs": None, "feet": None}
                if is_primary and labels is not None:
                    for label, name in ((1, "legs"), (2, "feet")):
                        mask = labels == label
                        record[name] = mask_geometry(mask)
                        record[f"{name}_mask"] = exporter.mask(mask, name, count, instance.track_id)
                records.append(record)
            temporal.prune(count)
            overlay = draw_overlay(frame, instances, primary_id, status, count,
                                   count / processing_seconds if processing_seconds else 0.,
                                   a.mode, labels, not a.no_bbox, contact_enabled=a.first_contact)
            if contact:
                overlay = draw_contact_overlay(overlay, contact_snapshot)
            exporter.frame({"frame_id": count, "timestamp": timestamp, "primary_track_id": primary_id,
                            "primary_status": status, "selection_reason": getattr(primary, "reason", "largest"),
                            "candidate_count": candidate_count, "tracks": records,
                            "first_contact": contact_snapshot}, overlay)
            stage_time["features_export"] += time.perf_counter() - t
            processing_seconds += time.perf_counter() - started
            count += 1
            if count % 30 == 0:
                print(f"帧 {count}/{total_frames} | 主目标 {primary_id if primary_id is not None else '未锁定'} "
                      f"{TRACK_STATUS.get(status, '状态未知')} | {count / processing_seconds:.1f} 帧/秒", flush=True)
            if a.preview:
                preview = cv2.resize(overlay, None, fx=a.display_scale, fy=a.display_scale)
                cv2.imshow(WINDOW_NAME, preview)
                delay = max(1, round(1000 / fps - (time.perf_counter() - started) * 1000))
                key = cv2.waitKey(delay) & 0xFF
                if key == 32:
                    while True:
                        key = cv2.waitKey(50) & 0xFF
                        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                            key = 27
                        if key in (32, 27, ord("q")):
                            break
                if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    stop_reason = "user_quit"
                    break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as exc:
        stop_reason, failure = "error", str(exc)
        raise
    finally:
        capture.release()
        if exporter:
            exporter.close()
            if contact:
                write_json(output / "first_contact.json", {
                    **contact.snapshot(), "stop_reason": stop_reason, "frames_processed": count,
                    "method": "foot_mask_descent_ground_gate_stance_v1",
                    "limitations": ["单目像素估计，不保证真实首次接触或毫米级位置。",
                                    "自动地面参考仅覆盖检测到的靶盘，盘外需独立配置局部地面线。",
                                    "预览在连续帧确认后显示点位，first_contact_frame.jpg 回溯候选首帧。"]})
            import ultralytics
            write_json(output / "metrics.json", {
                "schema_version": 2, "stop_reason": stop_reason, "error": failure,
                "input_video": str(a.video.resolve()), "source_fps": fps,
                "frame_size": [width, height], "source_frame_count": total_frames, "frames_processed": count,
                "frames_with_detection": detected, "primary_detected_frames": primary_frames,
                "primary_lost_frames": lost_frames, "primary_id_changes": switches,
                "tracker_reassociations": getattr(primary, "reassociations", 0), "selection": a.selection,
                "unique_track_ids": sorted(seen_ids),
                "primary_coverage": primary_frames / count if count else 0.,
                "processing_seconds": processing_seconds,
                "processing_fps": count / processing_seconds if processing_seconds else 0.,
                "timing_seconds": stage_time, "ultralytics_version": ultralytics.__version__,
                "mode": a.mode, "parts": a.parts, "mask_format": a.mask_format,
                "first_contact_enabled": a.first_contact,
                "first_contact_status": contact.status if contact else "disabled",
                "notes": ["速度为 pixel/s，未标定；包含相机运动。", "倾角相对图像 +x 轴顺时针，模 180 度；不判定前后方向。",
                          "track_id 是主目标逻辑 ID；source_track_id 保存底层 ID，tracker_reassociations 记录接回次数。",
                          "primary_id_changes 无真值时不是 ID-switch 精度指标。frames_with_detection 仅统计被选中目标。",
                          "原始 Mask 不平滑；平滑值在 frames.jsonl 的 smoothed 中。",
                          "源时间戳优先使用 POS_MSEC，无有效值时按 FPS 递增；覆盖视频按源标称 FPS 编码且不含音频。",
                          "processing_fps 含解码、推理、特征和导出，不含模型加载、预览暂停。"]})
        if a.preview:
            cv2.destroyAllWindows()
    print(f"完成：{count} 帧，{count / processing_seconds if processing_seconds else 0.:.1f} 帧/秒；{output}", flush=True)
    return output


def main(argv=None):
    try:
        run(arguments(argv))
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Demo13：{exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
