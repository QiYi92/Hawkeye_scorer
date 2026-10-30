"""Read-only evaluation of a completed Demo15 run; writes a new review directory."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from demo15.extraction import save_json
from demo15.render import annotate, annotated_depth, read_frame


def review(run, destination, baseline=None):
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for folder in sorted(run.iterdir()):
        if not (folder/'fusion_result.json').exists():
            continue
        cache = folder/'extraction'
        if not cache.exists():
            cache = Path(json.loads((folder/'cache_source.json').read_text())['path'])
        meta = json.loads((cache/'metadata.json').read_text())
        rows = [json.loads(line) for line in (cache/'frames.jsonl').read_text().splitlines()]
        result = json.loads((folder/'fusion_result.json').read_text())
        depth_result = json.loads((folder/'depth_result.json').read_text())
        old_path = baseline/folder.name/'fusion_result.json' if baseline else None
        old = json.loads(old_path.read_text()) if old_path and old_path.exists() else None
        video = Path(meta['video'])
        out = destination/folder.name; out.mkdir()
        event = result.get('event')
        contact = event['frame_id'] if event else len(rows)//2
        frame_ids = sorted({len(rows)//5, contact, min(len(rows)-1, len(rows)*4//5)})
        counts = dict(fitted=0, unknown=0, image_and_geometry=0, registered_history=0)
        unknown_intervals = []
        for row in rows:
            model = row['target'].get('ring_model', {})
            counts[model.get('status', 'unknown')] += 1
            if model.get('source') in counts:
                counts[model['source']] += 1
            if model.get('status') != 'fitted':
                reason = model.get('reason', 'insufficient_ring_evidence')
                if (unknown_intervals and unknown_intervals[-1]['end_frame'] == row['frame_id']-1
                        and unknown_intervals[-1]['reason'] == reason):
                    unknown_intervals[-1]['end_frame'] = row['frame_id']
                else:
                    unknown_intervals.append(dict(start_frame=row['frame_id'], end_frame=row['frame_id'], reason=reason))
        summaries = []
        for n in frame_ids:
            row = rows[n]
            model = row['target'].get('ring_model', {})
            if event and n == contact:
                refined = event.get('target_evidence', {}).get('ring_model', {})
                if refined.get('status') == 'fitted':
                    row['target']['ring_model'] = model = refined
            frame = read_frame(video, n)
            before = frame.copy()
            for boundary in ('outer_boundary', 'bullseye_boundary'):
                pts = row['target'].get(boundary, [])
                for a, b in zip(pts, pts[1:]):
                    if a['source'] == b['source'] == 'observed' and a['point'] and b['point']:
                        cv2.line(before, tuple(map(round, a['point'])), tuple(map(round, b['point'])), (255, 220, 0), 1)
            after = frame.copy()
            if model.get('status') == 'fitted':
                for ring in reversed(model['rings']):
                    curve = np.rint(np.asarray(ring['points'])*256).astype(np.int32)
                    cv2.polylines(after, [curve], True,
                                  (0, 220, 255) if ring['ring_id'] == 1 else (255, 220, 0), 1, cv2.LINE_AA, shift=8)
                cv2.drawMarker(after, tuple(map(round, model['center'])), (0, 220, 255), cv2.MARKER_CROSS, 12, 1)
                keypoints = model.get('edge_keypoints') or [dict(point=p, source='fitted') for p in model['four_points']]
                for label, keypoint in zip(('L', 'T', 'R', 'B'), keypoints):
                    xy = tuple(map(round, keypoint['point']))
                    color = (0, 255, 0) if keypoint['source'] == 'observed' else (0, 150, 255)
                    cv2.circle(after, xy, 4, color, -1, cv2.LINE_AA)
                    cv2.putText(after, label, (xy[0]+6, xy[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
                outer = np.asarray(model['rings'][-1]['points'])
            else:
                outer = np.asarray(row.get('paper_outline', {}).get('points', []))
            h, w = frame.shape[:2]
            if len(outer):
                x1, y1 = np.maximum(0, np.floor(outer.min(axis=0)-25)).astype(int)
                x2, y2 = np.minimum([w, h], np.ceil(outer.max(axis=0)+25)).astype(int)
            else:
                x1, y1, x2, y2 = 0, h//2, w, h
            pair = np.hstack([before[y1:y2, x1:x2], after[y1:y2, x1:x2]])
            strip = np.zeros((38, pair.shape[1], 3), np.uint8)
            label = 'fitted 10 rings' if model.get('status') == 'fitted' else 'awaiting evidence'
            cv2.putText(strip, f'{folder.name} frame {n}: previous contours | {label}',
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(out/f'rings_before_after_{n:04d}.jpg'), np.vstack([strip, pair]))
            summaries.append(dict(frame_id=n, model=model))
            if n == contact:
                cv2.imwrite(str(out/'rings_contact.png'), annotate(frame, row, result, depth_result=depth_result))
                confirmed = min(len(rows)-1, event.get('confirmation_frame', n)) if event else n
                state_frame = frame if confirmed == n else read_frame(video, confirmed)
                state_row = rows[confirmed]
                cv2.imwrite(str(out/'rings_main.png'), annotate(state_frame, state_row, result, depth_result=depth_result))
                depth_path = cache/'depth_cache.npz'
                if depth_path.exists():
                    with np.load(depth_path) as archive:
                        cv2.imwrite(str(out/'rings_depth.png'), annotated_depth(archive['depth'][confirmed], state_row, result, depth_result=depth_result))
        timing = json.loads((folder/'processing_timing.json').read_text())
        metrics = result['evaluation']
        old_event = old.get('event') if old else None
        record = dict(video=folder.name, frames=len(rows), model_frames=counts,
                      unknown_intervals=unknown_intervals,
                      first_contact=contact if event else None, target_hit=result['target_hit'],
                      frame_error=metrics.get('frame_error'), position_error_px=metrics.get('position_error_px'),
                      time_pass=metrics.get('time_pass'), joint_pass=metrics.get('joint_pass'),
                      preview_seconds=meta['stage_seconds'].get('ring_preview'), full_ring_seconds=timing['ring_seconds'],
                      ring_calls=timing['ring_calls'], refinement_calls=timing['refinement_calls'],
                      contact_delta_vs_baseline=contact-old_event['frame_id'] if event and old_event else None,
                      point_delta_vs_baseline_px=(float(np.linalg.norm(np.asarray(event['point_px'])-old_event['point_px']))
                          if event and old_event and contact == old_event['frame_id'] else None),
                      selected_frames=summaries)
        save_json(out/'diagnostics.json', record)
        records.append(record)
    save_json(destination/'summary.json', records)
    lines = ['# Demo15 十环拟合实测', '',
             '完整曲线包含几何补全；拟合残差是模型对提取边缘的贴合程度，不等同于人工标注准确率。', '',
             '| 视频 | 拟合帧/总帧 | 触地帧 | 帧差 | 位置误差px | 落靶分类 | 预览拟合秒 | 完整环复核秒/次数 |',
             '|---|---:|---:|---:|---:|---|---:|---:|']
    for r in records:
        error = f'{r["position_error_px"]:.2f}' if r['position_error_px'] is not None else '无标注'
        lines.append(f'| {r["video"]} | {r["model_frames"]["fitted"]}/{r["frames"]} | {r["first_contact"]} | {r["frame_error"]} | {error} | {r["target_hit"]} | {r["preview_seconds"]:.3f} | {r["full_ring_seconds"]:.3f}/{r["ring_calls"]} |')
    lines += ['', '预览拟合耗时包含当前图像取证、拟合及跨帧模型投影；完整环复核为各模式合计。video19 脱靶粗坐标仅诊断，精确位置不参加验收。',
              '', '每段视频目录包含三个时刻的新旧绘图对照、主窗口及深度窗口截图、逐环观测比例和拟合误差。旧侧按旧绘制逻辑重绘同帧轮廓；首次触地结果另与历史运行比较。',
              '', '未知帧原因保存在提取缓存 frames.jsonl；未建立可靠模型时显示“靶环待确认”，不按规格凭空锁定。']
    lines += ['', '## 未建立模型的片段', '']
    for record in records:
        intervals = ', '.join(f'{i["start_frame"]}–{i["end_frame"]}: {i["reason"]}' for i in record['unknown_intervals'])
        lines.append(f'- {record["video"]}: {intervals or "无"}')
    (destination/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    review(args.run, args.output, args.baseline)
