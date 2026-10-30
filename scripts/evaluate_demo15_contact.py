"""Compare complete contact runs and export observation-only review evidence.

No frame/coordinate reference is supplied to inference by this reporting tool.
An unlabeled video stays unlabeled; rendered crops are for human inspection.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from demo15.extraction import save_json
from demo15.render import annotate, annotated_depth, read_frame


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def review(run, output, baselines):
    output.mkdir(parents=True, exist_ok=False)
    records=[]
    for folder in sorted(run.iterdir()):
        result_path=folder/'fusion_result.json'
        if not result_path.exists():continue
        result=read_json(result_path)
        cache=folder/'extraction'
        if not cache.exists():cache=Path(read_json(folder/'cache_source.json')['path'])
        meta=read_json(cache/'metadata.json')
        if not meta['complete']:raise ValueError(f'Incomplete video: {folder.name}')
        rows=[json.loads(line) for line in (cache/'frames.jsonl').read_text().splitlines()]
        video=Path(meta['video']);event=result.get('event')
        old_path=next((root/folder.name/'fusion_result.json' for root in baselines
                       if (root/folder.name/'fusion_result.json').exists()),None)
        old=read_json(old_path) if old_path else None
        previous=old.get('event') if old else None
        timing=read_json(folder/'processing_timing.json')
        out=output/folder.name;out.mkdir()
        recovered=[o['frame_id'] for o in rows if o.get('shoe_recovery',{}).get('recovered_shoes',0)]
        trusted_recovery=sum(f.get('observation_source')=='recovered_rgb' and
            f.get('association',{}).get('source')=='bracketed_rgb_recovery' for o in rows for f in o['feet'])
        record=dict(video=folder.name,status=result['status'],target_hit=result['target_hit'],
                    point_precision=result['point_precision'],frame_id=event['frame_id'] if event else None,
                    point_px=event['point_px'] if event else None,
                    confirmation_frame=event['confirmation_frame'] if event else None,
                    baseline_status=old['status'] if old else None,
                    baseline_frame=previous['frame_id'] if previous else None,
                    baseline_point=previous['point_px'] if previous else None,
                    evaluation=result['evaluation'],frames=meta['frames'],
                    recovery=meta.get('observation_recovery'),recovered_frames=recovered,
                    bracketed_recovered_shoes=trusted_recovery,
                    recovery_seconds=meta['stage_seconds'].get('observation_recovery',0.),
                    extraction_seconds=meta['elapsed_seconds'],postprocess_seconds=timing['postprocess_seconds'],
                    refinement_calls=timing['refinement_calls'],ring_calls=timing['ring_calls'],
                    candidate_diagnostics=result.get('candidate_diagnostics',[]))
        if event:
            first,confirmed=event['frame_id'],event['confirmation_frame']
            depth_result=read_json(folder/'depth_result.json') if (folder/'depth_result.json').exists() else None
            cv2.imwrite(str(out/'contact_frame.png'),annotate(read_frame(video,first),rows[first],result,depth_result=depth_result))
            cv2.imwrite(str(out/'confirmed_frame.png'),annotate(read_frame(video,confirmed),rows[confirmed],result,depth_result=depth_result))
            depth_path=cache/'depth_cache.npz'
            if depth_path.exists():
                with np.load(depth_path) as archive:
                    cv2.imwrite(str(out/'confirmed_depth.png'),annotated_depth(archive['depth'][confirmed],rows[confirmed],result,depth_result=depth_result))
            # All crops use the same fixed image region around the predicted
            # contact. No reference frame or marked coordinate chooses this ROI.
            x,y=event['point_px'];w,h=meta['image_size']
            x1=max(0,round(x)-140);x2=min(w,round(x)+140)
            y1=max(0,round(y)-145);y2=min(h,round(y)+65)
            tiles=[]
            for n in range(max(0,first-4),min(len(rows),first+5)):
                frame=read_frame(video,n);tile=frame[y1:y2,x1:x2].copy()
                for foot in rows[n]['feet']:
                    boundary=foot.get('sole_boundary',[])
                    if boundary:
                        pts=np.asarray(boundary,np.int32)-[x1,y1]
                        color=(0,180,255) if foot.get('observation_source')=='recovered_rgb' else (220,60,220)
                        cv2.polylines(tile,[pts],False,color,1,cv2.LINE_AA)
                tile=cv2.resize(tile,(420,315))
                cv2.rectangle(tile,(0,0),(420,26),(15,15,15),-1)
                cv2.putText(tile,f'{folder.name} frame {n}',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                tiles.append(tile)
            while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(out/'contact_sequence.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
            record['contact_observations']=[dict(frame_id=o['frame_id'],feet=o['feet'],shoe_recovery=o.get('shoe_recovery'))
                for o in rows[max(0,first-9):min(len(rows),confirmed+1)]]
        save_json(out/'diagnostics.json',record);records.append(record)
    save_json(output/'summary.json',records)
    lines=['# Demo15 无人工标注触地优化实测','',
           '推理使用原视频，不使用人工帧号、坐标或落地区域。已有标注仅在结果生成后评估；无标注视频不计算正式像素准确率。','',
           '| 视频 | 原状态/帧 | 新状态/帧 | 标注帧差 | 位置误差px | 分类 |',
           '|---|---|---|---:|---:|---|']
    for r in records:
        e=r['evaluation'];error=e.get('position_error_px')
        position=f'{error:.2f}' if error is not None else '无标注'
        if r['point_precision']=='coarse' and error is not None:position+='（粗坐标诊断）'
        lines.append(f'| {r["video"]} | {r["baseline_status"]}/{r["baseline_frame"]} | {r["status"]}/{r["frame_id"]} | {e.get("frame_error", "—")} | {position} | {r["target_hit"]} |')
    lines+=['','## 补检与耗时','',
            '| 视频 | 补检调用/恢复观测/可信接入 | 补检秒 | 基础提取秒 | 后处理秒 | 精确精修/完整环调用 |',
            '|---|---:|---:|---:|---:|---:|']
    for r in records:
        recovery=r['recovery'] or {}
        lines.append(f'| {r["video"]} | {recovery.get("calls",0)}/{recovery.get("recovered_shoes",0)}/{r["bracketed_recovered_shoes"]} | {r["recovery_seconds"]:.3f} | {r["extraction_seconds"]:.3f} | {r["postprocess_seconds"]:.3f} | {r["refinement_calls"]}/{r["ring_calls"]} |')
    lines+=['','补检是首次触地的基础观测，明确脱靶之后的精确精修与完整环复核单列。补检秒包含基础观测检查及实际补检；调用数为额外RGB分割次数。耗时来自各段完整运行，未声称为受控整机加速对照。',
            '', '每段目录包含原图鞋底逐帧放大、触地帧及确认帧主/深度视图、候选淘汰原因和轨迹关联诊断。video33 无人工真值，需逐帧复核首次接触，不宣称10像素正式通过。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,nargs='*',default=[])
    args=parser.parse_args();review(args.run,args.output,args.baseline)
