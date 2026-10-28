"""Compare recorded baseline predictions with a complete Demo15 evaluation.

All annotations enter this reporting tool after inference. Historical results
are rescored using the same strict frame/pixel tolerances as the new run.
"""
import argparse
import csv
import json
from pathlib import Path

from demo15.extraction import digest, save_json
from demo15.references import load_reference, evaluate

ROOT=Path(__file__).resolve().parents[1]


def resolve(value):
    path=Path(value)
    return path if path.is_absolute() else ROOT/path


def compare(current,baseline_manifest,output,additional_current=None):
    current,output=resolve(current),resolve(output)
    baseline=json.loads(resolve(baseline_manifest).read_text())
    if output.exists():raise ValueError('比较输出目录必须不存在，避免覆盖历史报告')
    output.mkdir(parents=True)
    records=[]
    evaluation=json.loads((current/'evaluation.json').read_text())
    for extra in additional_current or []:
        other=json.loads((resolve(extra)/'evaluation.json').read_text())
        if other['config']!=evaluation['config']:
            raise ValueError('不同参数的评估不能合并为同一冻结配置成绩')
        evaluation['videos'].extend(other['videos'])
    seen=set()
    for row in evaluation['videos']:
        if row['mode']!='fusion' or not row.get('labeled'):continue
        video=row['video'];entry=baseline['videos'].get(video)
        if video in seen:raise ValueError(f'重复计入的视频：{video}')
        seen.add(video)
        if entry is None:raise ValueError(f'缺少基线：{video}')
        prediction=json.loads(resolve(entry['result']).read_text())
        cache=resolve(entry['extraction'])
        meta=json.loads((cache/'metadata.json').read_text())
        source=ROOT/'training_video'/video
        if not meta['complete'] or meta['video_sha256']!=digest(source):
            raise ValueError(f'基线不完整或输入不匹配：{video}')
        observations=[json.loads(line) for line in (cache/'frames.jsonl').read_text().splitlines()]
        if len(observations)!=meta['frames']:raise ValueError(f'基线帧数不一致：{video}')
        ref=load_reference(source,ROOT/'training_video_mark')
        old=evaluate(prediction,ref,observations)
        event=prediction.get('event') or {}
        records.append(dict(video=video,split=row['split'],reference_frame=ref['frame_id'],reference_point=ref['point_px'],
                            baseline_status=prediction['status'],baseline_frame=event.get('frame_id'),
                            baseline_frame_error=old['frame_error'],baseline_position_error_px=old['position_error_px'],
                            baseline_pass=old['joint_pass'],status=row['status'],predicted_frame=row['predicted_frame'],
                            predicted_point=row['predicted_point'],frame_error=row['frame_error'],
                            position_error_px=row['position_error_px'],joint_pass=row['joint_pass'],
                            baseline_result=entry['result'],annotation_sha256=digest(ref['annotation_path']),
                            complete=not row.get('partial_run',False)))
    save_json(output/'comparison.json',dict(thresholds=dict(frames=2,position_px=10),videos=records,
                                           current_run=str(current),additional_runs=[str(resolve(p)) for p in additional_current or []],
                                           baseline_manifest=str(resolve(baseline_manifest))))
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]) if records else ['video'])
        writer.writeheader();writer.writerows(records)
    lines=['# Demo15 优化前后实测','',
           '统一验收：帧差绝对值≤2，配准后坐标误差≤10原图像素；只接受完整视频的 confirmed_estimate。',
           'video18–video23 为优化开发/回归集；其他已有标注视频单独报告，不视为独立泛化成绩。','',
           '| 分组 | 优化前通过 | 优化后通过 |','|---|---:|---:|']
    for split,label in [('development','video18–video23'),('regression','其他标注回归')]:
        group=[r for r in records if r['split']==split]
        lines.append(f'| {label} | {sum(r["baseline_pass"] for r in group)}/{len(group)} | {sum(r["joint_pass"] for r in group)}/{len(group)} |')
    lines+=['','| 视频 | 标注帧 | 原帧差 | 原位置误差 | 新帧差 | 新位置误差 | 新状态 | 通过 |',
            '|---|---:|---:|---:|---:|---:|---|---|']
    def fmt(value):return '—' if value is None else f'{value:.2f}'
    for r in records:
        lines.append(f'| {r["video"]} | {r["reference_frame"]} | {fmt(r["baseline_frame_error"])} | {fmt(r["baseline_position_error_px"])} | {fmt(r["frame_error"])} | {fmt(r["position_error_px"])} | {r["status"]} | {r["joint_pass"]} |')
    lines+=['','逐帧放大图、叠加视频、模型/输入校验值和详细证据位于当前运行目录各视频子目录。',
            '未通过项保留在统计中；不会用参考帧附近的定位诊断替代实际预测结果。']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(output/'comparison.md')
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',required=True)
    parser.add_argument('--baseline-manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--additional-current',nargs='*',help='同一冻结配置下的其他回归运行目录')
    args=parser.parse_args()
    compare(args.current,args.baseline_manifest,args.output,args.additional_current)


if __name__=='__main__':main()
