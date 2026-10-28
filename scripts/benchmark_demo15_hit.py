"""Paired shortcut timing on identical coarse evidence, with fresh precision."""
import argparse
import csv
import gc
import json
from pathlib import Path
from statistics import median

from demo15.core import ContactConfig,MODES
from demo15.extraction import load_cache,save_json,digest
from demo15.postprocess import analyze_modes
from run_demo15 import cache_for,resolve,ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extraction',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--videos',nargs='+',default=[f'training_video/video{n}.mp4' for n in range(18,24)])
    p.add_argument('--device',default='mps')
    p.add_argument('--miss-repeats',type=int,default=3,help='video19 脱靶样例的配对重复次数')
    args=p.parse_args()
    if args.miss_repeats<1:p.error('重复次数必须为正')
    out=resolve(args.output)
    if out.exists():raise ValueError('耗时比较输出目录必须不存在')
    out.mkdir(parents=True)
    config=ContactConfig(**json.loads((ROOT/'configs/demo15_contact.json').read_text()))
    records=[];summary=[]
    for source in args.videos:
        video=resolve(source);cache=cache_for(args.extraction,video.stem)
        if cache is None:raise ValueError(f'缺少基础证据：{video}')
        rows,meta=load_cache(cache,video)
        if not meta['complete']:raise ValueError('耗时验收需要完整视频')
        repeats=args.miss_repeats if video.stem=='video19' else 1
        for repeat in range(repeats):
            # Alternating order limits warm-up/order bias. Every call constructs
            # a new lazy refiner: no persisted or previous precision is reused.
            for enabled in ((False,True) if repeat%2==0 else (True,False)):
                gc.collect()
                outputs,_,timing,_=analyze_modes(video,rows,config,MODES,cache,ROOT,args.device,enabled)
                event=outputs['fusion'].get('event') or {}
                record=dict(video=video.name,repeat=repeat,shortcut_enabled=enabled,**{k:v for k,v in timing.items() if k!='shortcut_enabled'},
                            fresh_base_seconds=meta['elapsed_seconds'],
                            reconstructed_analysis_seconds=meta['elapsed_seconds']+timing['postprocess_seconds'],
                            target_hit=outputs['fusion']['target_hit'],point_precision=outputs['fusion']['point_precision'],
                            frame_id=event.get('frame_id'),point_px=event.get('point_px'))
                records.append(record)
                save_json(out/'measurements.json',records)
                print(video.name,repeat,'on' if enabled else 'off',f'{timing["postprocess_seconds"]:.3f}s',
                      'refine',timing['refinement_calls'],'rings',timing['ring_calls'],flush=True)
        own=[r for r in records if r['video']==video.name]
        off=[r for r in own if not r['shortcut_enabled']];on=[r for r in own if r['shortcut_enabled']]
        before=median(r['postprocess_seconds'] for r in off);after=median(r['postprocess_seconds'] for r in on)
        summary.append(dict(video=video.name,repeats=repeats,base_seconds=meta['elapsed_seconds'],
                            off_postprocess_seconds=before,on_postprocess_seconds=after,saved_seconds=before-after,
                            reconstructed_total_off=meta['elapsed_seconds']+before,
                            reconstructed_total_on=meta['elapsed_seconds']+after,
                            reconstructed_total_saved_percent=100*(before-after)/(meta['elapsed_seconds']+before),
                            on_refinement_calls=on[0]['refinement_calls'],on_ring_calls=on[0]['ring_calls'],
                            target_hit=on[0]['target_hit']))
    save_json(out/'summary.json',dict(summary=summary,config=config.to_dict(),
              method='identical full-video coarse evidence; fresh precision per variant; median paired repetitions',
              source_sha256={f.name:digest(f) for f in (ROOT/'scripts/demo15').glob('*.py')}))
    with (out/'measurements.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# 落靶分类与脱靶短路耗时对比','',
           '每组使用同一段完整视频的同一份 v5 基础证据；每次新建精修器，不复用任何精修结果。跨模式仅共享该次运行内相同帧的工作。',
           'video19 配对重复3次（或命令指定次数），交换开关顺序，报告中位数；其他视频各做一次对照。',
           '实测时间为分类、精修和靶环复核。总分析时间用一次真实完整基础提取耗时加实测后处理耗时重建，排除预览与视频编码；不是两次全模型端到端计时。','',
           '| 视频 | 分类(fusion) | 关闭后处理(s) | 开启后处理(s) | 节省(s) | 开启精修次数(各模式合计) | 开启靶环次数(各模式合计) | 重建总耗时节省 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["video"]} | {r["target_hit"]} | {r["off_postprocess_seconds"]:.3f} | {r["on_postprocess_seconds"]:.3f} | {r["saved_seconds"]:.3f} | {r["on_refinement_calls"]} | {r["on_ring_calls"]} | {r["reconstructed_total_saved_percent"]:.2f}% |')
    lines+=['','主模式落靶的视频，其余对照模式仍可能判为脱靶并短路；精修或靶环调用数相同时，小幅正负耗时差属于测量波动，不能作为稳定提速结论。',
            '基础人体跟踪、基础鞋部分割和首次触地确认仍执行。脱靶粗坐标仅用于诊断，不声称为精确坐标验收。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
