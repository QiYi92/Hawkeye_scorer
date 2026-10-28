"""Demo15 CLI: inference, development-only tuning, held-out evaluation and review."""
import argparse
from datetime import datetime
from pathlib import Path
import csv
import json
import itertools
import sys
import time
import cv2
from demo15.core import ContactConfig, MODES, estimate
from demo15.extraction import extract, load_cache, save_json, digest, StaleEvidenceError
from demo15.references import load_reference, evaluate
from demo15.render import export, preview, contact_grid
from demo15.postprocess import analyze_modes

ROOT=Path(__file__).resolve().parents[1]
DEVELOPMENT = tuple(f'video{n}' for n in range(18,24))


def resolve(path):
    path=Path(path).expanduser()
    return path if path.is_absolute() else ROOT/path


def cache_for(root,stem):
    base=resolve(root)/stem
    cache=base/'extraction'
    if (cache/'metadata.json').exists():return cache
    source=base/'cache_source.json'
    if source.exists():
        actual=Path(json.loads(source.read_text())['path'])
        if (actual/'metadata.json').exists():return actual
    return None


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Demo15：深度驱动首次脚部触地识别与用户标记评估')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--video',default=None)
    group.add_argument('--video-dir')
    group.add_argument('--videos',nargs='+',help='指定多段视频，逐段完整推理')
    p.add_argument('--mark-dir',default='training_video_mark')
    p.add_argument('--output')
    p.add_argument('--reuse-extraction',help='既有运行的输出根目录；仅复用校验通过的模型证据')
    p.add_argument('--device',default='mps')
    p.add_argument('--mode',choices=MODES,default='fusion')
    p.add_argument('--no-depth',action='store_true',help='明确关闭深度：只运行视觉模式')
    p.add_argument('--no-preview',action='store_true')
    p.add_argument('--no-depth-view',action='store_true',help='隐藏独立深度窗口，不影响深度推理和落点标记')
    p.add_argument('--display-scale',type=float,default=.75)
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--tune',action='store_true',help='仅用 video18–video23 调试并冻结参数')
    p.add_argument('--config',help='加载已冻结的 ContactConfig JSON')
    p.add_argument('--no-miss-shortcut',action='store_true',help='关闭脱靶计算短路，用于耗时对照')
    p.add_argument('--no-export',action='store_true',help='只输出数据与评估，用于参数实验')
    args=p.parse_args(argv)
    if args.no_depth:args.mode='visual'
    if args.max_frames<0 or args.display_scale<=0:p.error('最大帧数不可负，显示缩放必须为正')
    if args.tune and (not (args.video_dir or args.videos) or args.max_frames or args.config):p.error('调参要求完整目录模式，且不能同时指定 config')
    if args.tune and args.no_depth:p.error('深度参数调试不能与关闭深度同时使用')
    return args


def tune(development):
    trials=[];best=None
    for descent,motion,support in itertools.product((8.,12.,16.),(6.,8.,10.),(.10,.15)):
        cfg=ContactConfig(near=.04,separation=.06,support_seconds=support,
                          descent_px_720=descent,motion_tolerance_720=motion)
        scores=[]
        for rows,ref in development:
            result=estimate(rows,'fusion',cfg)
            scores.append({**evaluate(result,ref,rows),'accepted':result['status']=='confirmed_estimate'})
        success=sum(s['joint_pass'] for s in scores)
        time_error=sum(s['time_error_seconds'] if s['accepted'] and s['time_error_seconds'] is not None else 100 for s in scores)
        position_error=sum(s['position_error_px'] if s['accepted'] and s['position_error_px'] is not None else 1000 for s in scores)
        # Prefer the default on a complete tie, rather than arbitrarily widening gates.
        tie=abs(descent-12)+abs(motion-10)+abs(support-.10)
        objective=(-success,time_error,position_error,tie)
        trials.append(dict(config=cfg.to_dict(),joint_success=success,time_error_sum=time_error,position_error_sum=position_error))
        if best is None or objective<best[0]:best=(objective,cfg)
    return best[1],trials


def report(root,records,config):
    totals={}
    for mode in MODES:
        rows=[r for r in records if r['mode']==mode and r.get('labeled')]
        totals[mode]=dict(labeled_videos=len(rows),joint_success=sum(r.get('joint_pass',False) for r in rows),
                          time_success=sum(r.get('time_pass',False) for r in rows),position_success=sum(r.get('position_pass',False) for r in rows),
                          unconfirmed=sum(r['status']!='confirmed_estimate' for r in rows),
                          coordinate_videos=sum(r.get('coordinate_acceptance_applicable',True) for r in rows),
                          coarse_diagnostics=sum(not r.get('coordinate_acceptance_applicable',True) for r in rows),
                          joint_rate=(sum(r.get('joint_pass',False) for r in rows)/sum(r.get('coordinate_acceptance_applicable',True) for r in rows)
                                      if any(r.get('coordinate_acceptance_applicable',True) for r in rows) else None))
    save_json(root/'evaluation.json',dict(config=config.to_dict(),summary=totals,videos=records,thresholds=dict(frames=2,position_px=10.)))
    fields=['video','split','mode','status','labeled','target_hit','point_precision','shortcut_reason','coordinate_evaluation','coordinate_acceptance_applicable','predicted_frame','predicted_time','predicted_point','time_error_ms','frame_error','position_error_px','raw_position_error_px','registration','reference_frame_sole_error_px','time_pass','position_pass','joint_pass','failure_reason']
    with (root/'evaluation.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    lines=['# Demo15 实测报告','','时间阈值：±2 帧；位置阈值：10 原图像素；缺失或待复核结果计为失败。','',
           '| 模式 | 标记视频 | 时间达标 | 精确位置达标/适用数 | 精确联合达标/适用数 | 未确认/待复核 |','|---|---:|---:|---:|---:|---:|']
    for mode,s in totals.items():lines.append(f'| {mode} | {s["labeled_videos"]} | {s["time_success"]} | {s["position_success"]}/{s["coordinate_videos"]} | {s["joint_success"]}/{s["coordinate_videos"]} | {s["unconfirmed"]} |')
    lines+=['','| 视频 | 模式 | 状态 | 帧差 | 时间误差(ms) | 位置误差(px) | 联合达标 |','|---|---|---|---:|---:|---:|---|']
    for r in records:
        def number(v):return f'{v:.2f}' if v is not None else '—'
        if r.get('labeled'):lines.append(f'| {r["video"]} | {r["mode"]} | {r["status"]} | {r.get("frame_error")} | {number(r.get("time_error_ms"))} | {number(r.get("position_error_px"))} | {r["joint_pass"]} |')
    lines+=['','## 落靶分类','', '| 视频 | 模式 | 识别情况 | 坐标精度 | 短路原因 |','|---|---|---|---|---|']
    for r in records:
        lines.append(f'| {r["video"]} | {r["mode"]} | {r.get("target_hit","unknown")} | {r.get("point_precision","coarse")} | {r.get("shortcut_reason") or "—"} |')
    lines+=['','## 参考帧鞋底定位（仅诊断）','', '| 视频 | 最近鞋底点误差(px) |','|---|---:|']
    for r in records:
        if r.get('labeled') and r['mode'] in (('visual',) if not any(x['mode']=='fusion' for x in records) else ('fusion',)):
            value=r.get('reference_frame_sole_error_px')
            lines.append(f'| {r["video"]} | {value:.2f} |' if value is not None else f'| {r["video"]} | 无可见鞋部结果 |')
    lines+=['','## 分组验收','', '| 模式 | 优化回归集联合通过 | 其他标注回归集联合通过 |','|---|---:|---:|']
    for mode in MODES:
        dev=[r for r in records if r['mode']==mode and r.get('labeled') and r['split']=='development' and r.get('coordinate_acceptance_applicable',True)]
        held=[r for r in records if r['mode']==mode and r.get('labeled') and r['split']=='regression' and r.get('coordinate_acceptance_applicable',True)]
        lines.append(f'| {mode} | {sum(r["joint_pass"] for r in dev)}/{len(dev)} | {sum(r["joint_pass"] for r in held)}/{len(held)} |')
    lines+=['','## 全部视频处理','', '| 视频 | 完整帧数 | 证据提取秒数 | fusion 状态/原因 |','|---|---:|---:|---|']
    for r in records:
        if r['mode']=='fusion':lines.append(f'| {r["video"]} | {r.get("decoded_frames", "—")} | {r.get("extraction_seconds", 0):.1f} | {r["status"]} / {r["failure_reason"] or "已输出估计"} |')
    lines+=['','video9：保留原标记帧 192，按原截图顺序解码匹配至 193。video8 按用户参考纳入，不因视觉歧义排除。',
            'video18–video23 为本次优化开发/回归集；其他标注视频为额外回归集（含历史调参视频），不宣称为独立泛化测试。未标记视频不计算准确率。',
            '脱靶短路保留的粗坐标只作诊断，不参加精确落点联合验收；落靶分类结果与坐标验收分开报告。',
            '脚底定位诊断不代替实际事件的联合成绩。背景配准不可用时位置验收失败；原始像素误差仍保留。',
            '深度残差是模型预测差异，不是离地高度或物理接触测量。草地与地面范围采用外观启发式；严重遮挡、机位移动和鞋部分割误差可能导致失败。',
            '时间通过要求±2帧，且必须同时通过位置验收；首次接触含义还需检查复核图。',
            '视频输出使用源标称 FPS 编码，无音轨；源时间戳以 JSONL 为准。',
            '测试结果由本次独立测试日志提供；本报告不预先声称测试通过。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(args):
    output=resolve(args.output) if args.output else ROOT/'output'/f'demo15_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if output.exists() and any(output.iterdir()):raise ValueError('输出必须是新目录或空目录，避免覆盖历史结果')
    output.mkdir(parents=True,exist_ok=True)
    if args.video_dir:
        videos=sorted(resolve(args.video_dir).glob('*.mp4'),key=lambda p:int(''.join(filter(str.isdigit,p.stem)) or 0))
    elif args.videos:videos=[resolve(p) for p in args.videos]
    else:videos=[resolve(args.video or 'training_video/video6.mp4')]
    if not videos:raise ValueError('没有输入视频')
    if args.tune:
        dev=[p for p in videos if p.stem in DEVELOPMENT]
        if len(dev)!=len(DEVELOPMENT):raise ValueError('调试输入必须包含 video18–video23')
        videos=dev+[p for p in videos if p not in dev]
    config_path=resolve(args.config) if args.config else (ROOT/'configs/demo15_contact.json' if not args.tune else None)
    config=ContactConfig(**json.loads(config_path.read_text())) if config_path and config_path.exists() else ContactConfig()
    if args.config and not config_path.exists():raise ValueError(f'冻结参数文件不存在：{config_path}')
    manifest=dict(arguments=vars(args),initial_contact_config=config.to_dict(),config_source=str(config_path) if config_path and config_path.exists() else 'built_in_defaults',code_sha256={p.name:digest(p) for p in [Path(__file__),*sorted((ROOT/'scripts/demo15').glob('*.py'))]},reference_access='evaluation_only',development_videos=[v+'.mp4' for v in DEVELOPMENT])
    save_json(output/'run.json',manifest)
    records=[];development=[];actual_caches={};actual_metadata={}
    for video in videos:
        target=output/video.stem;target.mkdir()
        cache=cache_for(args.reuse_extraction,video.stem) if args.reuse_extraction else None
        if cache and (cache/'metadata.json').exists():
            try:
                rows,meta=load_cache(cache,video)
                if not args.max_frames and not meta['complete']:
                    raise StaleEvidenceError('短测缓存不能代替完整视频，重新提取')
            except StaleEvidenceError as error:
                print(f'{video.name}：{error}；重新提取',flush=True)
                cache=target/'extraction'
                rows,meta=extract(video,cache,ROOT,args.device,not args.no_depth,args.max_frames)
            if not args.no_depth and not meta['with_depth']:raise ValueError('无深度缓存不能用于深度实验')
            save_json(target/'cache_source.json',dict(path=str(cache.resolve())))
        else:
            cache=target/'extraction';rows,meta=extract(video,cache,ROOT,args.device,not args.no_depth,args.max_frames)
        actual_caches[video.stem]=cache
        actual_metadata[video.stem]=meta
        if args.max_frames and len(rows)>args.max_frames:
            rows=rows[:args.max_frames]
            meta={**meta,'frames':len(rows),'complete':len(rows)==meta['expected_frames']}
        if args.no_depth:
            for row in rows:
                for foot in row['feet']:
                    foot['depth_residual']=None
        # Only now, after full inference, may annotations enter the evaluation path.
        ref=load_reference(video,resolve(args.mark_dir))
        if ref:save_json(target/'reference_check.json',{k:v for k,v in ref.items() if k!='timeline'})
        if args.tune and video.stem in DEVELOPMENT:
            if ref is None:raise ValueError('调试视频缺少参考')
            development.append((rows,ref))
            if len(development)==len(DEVELOPMENT):
                config,trials=tune(development);save_json(output/'tuning.json',dict(trials=trials,selected=config.to_dict(),scope='video18_video23_only'))
                save_json(output/'contact_config.json',config.to_dict())
                # Re-evaluate the first development video using frozen parameters.
                records=[]
                for v,(devrows,devref) in zip(videos[:len(DEVELOPMENT)],development):
                    devcache=actual_caches[v.stem]
                    devmeta=actual_metadata[v.stem]
                    records+=process(v,output/v.stem,devrows,devmeta,devref,config,args,devcache)
                report(output,records,config)
            continue
        records+=process(video,target,rows,meta,ref,config,args,cache)
        report(output,records,config)
    save_json(output/'contact_config.json',config.to_dict())
    print(f'完成：{output}/report.md',flush=True)
    return output


def process(video,target,rows,meta,ref,config,args,cache):
    records=[];results={}
    modes=('visual',) if args.no_depth else MODES
    results,display_rows,timing,refinements=analyze_modes(video,rows,config,modes,cache,ROOT,args.device,
                                                        shortcut=not args.no_miss_shortcut)
    save_json(target/'processing_timing.json',dict(**timing,extraction_seconds=meta['elapsed_seconds'],
              analysis_total_seconds=meta['elapsed_seconds']+timing['postprocess_seconds']))
    save_json(target/'refinement.json',refinements)
    for mode in modes:
        result=results[mode]
        result['estimation_seconds']=timing['postprocess_seconds']
        result.update(frame_index_base=0,coordinate_system='original_image_pixels_x_right_y_down',is_estimate=True)
        result['diagnostics']=dict(tracked_frames=sum(o['track_status']=='tracked' for o in rows),
                                   foot_frames=sum(bool(o['feet']) for o in rows),
                                   camera_stable_frames=sum(o['camera_stable'] for o in rows),
                                   camera_moved_frames=sum(o.get('camera_moved',False) for o in rows),
                                   depth_ground_frames=sum(any(f.get('depth_residual') is not None for f in o['feet']) for o in rows))
        if result['event'] is None:
            d=result['diagnostics']
            if d['foot_frames']==0:result['failure_reason']='no_observed_shoes'
            elif d['camera_stable_frames']==0:result['failure_reason']='unreliable_background_registration'
            elif mode=='depth' and d['depth_ground_frames']==0:result['failure_reason']='no_reliable_depth_ground_reference'
        metrics=evaluate(result,ref,rows)
        metrics['coordinate_evaluation']='precise' if result['point_precision']=='refined' else 'coarse_diagnostic'
        if result['point_precision']!='refined':
            metrics.update(position_pass=False,joint_pass=False,coordinate_acceptance_applicable=False)
        else:metrics['coordinate_acceptance_applicable']=True
        if not meta['complete']:
            metrics.update(joint_pass=False,time_pass=False,position_pass=False,partial_run=True)
        result['evaluation']=metrics;result['configuration']=config.to_dict()
        save_json(target/f'{mode}_result.json',result);results[mode]=result
        event=result['event']
        records.append(dict(video=video.name,split='development' if video.stem in DEVELOPMENT else ('regression' if ref else 'unlabeled'),mode=mode,status=result['status'],failure_reason=result['failure_reason'],
                            target_hit=result['target_hit'],point_precision=result['point_precision'],shortcut_reason=result['shortcut_reason'],
                            decoded_frames=meta['frames'],extraction_seconds=meta['elapsed_seconds'],estimation_seconds=result['estimation_seconds'],diagnostics=result['diagnostics'],
                            predicted_frame=event['frame_id'] if event else None,predicted_time=event['timestamp'] if event else None,predicted_point=event['point_px'] if event else None,**metrics))
    chosen=results[args.mode]
    depth_path=None if args.no_depth else cache/'depth_cache.npz'
    depth_result=results.get('depth')
    contact_grid(video,target,display_rows,chosen,ref)
    if not args.no_export:export(video,target,display_rows,meta,chosen,ref,depth_path,depth_result=depth_result)
    else:save_json(target/'first_contact.json',chosen)
    if not args.no_preview and not args.video_dir and not args.videos:preview(video,target,display_rows,chosen,ref,depth_path,args.display_scale,show_depth=not args.no_depth_view,depth_result=depth_result)
    return records


def main(argv=None):
    try:run(parse_args(argv))
    except (ValueError,RuntimeError,OSError) as error:
        print(f'Demo15 失败：{error}',file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
