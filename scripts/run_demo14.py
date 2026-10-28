#!/usr/bin/env python3
"""Sequential preview / reproducible headless export of deformable target rings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import cv2
from demo14.tracker import Config, TargetTracker
from demo14.render import annotate

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT/p


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Demo14：全自动变形靶环与靶心追踪（像素坐标）')
    p.add_argument('--video',default='training_video/video6.mp4')
    p.add_argument('--display-scale',type=float,default=.75)
    p.add_argument('--ring-count',type=int,default=10)
    p.add_argument('--no-preview',action='store_true')
    p.add_argument('--output-video')
    p.add_argument('--output-jsonl')
    p.add_argument('--max-frames',type=int,default=0,help='调试用；0 处理全片')
    args = p.parse_args(argv)
    if args.display_scale<=0 or args.ring_count<2 or args.max_frames<0:
        p.error('缩放必须为正，环数至少 2，最大帧数非负')
    paths = [resolve(v).resolve() for v in (args.video,args.output_video,args.output_jsonl) if v]
    if len(set(paths))!=len(paths):
        p.error('输入和各输出路径必须互不相同')
    return args


def run(args):
    source = resolve(args.video)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f'无法打开视频：{source}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not 0<fps<1000:
        fps = 30.
    width,height = int(capture.get(3)),int(capture.get(4))
    tracker = TargetTracker(Config(ring_count=args.ring_count))
    writer = None
    stream = None
    window = 'Demo14 - Deformable target calibration'
    count = 0; center_count = 0; fraction_sum = 0.; elapsed = 0.
    paused = False; advance = True; overlay = True
    frame = None; result = None; speed = 0.
    try:
        if args.output_video:
            path = resolve(args.output_video); path.parent.mkdir(parents=True,exist_ok=True)
            writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
            if not writer.isOpened():
                raise RuntimeError(f'无法写入视频：{path}')
        if args.output_jsonl:
            path = resolve(args.output_jsonl); path.parent.mkdir(parents=True,exist_ok=True)
            stream = path.open('w',encoding='utf-8')
        while True:
            if advance:
                ok,frame = capture.read()
                if not ok:
                    break
                start = time.perf_counter()
                result = tracker.update(frame,count,count/fps)
                dt = time.perf_counter()-start
                elapsed += dt; speed = 1/max(dt,1e-9)
                if stream:
                    stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                if writer:
                    writer.write(annotate(frame,result,speed))
                center_count += result['center']['source']=='observed'
                fraction_sum += sum(r['observed_fraction'] for r in result['rings'])/max(args.ring_count-2,1)
                count += 1
                if count%100==0:
                    print(f'{source.name}: {count} 帧，处理 {count/max(elapsed,1e-9):.1f} FPS',flush=True)
                if args.max_frames and count>=args.max_frames:
                    break
            if args.no_preview:
                continue
            canvas = annotate(frame,result,speed,overlay)
            if args.display_scale!=1:
                canvas = cv2.resize(canvas,None,fx=args.display_scale,fy=args.display_scale)
            cv2.imshow(window,canvas)
            key = cv2.waitKey(20 if paused else 1)&255
            if key in (27,ord('q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                break
            if key==ord(' '): paused = not paused
            if key==ord('o'): overlay = not overlay
            if key==ord('n'): paused = True
            if key==ord('r'):
                # Restart outputs too: one JSON line / output frame per source frame.
                capture.set(cv2.CAP_PROP_POS_FRAMES,0); tracker.reset()
                count=0; center_count=0; fraction_sum=0.; elapsed=0.
                if stream: stream.seek(0); stream.truncate()
                if writer:
                    writer.release()
                    writer = cv2.VideoWriter(str(resolve(args.output_video)),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
                    if not writer.isOpened(): raise RuntimeError('重启输出视频失败')
                advance=True
                continue
            advance = not paused or key==ord('n')
    finally:
        capture.release()
        if writer: writer.release()
        if stream: stream.close()
        if not args.no_preview: cv2.destroyAllWindows()
    summary = dict(video=source.name,frames=count,observed_center_fraction=center_count/max(count,1),
                   observed_ring_sample_fraction=fraction_sum/max(count,1),processing_fps=count/max(elapsed,1e-9))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    try:
        run(parse_args())
    except (ValueError,RuntimeError,OSError) as exc:
        raise SystemExit(str(exc))
