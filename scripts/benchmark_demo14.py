#!/usr/bin/env python3
"""Full sequential video evaluation; optional independent workers per video."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from demo14.tracker import TargetTracker
from demo14.render import annotate

ROOT=Path(__file__).resolve().parents[1]
TRACKER_SHA256=hashlib.sha256((ROOT/'scripts/demo14/tracker.py').read_bytes()).hexdigest()


def evaluate(job):
    video,output,export_example=job
    cv2.setNumThreads(1)
    source=ROOT/'training_video'/f'video{video}.mp4'
    directory=Path(output)/source.stem;directory.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():raise RuntimeError(f'Cannot open {source}')
    fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    checkpoints={7,int(total*.3),int(total*.6),int(total*.9),total-1}
    writer=None;stream=None
    if export_example and video==6:
        writer=cv2.VideoWriter(str(directory.parent.parent/'video6_overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,
                               (int(cap.get(3)),int(cap.get(4))))
        if not writer.isOpened():raise RuntimeError('Cannot open example video writer')
        stream=(directory.parent.parent/'video6_points.jsonl').open('w',encoding='utf-8')
    tracker=TargetTracker();n=0;observed=0;predicted=0;coverage=[];times=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            start=time.perf_counter();r=tracker.update(frame,n,n/fps);dt=time.perf_counter()-start
            times.append(dt);observed+=r['center']['source']=='observed';predicted+=r['center']['source']=='predicted'
            coverage.append(sum(v['observed_fraction'] for v in r['rings'])/8)
            if stream:
                stream.write(json.dumps(r,allow_nan=False)+'\n')
                writer.write(annotate(frame,r,1/dt))
            if n in checkpoints:
                cv2.imwrite(str(directory/f'frame_{n:04d}.jpg'),annotate(frame,r,1/dt))
                (directory/f'frame_{n:04d}.json').write_text(json.dumps(r,allow_nan=False),encoding='utf-8')
            n+=1
    finally:
        cap.release()
        if writer:writer.release()
        if stream:stream.close()
    summary=dict(video=source.name,decoded_frames=n,expected_frames=total,
                 observed_center_fraction=observed/max(n,1),predicted_center_fraction=predicted/max(n,1),
                 observed_ring_sample_fraction=float(np.mean(coverage)) if coverage else 0,
                 processing_fps=n/max(sum(times),1e-9),p95_frame_seconds=float(np.percentile(times,95)) if times else None)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos',nargs='+',type=int,default=list(range(5,18)))
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--export-example',action='store_true',help='Export full video6 MP4 and JSONL into the parent of output')
    p.add_argument('--output',default=str(ROOT/'output/demo14/validation'))
    args=p.parse_args()
    if args.workers<1:p.error('workers must be positive')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(evaluate,(v,str(output),args.export_example)) for v in args.videos]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    report=dict(tracker_sha256=TRACKER_SHA256,
                note='Coverage is the fraction of all candidate ring samples observed, not ground-truth recall; concurrent FPS is not standalone FPS.',
                videos=sorted(results,key=lambda r:int(r['video'][5:-4])))
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
