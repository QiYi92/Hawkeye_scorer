"""Classify confirmed contact before lazily computing precise shoes and rings."""
from pathlib import Path
import copy
import time

import cv2
import numpy as np

from demo13.tracking import Instance
from demo13.parsing import SCHPParser
from demo14.tracker import TargetTracker
from .core import estimate, region
from .evidence import parse_athlete, shoe_observations, add_refined_points
from .target_hit import target_decision
from .rings import RingTracker


class ContactRefiner:
    def __init__(self, video, cache, root, device):
        self.video=video;self.cache=Path(cache);self.root=Path(root);self.device=device
        self.parser=None;self.frames={};self.masks=None
        self.seconds=0.;self.calls=0;self.parser_inferences=0;self.reuses=0

    def refine(self, row):
        n=row['frame_id']
        if n in self.frames:
            self.reuses+=1
            return copy.deepcopy(self.frames[n])
        started=time.perf_counter()
        if self.masks is None:
            with np.load(self.cache/'athlete_masks.npz',allow_pickle=False) as archive:
                self.masks=archive['masks'];self.shape=tuple(archive['image_shape'])
                self.valid=archive['valid']
        if not self.valid[n]:return copy.deepcopy(row['feet'])
        if self.parser is None:
            self.parser=SCHPParser(self.root/'third_party/Self-Correction-Human-Parsing',
                                   self.root/'models/exp-schp-201908261155-lip.pth',device=self.device)
        cap=cv2.VideoCapture(str(self.video));frame=None
        try:
            for _ in range(n+1):
                ok,frame=cap.read()
                if not ok:raise RuntimeError('按需鞋底精修解码失败')
        finally:cap.release()
        mask=np.unpackbits(self.masks[n],count=int(np.prod(self.shape))).reshape(self.shape).astype(bool)
        person=Instance(1,1.,row['person_bbox'],mask,0)
        fine=shoe_observations(parse_athlete(frame,person,self.parser,refine=True))
        feet=copy.deepcopy(row['feet'])
        add_refined_points(feet,fine)
        self.frames[n]=feet;self.calls+=1;self.parser_inferences+=2
        self.seconds+=time.perf_counter()-started
        return copy.deepcopy(feet)


def full_target(video, rows, frame_id):
    tracker=TargetTracker();cap=cv2.VideoCapture(str(video))
    rings=RingTracker()
    target=rows[frame_id]['target']
    try:
        for i in range(frame_id+1):
            ok,frame=cap.read()
            if not ok:raise RuntimeError('触地帧靶盘复核解码失败')
            if i>=max(0,frame_id-6):
                target=tracker.update(frame,i,rows[i]['timestamp'])
                excluded=np.zeros(frame.shape[:2],np.uint8)
                box=rows[i].get('person_bbox')
                if box is not None:
                    x1,y1,x2,y2=map(round,box)
                    excluded[max(0,y1):min(len(frame),y2),max(0,x1):min(frame.shape[1],x2)]=1
                model=rings.update(frame,target,rows[i],excluded,full=True)
                preview_model=rows[i]['target'].get('ring_model',{})
                target['ring_model']=(model if model.get('status')=='fitted' else preview_model)

    finally:cap.release()
    return target


def analyze_modes(video, rows, config, modes, cache, root, device='mps', shortcut=True):
    started=time.perf_counter()
    # Estimate all modes from exactly the same coarse evidence. Precision is
    # computed afterwards and cannot leak into another mode's event selection.
    results={mode:estimate(rows,mode,config) for mode in modes}
    refiner=ContactRefiner(video,cache,root,device)
    targets={};ring_seconds=0.;ring_calls=0;skipped=0
    rendered=copy.deepcopy(rows)
    for mode,result in results.items():
        result.update(target_hit='unknown',point_precision='coarse',shortcut_reason=None)
        event=result.get('event')
        if event is None:continue
        n=event['frame_id'];row=rows[n]
        foot=next((f for f in row['feet'] if f.get('foot_track_id')==event.get('foot_track_id')),None)
        if foot is None and row['feet']:
            foot=min(row['feet'],key=lambda f:np.linalg.norm(np.asarray(f['point'])-event['point_px']))
        decision=target_decision(rows,result,foot)
        before=refiner.calls;before_rings=ring_calls
        if shortcut and decision['early_miss']:
            skipped+=1
            result.update(target_hit='miss',shortcut_reason=decision['reason'])
            event['target_region']='outside'
            event['target_evidence']=row['target']
        else:
            refined=refiner.refine(row)
            rendered[n]['feet']=refined
            match=next((f for f in refined if foot is not None and f.get('foot_track_id')==foot.get('foot_track_id')),None)
            if match and match.get('contact_point_source')=='two_scale_rgb_agreement':
                event['point_px']=match['contact_point']
                event['point_source']=match['contact_point_source']
                result['point_precision']='refined'
            decision=target_decision(rows,result)
            result['target_hit']=decision['target_hit']
            if shortcut and decision['target_hit']=='miss':
                result['shortcut_reason']='refined_miss_skip_rings';skipped+=1
                event['target_evidence']=row['target'];event['target_region']='outside'
            else:
                if n not in targets:
                    tic=time.perf_counter();targets[n]=full_target(video,rows,n)
                    ring_seconds+=time.perf_counter()-tic;ring_calls+=1
                event['target_evidence']=targets[n]
                rendered[n]['target']['ring_model']=targets[n].get('ring_model',{})
                event['target_region']=region(event['point_px'],targets[n])
        event.update(target_hit=result['target_hit'],point_precision=result['point_precision'],
                     shortcut_reason=result['shortcut_reason'],camera_transform=row['camera_transform'])
        result['target_classification']=decision
        result['processing']=dict(refinement_calls=refiner.calls-before,ring_calls=ring_calls-before_rings,
                                  refinement_skipped=bool(shortcut and decision['early_miss']),
                                  ring_skipped=bool(shortcut and result['target_hit']=='miss'))
    timing=dict(postprocess_seconds=time.perf_counter()-started,refinement_seconds=refiner.seconds,
                refinement_calls=refiner.calls,parser_inferences=refiner.parser_inferences,
                refinement_reuses=refiner.reuses,ring_seconds=ring_seconds,ring_calls=ring_calls,
                shortcut_modes=skipped,shortcut_enabled=shortcut)
    refinements={str(n):feet for n,feet in refiner.frames.items()}
    return results,rendered,timing,refinements
