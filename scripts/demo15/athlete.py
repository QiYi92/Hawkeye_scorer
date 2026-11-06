"""Demo15-only person tracklets and landing-role selection.

Detector identities and temporary 960px detections are associated before any
landing role is assigned. No image-entry height or target-center prior is used.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from demo13.tracking import box_iou


@dataclass(frozen=True)
class AthleteConfig:
    min_observations: int = 3
    min_height_fraction: float = .05
    min_motion_body_fraction: float = .12
    max_gap_seconds: float = .4
    max_appearance_distance: float = .75
    seated_aspect: float = .55


class CandidateAssociator:
    def __init__(self, fps, config=None):
        self.config=config or AthleteConfig()
        self.max_gap=max(3,round(fps*self.config.max_gap_seconds))
        self.active={};self.next_id=0

    def update(self, detections, frame_id, image_size):
        w,h=image_size
        self.active={k:v for k,v in self.active.items() if frame_id-v['frame']<=self.max_gap}
        ids=list(self.active);cost=np.full((len(detections),len(ids)+len(detections)),1.6)
        for j,item in enumerate(detections):
            box=np.asarray(item['bbox'],float);center=(box[:2]+box[2:])/2
            for column,key in enumerate(ids):
                old=self.active[key];gap=frame_id-old['frame']
                predicted=old['box']+np.tile(old['velocity']*gap,2)
                size=max(12.,max(old['box'][2:]-old['box'][:2]))
                distance=np.linalg.norm(center-(predicted[:2]+predicted[2:])/2)/size
                overlap=box_iou(box,predicted)
                ratio=np.prod(box[2:]-box[:2])/max(1.,np.prod(old['box'][2:]-old['box'][:2]))
                clipped=bool(box[0]<2 or box[2]>w-2 or old['box'][0]<2 or old['box'][2]>w-2)
                appearance=float(cv2.compareHist(np.asarray(item['appearance'],np.float32),
                    np.asarray(old['appearance'],np.float32),cv2.HISTCMP_BHATTACHARYYA))
                same=item['track_id']>=0 and item['track_id']==old['source_id']
                if (distance<.85 and overlap>.05 and (.15 if clipped else .35)<=ratio<=(6 if clipped else 3)
                        and appearance<=self.config.max_appearance_distance):
                    cost[j,column]=distance+.3*(1-overlap)+.2*appearance-.15*same
        assignments={}
        if detections:
            a,b=linear_sum_assignment(cost)
            assignments=dict(zip(a,b))
        for j,item in enumerate(detections):
            column=assignments[j];key=ids[column] if column<len(ids) and cost[j,column]<1.6 else None
            old=self.active.get(key)
            if key is None:key=self.next_id;self.next_id+=1
            box=np.asarray(item['bbox'],float);center=(box[:2]+box[2:])/2
            velocity=(center-(old['box'][:2]+old['box'][2:])/2)/(frame_id-old['frame']) if old else np.zeros(2)
            if old:velocity=.7*velocity+.3*old['velocity']
            item['candidate_id']=key
            item['candidate_association']='motion_appearance_match' if old else 'new_detection_tracklet'
            self.active[key]=dict(frame=frame_id,box=box,velocity=velocity,
                appearance=item['appearance'],source_id=item['track_id'])
        return detections


def candidate_sequences(detections, legacy, image_size, config=None):
    cfg=config or AthleteConfig();n=len(detections);w,h=image_size
    grouped={}
    for frame_id,items in enumerate(detections):
        for item in items:grouped.setdefault(item['candidate_id'],{})[frame_id]=item
    legacy_ids={item['candidate_id'] for item in legacy if item is not None}
    streams={};profiles=[]
    if legacy_ids:
        streams['descent']=legacy
        profiles.append(dict(candidate='descent',path='strong_descent',eligible=True,source_ids=sorted(legacy_ids)))
    for key,items in grouped.items():
        if key in legacy_ids:continue
        boxes=np.asarray([item['bbox'] for item in items.values()],float)
        heights=boxes[:,3]-boxes[:,1];centers=(boxes[:,:2]+boxes[:,2:])/2
        motion=float(np.max(np.linalg.norm(centers-centers[0],axis=1)))
        eligible=(len(items)>=cfg.min_observations and np.median(heights)>=cfg.min_height_fraction*h
                  and motion>=cfg.min_motion_body_fraction*np.median(heights))
        reason=('persistent_moving_person' if eligible else
                ('short_tracklet' if len(items)<cfg.min_observations else
                 ('small_candidate' if np.median(heights)<cfg.min_height_fraction*h else 'stationary_candidate')))
        label=f'person_{key}'
        profiles.append(dict(candidate=label,path='low_or_lateral_approach',eligible=bool(eligible),reason=reason,
            frames=[min(items),max(items)],observations=len(items),motion_px=motion))
        if eligible:streams[label]=[items.get(i) for i in range(n)]
    return streams,profiles


def choose_landing_athlete(trajectories, config, actor_config=None):
    from .core import estimate
    cfg=actor_config or AthleteConfig();eligible=[];diagnostics=[]
    for key,rows in trajectories.items():
        result=estimate(rows,'fusion',config)
        event=result.get('event');prior=[]
        if event:
            prior=[row for row in rows if 0<event['timestamp']-row['timestamp']<=.8 and row.get('person_bbox')]
        seated_frames=[]
        for row in prior:
            box=row['person_bbox'];height=max(1.,box[3]-box[1]);cx=(box[0]+box[2])/2
            if ((box[2]-box[0])/height>=cfg.seated_aspect and
                    any(abs(f['point'][0]-cx)>=height*.2 for f in row['feet'])):
                seated_frames.append(row['frame_id'])
        seated=len(seated_frames)>=3
        separating=[row for row in prior if any(f.get('depth_residual') is not None and
            f['depth_residual']>=config.separation for f in row['feet'])]
        separation=len(separating)>=3
        body_descent=False
        if len(prior)>=3:
            centers=[(row['person_bbox'][1]+row['person_bbox'][3])/2 for row in prior]
            heights=[row['person_bbox'][3]-row['person_bbox'][1] for row in prior]
            body_descent=centers[-1]-centers[0]>=.1*float(np.median(heights))
        strong=key=='descent'
        # A walker can have step transitions, but that alone never creates a
        # new low-approach landing role. Sitting/airborne approach is required.
        accepted=strong or bool(event and result['status']=='confirmed_estimate' and (seated or (separation and body_descent)))
        diagnostics.append(dict(candidate=key,path='strong_descent' if strong else 'low_or_lateral_approach',
            role_accepted=accepted,seated_approach=seated,prior_depth_separation=separation,
            seated_evidence_frames=seated_frames,body_approach=bool(body_descent),
            contact_status=result['status'],contact_frame=event['frame_id'] if event else None,
            contact_failure=result.get('failure_reason')))
        if accepted:eligible.append(key)
    if len(eligible)==1:return eligible[0],diagnostics,'landing_athlete_selected'
    return None,diagnostics,'ambiguous_landing_athletes' if eligible else 'no_supported_landing_athlete'
