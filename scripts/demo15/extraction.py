"""Model inference and compact reusable evidence cache; no annotation access."""
from pathlib import Path
import hashlib
import inspect
import json
import os
import time
import cv2
import numpy as np
from demo13.tracking import YoloSegmenter, DescendingAthleteTrack, Instance, box_iou
from demo13.parsing import SCHPParser
from demo14.tracker import TargetTracker, Config, paper_region
from .references import timestamp
from .core import aligned_depth, add_depth_features, _segment_depth_features, background_reference
from .evidence import (CameraRegistration, pack_instance, unpack_instance, complete_athlete_track,
                       parse_athlete, shoe_observations, associate_feet)
from .target_hit import paper_outline

EVIDENCE_VERSION = 5
EXTRACTION_CONFIG = dict(detector_sizes=[640,960], detector_conf=.15, parser="expanded_person_roi_coarse", camera="background_lk_640")


class StaleEvidenceError(ValueError):
    """Model evidence must be recomputed, without changing historical caches."""


def feature_code_digest():
    source=''.join(inspect.getsource(f) for f in
                   (aligned_depth,add_depth_features,_segment_depth_features,background_reference))
    return hashlib.sha256(source.encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def ground_mask(frame, found=None):
    """Conservative appearance gate: observed paper plus adjacent grassy terrain."""
    h,w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = ((hsv[:,:,0]>=18)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>=30)&(hsv[:,:,2]>=40)&(hsv[:,:,2]<=230)).astype(np.uint8)
    # Perspective sky/trees are excluded by adjacency to the observed target.
    found = paper_region(frame) if found is None else found
    if found is None:
        return np.zeros((h,w), bool)
    _, ellipse, paper = found
    candidate = grass | (paper>0).astype(np.uint8)
    y0 = max(0, int(ellipse[0][1]-ellipse[1][1]*1.5))
    candidate[:y0] = 0
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    count,labels,stats,_ = cv2.connectedComponentsWithStats(candidate)
    values,counts = np.unique(labels[paper>0], return_counts=True)
    values = values[values>0]
    if not len(values):
        return paper>0
    selected = max(values,key=lambda k: np.count_nonzero((labels==k)&(paper>0)))
    result = labels==selected
    return result


def target_coordinates(result,width,height):
    """Restore auxiliary target preview coordinates; sole inference stays native."""
    small_w,small_h=result['image_size'];sx,sy=width/small_w,height/small_h
    result['image_size']=[width,height]
    center=result['center']['point']
    if center is not None:result['center']['point']=[center[0]*sx,center[1]*sy]
    for key in ('outer_boundary','bullseye_boundary'):
        for sample in result[key]:
            if sample['point'] is not None:
                x,y=sample['point'];sample['point']=[x*sx,y*sy]
    return result


def extract(video, destination, root, device='mps', with_depth=True, max_frames=0):
    config_directory=Path(os.environ.setdefault('YOLO_CONFIG_DIR',str(root/'.cache/demo15/ultralytics')))
    config_directory.mkdir(parents=True,exist_ok=True)
    # Older releases append this subdirectory to YOLO_CONFIG_DIR.
    (config_directory/'Ultralytics').mkdir(exist_ok=True)
    from ultralytics import YOLO, __version__
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    model_path = root/'models/yolo26n-depth.pt'
    depth_model = None
    if with_depth:
        if 'depth' not in YOLO.task_map.fget(None):
            raise RuntimeError('当前 Ultralytics 不支持 depth，请使用 .venv-demo15 的 Python 解释器')
        if not model_path.exists():
            raise RuntimeError(f'缺少深度权重：{model_path}')
        depth_model = YOLO(str(model_path),task='depth')
        if depth_model.task != 'depth' or 'depth' not in depth_model.task_map:
            raise RuntimeError('当前 Ultralytics 不支持 depth，请使用 .venv-demo15')
    segmenter = YoloSegmenter(str(root/'yolo26n-seg.pt'),device=device)
    supplemental = YOLO(str(root/'yolo26n-seg.pt'))
    parser = SCHPParser(root/'third_party/Self-Correction-Human-Parsing',root/'models/exp-schp-201908261155-lip.pth',device=device)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'无法打开视频：{video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width,height = int(cap.get(3)),int(cap.get(4))
    expected = int(cap.get(7))
    primary = DescendingAthleteTrack(fps=fps)
    camera = CameraRegistration()
    # Full ring tracing is unrelated to contact inference and dominated CPU
    # latency. Trace the observed paper and bullseye here; refine rings only
    # around an automatically predicted event in the reporting stage.
    target = TargetTracker(Config(ring_count=2))
    observations=[]; depths=[]; exclusions=[]; grounds=[]
    previous=-1.; started=time.perf_counter(); times=dict(segmentation=0.,parts=0.,depth=0.,scene=0.)
    factor=320/max(width,height)
    cache_size=(max(32,round(width*factor)),max(32,round(height*factor)))
    target_factor=min(1.,640/max(width,height))
    target_size=(round(width*target_factor),round(height*target_factor))
    # First pass: retain actual detections so a late identity lock can be
    # propagated backwards without using a reference frame or annotation.
    detections=[]; selections=[]; selection_reasons=[]
    detection_started=time.perf_counter()
    try:
        while not max_frames or len(detections)<max_frames:
            ok,frame=cap.read()
            if not ok:break
            n=len(detections)
            instances=segmenter.infer(frame)
            # Independent detection covers track-ID dropouts; only geometric
            # and appearance-consistent observations can join the target later.
            extra=supplemental.predict(frame,imgsz=960,conf=.15,classes=[0],
                                       retina_masks=True,device=device,verbose=False)[0]
            if extra.masks is not None:
                masks=extra.masks.data.cpu().numpy()>.5
                for k,(box,confidence,mask) in enumerate(zip(extra.boxes.xyxy.cpu().numpy(),extra.boxes.conf.cpu().numpy(),masks)):
                    if all(box_iou(np.asarray(item.bbox),box)<.5 for item in instances):
                        instances.append(Instance(-1-n*100-k,float(confidence),box.tolist(),mask,0))
            logical,status=primary.update(instances,n,frame)
            packed=[pack_instance(item,frame) for item in instances]
            choice=next((item for item in packed if item['track_id']==primary.source_track_id),None) if status=='tracked' else None
            detections.append(packed);selections.append(choice);selection_reasons.append(primary.reason)
            if n%60==0:print(f'{Path(video).name}：目标检测 {n}/{expected}',flush=True)
    finally:
        cap.release()
    times['segmentation']=time.perf_counter()-detection_started
    selections=complete_athlete_track(detections,selections,fps)
    cap=cv2.VideoCapture(str(video))
    i=0
    try:
        while not max_frames or i<max_frames:
            ok,frame=cap.read()
            if not ok:
                break
            t,source=timestamp(cap.get(cv2.CAP_PROP_POS_MSEC)/1000,previous,fps,i); previous=t
            instances=[unpack_instance(item) for item in detections[i]]
            selected=selections[i]
            person=unpack_instance(selected,1) if selected is not None else None
            logical,status=(1,'tracked') if person is not None else (None,'lost')
            tic=time.perf_counter()
            labels=parse_athlete(frame,person,parser) if person is not None else None
            parts_source='inferred' if labels is not None else 'unavailable'
            times['parts']+=time.perf_counter()-tic
            excluded=np.zeros((height,width),np.uint8)
            for item in instances:
                excluded[item.mask]=1
            # Scene alignment excludes all observed people, not only the athlete.
            tic=time.perf_counter(); camera_info=camera.update(frame,excluded>0)
            target_result=target_coordinates(target.update(cv2.resize(frame,target_size),i,t),width,height)
            paper=paper_region(frame)
            terrain=ground_mask(frame,paper)
            outline=paper_outline(paper,excluded>0)
            times['scene']+=time.perf_counter()-tic
            tic=time.perf_counter()
            depth=None
            if depth_model is not None:
                result=depth_model.predict(frame,imgsz=768,device=device,verbose=False)[0]
                if getattr(result,'depth',None) is None:
                    raise RuntimeError('模型未返回深度图；拒绝无深度回退')
                depth=aligned_depth(result.depth.data.cpu().numpy(),(height,width))
            times['depth']+=time.perf_counter()-tic
            feet=shoe_observations(labels) if labels is not None else []
            for foot in feet:
                x,y=foot['point']; foot['ground_region']=bool(terrain[max(0,y-2):min(height,y+3),max(0,x-2):min(width,x+3)].any())
                foot['shoe_depth']=None
                if depth is not None:
                    x1,y1,x2,y2=foot['bbox']
                    shoe=(labels[y1:y2,x1:x2]==2).astype(np.uint8)
                    interior=cv2.erode(shoe,np.ones((3,3),np.uint8))>0
                    if not interior.any(): interior=shoe>0
                    values=depth[y1:y2,x1:x2][interior]
                    values=values[np.isfinite(values)&(values>0)]
                    if len(values): foot['shoe_depth']=float(np.median(values))
            observation=dict(frame_id=i,timestamp=t,timestamp_source=source,image_size=[width,height],
                             track_status=status,track_id=logical,
                             selection_reason='offline_observed_track' if selected is not None else selection_reasons[i],
                             person_bbox=person.bbox if person is not None else None,
                             parts_source=parts_source,feet=feet,target=target_result,paper_outline=outline,**camera_info)
            observations.append(observation)
            if depth is not None:
                depths.append(cv2.resize(depth,cache_size,interpolation=cv2.INTER_AREA))
                edges=cv2.Canny(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),80,180)
                grass_edges=(cv2.dilate(edges,np.ones((3,3),np.uint8))>0)&(cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,1]>40)&terrain
                fit_exclusion=cv2.dilate(excluded,np.ones((17,17),np.uint8))>0
                fit_exclusion|=grass_edges
                exclusions.append(cv2.resize(fit_exclusion.astype(np.uint8),cache_size,interpolation=cv2.INTER_NEAREST)>0)
                grounds.append(cv2.resize(terrain.astype(np.uint8),cache_size,interpolation=cv2.INTER_NEAREST)>0)
            if i%30==0:
                print(f'{Path(video).name}：{i}/{expected} 帧，主目标 {status}，脚部 {len(feet)}，耗时 {time.perf_counter()-started:.1f}秒',flush=True)
            i+=1
    finally:
        cap.release()
    if not max_frames and i!=expected:
        raise RuntimeError(f'视频未完整解码：{i}/{expected}，不执行正式评估')
    if not i:
        raise ValueError('视频为空')
    associate_feet(observations)
    # Persist real selected masks for lazy, per-event precision after the gate.
    packed_size=(width*height+7)//8
    masks=np.zeros((len(observations),packed_size),np.uint8)
    valid=np.zeros(len(observations),bool)
    for n,item in enumerate(selections):
        if item is not None:
            masks[n]=item['mask'];valid[n]=True
    np.savez_compressed(destination/'athlete_masks.npz',masks=masks,valid=valid,
                        image_shape=np.array([height,width]))
    times['contact_refinement']=0.
    ref=reliable=None
    if with_depth:
        depths=np.asarray(depths,np.float32)
        ref,reliable=add_depth_features(observations,depths,np.asarray(exclusions),np.asarray(grounds))
        np.savez_compressed(destination/'depth_cache.npz',depth=depths,exclusions=np.asarray(exclusions),grounds=np.asarray(grounds),ground_reference=ref,reliable=reliable)
    metadata=dict(athlete_masks_sha256=digest(destination/'athlete_masks.npz'),
                  target_outline_code_sha256=hashlib.sha256(inspect.getsource(paper_outline).encode()).hexdigest(),evidence_version=EVIDENCE_VERSION, extraction_config=EXTRACTION_CONFIG, video=str(Path(video).resolve()),video_sha256=digest(video),fps=fps,
                  image_size=[width,height],frames=i,expected_frames=expected,complete=i==expected,
                  with_depth=with_depth,device=device,ultralytics=__version__,elapsed_seconds=time.perf_counter()-started,
                  stage_seconds=times,depth_cache_size=list(cache_size),code_sha256=digest(__file__), evidence_code_sha256=digest(Path(__file__).with_name('evidence.py')),
                  feature_code_sha256=feature_code_digest(),
                  dependency_sha256={name:digest(root/'scripts/demo13'/name) for name in ('tracking.py','parsing.py','contact.py')},
                  target_tracking='paper_and_bullseye_640_preview_native_event_refinement',models={p.name:digest(p) for p in [root/'yolo26n-seg.pt',root/'models/exp-schp-201908261155-lip.pth']+([model_path] if with_depth else [])})
    save_json(destination/'metadata.json',metadata)
    with (destination/'frames.jsonl').open('w',encoding='utf-8') as stream:
        for o in observations:stream.write(json.dumps(o,ensure_ascii=False,allow_nan=False)+'\n')
    return observations,metadata


def load_cache(path,video=None):
    path=Path(path)
    metadata=json.loads((path/'metadata.json').read_text())
    if metadata.get('evidence_version') != EVIDENCE_VERSION or metadata.get('extraction_config') != EXTRACTION_CONFIG:
        raise StaleEvidenceError('证据缓存版本或提取配置已过期，请重新提取')
    if metadata.get('code_sha256') != digest(__file__) or metadata.get('evidence_code_sha256') != digest(Path(__file__).with_name('evidence.py')):
        raise StaleEvidenceError('证据提取代码已变化，请重新提取')
    if metadata.get('target_outline_code_sha256') != hashlib.sha256(inspect.getsource(paper_outline).encode()).hexdigest():
        raise StaleEvidenceError('外轮廓提取代码已变化，请重新提取')
    mask_path=path/'athlete_masks.npz'
    if not mask_path.exists() or digest(mask_path)!=metadata.get('athlete_masks_sha256'):
        raise StaleEvidenceError('按需精修掩膜缺失或校验失败，请重新提取')
    if metadata.get('feature_code_sha256') != feature_code_digest():
        raise StaleEvidenceError('深度特征提取代码已变化，请重新提取')
    root=Path(__file__).resolve().parents[2]
    for name,expected in metadata.get('dependency_sha256',{}).items():
        if digest(root/'scripts/demo13'/name)!=expected:
            raise StaleEvidenceError('共享提取代码已变化，请重新提取')
    for name,expected in metadata['models'].items():
        weight=root/name if name=='yolo26n-seg.pt' else root/'models'/name
        if not weight.exists() or digest(weight)!=expected:
            raise StaleEvidenceError('模型权重已变化，请重新提取')
    if video is not None and metadata['video_sha256']!=digest(video):
        raise ValueError('缓存与输入视频校验值不一致')
    rows=[json.loads(v) for v in (path/'frames.jsonl').read_text().splitlines()]
    if any(row['frame_id']!=i or (i and row['timestamp']<=rows[i-1]['timestamp']) for i,row in enumerate(rows)):
        raise ValueError('缓存帧号或时间戳不连续')
    if len(rows)!=metadata['frames']:
        raise ValueError('缓存帧数不一致')
    return rows,metadata
