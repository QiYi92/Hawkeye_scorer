"""Offline evidence construction and three comparable contact estimators."""
from dataclasses import dataclass, asdict
import warnings
import cv2
import numpy as np

MODES = ('depth', 'fusion', 'visual')


@dataclass(frozen=True)
class ContactConfig:
    near: float = .08
    separation: float = .10
    support_seconds: float = .10
    window_seconds: float = 1.
    descent_px_720: float = 12.
    motion_tolerance_720: float = 10.

    def __post_init__(self):
        if not (0<self.near<self.separation and 0<self.support_seconds<=self.window_seconds<=2
                and self.descent_px_720>0 and self.motion_tolerance_720>0):
            raise ValueError('触地参数必须满足：0<接近阈值<分离阈值，0<支撑时间≤窗口≤2秒，像素容差为正')

    def to_dict(self):
        return asdict(self)


def aligned_depth(value, shape):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).any() or not (array > 0).any():
        raise ValueError('深度输出必须是有效的二维浮点图')
    if array.shape != tuple(shape):
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return array


def background_reference(depths, exclusions, grounds, stable):
    """Robust static-background median; masks and scale fit never use labels."""
    values = np.asarray(depths, np.float32).copy()
    valid = np.isfinite(values) & (values > 0) & ~exclusions & grounds
    valid[~np.asarray(stable, bool)] = False
    values[~valid] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        base = np.nanmedian(values, axis=0)
    scales = []
    normalized = values.copy()
    for i, value in enumerate(values):
        fit = np.isfinite(base) & np.isfinite(value)
        ratio = np.log(value[fit]/base[fit])
        scale = float(np.exp(np.median(ratio))) if len(ratio) >= 100 else None
        scales.append(scale)
        if scale is not None and .25 <= scale <= 4:
            normalized[i] /= scale
        else:
            normalized[i] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        reference = np.nanmedian(normalized, axis=0)
        spread = np.nanmedian(np.abs(np.log(normalized/reference)), axis=0)
    counts = np.isfinite(normalized).sum(axis=0)
    reliable = (counts >= 5) & np.isfinite(reference) & (spread < .12)
    return reference, reliable, scales


def add_depth_features(observations, depths, exclusions, grounds):
    # Registration gaps start independent reference systems. Never combine
    # their depth pixels as if they shared a known camera transform.
    segments = sorted(set(o.get('camera_segment', 0) for o in observations))
    if len(segments) > 1:
        largest = (0, None)
        for segment in segments:
            indices = [i for i,o in enumerate(observations) if o.get('camera_segment',0)==segment]
            result = _segment_depth_features([observations[i] for i in indices],
                                             depths[indices], exclusions[indices], grounds[indices])
            if len(indices)>largest[0]:largest=(len(indices),result)
        return largest[1]
    return _segment_depth_features(observations, depths, exclusions, grounds)


def _segment_depth_features(observations, depths, exclusions, grounds):
    stable = [o['camera_stable'] for o in observations]
    dh,dw=depths.shape[1:]
    warped=[];warped_exclusions=[];warped_grounds=[]
    for i,o in enumerate(observations):
        w,h=o['image_size'];s=np.diag([dw/w,dh/h,1.])
        transform=s@np.linalg.inv(np.asarray(o['camera_transform']))@np.linalg.inv(s)
        warped.append(cv2.warpPerspective(depths[i],transform,(dw,dh),flags=cv2.INTER_LINEAR,borderValue=float('nan')))
        warped_exclusions.append(cv2.warpPerspective(exclusions[i].astype(np.uint8),transform,(dw,dh),flags=cv2.INTER_NEAREST,borderValue=1)>0)
        warped_grounds.append(cv2.warpPerspective(grounds[i].astype(np.uint8),transform,(dw,dh),flags=cv2.INTER_NEAREST)>0)
    ref, reliable, scales = background_reference(np.asarray(warped),np.asarray(warped_exclusions),np.asarray(warped_grounds),stable)
    dh, dw = ref.shape
    for i, o in enumerate(observations):
        h, w = o['image_size'][1], o['image_size'][0]
        o['depth_scale'] = scales[i]
        for foot in o['feet']:
            x, y = foot['point']
            mapped=np.linalg.inv(np.asarray(o['camera_transform']))@np.array([x,y,1.])
            u,v=round(mapped[0]/mapped[2]*dw/w),round(mapped[1]/mapped[2]*dh/h)
            scale = scales[i]
            inside=0<=u<dw and 0<=v<dh
            # The shoe occludes terrain in the current frame. Ground validity
            # must come from visible background frames, not from current shoe
            # appearance. A tiny robust patch tolerates printed-line holes.
            patch=ref[max(0,v-1):min(dh,v+2),max(0,u-1):min(dw,u+2)] if inside else np.array([])
            ok=reliable[max(0,v-1):min(dh,v+2),max(0,u-1):min(dw,u+2)] if inside else np.array([],bool)
            valid = bool(inside and ok.sum()>=3 and scale is not None)
            ground_depth=float(np.median(patch[ok])) if valid else None
            foot['ground_valid'] = valid
            foot['depth_residual'] = None
            foot['shoe_depth'] = foot.get('shoe_depth')
            foot['ground_depth'] = ground_depth
            if valid and foot['shoe_depth'] is not None and foot['shoe_depth'] > 0:
                foot['depth_residual'] = float(abs(np.log(foot['shoe_depth']/scale/ground_depth)))
    return ref, reliable


def nearest_foot(observation, anchor, max_distance=120.):
    if observation.get('parts_source','inferred')!='inferred' or not observation['camera_stable']:
        return None
    feet = observation['feet']
    if not feet:
        return None
    f = min(feet, key=lambda x: np.linalg.norm(np.asarray(x['point'])-anchor))
    return f if np.linalg.norm(np.asarray(f['point'])-anchor) <= max_distance else None


def estimate(observations, mode='fusion', config=None):
    """First supported descent break of one observed shoe, in background space.

    Confirmation can look ahead, but the returned point is always an observed
    sole at the beginning of support. Depth remains an independent comparison
    mode; noisy depth cannot veto a well observed visual support transition.
    """
    if mode not in MODES:raise ValueError(mode)
    cfg=config or ContactConfig()
    if mode=='depth':return estimate_depth(observations,mode,cfg)
    from .evidence import associate_feet
    associate_feet(observations)
    tracks={}
    for row in observations:
        if not row['camera_stable'] or row.get('parts_source','inferred')!='inferred':continue
        for foot in row['feet']:
            tracks.setdefault(foot['foot_track_id'],[]).append((row,foot))
    candidates=[]
    for track_id, track in tracks.items():
        for j,(row,foot) in enumerate(track):
            t=row['timestamp'];scale=row['image_size'][1]/720
            before=[item for item in track[max(0,j-12):j+1] if 0<=t-item[0]['timestamp']<=.30]
            after=[item for item in track[j:j+12] if 0<=item[0]['timestamp']-t<=max(cfg.support_seconds,.10)+1e-6]
            if len(before)<3 or len(after)<3:continue
            if after[-1][0]['timestamp']-t<.065:continue
            local=before[-4:]+after[1:]
            if any(b[0]['frame_id']-a[0]['frame_id']>2 for a,b in zip(local,local[1:])):
                continue
            # Fit the approach locally: a distant airborne height alone is not
            # evidence that the current step belongs to the original landing.
            pre=before[-5:]
            pre_t=np.array([o['timestamp']-t for o,f in pre])
            pre_y=np.array([f['stabilized_point'][1] for o,f in pre])
            post_t=np.array([o['timestamp']-t for o,f in after])
            post_y=np.array([f['stabilized_point'][1] for o,f in after])
            if np.ptp(pre_t)<.05 or pre_y[-1]-pre_y[0]<cfg.descent_px_720*scale:continue
            descent_speed=float(np.polyfit(pre_t,pre_y,1)[0])
            support_speed=float(np.polyfit(post_t,post_y,1)[0])
            fit_error=float(np.mean((np.polyval(np.polyfit(pre_t,pre_y,1),pre_t)-pre_y)**2)
                            +np.mean((np.polyval(np.polyfit(post_t,post_y,1),post_t)-post_y)**2))
            if descent_speed<30*scale or support_speed>descent_speed*.35:continue
            if np.ptp(post_y)>cfg.motion_tolerance_720*scale:continue
            post_x=[f['stabilized_point'][0] for o,f in after]
            # Supporting feet can roll/run, but a shoe travelling several
            # lengths during a vertical plateau is still flying forward.
            if np.ptp(post_x)>max(cfg.motion_tolerance_720*scale,
                                  (foot['bbox'][2]-foot['bbox'][0])*.75):continue
            grounded=any(f.get('ground_region',False) or
                         (mode=='fusion' and f.get('ground_valid',False)) for o,f in after)
            if not grounded:continue
            # A short plateau followed immediately by continued descent is an
            # airborne posture change, not contact. Upward release is allowed.
            future=[(o,f) for o,f in track[j+1:j+16] if 0<o['timestamp']-t<=.23]
            baseline=foot['stabilized_point'][1]
            released=False;resumed=False
            for o,f in future:
                dy=f['stabilized_point'][1]-baseline
                if dy < -cfg.motion_tolerance_720*scale:released=True
                if not released and dy>max(12*scale,(foot['bbox'][3]-foot['bbox'][1])*.6):resumed=True
            if resumed:continue
            gaps=any(b[0]['frame_id']-a[0]['frame_id']>1 for a,b in zip(pre,pre[1:]))
            # A one-frame missing segmentation may be bridged by consistent
            # real observations on both sides, without inventing a contact.
            adequate_prior=len(pre)>=4 and pre[-1][0]['frame_id']-pre[0][0]['frame_id']<=6
            adequate_prior |= (len(pre)>=3 and not gaps and
                               pre_y[-1]-pre_y[0]>=2*cfg.descent_px_720*scale)
            status='confirmed_estimate' if adequate_prior else 'needs_review'
            residuals=[f.get('depth_residual') for o,f in after]
            depth_support=sum(v is not None and v<=cfg.near for v in residuals)
            evidence=['observed_sole','same_foot_trajectory','background_compensated_descent',
                      'vertical_support_transition','future_descent_rejected']
            if mode=='fusion' and depth_support:evidence.append('auxiliary_depth_proximity')
            if gaps:evidence.append('bounded_observation_gap')
            confirmation=max(after+future,key=lambda item:item[0]['frame_id'])[0]
            event=dict(frame_id=row['frame_id'],timestamp=t,point_px=foot.get('contact_point',foot['point']),
                       point_source=foot.get('contact_point_source','observed_sole'),
                       foot_track_id=track_id,camera_segment=row.get('camera_segment',0),
                       candidate_interval=[t,after[-1][0]['timestamp']],
                       confirmation_frame=confirmation['frame_id'],confirmation_timestamp=confirmation['timestamp'],
                       evidence=evidence,depth_residual=foot.get('depth_residual'),target_region='unknown',
                       motion_diagnostics=dict(descent_px_per_second=descent_speed,
                                               support_px_per_second=support_speed,
                                               support_range_px=float(np.ptp(post_y)),change_point_error=fit_error,
                                               depth_support_frames=depth_support))
            candidates.append(dict(status=status,event=event,mode=mode,
                                   failure_reason=None if adequate_prior else 'insufficient_observed_approach'))
    if candidates:
        earliest=min(c['event']['timestamp'] for c in candidates)
        onset=[c for c in candidates if c['event']['timestamp']<=earliest+.067]
        return min(onset,key=lambda c:(c['event']['motion_diagnostics']['change_point_error'],c['event']['frame_id']))
    return dict(status='unconfirmed',event=None,mode=mode,failure_reason='no_observed_support_transition')


def estimate_depth(observations, mode='depth', config=None):
    if mode not in MODES:
        raise ValueError(mode)
    cfg = config or ContactConfig()
    candidates = []
    for i, current in enumerate(observations):
        if not current['camera_stable'] or not current['feet'] or current.get('parts_source','inferred')!='inferred':
            continue
        t = current['timestamp']; h = current['image_size'][1]
        pre = [o for o in observations[max(0, i-90):i] if 0 < t-o['timestamp'] <= cfg.window_seconds]
        post = [o for o in observations[i:i+91] if 0 <= o['timestamp']-t <= cfg.window_seconds]
        for foot in current['feet']:
            anchor = np.array(foot['point'])
            before = [(o, nearest_foot(o, anchor, h*.25)) for o in pre]
            after = [(o, nearest_foot(o, anchor, h*.25)) for o in post]
            before = [(o, f) for o, f in before if f is not None]
            after = [(o, f) for o, f in after if f is not None]
            if len(before) < 2 or len(after) < 3:
                continue
            descent = anchor[1]-min(f['point'][1] for _, f in before) >= cfg.descent_px_720*h/720
            # Short support window admits horizontal running and shoe rotation.
            support = [(o, f) for o, f in after if o['timestamp']-t <= cfg.support_seconds]
            continuous = len(support) >= 3 and support[-1][0]['timestamp']-t >= cfg.support_seconds*.65
            continuous &= all(b[0]['frame_id'] == a[0]['frame_id']+1 for a,b in zip(support,support[1:]))
            grounded = foot.get('ground_region',False) if mode=='visual' else foot.get('ground_valid',False)
            near = foot.get('depth_residual') is not None and foot['depth_residual'] <= cfg.near
            separation = any(f.get('depth_residual') is not None and f['depth_residual'] >= cfg.separation for _,f in before)
            depth_support = continuous and all(f.get('depth_residual') is not None and f['depth_residual'] <= cfg.near for _,f in support)
            ys = [f['point'][1] for _,f in support]
            motion_support = continuous and max(ys)-min(ys) <= cfg.motion_tolerance_720*h/720
            evidence = ['fresh_shoe_masks', 'prior_descent']
            gaps = any(not o['feet'] or not o['camera_stable'] or o.get('parts_source','inferred')!='inferred'
                       for o in pre if t-o['timestamp'] <= .25)
            qualifies = grounded and descent and len(before)>=3
            if mode in ('depth', 'fusion'):
                qualifies &= near and separation and depth_support
                evidence += ['depth_separation_to_ground_convergence', 'persistent_depth_support']
            if mode in ('fusion','visual'):
                qualifies &= motion_support
                evidence += ['vertical_support_or_running_transition']
            # A late track lock can miss the airborne separation completely.
            # Latch the earliest observed depth proximity + motion support as
            # review-only evidence instead of searching a later walking step.
            # This never upgrades missing separation into a confirmed contact.
            incomplete_prior = (mode in ('depth','fusion') and grounded and descent and
                                near and depth_support and motion_support and gaps and not separation)
            if not qualifies and not incomplete_prior:
                continue
            if incomplete_prior and not qualifies:
                evidence=['fresh_shoe_masks','prior_descent','persistent_depth_proximity',
                          'vertical_support_or_running_transition','missing_prior_depth_separation']
            status = 'needs_review' if gaps or not qualifies else 'confirmed_estimate'
            event = dict(frame_id=i, timestamp=t, point_px=foot['point'],
                         candidate_interval=[t, support[-1][0]['timestamp']],
                         confirmation_frame=support[-1][0]['frame_id'],
                         confirmation_timestamp=support[-1][0]['timestamp'], evidence=evidence,
                         depth_residual=foot.get('depth_residual'), target_region='unknown')
            candidates.append(dict(status=status,event=event,mode=mode,
                                   failure_reason=('missing_prior_depth_separation' if not qualifies else
                                                   ('preceding_observation_gap' if gaps else None))))
        if candidates:
            # Earliest plausible event is never replaced by a later running step.
            return candidates[0]
    return dict(status='unconfirmed', event=None, mode=mode,
                failure_reason='no_depth_ground_transition' if mode != 'visual' else 'no_visual_support_transition')


def region(point, target):
    if not target:
        return 'unknown'
    def polygon(samples):
        if not samples or any(s.get('source') != 'observed' or s.get('point') is None for s in samples):
            return None
        return np.array([s['point'] for s in samples],np.float32)
    bull = polygon(target.get('bullseye_boundary', []))
    if bull is not None and len(bull)>=3:
        distance=cv2.pointPolygonTest(bull,tuple(map(float,point)),True)
        if abs(distance)<=5:return 'unknown'
        if distance>0:return 'bullseye'
    boundary = polygon(target.get('outer_boundary', []))
    if boundary is None or len(boundary) < 3:
        return 'unknown'
    distance = cv2.pointPolygonTest(boundary, tuple(map(float,point)), True)
    if abs(distance) <= 5:
        return 'unknown'
    if distance < 0:
        return 'outside'
    # Ring number needs reliable curves; do not infer it from concentric priors.
    for ring in target.get('rings', []):
        poly = polygon(ring.get('samples', []))
        if ring.get('ring_id') is not None and poly is not None and len(poly)>=3:
            d = cv2.pointPolygonTest(poly,tuple(map(float,point)),True)
            if abs(d)<=5:
                return 'unknown'
            if d>0:
                return f'ring_{ring["ring_id"]}'
    return 'inside_target_ring_unknown'
