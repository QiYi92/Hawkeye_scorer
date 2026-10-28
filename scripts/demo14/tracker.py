"""Conservative polar-path ring detector; all public coordinates are image pixels.

An ellipse supplies search coordinates only. A path is observable only where a
narrow dark ridge and temporal agreement support it. Unknown samples stay null.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Config:
    ring_count: int = 10
    angles: int = 240
    radial_samples: int = 420
    confirm_frames: int = 3
    prediction_seconds: float = 0.5
    min_contrast: float = 7.0

    def __post_init__(self):
        if self.ring_count < 2 or self.angles < 32 or self.radial_samples < 64:
            raise ValueError('Invalid ring count or sampling resolution')
        if self.confirm_frames < 2 or self.prediction_seconds < 0:
            raise ValueError('Invalid confirmation/prediction settings')


def cyclic_path(score: np.ndarray, smoothness: float = 1.4) -> np.ndarray:
    """Bounded-step dynamic programming, with duplicated angular seam context."""
    n, width = score.shape
    pad = min(32, n // 4)
    extended = np.concatenate([score[-pad:], score, score[:pad]])
    cost = -extended[0].copy()
    back = np.zeros(extended.shape, np.int16)
    indices = np.arange(width)
    for t in range(1, len(extended)):
        candidates = []
        for delta in range(-2, 3):
            previous = indices + delta
            valid = (previous >= 0) & (previous < width)
            candidates.append(np.where(valid, cost[np.clip(previous, 0, width-1)] + smoothness*delta*delta, 1e9))
        candidates = np.stack(candidates)
        choice = candidates.argmin(axis=0)
        back[t] = np.clip(indices + choice-2, 0, width-1)
        cost = candidates[choice, indices] - extended[t]
    path = np.empty(len(extended), np.int32)
    path[-1] = cost.argmin()
    for t in range(len(extended)-1, 0, -1):
        path[t-1] = back[t, path[t]]
    return path[pad:pad+n]


def samples(points, confidence=None, sources=None):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    confidence = np.ones(len(points)) if confidence is None else confidence
    sources = ['observed'] * len(points) if sources is None else sources
    return [dict(point=[round(float(x), 3), round(float(y), 3)] if np.isfinite([x, y]).all() else None,
                 confidence=round(float(c), 3) if np.isfinite([x, y]).all() else 0.0, source=s if np.isfinite([x, y]).all() else 'unknown')
            for (x, y), c, s in zip(points, confidence, sources)]


def enforce_ring_order(rings,center,state):
    """Reject crossings introduced when current observations merge with flow.

    Detection paths alone are ordered; historical flow samples need a separate
    check. Discard their confirmation history so they cannot reappear as ghosts.
    """
    center=np.asarray(center)
    rejected=set()
    for inner,outer in zip(rings,rings[1:]):
        for i,(a,b) in enumerate(zip(inner['samples'],outer['samples'])):
            if a['point'] is None or b['point'] is None:continue
            if np.linalg.norm(np.asarray(a['point'])-center)>=np.linalg.norm(np.asarray(b['point'])-center):
                rejected.add((inner['candidate_index'],i))
                rejected.add((outer['candidate_index'],i))
    for ring in rings:
        key=ring['candidate_index']
        for i,sample in enumerate(ring['samples']):
            if (key,i) in rejected:
                ring['samples'][i]=dict(point=None,source='unknown',confidence=0.)
                state[key]['hits'][i]=0
                state[key]['seen'][i]=-np.inf
                state[key]['points'][i]=np.nan
        fraction=sum(p['source']=='observed' for p in ring['samples'])/max(len(ring['samples']),1)
        ring['observed_fraction']=round(fraction,4)
        ring['ring_id']=key if fraction>=.25 else None


def paper_region(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 155), (179, 65, 255))
    mask[:int(h*.28)] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, bw, bh = cv2.boundingRect(contour)
        if not (.012*w*h < area < .65*w*h and bw > .18*w and bh > .04*h):
            continue
        if not (1.4 < bw/max(bh, 1) < 12 and y+bh > .5*h and len(contour) >= 5):
            continue
        ellipse = cv2.fitEllipse(contour)
        (_, cy), (a, b), _ = ellipse
        if min(a, b) < 10 or max(a, b)/min(a, b) > 14:
            continue
        candidates.append((area * (cy/h), contour, ellipse))
    if not candidates:
        return None
    _, contour, ellipse = max(candidates, key=lambda item: item[0])
    # Adapt the paper/soil threshold locally. A permissive global white mask
    # otherwise attaches sunlit, desaturated grass to the paper silhouette.
    x,y,bw,bh = cv2.boundingRect(contour)
    x0,y0,x1,y1 = max(0,x-15),max(0,y-15),min(w,x+bw+15),min(h,y+bh+15)
    local = hsv[y0:y1,x0:x1]
    threshold,_ = cv2.threshold(local[:,:,2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    threshold = int(np.clip(threshold,155,195))
    refined = cv2.inRange(local,(0,0,threshold),(179,65,255))
    refined = cv2.morphologyEx(refined,cv2.MORPH_CLOSE,kernel,iterations=2)
    refined = cv2.morphologyEx(refined,cv2.MORPH_OPEN,kernel)
    outlines,_ = cv2.findContours(refined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if outlines:
        chosen = max(outlines,key=cv2.contourArea)
        if len(chosen)>=5 and cv2.contourArea(chosen)>.5*cv2.contourArea(contour):
            contour = chosen+np.array([[[x0,y0]]],dtype=np.int32)
            ellipse = cv2.fitEllipse(contour)
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, -1)
    return contour, ellipse, filled


def dark_rim_center(gray,ellipse,paper):
    """Fit the actually visible dark rim, retaining only its observed contour.

    Low-threshold crescents survive glare that joins the gray disk interior to
    wrinkles. Several thresholds must independently agree on the fitted center.
    """
    (cx,cy),axes,_ = ellipse
    major,minor = max(axes),min(axes)
    clipped = bool(paper[0].any() or paper[-1].any() or paper[:,0].any() or paper[:,-1].any())
    expected_y = cy if clipped else cy-.12*minor
    candidates=[]
    for threshold in (90,110,130,150):
        mask=((gray<threshold)&(paper>0)).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((1,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x,y,bw,bh=cv2.boundingRect(contour)
            if not (.065*major<bw<.15*major and 2<bh<.22*minor and len(contour)>=12):
                continue
            if abs(x+bw/2-cx)>.16*major or abs(y+bh/2-expected_y)>.25*minor:
                continue
            area=cv2.contourArea(contour)
            if area<max(12,.0003*major*minor):continue
            fitted=cv2.fitEllipse(contour)
            point=np.array(fitted[0]);pa,pb=sorted(fitted[1])
            if not (.045*minor<pa<.23*minor and .80*bw<pb<1.3*bw and pb/pa>1.8):continue
            if not (x<=point[0]<=x+bw and y-.2*minor<=point[1]<=y+bh):continue
            location=((point[0]-cx)/(major/2))**2+((point[1]-expected_y)/(minor/2))**2
            score=1/(1+16*location+8*abs(pb/major-.10))
            candidates.append((score,point,contour,threshold))
    if not candidates:return None
    groups=[]
    for candidate in candidates:
        nearby=[v for v in candidates if np.linalg.norm(v[1]-candidate[1])<5]
        if len(set(v[3] for v in nearby))>=2:
            groups.append((candidate[0],nearby))
    if not groups:return None
    group=max(groups,key=lambda v:v[0])[1]
    point=np.median([v[1] for v in group],axis=0)
    selected=min(group,key=lambda v:np.linalg.norm(v[1]-point))
    return selected[1],selected[2],'dark_rim_fit'


def center_region(gray, ellipse, paper):
    """Filled dark patch, not the midpoint of the paper's outer ellipse."""
    rim = dark_rim_center(gray,ellipse,paper)
    if rim is not None:
        return rim
    (cx, cy), (a, b), angle = ellipse
    theta = np.deg2rad(angle)
    yy, xx = np.indices(gray.shape)
    u = ((xx-cx)*np.cos(theta)+(yy-cy)*np.sin(theta))/(a/2)
    v = (-(xx-cx)*np.sin(theta)+(yy-cy)*np.cos(theta))/(b/2)
    roi = (u*u+v*v < .30) & (paper > 0)
    values = gray[roi]
    if len(values) < 20:
        return None
    background_level = float(np.percentile(values, 70))
    candidates = []
    # For a complete ground target, perspective expands the near half: the
    # printed center is at/above the silhouette midpoint. Disable this prior
    # for clipped paper, where the fitted silhouette center is biased.
    clipped = bool(paper[0].any() or paper[-1].any() or paper[:,0].any() or paper[:,-1].any())
    disk_area = np.pi*a*b/4
    # A broad opening disconnects the thin center cable without erasing the disk.
    kw = max(7, int(max(a,b)*.028)//2*2+1)
    kh = max(3, int(min(a,b)*.045)//2*2+1)
    kernels = [(kw,kh),(max(kw,int(max(a,b)*.052)//2*2+1),max(kh,int(min(a,b)*.075)//2*2+1))]
    for kernel_size in kernels:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
        for threshold in np.arange(max(80,background_level-75),min(210,background_level-8),5):
            dark = ((gray < threshold) & roi).astype(np.uint8)*255
            dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
            dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if not max(18,disk_area*.003) < area < disk_area*.04 or len(contour)<5:
                    continue
                moments = cv2.moments(contour)
                x,y = moments['m10']/moments['m00'],moments['m01']/moments['m00']
                if not clipped and y > cy+min(a,b)*.04:
                    continue
                solidity = area/max(cv2.contourArea(cv2.convexHull(contour)),1)
                patch_ellipse = cv2.fitEllipse(contour)
                pa,pb = patch_ellipse[1]
                fill = area/max(np.pi*pa*pb/4,1)
                if solidity<.80 or not .75<fill<1.2:
                    continue
                distance = float(u[int(y),int(x)]**2+v[int(y),int(x)]**2)
                if distance > .15:
                    continue
                expected_y = cy if clipped else cy-min(a,b)*.12
                location = ((x-cx)/(max(a,b)/2))**2+((y-expected_y)/(min(a,b)/2))**2
                score = np.sqrt(area)*solidity**3/(1+24*location)
                candidates.append((score,np.array([x,y]),contour,area))
    if not candidates:
        return None
    best = max(candidates,key=lambda item:item[0])
    nearby = [item for item in candidates if np.linalg.norm(item[1]-best[1])<max(6,min(a,b)*.09)
              and .65<item[3]/best[3]<1.5]
    point = np.median([item[1] for item in nearby],axis=0)
    boundary = min(nearby,key=lambda item:np.linalg.norm(item[1]-point))[2]
    # Use the same contour for center and boundary; no ellipse midpoint substitution.
    moments = cv2.moments(boundary)
    point = np.array([moments['m10']/moments['m00'],moments['m01']/moments['m00']])
    return point,boundary,'filled_contour'



class TargetTracker:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.reset()

    def reset(self):
        self.motion = None
        self.previous_paper = None
        self.previous_gray = None
        self.previous_time = None
        self.previous_center = None
        self.previous_ellipse = None
        self.center_seen = -np.inf
        self.center_hits = 0
        self.ring_state = {}
        self.target_hits = 0

    def _camera_motion(self,gray):
        self.motion = None
        if self.previous_gray is None or self.previous_paper is None:
            return
        features = cv2.goodFeaturesToTrack(self.previous_gray,160,.02,10,mask=self.previous_paper)
        if features is None or len(features)<8:
            return
        points = features[:,0,:]
        predicted = self._flow(gray,points)
        good = np.isfinite(predicted).all(axis=1)
        if good.sum()<8:
            return
        motion,inliers = cv2.estimateAffinePartial2D(points[good],predicted[good],method=cv2.RANSAC,ransacReprojThreshold=2.5)
        if motion is not None and inliers.sum()>=8 and inliers.mean()>.55:
            self.motion = motion

    def _flow(self, gray, points):
        points = np.asarray(points, np.float32).reshape(-1, 2)
        output = np.full_like(points, np.nan)
        valid = np.isfinite(points).all(axis=1)
        if self.previous_gray is None or not valid.any():
            return output
        p = points[valid].reshape(-1, 1, 2)
        q, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, p, None,
                                             winSize=(21,21), maxLevel=3)
        if q is None:
            return output
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, q, None,
                                                     winSize=(21,21), maxLevel=3)
        if back is None:
            return output
        good = (status.ravel()>0) & (status_back.ravel()>0) & (np.linalg.norm(back-p, axis=2).ravel()<1.5)
        h,w = gray.shape
        q = q.reshape(-1,2)
        good &= (q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
        if self.motion is not None:
            expected = points[valid] @ self.motion[:,:2].T+self.motion[:,2]
            good &= np.linalg.norm(q-expected,axis=1)<4.0
        output[np.flatnonzero(valid)[good]] = q[good]
        return output

    def _rings(self, gray, paper, ellipse, center):
        cfg = self.config
        n, nr = cfg.angles, cfg.radial_samples
        angles = np.arange(n)*2*np.pi/n
        (_, _), (a,b), rotation = ellipse
        theta = np.deg2rad(rotation)
        basis = np.array([[np.cos(theta)*a/2, -np.sin(theta)*b/2],
                          [np.sin(theta)*a/2, np.cos(theta)*b/2]])
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        displacement = np.linalg.solve(basis,center-np.array(ellipse[0]))
        projection = directions @ displacement
        reach = -projection+np.sqrt(np.maximum(.01,projection**2+1-displacement@displacement))
        rays = (directions @ basis.T)*reach[:,None]
        radius = np.linspace(0,1.16,nr)
        grid = center[None,None,:]+rays[:,None,:]*radius[None,:,None]
        mx, my = grid[:,:,0].astype(np.float32), grid[:,:,1].astype(np.float32)
        # A local background subtraction removes broad illumination gradients.
        background = cv2.GaussianBlur(gray, (0,0), 5).astype(np.float32)
        normalized = gray.astype(np.float32)-background
        polar = cv2.remap(normalized,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        allowed = cv2.remap(paper,mx,my,cv2.INTER_NEAREST)>0
        saturation_mask = cv2.inRange(self.current_hsv, (0, 0, 0), (179, 100, 255))
        allowed &= cv2.remap(saturation_mask,mx,my,cv2.INTER_NEAREST)>0
        # Radial two-sided ridge evidence rejects broad shadows and silhouette edges.
        step = max(2, round(nr*.009))
        left, right = np.roll(polar,step,axis=1), np.roll(polar,-step,axis=1)
        ridge = np.minimum(left-polar,right-polar)
        ridge[:,:step] = 0; ridge[:,-step:] = 0
        ridge[~allowed] = -100
        ridge = cv2.GaussianBlur(ridge,(1,3),0)
        output = []
        # Ring 1 is the central disk boundary; outer rim is a separate observation.
        for ring_id in range(2,cfg.ring_count):
            expected = ring_id/cfg.ring_count
            lo = max(0, int((expected-.065)/1.16*(nr-1)))
            hi = min(nr, int((expected+.065)/1.16*(nr-1))+1)
            score = ridge[:,lo:hi].copy()
            score -= 12*((radius[lo:hi]-expected)/.065)**2
            old = self.ring_state.get(ring_id)
            predicted = None
            if old is not None:
                predicted = self._flow(gray, old['points'])
                local = (predicted-center) @ np.linalg.inv(basis).T
                old_radius = np.linalg.norm(local,axis=1)/reach
                ok = np.isfinite(old_radius)
                score[ok] -= np.minimum(10, ((radius[None,lo:hi]-old_radius[ok,None])/.025)**2)
            path = cyclic_path(score)+lo
            row = np.arange(n)
            # Subpixel quadratic interpolation of the image response, not the prior.
            prev = ridge[row,np.maximum(path-1,0)]
            cur = ridge[row,path]
            nxt = ridge[row,np.minimum(path+1,nr-1)]
            denom = prev-2*cur+nxt
            offset = np.clip(np.divide(.5*(prev-nxt),denom,out=np.zeros(n),where=np.abs(denom)>1e-5),-.5,.5)
            rr = (path+offset)/(nr-1)*1.16
            points = center+rays*rr[:,None]
            evidence = cur
            observable = (evidence>=cfg.min_contrast) & allowed[row,path]
            observable &= (path>lo+1)&(path<hi-2)
            # Require neighboring angular support; isolated glyph strokes disappear.
            support = sum(np.roll(observable,k) for k in range(-2,3))
            observable &= support>=3
            confidence = np.clip((evidence-3)/24,0,1)
            output.append((ring_id,points,observable,confidence,predicted,rr))
        # Reject local ring crossings; never repair them with fabricated circles.
        for i in range(1,len(output)):
            bad = output[i][5] <= output[i-1][5]+.015
            output[i][2][bad] = False
            output[i-1][2][bad] = False
        return output

    def update(self, frame: np.ndarray, frame_index: int, timestamp: float) -> dict:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.current_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        h,w = gray.shape
        if self.previous_gray is not None:
            discontinuity = (gray.shape != self.previous_gray.shape or timestamp<=self.previous_time or timestamp-self.previous_time>.6)
            if not discontinuity:
                discontinuity = np.mean(cv2.absdiff(cv2.resize(gray,(64,36)),cv2.resize(self.previous_gray,(64,36))))>48
            if discontinuity:
                self.reset()
        self._camera_motion(gray)
        result = dict(frame_index=int(frame_index),timestamp=round(float(timestamp),6),image_size=[w,h],
                      status='searching',center=dict(point=None,source='unknown',confidence=0.0),
                      outer_boundary=[],bullseye_boundary=[],rings=[])
        region = paper_region(frame)
        center = None
        ellipse = None
        if region is not None:
            contour, ellipse, paper = region
            if self.previous_ellipse is not None:
                delta = np.linalg.norm(np.array(ellipse[0])-self.previous_ellipse[0])
                if delta > max(ellipse[1])*.25:
                    self.reset()
            self.target_hits += 1
            patch = center_region(gray,ellipse,paper)
            predicted_center = self._flow(gray,[self.previous_center])[0] if self.previous_center is not None else np.array([np.nan,np.nan])
            if patch is not None:
                candidate, boundary, method = patch
                agrees = not np.isfinite(predicted_center).all() or np.linalg.norm(candidate-predicted_center)<max(5,min(ellipse[1])*.12)
                self.center_hits = self.center_hits+1 if agrees else 1
                if not agrees:
                    self.ring_state.clear()
                    self.center_seen = -np.inf
                center = candidate
                if self.center_hits>=self.config.confirm_frames:
                    self.center_seen = timestamp
                    result['center'] = dict(point=candidate.tolist(),source='observed',confidence=.75 if method=='dark_rim_fit' else .85,method=method)
                    result['bullseye_boundary'] = samples(boundary[:,0,:])
            else:
                self.center_hits = 0
            if result['center']['point'] is None and self.motion is not None and timestamp-self.center_seen<=self.config.prediction_seconds and np.isfinite(predicted_center).all():
                center = predicted_center
                result['center'] = dict(point=center.tolist(),source='predicted',confidence=.5*(1-(timestamp-self.center_seen)/max(self.config.prediction_seconds,1e-6)))
            if self.target_hits>=self.config.confirm_frames:
                boundary = contour[:,0,:][::max(1,len(contour)//480)].astype(float)
                # Image borders are clipping boundaries, not paper edges.
                clipped = (boundary[:,0]<=1)|(boundary[:,0]>=w-2)|(boundary[:,1]<=1)|(boundary[:,1]>=h-2)
                (ex,ey),(ea,eb),angle = ellipse
                theta = np.deg2rad(angle)
                dx,dy = boundary[:,0]-ex,boundary[:,1]-ey
                radial = ((dx*np.cos(theta)+dy*np.sin(theta))/(ea/2))**2+((-dx*np.sin(theta)+dy*np.cos(theta))/(eb/2))**2
                clipped |= (radial>1.30)|(radial<.65)
                boundary[clipped] = np.nan
                result['outer_boundary'] = samples(boundary)
            if center is not None and result['center']['point'] is not None:
                for ring_id,points,visible,conf,predicted,rr in self._rings(gray,paper,ellipse,center):
                    old = self.ring_state.get(ring_id)
                    n = len(points)
                    hits = np.zeros(n,int)
                    seen = np.full(n,-np.inf)
                    if old is not None:
                        distance = np.linalg.norm(predicted-points,axis=1)
                        agreement = distance < 4.0
                        hits = np.where(visible & agreement,old['hits']+1,0)
                        seen = old['seen'].copy()
                    hits = np.where(visible,np.maximum(hits,1),0)
                    observed = visible & (hits>=self.config.confirm_frames)
                    stored = points.copy()  # candidates are retained only for confirmation
                    sources = np.full(n,'unknown',dtype=object)
                    exported = np.full((n,2),np.nan)
                    exported[observed] = points[observed]
                    sources[observed] = 'observed'
                    seen[observed] = timestamp
                    confidence = np.where(observed,conf,0)
                    if old is not None:
                        estimate = (~observed)&np.isfinite(predicted).all(axis=1)&(timestamp-seen<=self.config.prediction_seconds)&(self.motion is not None)
                        # Do not extend a prediction through conflicting strong evidence.
                        estimate &= (~visible)|(np.linalg.norm(predicted-points,axis=1)<4)
                        exported[estimate] = predicted[estimate]
                        stored[estimate] = predicted[estimate]
                        sources[estimate] = 'predicted'
                        confidence[estimate] = .45*np.maximum(0,1-(timestamp-seen[estimate])/max(self.config.prediction_seconds,1e-6))
                    self.ring_state[ring_id] = dict(points=stored,hits=hits,seen=seen)
                    # Numbering is tentative unless enough of the ordered ring is visible.
                    confirmed_id = ring_id if np.mean(observed) >= .25 else None
                    result['rings'].append(dict(ring_id=confirmed_id,candidate_index=ring_id,
                                               observed_fraction=round(float(np.mean(observed)),4),
                                               samples=samples(exported,confidence,sources)))
                enforce_ring_order(result['rings'],center,self.ring_state)
                result['status'] = 'tracking' if result['center']['source']=='observed' else 'predicting'
            if result['center']['point'] is None:
                self.ring_state.clear()
            self.previous_center = center
            self.previous_ellipse = ellipse
            self.previous_paper = paper
        else:
            # No paper support: abandon predictions rather than drifting onto a person.
            self.target_hits = 0
            self.previous_paper = None
            self.center_hits = 0
            self.previous_center = None
            self.previous_ellipse = None
            self.ring_state.clear()
            self.center_seen = -np.inf
        self.previous_gray = gray
        self.previous_time = timestamp
        return result
