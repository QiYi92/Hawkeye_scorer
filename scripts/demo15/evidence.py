"""Demo15-only, annotation-free camera, athlete and shoe observations."""
from itertools import permutations

import cv2
import numpy as np

from demo13.tracking import Instance, box_iou
from demo13.contact import foot_bottom_points


class CameraRegistration:
    """A moving camera is usable when background registration is reliable.

    Failed links start a new coordinate segment: no comparison across a gap
    silently treats an unknown transform as identity.
    """
    def __init__(self):
        self.previous = self.exclusion = None
        self.transform = np.eye(3)
        self.segment = 0

    def update(self, frame, excluded):
        h, w = frame.shape[:2]
        size = (640, round(h * 640 / w))
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size)
        mask = cv2.resize(excluded.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
        valid = self.previous is None
        count, error, moved = 0, None, False
        if self.previous is not None:
            points = cv2.goodFeaturesToTrack(self.previous, 400, .01, 8,
                                            mask=((mask == 0) & (self.exclusion == 0)).astype(np.uint8)*255)
            if points is not None and len(points) >= 16:
                forward, good, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None)
                backward, back_good, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, forward, None)
                keep = (good.ravel() > 0) & (back_good.ravel() > 0)
                keep &= np.linalg.norm((backward-points)[:, 0], axis=1) < 1.
                if keep.sum() >= 16:
                    a, inliers = cv2.estimateAffinePartial2D(points[keep], forward[keep],
                                                           method=cv2.RANSAC, ransacReprojThreshold=1.)
                    count = int(inliers.sum()) if inliers is not None else 0
                    if a is not None and count >= max(16, .65*keep.sum()):
                        residual = cv2.transform(points[keep], a)-forward[keep]
                        error = float(np.median(np.linalg.norm(residual[:, 0], axis=1)[inliers.ravel() > 0]))
                        scale = float(np.hypot(a[0, 0], a[1, 0]))
                        valid = bool(np.isfinite(a).all() and .95 <= scale <= 1.05 and error < .75)
                        if valid:
                            s = np.diag([size[0]/w, size[1]/h, 1.])
                            step = np.linalg.inv(s) @ np.vstack([a, [0., 0., 1.]]) @ s
                            self.transform = step @ self.transform
                            moved = bool(np.linalg.norm(step[:2, 2]) > 1.)
            if not valid:
                self.segment += 1
                self.transform = np.eye(3)
        self.previous, self.exclusion = gray, mask
        return dict(camera_stable=valid, camera_moved=moved,
                    camera_transform=self.transform.tolist(), camera_segment=self.segment,
                    registration_quality=dict(valid=valid, inliers=count, median_error_640=error))


def pack_instance(instance, frame):
    from demo13.tracking import DescendingAthleteTrack
    return dict(track_id=instance.track_id, confidence=instance.confidence,
                bbox=instance.bbox, mask=np.packbits(instance.mask), shape=instance.mask.shape,
                appearance=DescendingAthleteTrack.appearance_of(instance, frame))


def unpack_instance(item, logical_id=None):
    mask = np.unpackbits(item['mask'], count=int(np.prod(item['shape']))).reshape(item['shape']).astype(bool)
    return Instance(item['track_id'] if logical_id is None else logical_id,
                    item['confidence'], item['bbox'], mask, 0, item['track_id'])


def complete_athlete_track(detections, selections, fps):
    """Extend confirmed forward selections backwards using observed detections.

    Same-ID detections still require geometric consistency. Ambiguous matches
    and long gaps stay missing; no synthesized boxes or masks are returned.
    """
    selected = list(selections)
    anchors = [i for i, item in enumerate(selected) if item is not None]
    if not anchors:
        return selected
    for direction, start, stop in ((-1, anchors[-1], -1), (1, anchors[0], len(selected))):
        last, last_i, velocity = None, None, np.zeros(2)
        for i in range(start, stop, direction):
            item = selected[i]
            if item is None and last is not None and abs(i-last_i) <= round(fps*.4):
                box = np.asarray(last['bbox'], float)
                predicted = box + np.tile(velocity*(i-last_i), 2)
                size = max(1., max(box[2:]-box[:2]))
                scores = []
                for candidate in detections[i]:
                    b = np.asarray(candidate['bbox'], float)
                    ratio = np.prod(b[2:]-b[:2])/max(1., np.prod(box[2:]-box[:2]))
                    distance = np.linalg.norm((b[:2]+b[2:]-predicted[:2]-predicted[2:])/2)/size
                    overlap = box_iou(b, predicted)
                    appearance = cv2.compareHist(last['appearance'], candidate['appearance'], cv2.HISTCMP_BHATTACHARYYA)
                    if .4 <= ratio <= 2.5 and distance <= .55 and overlap >= .15 and appearance < .7:
                        scores.append((overlap-.3*distance-.2*appearance + .15*(candidate['track_id']==last['track_id']), candidate))
                scores.sort(key=lambda pair: pair[0], reverse=True)
                if scores and (len(scores)==1 or scores[0][0]-scores[1][0] > .12):
                    item = selected[i] = scores[0][1]
            if item is not None:
                if last is not None:
                    a, b = np.asarray(last['bbox']), np.asarray(item['bbox'])
                    velocity = ((b[:2]+b[2:]-a[:2]-a[2:])/2)/(i-last_i)
                last, last_i = item, i
    return selected


def parse_athlete(frame, person, parser, refine=False):
    """Parse actual RGB beyond the coarse person-mask edge, with bounded support."""
    h, w = frame.shape[:2]
    x, y, bw, bh = cv2.boundingRect(person.mask.astype(np.uint8))
    pad = max(8, round(max(bw, bh)*.12))
    x1, y1, x2, y2 = max(0,x-pad), max(0,y-pad), min(w,x+bw+pad), min(h,y+bh+pad)
    labels = np.zeros((h,w), np.uint8)
    labels[y1:y2,x1:x2] = parser.parse(frame[y1:y2,x1:x2])
    # Keep shoe pixels near the athlete, including feet missed by coarse YOLO.
    radius = max(5, round(bh*.045))
    allowed = cv2.dilate(person.mask.astype(np.uint8), np.ones((2*radius+1,2*radius+1),np.uint8)) > 0
    labels[~allowed] = 0
    if refine:
        # A second, closer view retains torso/leg context while allocating
        # more network pixels to footwear. Require agreement with the full
        # person parse so printed targets/shadows do not become new shoes.
        ya=max(0,y+round(bh*.25)-15)
        xa,xb=max(0,x-30),min(w,x+bw+30)
        yb=min(h,y+bh+25)
        close=np.zeros((h,w),np.uint8)
        close[ya:yb,xa:xb]=parser.parse(frame[ya:yb,xa:xb])
        labels[(labels==2)&(close!=2)]=0
    return labels


def refinement_frames(rows):
    """Loose motion proposals only; never use reference frames to select ROIs."""
    associate_feet(rows)
    tracks={};selected=set()
    for row in rows:
        for foot in row['feet']:tracks.setdefault(foot['foot_track_id'],[]).append((row,foot))
    for track in tracks.values():
        for j in range(2,len(track)-2):
            row,foot=track[j];before=track[j-2];after=track[j+2]
            if row['frame_id']-before[0]['frame_id']>4 or after[0]['frame_id']-row['frame_id']>4:continue
            y=foot['stabilized_point'][1]
            descent=y-before[1]['stabilized_point'][1]
            support=after[1]['stabilized_point'][1]-y
            if descent>6*row['image_size'][1]/720 and support<descent*.6:
                selected.update(range(max(0,row['frame_id']-3),min(len(rows),row['frame_id']+5)))
    return selected


def add_refined_points(feet, refined):
    """Refine location independently of the coarse trajectory used for timing."""
    used=set()
    for foot in feet:
        matches=[]
        for j,fine in enumerate(refined):
            if j in used:continue
            overlap=box_iou(np.asarray(foot['bbox'],float),np.asarray(fine['bbox'],float))
            distance=np.linalg.norm(np.asarray(foot['point'])-fine['point'])
            if overlap>=.2 and distance<=max(12.,(foot['bbox'][2]-foot['bbox'][0])*.4):
                matches.append((overlap,j,fine))
        if matches:
            _,j,fine=max(matches,key=lambda item:item[0]);used.add(j)
            foot['contact_point']=fine['contact_point']
            foot['refined_sole_boundary']=fine['sole_boundary']
            foot['contact_point_source']='two_scale_rgb_agreement'


def contact_boundary_point(boundary, fallback):
    """Middle of the lowest supported horizontal sole run, not a lone pixel.

    The coarse trajectory tolerates the bottom few rows to suppress jitter.
    A contact location instead belongs to the actual lower contour; averaging
    the entire bottom band can move it sideways on a tall, rotating boot.
    """
    for y in sorted({p[1] for p in boundary},reverse=True):
        xs=sorted(p[0] for p in boundary if p[1]==y)
        runs=[]
        for x in xs:
            if not runs or x!=runs[-1][-1]+1:runs.append([x])
            else:runs[-1].append(x)
        runs=[r for r in runs if len(r)>=3]
        if runs:
            run=max(runs,key=len)
            return [run[len(run)//2],y]
    return list(fallback)


def shoe_observations(labels):
    feet = foot_bottom_points(labels)
    for foot in feet:
        x1,y1,x2,y2=foot['bbox']
        shoe=(labels[y1:y2,x1:x2]==2).astype(np.uint8)
        # Retain one lower-envelope pixel per occupied column, in native pixels.
        boundary=[]
        for x in range(shoe.shape[1]):
            ys=np.flatnonzero(shoe[:,x])
            if len(ys):boundary.append([int(x+x1),int(ys[-1]+y1)])
        foot.update(sole_boundary=boundary, observation_source='inferred',
                    contact_point=contact_boundary_point(boundary,foot['point']),
                    contact_point_source='observed_sole_run',
                    observation_quality=float(min(1.,foot['area']/max(1,(x2-x1)*(y2-y1)))))
    return feet


def stabilized(point, row):
    value = np.linalg.inv(np.asarray(row['camera_transform'])) @ np.array([*point,1.])
    return value[:2]/value[2]


def associate_feet(rows, max_gap=3):
    """Two-shoe global assignment with velocity, overlap and scale evidence."""
    active, next_id = {}, 0
    for row in rows:
        i=row['frame_id'];feet=row['feet']
        active={k:v for k,v in active.items() if i-v['frame']<=max_gap and v['segment']==row.get('camera_segment',0)}
        ids=list(active)
        costs={}
        for j,foot in enumerate(feet):
            p=stabilized(foot['point'],row)
            foot['stabilized_point']=p.tolist()
            for k in ids:
                old=active[k];gap=i-old['frame']
                predicted=old['point']+old['velocity']*gap
                scale=max(12.,old['width'])
                distance=np.linalg.norm(p-predicted)/scale
                width=max(1.,foot['bbox'][2]-foot['bbox'][0])
                size_error=abs(np.log(width/scale))
                # Shoe rotation/partial occlusion can change area considerably.
                delta=p-old['point']
                shifted=np.asarray(old['bbox'])+np.tile(delta,2)
                overlap=box_iou(np.asarray(foot['bbox'],float),shifted)
                if distance<1.35:
                    costs[j,k]=distance+.15*size_error+.2*(1-overlap)
        best=(float('inf'),None)
        # Add a distinct dummy ID for each observation; unmatched cost is bounded.
        for assignment in permutations(ids+list(range(-len(feet),0)),len(feet)):
            cost=sum(1.5 if k<0 else costs.get((j,k),1e6) for j,k in enumerate(assignment))
            if cost<best[0]:best=cost,assignment
        for j,foot in enumerate(feet):
            k=best[1][j]
            if k<0:k=next_id;next_id+=1
            p=np.asarray(foot['stabilized_point']);old=active.get(k)
            velocity=(p-old['point'])/(i-old['frame']) if old else np.zeros(2)
            if old is not None:velocity=.7*velocity+.3*old['velocity']
            foot['foot_track_id']=k
            active[k]=dict(point=p,velocity=velocity,frame=i,bbox=foot['bbox'],
                           width=foot['bbox'][2]-foot['bbox'][0],segment=row.get('camera_segment',0))
