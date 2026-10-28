"""Observed paper silhouette classification, independent of center and rings."""
import cv2
import numpy as np


def paper_outline(found, excluded):
    if found is None:
        return dict(points=[], usable=False, reason='paper_missing')
    contour, _, _ = found
    h,w=excluded.shape
    points=contour[:,0,:]
    clipped=bool((points[:,0]<=1).any() or (points[:,0]>=w-2).any() or
                 (points[:,1]<=1).any() or (points[:,1]>=h-2).any())
    near=cv2.dilate(excluded.astype(np.uint8),np.ones((7,7),np.uint8))
    occlusion=float(np.mean(near[points[:,1],points[:,0]]>0))
    return dict(points=points.tolist(),usable=not clipped and occlusion<=.02,
                reason='clipped_or_occluded_outline' if clipped or occlusion>.02 else None,
                boundary_occlusion_fraction=occlusion)


def target_decision(rows, result, foot=None):
    """Require three consistent registered, observed silhouettes.

    A union of these silhouettes protects the early-miss gate against small
    paper motion and contour jitter. A coarse point alone never triggers it.
    """
    unknown=dict(target_hit='unknown',early_miss=False,reason='contact_not_confirmed',outline_frames=[])
    event=result.get('event')
    if not event or result['status']!='confirmed_estimate':return unknown
    n=event['frame_id'];anchor=rows[n]
    if not anchor['camera_stable']:return {**unknown,'reason':'registration_unavailable'}
    w,h=anchor['image_size'];candidates=[]
    small_size=(320,max(1,round(h*320/w)))
    for row in reversed(rows[max(0,n-90):n+1]):
        if anchor['timestamp']-row['timestamp']>.75+1e-6:break
        outline=row.get('paper_outline',{})
        if (not outline.get('usable') or not row['camera_stable'] or
                row.get('camera_segment',0)!=anchor.get('camera_segment',0)):continue
        points=np.asarray(outline['points'],np.float32).reshape(-1,1,2)
        if len(points)<3:continue
        transform=np.asarray(anchor['camera_transform'])@np.linalg.inv(np.asarray(row['camera_transform']))
        polygon=cv2.perspectiveTransform(points,transform)[:,0,:]
        if not np.isfinite(polygon).all():continue
        mask=np.zeros((small_size[1],small_size[0]),np.uint8)
        scaled=polygon*np.array([small_size[0]/w,small_size[1]/h])
        cv2.fillPoly(mask,[np.rint(scaled).astype(np.int32)],1)
        candidates.append((polygon,mask,row['frame_id']))
    groups=[]
    for seed in candidates:
        group=[]
        for candidate in candidates:
            intersection=np.count_nonzero(candidate[1] & seed[1])
            union=np.count_nonzero(candidate[1] | seed[1])
            if intersection/max(1,union)>=.85:group.append(candidate)
        groups.append(group)
    group=max(groups,key=lambda g:(len(g),np.median([cv2.contourArea(c[0]) for c in g])),default=[])
    if len(group)<3:
        return {**unknown,'reason':'insufficient_consistent_outlines','outline_frames':[c[2] for c in group]}
    # Small occlusion notches can survive the boundary-quality gate and
    # still have high IoU. Prefer the three largest *observed* silhouettes in
    # the consistent cluster, rather than connecting across a missing edge.
    selected=sorted(group,key=lambda c:(cv2.contourArea(c[0]),c[2]),reverse=True)[:3]
    polygons=[c[0] for c in selected];frame_ids=[c[2] for c in selected]
    masks=[]
    for polygon in polygons:
        mask=np.zeros((h,w),np.uint8)
        cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
        masks.append(mask)
    distances=[float(cv2.pointPolygonTest(p,tuple(map(float,event['point_px'])),True)) for p in polygons]
    # Signed distance zero is the actual outer line, which counts as a hit.
    hit='hit' if all(d>=-1e-6 for d in distances) else ('miss' if all(d<0 for d in distances) else 'unknown')
    early=False
    if foot is not None and hit=='miss':
        margin=max(1,int(np.ceil(10*h/720)))
        union=np.maximum.reduce(masks)
        safe=cv2.dilate(union,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*margin+1,2*margin+1)))
        x1,y1,x2,y2=map(int,foot['bbox'])
        x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
        early=bool(x2>x1 and y2>y1 and not safe[y1:y2,x1:x2].any())
    return dict(target_hit=hit,early_miss=early,
                reason='whole_shoe_outside_margin' if early else ('boundary_disagreement' if hit=='unknown' else 'observed_outer_boundary'),
                outline_frames=frame_ids,signed_distances_px=distances,margin_px=10*h/720)
