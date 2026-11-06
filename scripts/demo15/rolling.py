"""Region-supported contact evidence when a sole point switches toe/heel."""
import cv2
import numpy as np
from .evidence import stabilized
from demo13.tracking import box_iou


def sole_level(row,foot):
    boundary=foot.get('sole_boundary',[])
    if len(boundary)<6:return None
    points=np.asarray([stabilized(point,row) for point in boundary])
    return float(np.percentile(points[:,1],85))


def rolling_support(before,after,config):
    """Return measured region support, never widen a point threshold by itself."""
    levels_pre=[sole_level(o,f) for o,f in before]
    levels_post=[sole_level(o,f) for o,f in after]
    if any(level is None for level in levels_pre+levels_post):return None
    t=after[0][0]['timestamp'];scale=after[0][0]['image_size'][1]/720
    pre_t=np.array([o['timestamp']-t for o,f in before]);post_t=np.array([o['timestamp']-t for o,f in after])
    pre=np.asarray(levels_pre);post=np.asarray(levels_post)
    descent=float(np.polyfit(pre_t,pre,1)[0]);support=float(np.polyfit(post_t,post,1)[0])
    width=float(np.median([f['bbox'][2]-f['bbox'][0] for o,f in after]))
    height=float(np.median([f['bbox'][3]-f['bbox'][1] for o,f in after]))
    if (pre[-1]-pre[0]<config.descent_px_720*scale or descent<30*scale or
            support>descent*.35 or np.ptp(post)>max(config.motion_tolerance_720*scale,height*.35)):
        return None
    overlaps=[];centers=[]
    for row,foot in after:
        box=np.asarray(foot['bbox'],float);centers.append(stabilized((box[:2]+box[2:])/2,row))
    for (a,fa),(b,fb),pa,pb in zip(after,after[1:],centers,centers[1:]):
        shifted=np.asarray(fa['bbox'],float)+np.tile(pb-pa,2)
        overlaps.append(box_iou(shifted,np.asarray(fb['bbox'],float)))
    if not overlaps or np.median(overlaps)<.35:return None
    centers=np.asarray(centers)
    residual=np.max(np.linalg.norm(centers-np.column_stack([
        np.polyval(np.polyfit(post_t,centers[:,axis],1),post_t) for axis in (0,1)]),axis=1))
    if residual>max(4*scale,width*.35):return None
    pre_x=np.array([stabilized(((np.asarray(f['bbox'])[:2]+np.asarray(f['bbox'])[2:])/2),o)[0] for o,f in before])
    old_v=float(np.polyfit(pre_t,pre_x,1)[0]);new_v=float(np.polyfit(post_t,centers[:,0],1)[0])
    deceleration=abs(old_v)>30*scale and abs(new_v)<abs(old_v)*.65
    return dict(pre_y=pre,post_y=post,descent=descent,support=support,
                fit_error=float(np.mean((np.polyval(np.polyfit(pre_t,pre,1),pre_t)-pre)**2)+
                                np.mean((np.polyval(np.polyfit(post_t,post,1),post_t)-post)**2)),
                median_region_overlap=float(np.median(overlaps)),region_motion_error_px=float(residual),
                tangential_speed_before=old_v,tangential_speed_after=new_v,
                tangential_travel_fraction=float(np.ptp(centers[:,0])/max(1.,width)),
                tangential_deceleration=bool(deceleration))
