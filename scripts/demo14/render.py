"""Draw only exported coordinates so preview and JSONL cannot disagree."""
import cv2
import numpy as np


def draw_samples(frame, entries, color, thickness=1, closed=True):
    for i in range(len(entries) if closed else max(0,len(entries)-1)):
        a,b = entries[i], entries[(i+1)%len(entries)]
        if a['point'] is None or b['point'] is None:
            continue
        predicted = a['source']=='predicted' or b['source']=='predicted'
        if predicted and i%6>=3:
            continue
        p,q = np.rint(a['point']).astype(int),np.rint(b['point']).astype(int)
        # A discontinuous seam is never connected across the target.
        if np.linalg.norm(p-q)>max(12,frame.shape[1]*.025):
            continue
        cv2.line(frame,tuple(p),tuple(q),(0,165,255) if predicted else color,thickness,cv2.LINE_AA)


def annotate(frame,result,process_fps,overlay=True,inset=True):
    out = frame.copy()
    if overlay:
        draw_samples(out,result['outer_boundary'],(50,255,90),2)
        draw_samples(out,result['bullseye_boundary'],(0,230,255),1)
        for ring in result['rings']:
            draw_samples(out,ring['samples'],(255,240,20))
            if ring['ring_id'] is not None:
                candidates = [s['point'] for s in ring['samples'] if s['source']=='observed' and s['point'] is not None]
                if candidates:
                    p = min(candidates,key=lambda p:p[0])
                    cv2.putText(out,str(ring['ring_id']),tuple(np.rint(p).astype(int)),0,.4,(20,20,255),1,cv2.LINE_AA)
        center = result['center']
        if center['point'] is not None:
            p = tuple(np.rint(center['point']).astype(int))
            color = (0,0,255) if center['source']=='observed' else (0,165,255)
            cv2.drawMarker(out,p,color,cv2.MARKER_CROSS,16,2)
        if inset and result['outer_boundary']:
            pts = np.array([s['point'] for s in result['outer_boundary'] if s['point'] is not None])
            if len(pts):
                h,w = out.shape[:2]
                x0,y0 = np.maximum(pts.min(axis=0).astype(int)-8,0)
                x1,y1 = np.minimum(pts.max(axis=0).astype(int)+9,[w,h])
                focus = result['center']['point']
                if focus is not None:
                    fx,fy = np.rint(focus).astype(int)
                    x0,x1 = max(x0,fx-160),min(x1,fx+160)
                    y0,y1 = max(y0,fy-60),min(y1,fy+60)
                crop = out[y0:y1,x0:x1].copy()
                available_h = min(h//3,h-72)
                if crop.size and available_h>0:
                    scale = min(min(w//2,640)/crop.shape[1],available_h/crop.shape[0])
                    iw,ih = max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))
                    enlarged = cv2.resize(crop,(iw,ih),interpolation=cv2.INTER_CUBIC)
                    out[72:72+ih,w-iw:] = enlarged
                    cv2.rectangle(out,(w-iw,72),(w-1,71+ih),(180,180,180),1)
                    cv2.putText(out,f'DETAIL {scale:.1f}x',(w-iw+5,88),0,.4,(0,0,255),1,cv2.LINE_AA)
    c = result['center']['point']
    xy = 'unknown' if c is None else f'({c[0]:.1f}, {c[1]:.1f}) {result["center"]["source"]}'
    lines = [f'Demo14 | {result["status"]} | frame {result["frame_index"]} | {process_fps:.1f} process fps | center {xy}',
             'solid: image-supported | orange dashed: predicted | Space pause / N step / R restart / O overlay / Q quit']
    for i,line in enumerate(lines):
        cv2.putText(out,line,(10,24+i*24),0,.48,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(out,line,(10,24+i*24),0,.48,(255,255,255),1,cv2.LINE_AA)
    return out
