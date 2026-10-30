"""Bounded RGB re-observation after short shoe segmentation gaps.

Predictions propose a crop only. Every returned pixel must be classified as
shoe in the current RGB image and supported by the athlete mask and appearance.
This is basic contact evidence, independent of precise contact-point refinement.
"""
import cv2
import numpy as np

from .evidence import MAX_SHOE_FRAME_GAP, shoe_observations, shoe_appearance


class ShoeRecovery:
    def __init__(self):
        self.last = None
        self.calls = self.recovered = 0

    def update(self, frame, labels, person, parser, frame_id, camera):
        result = dict(attempted=False, recovered_shoes=0, recovered_boxes=[])
        if person is None or labels is None or not camera['camera_stable']:
            self.last = None
            return labels, result
        current = shoe_observations(labels)
        box = np.asarray(person.bbox, float)
        if current:
            # Refresh from the ordinary parser only; repeated recovery cannot
            # extend the two-frame missing-observation budget indefinitely.
            for foot in current:
                foot['appearance'] = shoe_appearance(frame, labels, foot)
            self.last = dict(frame=frame_id, feet=current, box=box,
                             segment=camera.get('camera_segment', 0))
            return labels, result
        old = self.last
        if (old is None or not 1 <= frame_id-old['frame'] < MAX_SHOE_FRAME_GAP or
                old['segment'] != camera.get('camera_segment', 0)):
            return labels, result
        delta = (box[:2]+box[2:]-old['box'][:2]-old['box'][2:])/2
        ratio = (box[3]-box[1])/max(1., old['box'][3]-old['box'][1])
        if not .65 <= ratio <= 1.5:
            return labels, result
        h,w = labels.shape
        height = box[3]-box[1]
        padding = max(12., height*.12)
        x1 = max(0, int(box[0]-padding)); x2 = min(w, int(np.ceil(box[2]+padding)))
        y1 = max(0, int(box[1]+height*.22)); y2 = min(h, int(np.ceil(box[3]+padding)))
        if x2-x1 < 12 or y2-y1 < 12:
            return labels, result
        result['attempted'] = True; self.calls += 1
        close = np.zeros_like(labels)
        close[y1:y2,x1:x2] = parser.parse(frame[y1:y2,x1:x2])
        radius = max(5, round(height*.06))
        allowed = cv2.dilate(person.mask.astype(np.uint8),
                            np.ones((2*radius+1, 2*radius+1), np.uint8)) > 0
        close[~allowed] = 0
        proposals = shoe_observations(close)
        used = set()
        recovered = []
        for candidate in proposals:
            appearance = shoe_appearance(frame, close, candidate)
            matches = []
            for index, previous in enumerate(old['feet']):
                if index in used:
                    continue
                predicted = np.asarray(previous['point'])+delta
                width = max(12., previous['bbox'][2]-previous['bbox'][0])
                distance = np.linalg.norm(np.asarray(candidate['point'])-predicted)
                color_error = cv2.compareHist(np.asarray(appearance,np.float32),
                    np.asarray(previous['appearance'],np.float32),cv2.HISTCMP_BHATTACHARYYA)
                if (distance <= max(width*1.5, height*.12) and color_error < .6 and
                        .2*previous['area'] <= candidate['area'] <= 4*previous['area']):
                    matches.append((distance/width+color_error,index))
            if not matches:
                continue
            _, index = min(matches); used.add(index)
            a,b,c,d = candidate['bbox']
            recovered.append(candidate['bbox'])
            # Copy only observed class-2 pixels; never interpolate a shoe mask.
            patch = labels[b:d,a:c].copy()
            patch[close[b:d,a:c] == 2] = 2
            labels[b:d,a:c] = patch
        self.recovered += len(recovered)
        result.update(recovered_shoes=len(recovered), recovered_boxes=recovered)
        return labels, result
