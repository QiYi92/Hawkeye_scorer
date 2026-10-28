"""Ultralytics tracker IDs remain authoritative; no fabricated identity stitching."""
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np


@dataclass
class Instance:
    track_id: int
    confidence: float
    bbox: list
    mask: np.ndarray
    class_id: int
    source_track_id: int | None = None


class YoloSegmenter:
    def __init__(self, model, mode="person", tracker="botsort", conf=.25, imgsz=640, device=""):
        from ultralytics import YOLO
        self.model = YOLO(model)
        if self.model.task != "segment":
            raise ValueError("Demo13 必须使用 segmentation 权重（*-seg.pt），不能使用 pose/detect 权重")
        names = self.model.names
        target = "paraglider_athlete" if mode == "athlete" else "person"
        ids = [int(k) for k, v in names.items() if v == target]
        if len(ids) != 1 or (mode == "athlete" and (ids != [0] or len(names) != 1)):
            raise ValueError(f"模型类别不符合 {mode} 模式：需要 {'唯一 class 0 = ' if mode == 'athlete' else ''}{target}，实际 {names}")
        self.target_name = target
        self.options = dict(persist=True, tracker=f"{tracker}.yaml", classes=ids,
                            conf=conf, imgsz=imgsz, retina_masks=True, verbose=False)
        if device:
            self.options["device"] = device

    def infer(self, frame):
        result = self.model.track(frame, **self.options)[0]
        if result.masks is None or result.boxes.id is None:
            return []
        masks = result.masks.data.cpu().numpy() > .5
        if masks.shape[1:] != frame.shape[:2]:
            raise RuntimeError("retina_masks 未返回原图尺寸；拒绝导出错位 Mask")
        boxes = result.boxes
        return [Instance(int(t), float(c), b.tolist(), m, int(k)) for t, c, b, m, k in zip(
            boxes.id.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy(),
            masks, boxes.cls.cpu().numpy()) if m.any()]


class PrimaryTrack:
    """Confirm continuous observations, then hold the selected ID through short gaps."""
    def __init__(self, confirm_frames=3, lost_buffer=30):
        self.confirm_frames = confirm_frames
        self.lost_buffer = lost_buffer
        self.track_id = None
        self.last_seen = -1
        self.candidates = {}

    @property
    def source_track_id(self):
        return self.track_id

    def update(self, instances, frame_id):
        current = {i.track_id: i for i in instances}
        for i in instances:
            previous = self.candidates.get(i.track_id)
            count = previous[1] + 1 if previous and previous[0] == frame_id - 1 else 1
            self.candidates[i.track_id] = (frame_id, count)
        self.candidates = {k: v for k, v in self.candidates.items() if frame_id - v[0] <= self.lost_buffer}
        if self.track_id in current:
            self.last_seen = frame_id
            return self.track_id, "tracked"
        if self.track_id is not None and frame_id - self.last_seen <= self.lost_buffer:
            return self.track_id, "lost"
        self.track_id = None
        eligible = [i for i in instances if self.candidates[i.track_id][1] >= self.confirm_frames]
        if eligible:
            best = max(eligible, key=lambda i: np.count_nonzero(i.mask) * i.confidence)
            self.track_id, self.last_seen = best.track_id, frame_id
            return self.track_id, "tracked"
        return None, "searching"


def box_iou(a, b):
    extent = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(extent))
    return intersection / max(1., float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection))


class DescendingAthleteTrack:
    """Select one descending arrival, then keep one logical ID for this video.

    Image-plane descent is a heuristic, not ground-contact or 3D flight inference.
    Reassociation is bounded by time, motion, scale, appearance and ambiguity.
    Once the gap expires, remain lost instead of selecting another person/canopy.
    """
    def __init__(self, confirm_frames=3, lost_buffer=30, fps=30.,
                 entry_top=.45, descent_distance=.04, descent_seconds=.2):
        self.confirm_frames, self.lost_buffer, self.fps = confirm_frames, lost_buffer, fps
        self.entry_top, self.descent_distance = entry_top, descent_distance
        self.min_span = max(1, round(descent_seconds * fps))
        self.history = {}
        self.track_id = self.source_track_id = None
        self.last_seen = -1
        self.last_box = self.appearance = None
        self.motion = deque(maxlen=6)
        self.excluded_ids = set()
        self.reassociations = 0
        self.reason = "waiting_for_descent"

    @staticmethod
    def center(box):
        return (box[:2] + box[2:]) / 2

    @staticmethod
    def appearance_of(instance, frame):
        if frame is None:
            return None
        x, y, w, h = cv2.boundingRect(instance.mask.astype(np.uint8))
        hsv = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], instance.mask[y:y+h, x:x+w].astype(np.uint8),
                            [16, 16], [0, 180, 0, 256])
        return cv2.normalize(hist, None, alpha=1., norm_type=cv2.NORM_L1)

    def remember(self, instance, frame_id, frame):
        self.last_box = np.asarray(instance.bbox, dtype=float)
        self.last_seen = frame_id
        self.source_track_id = instance.track_id
        self.motion.append((frame_id, self.center(self.last_box)))
        hist = self.appearance_of(instance, frame)
        if hist is not None:
            self.appearance = hist if self.appearance is None else .8 * self.appearance + .2 * hist

    def descent_score(self, history, image_height):
        if len(history) < max(3, self.confirm_frames):
            return None
        start, end = history[0], history[-1]
        span = end[0] - start[0]
        if span < self.min_span or start[1][1] > image_height * self.entry_top:
            return None
        positions = np.array([self.center(row[1])[1] for row in history])
        net = positions[-1] - positions[0]
        body_height = float(np.median([row[1][3] - row[1][1] for row in history]))
        threshold = max(image_height * self.descent_distance, body_height * .25)
        consistency = float(np.mean(np.diff(positions) > 0))
        if net < threshold or consistency < .65 or net / span * self.fps < image_height * .10:
            return None
        return net / max(1., body_height) * consistency * float(np.mean([row[2] for row in history]))

    def update(self, instances, frame_id, frame=None):
        for item in instances:
            hist = self.history.setdefault(item.track_id, deque(maxlen=max(12, round(self.fps))))
            if hist and frame_id - hist[-1][0] > max(1, self.lost_buffer):
                hist.clear()
            hist.append((frame_id, np.asarray(item.bbox, dtype=float), item.confidence))
        self.history = {k: v for k, v in self.history.items()
                        if frame_id - v[-1][0] <= max(self.lost_buffer, round(self.fps))}
        if self.track_id is None:
            eligible = []
            for item in instances:
                score = self.descent_score(self.history[item.track_id], item.mask.shape[0])
                if score is not None:
                    eligible.append((score, item))
            eligible.sort(key=lambda pair: pair[0], reverse=True)
            if not eligible:
                self.reason = "waiting_for_descent"
                return None, "searching"
            if len(eligible) > 1 and eligible[1][0] >= eligible[0][0] * .85:
                self.reason = "ambiguous_descending_candidates"
                return None, "searching"
            selected = eligible[0][1]
            self.track_id = selected.track_id
            chosen_box = np.asarray(selected.bbox)
            self.excluded_ids = {k for k, hist in self.history.items()
                                 if k != selected.track_id and box_iou(hist[-1][1], chosen_box) < .2}
            for n, box, _ in list(self.history[selected.track_id])[-5:-1]:
                self.motion.append((n, self.center(box)))
            self.remember(selected, frame_id, frame)
            self.reason = "descending_arrival_confirmed"
            return self.track_id, "tracked"

        gap = frame_id - self.last_seen
        # No automatic new target after one athlete has been selected.
        if gap > max(1, self.lost_buffer):
            self.reason = "reacquisition_window_expired"
            return self.track_id, "lost"
        same = next((i for i in instances if i.track_id == self.source_track_id), None)
        if same is not None:
            self.remember(same, frame_id, frame)
            self.reason = "same_tracker_id"
            return self.track_id, "tracked"

        velocities = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(self.motion, list(self.motion)[1:])]
        velocity = np.median(velocities, axis=0) if velocities else np.zeros(2)
        predicted = self.last_box + np.tile(velocity * gap, 2)
        size = max(1., float(max(self.last_box[2:] - self.last_box[:2])))
        scores = []
        for item in instances:
            if item.track_id in self.excluded_ids or len(self.history[item.track_id]) < 2:
                continue
            box = np.asarray(item.bbox)
            ratio = np.prod(box[2:] - box[:2]) / max(1., np.prod(self.last_box[2:] - self.last_box[:2]))
            distance = np.linalg.norm(self.center(box) - self.center(predicted)) / size
            overlap = box_iou(box, predicted)
            if not .45 <= ratio <= 2.2 or distance > .65 or overlap < .15:
                continue
            hist = self.appearance_of(item, frame)
            appearance_distance = (cv2.compareHist(self.appearance, hist, cv2.HISTCMP_BHATTACHARYYA)
                                   if hist is not None and self.appearance is not None else 0.)
            if appearance_distance > .65:
                continue
            scores.append((overlap + .3 * (1 - distance) + .2 * (1 - appearance_distance), item))
        scores.sort(key=lambda pair: pair[0], reverse=True)
        if scores and (len(scores) == 1 or scores[0][0] - scores[1][0] > .15):
            self.remember(scores[0][1], frame_id, frame)
            self.reassociations += 1
            self.reason = "motion_appearance_reassociated"
            return self.track_id, "tracked"
        self.reason = "ambiguous_reassociation" if len(scores) > 1 else "target_not_observed"
        return self.track_id, "lost"
