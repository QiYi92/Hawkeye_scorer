"""Conservative, image-space first-foot-contact estimation without keypoints.

Ground support is independent of foot motion: a detected target surface region,
or an explicitly supplied local ground line. Neither proves physical contact.
Confirmed results are estimates with evidence and a separate confirmation frame.
"""
from collections import deque

import cv2
import numpy as np
from .display_text import CONTACT_REASON, CONTACT_STATUS, draw_panel


def foot_bottom_points(labels):
    """One observed sole point per nontrivial component; never average both shoes."""
    if labels is None:
        return []
    n, components, stats, _ = cv2.connectedComponentsWithStats((labels == 2).astype(np.uint8))
    minimum_area = max(6, round(labels.size * .00001))
    points = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < minimum_area:
            continue
        x, y, w, h, area = map(int, stats[k])
        if x == 0 or y == 0 or x + w == labels.shape[1] or y + h == labels.shape[0]:
            continue  # clipped shoe cannot provide an observed sole boundary
        yy, xx = np.where(components[y:y+h, x:x+w] == k)
        edge = yy >= yy.max() - max(1, round(h * .05))
        indices = np.flatnonzero(edge)
        chosen = indices[np.lexsort((-yy[indices], np.abs(xx[indices] - np.median(xx[indices]))))[0]]
        points.append({"point": [int(xx[chosen] + x), int(yy[chosen] + y)],
                       "area": area, "bbox": [x, y, x+w, y+h]})
    return sorted(points, key=lambda item: item["point"][1], reverse=True)[:2]


class GroundReference:
    """Static-camera ground gate, learned from scene geometry, never from feet."""
    def __init__(self, line=None, tolerance=6.):
        self.line = list(line) if line is not None else None
        self.tolerance = tolerance
        self.pixel_tolerance = tolerance
        self.ellipse = None
        self.last_seen = -1
        self.frame_id = -1
        self.camera_ok = False
        self.camera_moved = False
        self.previous = None
        self.camera_transform = np.eye(3)
        self.previous_exclusion = None
        self.background_inliers = 0
        self.detector = None
        if self.line is None:
            from browse_target_center_video import TargetCenterDetector
            self.detector = TargetCenterDetector()

    def update(self, frame, frame_id, athlete_mask=None):
        self.frame_id = frame_id
        h, w = frame.shape[:2]
        self.pixel_tolerance = self.tolerance * h / 720.
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, max(32, round(h*320/w))))
        gray = small
        exclusion = np.zeros(gray.shape, np.uint8)
        if athlete_mask is not None:
            exclusion = cv2.resize(athlete_mask.astype(np.uint8), (gray.shape[1], gray.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            exclusion = cv2.dilate(exclusion, np.ones((13, 13), np.uint8))
        if self.previous is not None:
            valid = ((exclusion == 0) & (self.previous_exclusion == 0)).astype(np.uint8)*255
            features = cv2.goodFeaturesToTrack(self.previous, 160, .02, 8, mask=valid)
            self.camera_ok = False
            self.background_inliers = 0
            if features is not None and len(features) >= 12:
                tracked, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, features, None)
                if tracked is not None and status is not None:
                    good = status.ravel().astype(bool) & np.isfinite(tracked).all(axis=(1, 2))
                    if good.sum() >= 12:
                        affine, inliers = cv2.estimateAffinePartial2D(features[good], tracked[good],
                                                                    method=cv2.RANSAC, ransacReprojThreshold=1.)
                        self.background_inliers = int(inliers.sum()) if inliers is not None else 0
                        if affine is not None and np.isfinite(affine).all() and self.background_inliers >= max(12, good.sum()*.6):
                            transform = np.vstack([affine, [0., 0., 1.]])
                            self.camera_transform = transform @ self.camera_transform
                            corners = np.array([[0, 0, 1], [320, 0, 1], [0, gray.shape[0], 1], [320, gray.shape[0], 1]]).T
                            step = np.linalg.norm((transform @ corners-corners)[:2], axis=0).max()
                            total = np.linalg.norm((self.camera_transform @ corners-corners)[:2], axis=0).max()
                            if step > 2. or total > 3.:
                                self.camera_moved = True
                            self.camera_ok = not self.camera_moved
        self.previous = gray
        self.previous_exclusion = exclusion
        if self.detector is not None and frame_id % 5 == 0 and not self.camera_moved:
            target = self.detector.detect(frame)
            if target is not None and self.detector.stable_disk_frames >= 5:
                self.ellipse = target.disk_ellipse
                self.last_seen = frame_id
        return self.summary()

    def supports(self, point):
        if not self.camera_ok:
            return False
        if self.line is not None:
            a, b = np.array(self.line[:2]), np.array(self.line[2:])
            delta = b-a
            fraction = float(np.dot(np.array(point)-a, delta) / np.dot(delta, delta))
            if not 0 <= fraction <= 1:
                return False
            return bool(np.linalg.norm(np.array(point) - (a + fraction*delta)) <= self.pixel_tolerance)
        if self.ellipse is None or self.frame_id - self.last_seen > 15:
            return False
        center, axes, angle = self.ellipse
        dx, dy = np.array(point) - center
        radians = np.deg2rad(angle)
        u = dx*np.cos(radians) + dy*np.sin(radians)
        v = -dx*np.sin(radians) + dy*np.cos(radians)
        return bool((u/max(axes[0]/2, 1))**2 + (v/max(axes[1]/2, 1))**2 <= 1.)

    def summary(self):
        available = self.line is not None or (self.ellipse is not None and self.frame_id-self.last_seen <= 15)
        return {"source": "configured_local_line" if self.line is not None else "detected_target_surface",
                "available": bool(available), "camera_stable": self.camera_ok,
                "camera_moved": self.camera_moved,
                "background_inliers": self.background_inliers,
                "line": self.line, "ellipse": self.ellipse,
                "meaning": "2D ground-region gate, not a height measurement"}


class FirstFootContact:
    def __init__(self, fps, confirm_frames=4, tolerance=6., ground_line=None):
        self.fps = fps
        self.confirm_frames = confirm_frames
        self.tolerance = tolerance
        self.ground = GroundReference(ground_line, tolerance)
        self.history = deque(maxlen=max(8, round(fps*.6)))
        self.candidate = self.event = None
        self.track_id = None
        self.status = "unconfirmed"
        self.reason = "waiting_for_target"
        self.event_image = self.candidate_image = None
        self.previous_image = None
        self.current_points = []
        self.prior_gaps = False
        self.earlier_candidate_obscured = False

    @property
    def needs_fresh_parts(self):
        return self.event is None and not self.ground.camera_moved

    def reset_candidate(self, reason):
        self.candidate = self.candidate_image = None
        self.status, self.reason = "unconfirmed", reason

    def update(self, frame, frame_id, timestamp, instance, labels, source):
        self.ground.update(frame, frame_id, instance.mask if instance is not None else None)
        self.current_points = foot_bottom_points(labels)
        if self.event is not None:
            return self.snapshot()
        if instance is None or not self.current_points or source != "inferred" or not self.ground.camera_ok:
            if self.candidate is not None:
                self.earlier_candidate_obscured = True
            if self.history:
                self.prior_gaps = True
            if self.ground.camera_moved:
                reason = "camera_moved"
                self.history.clear()
            elif instance is None:
                reason = "target_missing"
            elif not self.ground.camera_ok:
                reason = "camera_motion_unreliable"
            else:
                reason = "fresh_feet_mask_missing"
            self.reset_candidate(reason)
            return self.snapshot()
        if self.earlier_candidate_obscured:
            self.reset_candidate("earlier_contact_candidate_obscured")
            return self.snapshot()
        if self.candidate is None:
            self.status, self.reason = "unconfirmed", "observing_descent"
        if self.track_id is not None and instance.track_id != self.track_id:
            self.history.clear()
            self.reset_candidate("target_changed")
            self.prior_gaps = True
        self.track_id = instance.track_id
        observation = {"frame_id": frame_id, "timestamp": timestamp, "feet": self.current_points,
                       "ground_points": [p for p in self.current_points if self.ground.supports(p["point"])]}
        previous = self.history[-1] if self.history else None
        tolerance = self.tolerance * frame.shape[0] / 720.
        # Allow shoe rotation horizontally, but require a much tighter vertical stance.
        x_tolerance, y_tolerance = max(3., tolerance*2.5), max(2., tolerance)

        def near(point, anchor):
            return abs(point[0]-anchor[0]) <= x_tolerance and abs(point[1]-anchor[1]) <= y_tolerance

        if self.candidate is not None:
            if previous is None or frame_id != previous["frame_id"]+1:
                self.reset_candidate("nonconsecutive_support_frames")
            elif not any(near(p["point"], self.candidate["point"]) for p in observation["ground_points"]):
                self.reset_candidate("foot_not_stable_at_candidate")
            else:
                self.candidate["support_frames"] += 1
                if self.candidate["support_frames"] >= self.confirm_frames:
                    self.event = {**self.candidate, "confirmation_frame": frame_id,
                                  "confirmation_timestamp": timestamp, "track_id": self.track_id,
                                  "status": "confirmed_estimate", "is_estimate": True,
                                  "confidence": "low" if self.prior_gaps else "medium",
                                  "prior_observation_gaps": self.prior_gaps,
                                  "coordinate_system": "contact-frame image pixels",
                                  "uncertainty_px": [x_tolerance, y_tolerance],
                                  "ground_reference": self.ground.summary(),
                                  "evidence": ["prior_foot_descent", "independent_ground_region",
                                               "consecutive_fresh_foot_masks", "local_sole_stability"],
                                  "first_contact_guaranteed": False}
                    self.event_image = self.candidate_image
                    self.status, self.reason = "confirmed_estimate", "descent_followed_by_supported_stance"

        if self.candidate is None and self.event is None and previous and frame_id == previous["frame_id"]+1:
            # The start of support is the previous frame, not the later confirmation frame.
            pre = [o for o in self.history if .03 <= previous["timestamp"]-o["timestamp"] <= .4]
            descended = (len(pre) >= 3 and previous["feet"][0]["point"][1] -
                         min(o["feet"][0]["point"][1] for o in pre) >= max(12., frame.shape[0]*.035))
            if descended:
                eligible = [p for p in previous["ground_points"] if any(
                    near(q["point"], p["point"]) for q in observation["ground_points"])]
                if len(eligible) == 1:
                    self.candidate = {"frame_id": previous["frame_id"], "timestamp": previous["timestamp"],
                                      "point": eligible[0]["point"], "support_frames": 2,
                                      "foot": "visible_foot_region", "parts_source": "inferred"}
                    self.candidate_image = self.previous_image
                    self.status, self.reason = "candidate", "waiting_for_followup_support"
                elif len(eligible) > 1:
                    self.reason = "ambiguous_simultaneous_foot_regions"
            elif self.event is None:
                self.reason = "insufficient_prior_descent"
        if not observation["ground_points"] and self.event is None:
            self.reason = "foot_outside_ground_reference" if self.ground.summary()["available"] else "ground_reference_unavailable"
        self.history.append(observation)
        self.previous_image = frame.copy()
        return self.snapshot()

    def snapshot(self):
        return {"enabled": True, "status": self.status, "reason": self.reason,
                "foot_points": self.current_points, "candidate": self.candidate,
                "earlier_candidate_obscured": self.earlier_candidate_obscured,
                "event": self.event, "ground_reference": self.ground.summary()}


def draw_contact_overlay(canvas, snapshot):
    if snapshot is None:
        return canvas
    state = snapshot["status"]
    event = snapshot["event"]
    message = f"首次触地：{CONTACT_STATUS.get(state, '状态未知')}"
    if event:
        message += f"  |  候选首帧：{event['frame_id']}（{event['timestamp']:.3f} 秒）"
        messages = [message, f"估计点位：({event['point'][0]}, {event['point'][1]}) 像素  |  "
                    f"第 {event['confirmation_frame']} 帧完成复核，结果需人工核验"]
    else:
        messages = [message + "  |  " + CONTACT_REASON.get(snapshot['reason'], "等待更多有效证据")]
    draw_panel(canvas, messages, bottom=True, max_width=canvas.shape[1], size=20)
    point = event["point"] if event else (snapshot["candidate"]["point"] if snapshot["candidate"] else None)
    if point and snapshot["ground_reference"]["camera_stable"]:
        color = (0, 0, 255) if event else (0, 220, 255)
        cv2.drawMarker(canvas, tuple(point), color, cv2.MARKER_CROSS, 22, 2)
        cv2.circle(canvas, tuple(point), 9, color, 2)
    return canvas
