"""Pixel geometry. Boxes use exclusive x2/y2; angles are axial, in [0, 180)."""
import math

import cv2
import numpy as np


def encode_rle(mask):
    """Uncompressed COCO RLE, column-major, starting with background count."""
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    edges = np.flatnonzero(np.diff(np.r_[0, flat, 0]))
    counts = np.diff(np.r_[0, edges, flat.size]).tolist()
    if flat.size and flat[-1]:
        counts.pop()  # terminal zero-length background run
    return {"size": list(mask.shape), "counts": counts}


def mask_geometry(mask, contours=False):
    binary = np.asarray(mask, dtype=np.uint8)
    m = cv2.moments(binary, binaryImage=True)
    area = int(m["m00"])
    if not area:
        return None
    x, y, w, h = cv2.boundingRect(binary)
    cx, cy = m["m10"] / area, m["m01"] / area
    covariance = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / area
    values, vectors = np.linalg.eigh(covariance)
    # An isotropic/single-pixel region has no meaningful major axis.
    anisotropy = float((values[1] - values[0]) / max(float(values.sum()), 1e-9))
    axis = vectors[:, 1]
    angle = math.degrees(math.atan2(axis[1], axis[0])) % 180 if anisotropy > .05 else None
    result = {"bbox": [x, y, x + w, y + h], "centroid": [cx, cy],
              "area": area, "width": w, "height": h, "aspect_ratio": w / h,
              "orientation": angle, "axis_anisotropy": anisotropy,
              "top_y": y, "bottom_y": y + h - 1}
    if contours:
        curves, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result["contours"] = [c.reshape(-1, 2).tolist() for c in curves]
        result["contour_hierarchy"] = hierarchy[0].tolist() if hierarchy is not None else []
    return result


class TemporalFeatures:
    """Smooth derived values only; never alter evidence masks or bridge detection gaps."""
    def __init__(self, alpha=.65):
        if not 0 < alpha <= 1:
            raise ValueError("EMA alpha must be in (0, 1]")
        self.alpha = alpha
        self.states = {}

    def update(self, track_id, frame_id, timestamp, geometry):
        old = self.states.get(track_id)
        continuous = old is not None and old[0] == frame_id - 1 and timestamp > old[1]
        smoothed = {k: geometry[k] for k in ("bbox", "centroid", "area", "width", "height", "orientation")}
        velocity = [None, None]
        if continuous:
            velocity = ((np.array(geometry["centroid"]) - old[2]) / (timestamp - old[1])).tolist()
            for k in ("bbox", "centroid", "area", "width", "height"):
                value = self.alpha * np.array(smoothed[k]) + (1 - self.alpha) * np.array(old[3][k])
                smoothed[k] = value.tolist()
            a, b = geometry["orientation"], old[3]["orientation"]
            if a is not None and b is not None:
                # Axial angles wrap at 180 degrees, not 360 degrees.
                smoothed["orientation"] = (b + self.alpha * ((a - b + 90) % 180 - 90)) % 180
        self.states[track_id] = (frame_id, timestamp, np.array(geometry["centroid"]), smoothed)
        return smoothed, velocity

    def prune(self, frame_id, max_age=90):
        self.states = {k: v for k, v in self.states.items() if frame_id - v[0] <= max_age}
