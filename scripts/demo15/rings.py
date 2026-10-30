"""Demo15's shared projective ten-circle model. Fitted curves are not observations.

The paper silhouette is only an initializer: it never replaces the observed
paper polygon used for hit/miss. All public geometry uses native image pixels.
"""
from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np
from scipy.optimize import least_squares


MODEL_VERSION = 1


@dataclass(frozen=True)
class RingConfig:
    ring_count: int = 10
    preview_angles: int = 96
    full_angles: int = 192
    curve_samples: int = 361
    update_interval: int = 5
    history_seconds: float = .5
    min_contrast: float = 5.
    max_error_px_720: float = 3.

    def __post_init__(self):
        if (self.ring_count != 10 or self.preview_angles < 32 or
                self.full_angles < 32 or self.curve_samples < 65 or
                self.update_interval < 1 or self.history_seconds <= 0 or
                self.min_contrast <= 0 or self.max_error_px_720 <= 0):
            raise ValueError('Invalid Demo15 ring configuration')


def load_config():
    path = Path(__file__).resolve().parents[2]/'configs/demo15_rings.json'
    return RingConfig(**json.loads(path.read_text(encoding='utf-8')))


def project(matrix, points):
    points = np.asarray(points, float).reshape(-1, 2)
    return cv2.perspectiveTransform(points[:, None, :], np.asarray(matrix, float))[:, 0, :]


def initial_model(ellipse, center):
    """Map a disk and its physical center to the observed paper ellipse/center.

    A disk-preserving projective boost moves the origin to the actual bullseye;
    simply translating an ellipse to that point loses the perspective offset.
    The small paper margin is an initializer, subsequently corrected by ridges.
    """
    ellipse_center, (a, b), angle = ellipse
    t = np.deg2rad(angle)
    basis = np.array([[np.cos(t)*a/2, -np.sin(t)*b/2],
                      [np.sin(t)*a/2, np.cos(t)*b/2]])
    displacement = np.linalg.solve(basis, np.asarray(center)-ellipse_center)
    distance = np.linalg.norm(displacement)
    if distance >= .8:
        return None
    gamma = 1/np.sqrt(1-distance**2)
    direction = displacement/max(distance, 1e-9)
    boost = np.eye(3)
    boost[:2, :2] += (gamma-1)*np.outer(direction, direction)
    boost[:2, 2] = gamma*displacement
    boost[2, :2] = gamma*displacement
    boost[2, 2] = gamma
    affine = np.vstack([np.column_stack([basis, ellipse_center]), [0., 0., 1.]])
    model = affine @ boost @ np.diag([.94, .94, 1.])
    return model/model[2, 2]


def radial_errors(matrix, points, radii):
    """Signed first-order orthogonal distance to projected circles, in pixels."""
    inv = np.linalg.inv(matrix)
    xy = np.column_stack([points, np.ones(len(points))]) @ inv.T
    uv = xy[:, :2] / xy[:, 2:]
    length = np.maximum(np.linalg.norm(uv, axis=1), 1e-8)
    jac = (inv[None, :2, :2] - uv[:, :, None]*inv[None, 2, :2]) / xy[:, 2, None, None]
    grad = np.einsum('ni,nij->nj', uv/length[:, None], jac)
    return (length-radii)/np.maximum(np.linalg.norm(grad, axis=1), 1e-8)


def fit_model(seed, points, radii, max_error=3.):
    """Fit one homography jointly to all rings with robust orthogonal residuals.

    Normalization keeps the eight homography parameters numerically comparable.
    A weak seed regularizer fixes the unobservable rotation of concentric circles.
    """
    points, radii = np.asarray(points, float), np.asarray(radii, float)
    center = project(seed, [[0, 0]])[0]
    scale = max(10., np.linalg.norm(seed[:2, :2]))
    normal = np.array([[1/scale, 0, -center[0]/scale],
                       [0, 1/scale, -center[1]/scale], [0, 0, 1.]])
    base = normal @ seed
    base /= base[2, 2]
    x0 = base.ravel()[:8]
    normalized = project(normal, points)

    def matrix(x):
        return np.r_[x, 1.].reshape(3, 3)

    def residual(x):
        try:
            e = radial_errors(matrix(x), normalized, radii)*scale
        except np.linalg.LinAlgError:
            e = np.full(len(normalized), 1e5)
        # Restrict unjustified drift in near-degenerate, strongly flattened views.
        return np.r_[e, (x-x0)*.3]

    opt = least_squares(residual, x0, loss='soft_l1', f_scale=max_error,
                        max_nfev=65)
    # Remove isolated distractors after the robust pass. They otherwise bias
    # projective terms disproportionately in the flattened ends of the disk.
    errors = np.abs(radial_errors(matrix(opt.x), normalized, radii))*scale
    keep = errors <= max(2*max_error, 3*float(np.median(errors)))
    if keep.sum() >= 32 and keep.mean() >= .55 and not keep.all():
        normalized, radii = normalized[keep], radii[keep]
        opt = least_squares(residual, opt.x, loss='soft_l1', f_scale=max_error,
                            max_nfev=65)
    model = np.linalg.inv(normal) @ matrix(opt.x)
    model /= model[2, 2]
    # A pole in/on the disk or collapsed geometry is never a valid target.
    if (not np.isfinite(model).all() or np.linalg.norm(model[2, :2]) >= .75 or
            np.linalg.cond(model[:2, :2]-model[:2, 2:]*model[2:3, :2]) > 25):
        return None
    return model


def ridge_observations(frame, found, excluded, model, config, full=False):
    """Search narrow radial neighborhoods; no clipped/masked points are evidence."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    background = cv2.GaussianBlur(gray, (0, 0), 4.)
    image = gray-background
    allowed = (found[2] > 0) & (hsv[:, :, 1] < 95)
    if excluded is not None:
        allowed &= cv2.dilate(np.asarray(excluded, np.uint8), np.ones((7, 7), np.uint8)) == 0
    allowed[:2] = False; allowed[-2:] = False
    allowed[:, :2] = False; allowed[:, -2:] = False
    n = config.full_angles if full else config.preview_angles
    theta = np.arange(n)*2*np.pi/n
    directions = np.column_stack([np.cos(theta), np.sin(theta)])
    points, radii, sectors, contrasts = [], [], [], []
    fractions = []
    for ring in range(1, config.ring_count+1):
        r = ring/config.ring_count
        offsets = np.linspace(-.043, .043, 33)
        rr = r+offsets
        grid = project(model, (directions[:, None, :]*rr[None, :, None]).reshape(-1, 2)).reshape(n, -1, 2)
        mx, my = grid[:, :, 0].astype(np.float32), grid[:, :, 1].astype(np.float32)
        profile = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.remap(allowed.astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
        if ring == 1:
            # The central disk is filled; its outward gradient is the boundary.
            response = np.roll(profile, -2, axis=1)-np.roll(profile, 2, axis=1)
        else:
            response = np.minimum(np.roll(profile, 3, axis=1)-profile,
                                  np.roll(profile, -3, axis=1)-profile)
        response[:, :3] = -1e4; response[:, -3:] = -1e4
        response[~valid] = -1e4
        choice = np.argmax(response-.8*(offsets/.043)**2, axis=1)
        row = np.arange(n)
        strength = response[row, choice]
        good = strength >= config.min_contrast
        # Isolated letters, cables and wrinkles cannot establish a circle.
        good &= sum(np.roll(good, k) for k in (-2, -1, 0, 1, 2)) >= 3
        fractions.append(float(np.mean(good)))
        points.extend(grid[row[good], choice[good]])
        radii.extend([r]*int(good.sum()))
        sectors.extend(row[good]/n)
        contrasts.extend(strength[good])
    return (np.asarray(points).reshape(-1, 2), np.asarray(radii),
            np.asarray(sectors), fractions, np.asarray(contrasts))


def geometry(matrix, fractions, error, source, frame_id, config, evidence_frames):
    angles = np.linspace(0, 2*np.pi, config.curve_samples)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    rings = []
    for ring in range(1, config.ring_count+1):
        points = project(matrix, unit*ring/config.ring_count)
        points[-1] = points[0]
        rings.append(dict(ring_id=ring, radius_ratio=ring/config.ring_count,
                          points=np.round(points, 3).tolist(), source='fitted',
                          observed_fraction=round(fractions[ring-1], 4)))
    outer = np.asarray(rings[-1]['points'])
    cardinal = [int(np.argmin(outer[:, 0])), int(np.argmin(outer[:, 1])),
                int(np.argmax(outer[:, 0])), int(np.argmax(outer[:, 1]))]
    return dict(version=MODEL_VERSION, status='fitted', source=source,
                homography=np.asarray(matrix).tolist(),
                center=project(matrix, [[0, 0]])[0].tolist(), rings=rings,
                four_points=[outer[i].tolist() for i in cardinal],
                four_points_source='fitted_extrema', fit_error_px=float(error),
                frame_id=frame_id, evidence_frames=list(evidence_frames),
                observed_ring_count=sum(f >= .15 for f in fractions))


class RingTracker:
    """Native-pixel temporal geometry; history never crosses failed registration."""
    def __init__(self, config=None):
        self.config = config or load_config()
        self.history = []
        self.last = None
        self.segment = None
        self.previous_time = None

    def update(self, frame, target, camera, excluded=None, found=None, full=False):
        cfg = self.config
        n, t = camera['frame_id'], camera['timestamp']
        segment = camera.get('camera_segment', 0)
        valid = camera.get('camera_stable', False)
        discontinuity = (self.previous_time is not None and
                         (t <= self.previous_time or t-self.previous_time > .6))
        self.previous_time = t
        if not valid or segment != self.segment or discontinuity:
            self.history.clear(); self.last = None
        self.segment = segment
        unknown = dict(version=MODEL_VERSION, status='unknown', rings=[],
                       reason='insufficient_ring_evidence')
        if not valid:
            return {**unknown, 'reason': 'registration_unavailable'}
        transform = np.asarray(camera['camera_transform'], float)
        warped = None
        if self.last is not None and t-self.last['time'] <= cfg.history_seconds:
            warped = transform @ self.last['reference_model']
        # A cheap projective warp supplies all intervening preview frames.
        if warped is not None and not full and n % cfg.update_interval:
            return geometry(warped, [0.]*10, self.last['error'], 'registered_history',
                            n, cfg, self.last['frames'])
        if found is None:
            from demo14.tracker import paper_region
            found = paper_region(frame)
        if found is None:
            if warped is None:
                return {**unknown, 'reason': 'paper_missing'}
            return geometry(warped, [0.]*10, self.last['error'], 'registered_history',
                            n, cfg, self.last['frames'])
        center = target.get('center', {}).get('point')
        if warped is None and center is None:
            return {**unknown, 'reason': 'center_missing'}
        seed = warped if warped is not None else initial_model(found[1], center)
        if seed is None:
            return {**unknown, 'reason': 'center_outside_target'}
        observations = ridge_observations(frame, found, excluded, seed, cfg, full)
        points, radii, sectors, fractions, _ = observations
        if len(points) < 32 or sum(f >= .15 for f in fractions) < 4:
            if warped is not None:
                return geometry(warped, fractions, self.last['error'], 'registered_history',
                                n, cfg, self.last['frames'])
            return unknown
        # Keep real samples in the registration anchor system, not filled curves.
        self.history = [entry for entry in self.history if t-entry['time'] <= cfg.history_seconds]
        current = dict(time=t, frame_id=n, points=project(np.linalg.inv(transform), points), radii=radii)
        entries = self.history[-2:]+[current]
        all_points = np.concatenate([project(transform, e['points']) for e in entries])
        all_radii = np.concatenate([e['radii'] for e in entries])
        limit = cfg.max_error_px_720*frame.shape[0]/720
        fitted = fit_model(seed, all_points, all_radii, max(.7, limit))
        if fitted is not None:
            errors = np.abs(radial_errors(fitted, points, radii))
            inliers = errors <= max(1., limit)
            # Require spatially distributed support, not one short arc.
            coverage = len(np.unique((sectors[inliers]*8).astype(int)))
            error = float(np.median(errors))
            supported = len(np.unique(np.round(radii[inliers]*10)))
            if error <= max(1., limit) and inliers.mean() >= .55 and coverage >= 6 and supported >= 4:
                fractions = [float(np.sum(inliers & np.isclose(radii, k/10)) /
                                   (cfg.full_angles if full else cfg.preview_angles))
                             for k in range(1, 11)]
                self.history.append(current)
                frames = [e['frame_id'] for e in entries]
                self.last = dict(time=t, reference_model=np.linalg.inv(transform)@fitted,
                                 error=error, frames=frames)
                result = geometry(fitted, fractions, error, 'image_and_geometry', n, cfg, frames)
                outer = points[inliers & np.isclose(radii, 1.)]
                result['edge_keypoints'] = []
                for name, keypoint in zip(('left', 'top', 'right', 'bottom'), result['four_points']):
                    distance = np.linalg.norm(outer-keypoint, axis=1)
                    observed = len(outer) and distance.min() <= max(6., limit*3)
                    result['edge_keypoints'].append(dict(direction=name, source='observed' if observed else 'fitted',
                        point=outer[distance.argmin()].tolist() if observed else keypoint))
                return result
        if warped is not None:
            return geometry(warped, fractions, self.last['error'], 'registered_history',
                            n, cfg, self.last['frames'])
        return unknown


def fitted_region(point, model, uncertainty_px=5.):
    """Conservative ring estimate; display-only completion cannot certify a score."""
    if not model or model.get('status') != 'fitted' or model.get('source') != 'image_and_geometry':
        return 'inside_target_ring_unknown'
    radius = np.linalg.norm(project(np.linalg.inv(model['homography']), [point])[0])
    index = max(1, int(np.ceil(radius*10)))
    if index > 10:
        return 'inside_target_ring_unknown'
    rings = model['rings']
    neighbors = rings[max(0, index-2):index]
    if any(r['observed_fraction'] < .25 for r in neighbors):
        return 'inside_target_ring_unknown'
    if any(abs(cv2.pointPolygonTest(np.asarray(r['points'], np.float32),
                                   tuple(map(float, point)), True)) <= uncertainty_px+model['fit_error_px']
           for r in neighbors):
        return 'unknown'
    return 'bullseye' if index == 1 else f'ring_{index}'
