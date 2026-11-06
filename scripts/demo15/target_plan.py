"""Display-only metric labels and fixed contact coordinates for Demo15."""
import math

import cv2
import numpy as np

from demo13.display_text import draw_label
from target_plan_view import draw_target_plan


TARGET_RADIUS_M = 2.0
RING_COUNT = 10
RING_INTERVAL_M = TARGET_RADIUS_M / RING_COUNT


def project_point(point, model):
    """Rectify into outer-radius units, retaining camera-oriented axes."""
    if model.get('status') != 'fitted':
        return None
    try:
        matrix = np.asarray(model['homography'], float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-9:
            return None
        matrix = matrix / matrix[2, 2]
        value = np.linalg.solve(matrix, np.array([*point, 1.]))
        if abs(value[2]) < 1e-9:
            return None
        jacobian = matrix[:2, :2] - np.outer(matrix[:2, 2], matrix[2, :2])
        u, _, vt = np.linalg.svd(jacobian)
        xy = (u @ vt) @ (value[:2] / value[2])
        return tuple(xy) if np.isfinite(xy).all() else None
    except (KeyError, ValueError, np.linalg.LinAlgError):
        return None


def contact_plan_point(rows, result):
    event = result.get('event')
    if not event or result.get('point_precision') == 'coarse':
        return None
    row = next((row for row in rows if row['frame_id'] == event['frame_id']), None)
    return project_point(event['point_px'], row['target'].get('ring_model', {})) if row else None


def ring_score(point, target_hit, point_precision=None):
    """Estimate a discrete ring score; exact ring boundaries favor the inside."""
    if target_hit == 'miss':
        return 200
    if target_hit != 'hit' or point_precision == 'coarse' or point is None:
        return None
    xy = np.asarray(point, float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        return None
    radius = float(np.linalg.norm(xy))
    # Homography inversion introduces roundoff at exact ring boundaries.
    ring = max(1, math.ceil(radius * RING_COUNT - 1e-9))
    return min(200, (ring - 1) * 20)


def score_text(result, frame_id, point):
    event = result.get('event')
    if (not event or result.get('status') != 'confirmed_estimate'
            or frame_id < max(event['frame_id'], event.get('confirmation_frame', event['frame_id']))):
        return None
    score = ring_score(point, result.get('target_hit'), result.get('point_precision'))
    if score is None:
        return '分数待确认'
    return '脱靶：200分' if result.get('target_hit') == 'miss' else f'估计分数：{score}分'


def plan_text(canvas, text, origin, color, scale, thickness):
    draw_label(canvas, text, *origin, color, size=max(10, round(scale * 26)))


def draw_plan(canvas, frame_id, result, point):
    event = result.get('event')
    visible = None
    if not event:
        status = '未确认首次触地'
    elif frame_id < event['frame_id']:
        status = '等待首次触地'
    elif result['status'] != 'confirmed_estimate' or frame_id < event.get('confirmation_frame', event['frame_id']):
        status = '触地候选待确认'
    elif result.get('point_precision') == 'coarse':
        status = '脱靶（粗略区域）' if result.get('target_hit') == 'miss' else '已确认触地，落点为粗略区域'
    else:
        visible = point
        status = '已确认首次触地' if point is not None else '已确认触地，暂无靶盘定位'
    draw_target_plan(canvas, plan_text, visible, status,
                     score_text(result, frame_id, point) or '',
                     radius_m=TARGET_RADIUS_M, ring_interval_m=RING_INTERVAL_M)


def draw_ring_labels(canvas, model):
    """Stagger labels along two projective radial directions inside the disk."""
    h, w = canvas.shape[:2]
    for ring in model['rings']:
        index = ring['ring_id']
        angle = np.deg2rad(-32 if index % 2 == 0 else 148)
        r = ring['radius_ratio'] * .93
        matrix = np.asarray(model.get('homography', []), float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            continue
        value = matrix @ np.array([r*np.cos(angle), r*np.sin(angle), 1.])
        if abs(value[2]) < 1e-9:
            continue
        x, y = value[:2] / value[2]
        if not np.isfinite([x, y]).all():
            continue
        text = f'{index * RING_INTERVAL_M:.1f}m'
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .34, 1)
        origin = (max(0, min(round(x)-size[0]//2, w-size[0]-2)),
                  max(size[1]+2, min(round(y), h-2)))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .34, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .34,
                    (0,220,255) if index == 1 else (255,220,0), 1, cv2.LINE_AA)
