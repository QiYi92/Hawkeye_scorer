"""Read-only reference reconciliation and honest end-to-end evaluation."""
from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np


def load_reference(video, mark_dir):
    path = Path(mark_dir) / Path(video).name / 'marks.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('frame_index_base') != 0 or data.get('coordinate_system') != 'original_image_pixels_x_right_y_down':
        raise ValueError(f'不支持的标记坐标约定：{path}')
    marks = data.get('marks', [])
    if len(marks) != 1:
        raise ValueError(f'首次触地评估需要恰好一个参考点：{path}')
    mark = marks[0]
    raw = cv2.imread(str(path.parent / mark['image']))
    if raw is None or raw.shape[:2] != (data['height'], data['width']):
        raise ValueError(f'标记原图不存在或尺寸不一致：{path}')
    point = mark['point']
    if not (0 <= point[0] < data['width'] and 0 <= point[1] < data['height']):
        raise ValueError('标记点超出原图')
    cap = cv2.VideoCapture(str(video))
    matches = []
    timeline = []
    previous = -1.
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t, source = timestamp(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, previous, fps, i)
            timeline.append(t)
            previous = t
            if abs(i-mark['frame_index']) <= 8 and np.array_equal(frame, raw):
                matches.append((i, t, source))
            i += 1
    finally:
        cap.release()
    if i != data['frame_count'] or len(matches) != 1:
        raise ValueError(f'标注无法唯一匹配原视频：{path}，匹配={matches}，帧数={i}')
    n, t, source = matches[0]
    return dict(original_frame_id=mark['frame_index'], frame_id=n, timestamp=t,
                original_timestamp=mark['timestamp_seconds'], timestamp_source=source,
                point_px=point, frame_reconciled=n != mark['frame_index'],
                annotation_path=str(path), image=str(path.parent/mark['image']),
                note='用户指定参考；video8 的接触画面存在歧义' if Path(video).stem == 'video8' else '用户手动参考',
                timeline=timeline)


def timestamp(raw, previous, fps, index):
    if np.isfinite(raw) and ((index == 0 and raw >= 0) or raw > previous):
        return float(raw), 'decoder_pos_msec'
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('视频时间戳与 FPS 均不可用')
    return (0. if index == 0 else previous+1/fps), 'fps_increment_fallback'


@dataclass(frozen=True)
class EvaluationConfig:
    frame_tolerance: int = 2
    position_tolerance_px: float = 10.

    def __post_init__(self):
        if (not isinstance(self.frame_tolerance, int) or self.frame_tolerance < 0
                or not np.isfinite(self.position_tolerance_px) or self.position_tolerance_px <= 0):
            raise ValueError('验收容差必须是非负整数帧数与正有限像素距离')


def evaluate(result, reference, observations, config=None):
    config = config or EvaluationConfig()
    if reference is None:
        return dict(labeled=False, joint_pass=None)
    ref = {k: v for k, v in reference.items() if k != 'timeline'}
    event = result.get('event')
    row = dict(labeled=True, reference=ref, time_error_seconds=None, frame_error=None,
               position_error_px=None, raw_position_error_px=None, registration='unavailable',
               time_pass=False, position_pass=False, joint_pass=False,
               reference_frame_sole_error_px=None)
    r = reference['frame_id']
    if r < len(observations):
        soles = observations[r]['feet']
        if soles:
            row['reference_frame_sole_error_px'] = float(min(np.linalg.norm(np.array(s['point'])-reference['point_px']) for s in soles))
    if event is None:
        return row
    accepted=result['status']=='confirmed_estimate'
    row['time_error_seconds'] = abs(event['timestamp']-reference['timestamp'])
    row['time_error_ms'] = 1000*row['time_error_seconds']
    row['frame_error'] = event['frame_id']-r
    row['time_pass'] = accepted and abs(row['frame_error']) <= config.frame_tolerance
    point = event.get('point_px')
    if point is None:
        return row
    row['raw_position_error_px'] = float(np.linalg.norm(np.asarray(point)-reference['point_px']))
    if r>=len(observations):return row
    pred = observations[event['frame_id']]
    target = observations[r]
    if event['frame_id'] == r:
        aligned = point
        row['registration'] = 'same_frame'
    elif (pred.get('camera_stable') and target.get('camera_stable')
          and pred.get('camera_segment',0)==target.get('camera_segment',0)):
        a, b = np.asarray(pred['camera_transform']), np.asarray(target['camera_transform'])
        v = b @ np.linalg.inv(a) @ np.array([*point, 1.])
        aligned = v[:2]/v[2]
        row['registration'] = 'background_affine'
    else:
        aligned = None
    if aligned is not None:
        row['position_error_px'] = float(np.linalg.norm(np.asarray(aligned)-reference['point_px']))
        row['registered_point_px'] = np.asarray(aligned).tolist()
        row['position_pass'] = accepted and row['position_error_px'] <= config.position_tolerance_px
    row['joint_pass'] = row['time_pass'] and row['position_pass']
    return row
