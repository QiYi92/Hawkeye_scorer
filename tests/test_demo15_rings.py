"""Projective geometry, real edge support, temporal failures and rendering."""
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from demo15.rings import (RingTracker, RingConfig, project, fit_model,
                          radial_errors, geometry, fitted_region)


def scene(noise=0, occluded=False):
    matrix = np.array([[235., 24., 330.], [9., 81., 245.], [.06, .08, 1.]])
    angles = np.linspace(0, 2*np.pi, 721)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    frame = np.full((400, 660, 3), 70, np.uint8)
    outline = np.rint(project(matrix, unit*1.065)).astype(np.int32)
    paper = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(paper, [outline], 255)
    frame[paper > 0] = 235
    for k in range(1, 11):
        points = np.rint(project(matrix, unit*k/10)).astype(np.int32)
        if k == 1:
            cv2.fillPoly(frame, [points], (60, 60, 60))
        else:
            cv2.polylines(frame, [points], True, (55, 55, 55), 1, cv2.LINE_AA)
    excluded = np.zeros_like(paper)
    if occluded:
        cv2.rectangle(excluded, (360, 180), (480, 320), 255, -1)
        frame[excluded > 0] = (190, 50, 15)
    if noise:
        rng = np.random.default_rng(23)
        frame = np.clip(frame.astype(float)+rng.normal(0, noise, frame.shape), 0, 255).astype(np.uint8)
    ellipse = cv2.fitEllipse(outline[:, None, :])
    target = dict(center=dict(point=project(matrix, [[0, 0]])[0].tolist()))
    return frame, target, (outline[:, None, :], ellipse, paper), excluded, matrix


def camera(n=0, **changes):
    return dict(frame_id=n, timestamp=n/30, camera_stable=True, camera_segment=0,
                camera_transform=np.eye(3).tolist(), **changes)


class ProjectiveFitTests(unittest.TestCase):
    def test_exact_projective_circles_fit_below_two_pixels(self):
        _, _, _, _, truth = scene()
        a = np.linspace(0, 2*np.pi, 96, endpoint=False)
        unit = np.column_stack([np.cos(a), np.sin(a)])
        uv = np.concatenate([unit*k/10 for k in range(1, 11)])
        radii = np.repeat(np.arange(1, 11)/10, len(a))
        observed = project(truth, uv)
        rng = np.random.default_rng(15)
        observed += rng.normal(0, .2, observed.shape)
        observed[::19] += rng.normal(0, 20, observed[::19].shape)
        seed = truth.copy(); seed[:2, 2] += [3, -2]
        fit = fit_model(seed, observed, radii, 1.)
        self.assertIsNotNone(fit)
        error = np.abs(radial_errors(fit, project(truth, uv), radii))
        self.assertLess(float(error.max()), 2.)

    def test_ten_closed_nested_curves_and_perspective_centers(self):
        *_, matrix = scene()
        model = geometry(matrix, [1.]*10, 0., 'image_and_geometry', 0, RingConfig(), [0])
        self.assertEqual(len(model['rings']), 10)
        for index, ring in enumerate(model['rings'], 1):
            self.assertEqual(ring['points'][0], ring['points'][-1])
            rectified = project(np.linalg.inv(matrix), ring['points'])
            np.testing.assert_allclose(np.linalg.norm(rectified, axis=1), index/10, atol=.0001)
        self.assertEqual(len(model['four_points']), 4)

    def test_image_evidence_noise_and_occlusion(self):
        for noise, occluded in ((0, False), (8, False), (0, True)):
            with self.subTest(noise=noise, occluded=occluded):
                frame, target, found, excluded, _ = scene(noise, occluded)
                model = RingTracker().update(frame, target, camera(), excluded, found, full=True)
                self.assertEqual(model['status'], 'fitted')
                self.assertEqual(len(model['rings']), 10)
                self.assertLess(model['fit_error_px'], 2.)
                self.assertTrue(all(r['source'] == 'fitted' for r in model['rings']))

    def test_ideal_image_curves_match_true_boundaries_within_two_pixels(self):
        frame, target, found, excluded, truth = scene()
        model = RingTracker().update(frame, target, camera(), excluded, found, full=True)
        self.assertEqual(model['status'], 'fitted')
        for ring in model['rings']:
            points = np.asarray(ring['points'])
            error = np.abs(radial_errors(truth, points, np.full(len(points), ring['radius_ratio'])))
            self.assertLess(float(error.max()), 2.)

    def test_blank_paper_and_isolated_false_edges_do_not_create_ten_observations(self):
        frame, target, found, excluded, _ = scene()
        frame[found[2] > 0] = 235
        cv2.line(frame, (250, 200), (450, 280), (20, 20, 20), 2)
        model = RingTracker().update(frame, target, camera(), excluded, found)
        self.assertEqual(model['status'], 'unknown')

    def test_fully_occluded_or_clipped_edge_evidence_cannot_lock(self):
        frame, target, found, excluded, _ = scene()
        excluded[:] = 1
        model = RingTracker().update(frame, target, camera(), excluded, found)
        self.assertEqual(model['status'], 'unknown')
        # A clipped image with no usable arcs must not count padding as edges.
        dark = np.zeros_like(frame)
        paper = np.ones(frame.shape[:2], np.uint8)*255
        clipped = (found[0], found[1], paper)
        model = RingTracker().update(dark, target, camera(), None, clipped)
        self.assertEqual(model['status'], 'unknown')

    def test_registered_motion_transforms_all_curves(self):
        frame, target, found, excluded, _ = scene()
        tracker = RingTracker()
        first = tracker.update(frame, target, camera(), excluded, found)
        self.assertEqual(first['status'], 'fitted')
        transform = np.array([[1., 0, 13], [0, 1, -4], [0, 0, 1]])
        current = camera(1); current['camera_transform'] = transform.tolist()
        moved = tracker.update(frame, target, current, excluded, found)
        np.testing.assert_allclose(np.asarray(moved['rings'][9]['points']),
                                   np.asarray(first['rings'][9]['points'])+[13, -4], atol=.002)
        self.assertEqual(moved['source'], 'registered_history')
        self.assertEqual(moved['observed_ring_count'], 0)

    def test_failed_registration_segment_change_and_scene_cut_clear_history(self):
        for change in ('registration', 'segment', 'time'):
            with self.subTest(change=change):
                frame, target, found, excluded, _ = scene()
                tracker = RingTracker()
                self.assertEqual(tracker.update(frame, target, camera(), excluded, found)['status'], 'fitted')
                current = camera(1)
                if change == 'registration': current['camera_stable'] = False
                if change == 'segment': current['camera_segment'] = 1
                if change == 'time': current['timestamp'] = 2.
                unknown = tracker.update(np.zeros_like(frame), {'center': {}}, current, excluded)
                self.assertEqual(unknown['status'], 'unknown')
                self.assertEqual(tracker.history, [])

    def test_short_missing_shoe_or_paper_does_not_renew_history_forever(self):
        frame, target, found, excluded, _ = scene()
        tracker = RingTracker()
        tracker.update(frame, target, camera(), excluded, found)
        blank = np.zeros_like(frame)
        with patch('demo14.tracker.paper_region', return_value=None):
            short = tracker.update(blank, {'center': {}}, camera(3))
            expired = tracker.update(blank, {'center': {}}, camera(16))
        self.assertEqual(short['status'], 'fitted')
        self.assertEqual(expired['status'], 'unknown')

    def test_completion_alone_does_not_certify_ring_number(self):
        *_, matrix = scene()
        model = geometry(matrix, [0.]*10, .2, 'image_and_geometry', 0, RingConfig(), [0])
        point = project(matrix, [[.35, 0]])[0]
        self.assertEqual(fitted_region(point, model), 'inside_target_ring_unknown')
        for ring in model['rings']: ring['observed_fraction'] = 1.
        self.assertEqual(fitted_region(point, model, uncertainty_px=0), 'ring_4')
        model['source'] = 'registered_history'
        self.assertEqual(fitted_region(point, model), 'inside_target_ring_unknown')

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError): RingConfig(ring_count=9)

    def test_main_depth_and_export_annotation_share_all_ten_curves(self):
        from demo15.render import annotate, annotated_depth
        *_, matrix = scene()
        model = geometry(matrix, [1.]*10, .2, 'image_and_geometry', 0, RingConfig(), [0])
        row = dict(frame_id=0, timestamp=0., image_size=[660, 400], feet=[], target={'ring_model': model})
        result = dict(mode='fusion', event=None, status='unconfirmed', failure_reason='not_found')
        with patch('demo15.render.draw_panel') as panel, patch('demo15.render.cv2.polylines', wraps=cv2.polylines) as curves:
            annotate(np.zeros((400, 660, 3), np.uint8), row, result)
            self.assertEqual(curves.call_count, 10)
            self.assertIn('靶环：10/10（拟合）', panel.call_args.args[1])
            main_points = [call.args[1][0] for call in curves.call_args_list]
            curves.reset_mock()
            annotated_depth(np.ones((40, 66)), row, result)
            self.assertEqual(curves.call_count, 10)
            for points, call in zip(main_points, curves.call_args_list):
                np.testing.assert_array_equal(points, call.args[1][0])

    def test_old_evidence_and_changed_model_code_are_rejected(self):
        from demo15.extraction import load_cache, StaleEvidenceError, EVIDENCE_VERSION, EXTRACTION_CONFIG
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'metadata.json').write_text(json.dumps({'evidence_version': 5}))
            with self.assertRaises(StaleEvidenceError): load_cache(path)
            (path/'metadata.json').write_text(json.dumps(dict(evidence_version=EVIDENCE_VERSION,
                extraction_config=EXTRACTION_CONFIG, ring_code_sha256='obsolete')))
            with self.assertRaisesRegex(StaleEvidenceError, '靶环模型'): load_cache(path)


if __name__ == '__main__':
    unittest.main()
