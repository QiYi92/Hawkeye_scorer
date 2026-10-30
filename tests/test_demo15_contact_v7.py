"""Airborne counter-evidence, bounded re-observation and first-contact regressions."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from demo13.tracking import Instance
from demo15.core import estimate, ContactConfig
from demo15.evidence import associate_feet
from demo15.recovery import ShoeRecovery
from demo15.render import annotate, mark_event
from test_demo15 import observations


def move_sole(row, y):
    foot = row['feet'][0]
    delta = y-foot['point'][1]
    foot['point'][1] = y
    foot['bbox'][1] += delta; foot['bbox'][3] += delta


def recovery_scene():
    frame = np.full((100,100,3), [45,75,110], np.uint8)
    mask = np.zeros((100,100), bool); mask[10:90,20:80] = True
    person = Instance(1,1.,[20,10,80,90],mask,0)
    labels = np.zeros((100,100),np.uint8); labels[75:85,40:60] = 2
    parser = Mock()
    def parse(crop):
        parsed = np.zeros(crop.shape[:2], np.uint8)
        parsed[48:58,32:52] = 2
        return parsed
    parser.parse.side_effect = parse
    camera = dict(camera_stable=True,camera_segment=0)
    return frame, person, labels, parser, camera


class ContactV7Tests(unittest.TestCase):
    def test_airborne_pause_is_explicitly_rejected_before_real_landing(self):
        rows = observations()
        for i,row in enumerate(rows):
            if i > 23: move_sole(row, min(124, 100+(i-23)*3))
        result = estimate(rows)
        self.assertEqual(result['status'], 'confirmed_estimate')
        self.assertEqual(result['event']['frame_id'], 31)
        self.assertTrue(any(c['frame_id']==20 and c['state']=='rejected_airborne'
                            for c in result['candidate_diagnostics']))

    def test_counter_evidence_after_old_point_two_three_second_window(self):
        rows = observations()
        for i,row in enumerate(rows):
            if i >= 28: move_sole(row, 100+(i-27)*5)
        result = estimate(rows)
        self.assertIsNone(result['event'])
        self.assertTrue(any(c['frame_id']==20 and c['reason']=='observed_resumed_descent'
                            for c in result['candidate_diagnostics']))

    def test_track_truncation_is_not_absence_of_counter_evidence(self):
        rows = observations()[:23]
        result = estimate(rows)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(result['failure_reason'], 'truncated_contact_track')
        self.assertFalse(result['event']['motion_diagnostics']['future_observed'])

    def test_flat_trace_and_grass_mask_with_truncated_tail_are_not_enough(self):
        rows=observations()[:26]
        for row in rows:row['feet'][0]['depth_residual']=None
        result=estimate(rows)
        self.assertEqual(result['status'],'needs_review')
        self.assertEqual(result['failure_reason'],'truncated_contact_track')
        self.assertFalse(result['event']['motion_diagnostics']['counter_evidence_complete'])

    def test_unresolved_early_contact_cannot_be_replaced_by_later_running(self):
        rows = observations()
        for row in rows[23:27]: row['feet'] = []
        for i in range(27,40): move_sole(rows[i], min(145,120+(i-27)*5))
        result = estimate(rows)
        self.assertEqual(result['event']['frame_id'], 20)
        self.assertEqual(result['status'], 'needs_review')

    def test_two_missing_frames_are_shared_by_association_and_event_gate(self):
        rows = observations()
        rows[18]['feet'] = []; rows[19]['feet'] = []
        result = estimate(rows)
        self.assertEqual(result['status'], 'confirmed_estimate')
        self.assertEqual(result['event']['frame_id'], 20)
        self.assertTrue(rows[20]['feet'][0]['association']['bounded_gap'])
        self.assertIn('bounded_observation_gap',result['event']['evidence'])

    def test_three_missing_frames_do_not_invent_an_approach(self):
        rows = observations()
        for row in rows[17:20]: row['feet'] = []
        self.assertIsNone(estimate(rows)['event'])

    def test_recent_shoe_scale_survives_tiny_last_fragment(self):
        rows = []
        for i in range(7):
            width = 4 if i==3 else 30
            feet = [] if i==4 else [dict(point=[80+i*8,100],
                bbox=[80+i*8-width/2,80,80+i*8+width/2,101], area=width*15)]
            rows.append(dict(frame_id=i,camera_transform=np.eye(3).tolist(),feet=feet))
        associate_feet(rows)
        self.assertEqual(len({f['foot_track_id'] for row in rows for f in row['feet']}),1)

    def test_appearance_conflict_across_gap_starts_a_new_track(self):
        rows = observations()[:4]
        for i,row in enumerate(rows):
            row['feet'][0]['appearance'] = [1.,0.] if i<2 else [0.,1.]
        rows[2]['feet'] = []
        associate_feet(rows)
        self.assertNotEqual(rows[1]['feet'][0]['foot_track_id'],rows[3]['feet'][0]['foot_track_id'])

    def test_wrong_recovered_shoe_cannot_change_primary_track_identity(self):
        rows=[]
        for i in range(4):
            feet=[]
            if i in (0,3):
                y=100+i*10
                feet=[dict(point=[100,y],bbox=[90,y-10,110,y+1],area=180,observation_source='inferred')]
            if i==2:
                feet=[dict(point=[100,80],bbox=[90,70,110,81],area=180,observation_source='recovered_rgb')]
            rows.append(dict(frame_id=i,timestamp=i/30,camera_transform=np.eye(3).tolist(),feet=feet))
        associate_feet(rows)
        self.assertEqual(rows[0]['feet'][0]['foot_track_id'],rows[3]['feet'][0]['foot_track_id'])
        self.assertNotEqual(rows[2]['feet'][0]['foot_track_id'],rows[3]['feet'][0]['foot_track_id'])

    def test_consistent_recovered_pixels_fill_only_a_bracketed_gap(self):
        rows=[]
        for i in range(4):
            y=100+i*3
            feet=[dict(point=[100,y],bbox=[90,y-10,110,y+1],area=180,
                observation_source='inferred' if i in (0,3) else 'recovered_rgb')]
            rows.append(dict(frame_id=i,timestamp=i/30,camera_transform=np.eye(3).tolist(),feet=feet))
        associate_feet(rows)
        self.assertEqual(len({row['feet'][0]['foot_track_id'] for row in rows}),1)
        self.assertEqual(rows[1]['feet'][0]['association']['source'],'bracketed_rgb_recovery')

    def test_rgb_recovery_observes_pixels_and_has_a_two_frame_budget(self):
        frame, person, labels, parser, camera = recovery_scene()
        recovery = ShoeRecovery(); recovery.update(frame,labels,person,parser,0,camera)
        for i in (1,2):
            observed, diagnostic = recovery.update(frame,np.zeros_like(labels),person,parser,i,camera)
            self.assertEqual(diagnostic['recovered_shoes'],1)
            np.testing.assert_array_equal(observed,labels)
        observed, diagnostic = recovery.update(frame,np.zeros_like(labels),person,parser,3,camera)
        self.assertFalse(diagnostic['attempted'])
        self.assertEqual(parser.parse.call_count,2)
        self.assertFalse(observed.any())

    def test_rgb_recovery_rejects_conflicting_appearance(self):
        frame, person, labels, parser, camera = recovery_scene()
        recovery = ShoeRecovery(); recovery.update(frame,labels,person,parser,0,camera)
        changed = np.full_like(frame,[230,20,20])
        observed, diagnostic = recovery.update(changed,np.zeros_like(labels),person,parser,1,camera)
        self.assertEqual(diagnostic['recovered_shoes'],0)
        self.assertFalse(observed.any())

    def test_prediction_without_current_rgb_shoe_pixels_never_returns_a_point(self):
        frame, person, labels, parser, camera = recovery_scene()
        recovery = ShoeRecovery(); recovery.update(frame,labels,person,parser,0,camera)
        parser.parse.side_effect = lambda crop: np.zeros(crop.shape[:2],np.uint8)
        observed, diagnostic = recovery.update(frame,np.zeros_like(labels),person,parser,1,camera)
        self.assertEqual(diagnostic['recovered_shoes'],0)
        self.assertFalse(observed.any())

    def test_registration_failure_clears_recovery_history(self):
        frame, person, labels, parser, camera = recovery_scene()
        recovery = ShoeRecovery(); recovery.update(frame,labels,person,parser,0,camera)
        bad = dict(camera_stable=False,camera_segment=1)
        recovery.update(frame,np.zeros_like(labels),person,parser,1,bad)
        _, diagnostic = recovery.update(frame,np.zeros_like(labels),person,parser,2,camera)
        self.assertFalse(diagnostic['attempted'])

    def test_pending_candidate_has_no_confirmed_point_or_crosshair(self):
        rows = observations(); result = estimate(rows); result['status']='needs_review'
        result['failure_reason']='truncated_contact_track'; result['target_hit']='unknown'
        canvas = np.zeros((240,320,3),np.uint8)
        with patch('demo15.render.cv2.drawMarker') as marker,patch('demo15.render.draw_label'):
            mark_event(canvas,rows[30],result,'落点',(0,255,0))
            marker.assert_not_called()
        with patch('demo15.render.draw_panel') as panel:
            annotate(canvas,rows[30],result)
        text='\n'.join(panel.call_args.args[1])
        self.assertIn('候选时间',text)
        self.assertNotIn('落点 [',text)

    def test_counter_evidence_configuration_is_validated(self):
        with self.assertRaises(ValueError): ContactConfig(counter_evidence_seconds=.05)


if __name__=='__main__': unittest.main()
