"""Strict acceptance and annotation-free motion/observation regressions."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from demo15.core import estimate, add_depth_features
from demo15.evidence import CameraRegistration, associate_feet, complete_athlete_track, contact_boundary_point
from demo15.extraction import load_cache
from demo15.references import EvaluationConfig, evaluate
from test_demo15 import observations


class OptimizedContactTests(unittest.TestCase):
    def test_acceptance_boundaries_and_review_failure(self):
        rows=observations();result=estimate(rows)
        for delta in (-3,-2,0,2,3):
            for distance in (10.,10.01):
                ref=dict(frame_id=20+delta,timestamp=(20+delta)/30,point_px=[100+distance,100])
                score=evaluate(result,ref,rows)
                self.assertEqual(score['joint_pass'],abs(delta)<=2 and distance<=10)
        result['status']='needs_review'
        self.assertFalse(evaluate(result,dict(frame_id=20,timestamp=20/30,point_px=[100,100]),rows)['joint_pass'])
        with self.assertRaises(ValueError):EvaluationConfig(frame_tolerance=-1)
        with self.assertRaises(ValueError):EvaluationConfig(position_tolerance_px=float('nan'))

    def test_visual_support_survives_bad_depth(self):
        rows=observations()
        for row in rows:row['feet'][0]['depth_residual']=None
        result=estimate(rows,'fusion')
        self.assertEqual(result['status'],'confirmed_estimate')
        self.assertEqual(result['event']['frame_id'],20)
        self.assertIsNone(estimate(rows,'depth')['event'])

    def test_no_ground_evidence_cannot_confirm(self):
        rows=observations()
        for row in rows:
            row['feet'][0].update(ground_valid=False,ground_region=False)
        self.assertIsNone(estimate(rows)['event'])

    def test_visual_mode_does_not_borrow_depth_ground_validity(self):
        rows=observations()
        for row in rows:row['feet'][0]['ground_region']=False
        self.assertEqual(estimate(rows,'fusion')['status'],'confirmed_estimate')
        self.assertIsNone(estimate(rows,'visual')['event'])

    def test_contact_uses_supported_lower_run_not_band_average_or_noise(self):
        boundary=[[x,100] for x in range(10,18)]+[[x,99] for x in range(18,35)]+[[35,103]]
        self.assertEqual(contact_boundary_point(boundary,[25,99]),[14,100])

    def test_strong_short_approach_can_confirm_with_observed_support(self):
        rows=observations()
        for row in rows:
            i=row['frame_id']
            if i<18:row['feet']=[]
            elif i<20:row['feet'][0]['point'][1]=100-(20-i)*5
        self.assertEqual(estimate(rows)['status'],'confirmed_estimate')

    def test_compensated_motion_and_origin_frame_are_preserved(self):
        rows=observations()
        for i,row in enumerate(rows):
            row['camera_transform']=[[1,0,i],[0,1,i*.5],[0,0,1]]
            row['camera_moved']=True
            row['feet'][0]['point'][0]+=i
            row['feet'][0]['point'][1]+=i*.5
        result=estimate(rows)
        self.assertEqual(result['event']['frame_id'],20)
        self.assertEqual(result['event']['point_px'],[120,110])
        self.assertGreater(result['event']['confirmation_frame'],20)

    def test_camera_registration_accepts_translation_and_rejects_blank(self):
        rng=np.random.default_rng(52)
        gray=rng.integers(0,256,(240,320),dtype=np.uint8)
        gray=cv2.GaussianBlur(gray,(3,3),0)
        first=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        moved=cv2.warpAffine(first,np.float32([[1,0,4],[0,1,2]]),(320,240))
        camera=CameraRegistration();mask=np.zeros((240,320),bool)
        camera.update(first,mask)
        result=camera.update(moved,mask)
        self.assertTrue(result['camera_stable']);self.assertTrue(result['camera_moved'])
        np.testing.assert_allclose(np.asarray(result['camera_transform'])[:2,2],[4,2],atol=.3)
        failed=camera.update(np.zeros_like(first),mask)
        self.assertFalse(failed['camera_stable'])
        self.assertGreater(failed['camera_segment'],result['camera_segment'])

    def test_different_registration_segments_cannot_pass_position(self):
        rows=observations();result=estimate(rows)
        rows[21]['camera_segment']=1
        score=evaluate(result,dict(frame_id=21,timestamp=.7,point_px=[100,100]),rows)
        self.assertFalse(score['position_pass'])

    def test_depth_reference_does_not_pool_across_registration_failure(self):
        rows=observations()[:6]
        for i,row in enumerate(rows):
            row['camera_segment']=i//3
            row['feet'][0]['shoe_depth']=2.
        depths=np.full((6,60,80),2.,np.float32)
        add_depth_features(rows,depths,np.zeros_like(depths,bool),np.ones_like(depths,bool))
        # Each coordinate segment has only three observations, below the
        # minimum support of five; combining the two would invent reliability.
        self.assertTrue(all(not row['feet'][0]['ground_valid'] for row in rows))

    def test_fast_horizontal_airborne_plateau_is_not_contact(self):
        rows=observations()
        for i,row in enumerate(rows):row['feet'][0]['point'][0]+=i*8
        self.assertIsNone(estimate(rows)['event'])

    def test_airborne_pause_then_resumed_descent_is_rejected(self):
        rows=observations()
        for i,row in enumerate(rows):
            if i>=24:row['feet'][0]['point'][1]=100+4*(i-23)
        self.assertIsNone(estimate(rows)['event'])

    def test_two_shoes_keep_identity_through_crossing_and_one_missing_frame(self):
        rows=[]
        for i in range(12):
            feet=[]
            for label,x,y in [('a',20+i*4,80),('b',64-i*4,88)]:
                if i==5 and label=='a':continue
                feet.append(dict(point=[x,y],bbox=[x-6,y-8,x+7,y+1],area=80,label=label))
            if i%2:feet.reverse()
            rows.append(dict(frame_id=i,camera_transform=np.eye(3).tolist(),feet=feet))
        associate_feet(rows)
        ids={label:{f['foot_track_id'] for row in rows for f in row['feet'] if f['label']==label} for label in ('a','b')}
        self.assertEqual(len(ids['a']),1);self.assertEqual(len(ids['b']),1)
        self.assertNotEqual(ids['a'],ids['b'])

    def test_long_observation_gap_does_not_claim_first_contact(self):
        rows=observations()
        for row in rows[14:21]:row['feet']=[]
        self.assertIsNone(estimate(rows)['event'])

    def test_missing_supporting_foot_does_not_jump_to_other_shoe_fragment(self):
        rows=[]
        for i in range(7):
            feet=[]
            if i!=3:
                feet.append(dict(point=[100,100],bbox=[70,75,130,101],area=900,label='support'))
            feet.append(dict(point=[230+i*5,95],bbox=[220+i*5,75,250+i*5,96],area=400,label='swing'))
            if i==3:feet.append(dict(point=[238,99],bbox=[237,96,241,100],area=10,label='fragment'))
            rows.append(dict(frame_id=i,camera_transform=np.eye(3).tolist(),feet=feet))
        associate_feet(rows)
        support={f['foot_track_id'] for row in rows for f in row['feet'] if f['label']=='support'}
        fragment={f['foot_track_id'] for row in rows for f in row['feet'] if f['label']=='fragment'}
        self.assertEqual(len(support),1)
        self.assertFalse(support & fragment)

    def test_offline_identity_uses_observed_boxes_not_interpolation(self):
        detections=[]
        for i in range(6):
            detections.append([dict(track_id=10,bbox=[10+i,20+i,30+i,60+i],
                                    appearance=np.ones((16,16),np.float32)/256)])
        selections=[None]*5+[detections[5][0]]
        detections[2]=[]
        completed=complete_athlete_track(detections,selections,30)
        self.assertIsNone(completed[2])
        self.assertIs(completed[0],detections[0][0])
        self.assertIs(completed[4],detections[4][0])

    def test_estimation_never_reads_reference_files(self):
        rows=observations()
        with patch('builtins.open',side_effect=AssertionError('inference must not read annotations')), \
             patch.object(Path,'read_text',side_effect=AssertionError('inference must not read annotations')):
            self.assertEqual(estimate(rows)['status'],'confirmed_estimate')

    def test_old_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'metadata.json').write_text(json.dumps(dict(frames=0)))
            with self.assertRaisesRegex(ValueError,'已过期'):load_cache(directory)


if __name__=='__main__':unittest.main()
