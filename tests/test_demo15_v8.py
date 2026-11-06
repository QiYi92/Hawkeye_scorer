"""Lateral discovery, identity ambiguity and observed rolling contacts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from demo15.athlete import CandidateAssociator,candidate_sequences,choose_landing_athlete
from demo15.core import ContactConfig,estimate
from demo15.target_hit import target_decision
from demo15.recovery import ShoeRecovery
from test_demo15 import observations
from test_demo15_contact_v7 import recovery_scene


def detection(source,box):
    return dict(track_id=source,bbox=box,confidence=.8,appearance=np.ones((16,16),np.float32)/256)


def region_observations(sliding=False):
    rows=observations()
    for i,row in enumerate(rows):
        foot=row['feet'][0];x=100+(i-20)*10 if sliding else 100
        y=foot['point'][1]
        foot.update(point=[x+(12 if i%2 else -12) if i>=20 and not sliding else x,y],
                    bbox=[x-15,y-20,x+16,y+1],area=350,
                    sole_boundary=[[x+j,y] for j in range(-15,16)])
        row['person_bbox']=[x-10+(i-20)*2 if i>=20 else x-10,y-100,x+20+(i-20)*2 if i>=20 else x+20,y-10]
    return rows


class DiscoveryTests(unittest.TestCase):
    def test_temporary_detector_ids_are_linked_before_role_selection(self):
        tracker=CandidateAssociator(30);ids=[]
        for n in range(9):
            item=detection(-100*n-1,[350-n*12,310,450-n*12,470])
            tracker.update([item],n,(640,480));ids.append(item['candidate_id'])
        self.assertEqual(len(set(ids)),1)

    def test_low_lateral_motion_is_eligible_without_upper_image_entry(self):
        tracker=CandidateAssociator(30);frames=[]
        for n in range(9):
            item=detection(-100*n-1,[350-n*12,310,450-n*12,470])
            frames.append(tracker.update([item],n,(640,480)))
        streams,profiles=candidate_sequences(frames,[None]*9,(640,480))
        self.assertEqual(len(streams),1)
        self.assertTrue(profiles[0]['eligible'])

    def test_clipped_arrival_can_grow_before_getting_a_tracker_id(self):
        tracker=CandidateAssociator(30);ids=[]
        for n,box in enumerate(([625,330,639,440],[610,330,639,440],[590,330,639,440],[570,330,639,440])):
            item=detection(-n-1,box);tracker.update([item],n,(640,480));ids.append(item['candidate_id'])
        self.assertEqual(len(set(ids)),1)

    def test_stationary_and_tiny_background_candidates_are_not_parsed(self):
        tracker=CandidateAssociator(30);frames=[]
        for n in range(8):
            frames.append(tracker.update([detection(1,[200,250,250,450]),
                detection(2,[20+n,300,30+n,310])],n,(640,480)))
        streams,profiles=candidate_sequences(frames,[None]*8,(640,480))
        self.assertFalse(streams)
        self.assertEqual({p['reason'] for p in profiles},{'stationary_candidate','small_candidate'})

    def test_opposing_people_do_not_merge_at_crossing(self):
        tracker=CandidateAssociator(30);observed={1:[],2:[]}
        for n in range(8):
            a=detection(1,[100+10*n,100,130+10*n,220]);b=detection(2,[240-10*n,130,270-10*n,250])
            a['appearance']=np.array([[1.,0.]],np.float32);b['appearance']=np.array([[0.,1.]],np.float32)
            for item in tracker.update([a,b] if n%2 else [b,a],n,(640,480)):
                observed[item['track_id']].append(item['candidate_id'])
        self.assertEqual(len(set(observed[1])),1);self.assertEqual(len(set(observed[2])),1)
        self.assertNotEqual(observed[1][0],observed[2][0])

    def test_walking_step_and_depth_noise_do_not_create_landing_role(self):
        rows=observations()
        for row in rows:
            x,y=row['feet'][0]['point'];row['person_bbox']=[x-12,20,x+12,110]
        chosen,diag,_=choose_landing_athlete({'walker':rows},ContactConfig())
        self.assertIsNone(chosen)
        self.assertFalse(diag[0]['role_accepted'])

    def test_seated_approach_with_contact_can_initialize_a_landing_role(self):
        rows=observations()
        for row in rows:
            x,y=row['feet'][0]['point'];row['person_bbox']=[x+10,y-80,x+100,y+5]
        chosen,_,_=choose_landing_athlete({'lateral':rows},ContactConfig())
        self.assertEqual(chosen,'lateral')

    def test_two_supported_arrivals_stay_ambiguous(self):
        rows=observations()
        for row in rows:
            x,y=row['feet'][0]['point'];row['person_bbox']=[x+10,y-80,x+100,y+5]
        chosen,_,reason=choose_landing_athlete({'one':rows,'two':rows},ContactConfig())
        self.assertIsNone(chosen);self.assertEqual(reason,'ambiguous_landing_athletes')

    def test_bootstrap_requires_current_observed_class_two_pixels(self):
        frame,person,labels,parser,camera=recovery_scene();recovery=ShoeRecovery()
        observed,diagnostic=recovery.update(frame,np.zeros_like(labels),person,parser,0,camera)
        self.assertTrue(diagnostic['bootstrap']);self.assertTrue(observed.any())
        self.assertEqual(recovery.bootstrap_calls,1)
        parser.parse.side_effect=lambda crop:np.zeros(crop.shape[:2],np.uint8)
        observed,_=ShoeRecovery().update(frame,np.zeros_like(labels),person,parser,0,camera)
        self.assertFalse(observed.any())


class RollingTests(unittest.TestCase):
    def test_toe_heel_point_switch_does_not_hide_observed_region_support(self):
        result=estimate(region_observations())
        self.assertEqual(result['status'],'confirmed_estimate')
        self.assertEqual(result['event']['frame_id'],20)
        self.assertEqual(result['event']['contact_kind'],'sliding_or_rolling')

    def test_sliding_contact_retains_original_off_target_event(self):
        rows=region_observations(sliding=True)
        for row in rows:
            row['paper_outline']=dict(usable=True,points=[[130,50],[300,50],[300,180],[130,180]])
        result=estimate(rows)
        self.assertEqual(result['event']['frame_id'],20)
        self.assertEqual(target_decision(rows,result)['target_hit'],'miss')
        self.assertGreater(rows[28]['feet'][0]['point'][0],130)

    def test_airborne_region_plateau_followed_by_descent_is_rejected(self):
        rows=region_observations(sliding=True)
        for i,row in enumerate(rows):
            if i>=24:
                dy=(i-23)*8;foot=row['feet'][0];foot['point'][1]+=dy
                foot['bbox'][1]+=dy;foot['bbox'][3]+=dy
                foot['sole_boundary']=[[x,y+dy] for x,y in foot['sole_boundary']]
        self.assertIsNone(estimate(rows)['event'])

    def test_fast_translation_with_truncated_region_cannot_certify_support(self):
        rows=region_observations(sliding=True)[:25]
        result=estimate(rows)
        self.assertNotEqual(result['status'],'confirmed_estimate')
        self.assertFalse(any(c['frame_id']==20 and c['state']=='support_confirmed'
                             for c in result['candidate_diagnostics']))

    def test_early_filter_failures_have_measured_diagnostics(self):
        rows=observations()
        for i,row in enumerate(rows):row['feet'][0]['point'][0]+=i*8
        result=estimate(rows)
        self.assertTrue(any(c.get('reason') and 'horizontal_point_range' in c['reason']
                            for c in result['candidate_diagnostics']))


if __name__=='__main__':unittest.main()
