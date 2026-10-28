"""Silhouette safety, compute-shortcut and recognition-panel behavior."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from demo15.core import estimate,ContactConfig
from demo15.target_hit import target_decision,paper_outline
from demo15.postprocess import analyze_modes,ContactRefiner
from demo15.render import annotate,recognition_status
from test_demo15 import observations


def scene(left=140):
    rows=observations()
    for row in rows:
        row['paper_outline']=dict(usable=True,points=[[left,70],[280,70],[280,160],[left,160]])
        row['person_bbox']=[90,60,120,105]
    result=estimate(rows)
    return rows,result,rows[20]['feet'][0]


class TargetHitTests(unittest.TestCase):
    def test_clear_miss_requires_whole_shoe_with_margin(self):
        rows,result,foot=scene()
        decision=target_decision(rows,result,foot)
        self.assertEqual(decision['target_hit'],'miss');self.assertTrue(decision['early_miss'])
        foot['bbox']=[90,80,142,101]
        self.assertFalse(target_decision(rows,result,foot)['early_miss'])

    def test_inside_and_outer_line_count_as_hit(self):
        rows,result,foot=scene(80)
        self.assertEqual(target_decision(rows,result,foot)['target_hit'],'hit')
        result['event']['point_px']=[80,100]
        self.assertEqual(target_decision(rows,result,foot)['target_hit'],'hit')

    def test_near_edge_must_refine(self):
        rows,result,foot=scene(107)
        self.assertFalse(target_decision(rows,result,foot)['early_miss'])

    def test_missing_or_occluded_outlines_are_unknown(self):
        rows,result,foot=scene()
        for row in rows:row['paper_outline']['usable']=False
        self.assertEqual(target_decision(rows,result,foot)['target_hit'],'unknown')
        self.assertFalse(target_decision(rows,result,foot)['early_miss'])
        self.assertFalse(paper_outline(None,np.zeros((40,60),bool))['usable'])

    def test_outline_quality_rejects_person_occlusion_and_image_clipping(self):
        paper=np.zeros((100,160),np.uint8)
        cv2.rectangle(paper,(40,30),(120,90),255,-1)
        contours,_=cv2.findContours(paper,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        found=(contours[0],None,paper)
        excluded=np.zeros_like(paper,bool)
        self.assertTrue(paper_outline(found,excluded)['usable'])
        excluded[20:60,35:50]=True
        self.assertFalse(paper_outline(found,excluded)['usable'])
        clipped=contours[0]-np.array([[[40,0]]],np.int32)
        self.assertFalse(paper_outline((clipped,None,paper),np.zeros_like(excluded))['usable'])

    def test_registration_failure_or_segment_gap_cannot_trigger_shortcut(self):
        rows,result,foot=scene();rows[20]['camera_stable']=False
        self.assertFalse(target_decision(rows,result,foot)['early_miss'])
        rows[20]['camera_stable']=True;rows[20]['camera_segment']=1
        self.assertEqual(target_decision(rows,result,foot)['target_hit'],'unknown')

    def test_camera_motion_maps_outline_into_contact_frame(self):
        rows,result,foot=scene()
        for i,row in enumerate(rows):
            row['camera_transform']=[[1,0,i],[0,1,0],[0,0,1]]
            row['paper_outline']['points']=[[140+i,70],[280+i,70],[280+i,160],[140+i,160]]
        result['event']['point_px']=[160,100]
        self.assertEqual(target_decision(rows,result,foot)['target_hit'],'hit')

    def test_unconfirmed_airborne_position_never_classifies_miss(self):
        rows,result,foot=scene();result['status']='needs_review'
        self.assertEqual(target_decision(rows,result,foot)['target_hit'],'unknown')
        self.assertFalse(target_decision(rows,result,foot)['early_miss'])
        rows,result,foot=scene(80)
        # Earlier airborne samples lie outside; the confirmed landing is inside.
        for row in rows[:15]:row['feet'][0]['point']=[20,20]
        self.assertEqual(target_decision(rows,result,foot)['target_hit'],'hit')

    def test_consistent_history_rejects_one_recent_partial_paper(self):
        rows,result,foot=scene(80)
        rows[20]['paper_outline']['points']=[[190,90],[280,90],[280,160],[190,160]]
        decision=target_decision(rows,result,foot)
        self.assertEqual(decision['target_hit'],'hit')
        self.assertNotIn(20,decision['outline_frames'])

    def test_miss_skips_refinement_and_rings_entirely(self):
        rows,result,_=scene()
        with patch('demo15.postprocess.estimate',side_effect=lambda *a:copy.deepcopy(result)), \
             patch('demo15.postprocess.ContactRefiner') as refiner, \
             patch('demo15.postprocess.full_target') as rings:
            refiner.return_value.calls=0;refiner.return_value.seconds=0
            refiner.return_value.parser_inferences=0;refiner.return_value.reuses=0;refiner.return_value.frames={}
            outputs,_,timing,_=analyze_modes('unused',rows,ContactConfig(),('fusion','visual'),'unused','unused')
            refiner.return_value.refine.assert_not_called();rings.assert_not_called()
            self.assertEqual(timing['refinement_calls'],0);self.assertEqual(timing['ring_calls'],0)
            for r in outputs.values():
                self.assertEqual(r['target_hit'],'miss');self.assertEqual(r['point_precision'],'coarse')

    def test_disabling_shortcut_runs_precision_and_shares_rings(self):
        rows,result,_=scene()
        with patch('demo15.postprocess.estimate',side_effect=lambda *a:copy.deepcopy(result)), \
             patch('demo15.postprocess.ContactRefiner') as refiner, \
             patch('demo15.postprocess.full_target',return_value={}) as rings:
            refiner.return_value.calls=0;refiner.return_value.seconds=0
            refiner.return_value.parser_inferences=0;refiner.return_value.reuses=0;refiner.return_value.frames={}
            refiner.return_value.refine.return_value=rows[20]['feet']
            analyze_modes('unused',rows,ContactConfig(),('fusion','visual'),'unused','unused',shortcut=False)
            self.assertEqual(refiner.return_value.refine.call_count,2)
            rings.assert_called_once()

    def test_border_miss_refines_but_skips_unnecessary_rings(self):
        rows,result,_=scene(107)
        fine=copy.deepcopy(rows[20]['feet'])
        fine[0].update(contact_point=[100,100],contact_point_source='two_scale_rgb_agreement')
        with patch('demo15.postprocess.estimate',side_effect=lambda *a:copy.deepcopy(result)), \
             patch('demo15.postprocess.ContactRefiner') as refiner, \
             patch('demo15.postprocess.full_target') as rings:
            refiner.return_value.calls=0;refiner.return_value.seconds=0
            refiner.return_value.parser_inferences=0;refiner.return_value.reuses=0;refiner.return_value.frames={}
            refiner.return_value.refine.return_value=fine
            outputs,_,_,_=analyze_modes('unused',rows,ContactConfig(),('fusion',),'unused','unused')
            refiner.return_value.refine.assert_called_once();rings.assert_not_called()
            self.assertEqual(outputs['fusion']['target_hit'],'miss')
            self.assertEqual(outputs['fusion']['point_precision'],'refined')
            self.assertFalse(outputs['fusion']['processing']['refinement_skipped'])

    def test_status_lifecycle_and_review(self):
        _,result,_=scene();result['target_hit']='miss'
        self.assertEqual(recognition_status(result,19),'待识别')
        self.assertEqual(recognition_status(result,20),'待确认')
        self.assertEqual(recognition_status(result,30),'脱靶')
        result['target_hit']='hit';self.assertEqual(recognition_status(result,30),'落靶')
        result['status']='needs_review';self.assertEqual(recognition_status(result,30),'待确认')

    def test_refinement_cache_reuses_model_work_and_returns_independent_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video=root/'source.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(64,64))
            for _ in range(3):writer.write(np.zeros((64,64,3),np.uint8))
            writer.release()
            np.savez(root/'athlete_masks.npz',masks=np.full((3,512),255,np.uint8),
                     valid=np.ones(3,bool),image_shape=[64,64])
            labels=np.zeros((64,64),np.uint8);labels[20:26,10:30]=2
            row=dict(frame_id=1,person_bbox=[0,0,64,64],feet=[dict(point=[20,25],bbox=[10,20,30,26],foot_track_id=0)])
            with patch('demo15.postprocess.SCHPParser') as parser, \
                 patch('demo15.postprocess.parse_athlete',return_value=labels) as parse:
                refiner=ContactRefiner(video,root,root,'cpu')
                first=refiner.refine(row);second=refiner.refine(row)
                parser.assert_called_once();parse.assert_called_once()
                self.assertEqual(refiner.calls,1);self.assertEqual(refiner.reuses,1)
                self.assertEqual(refiner.parser_inferences,2)
                first[0]['point'][0]=-1
                self.assertNotEqual(second[0]['point'][0],-1)

    def test_miss_panel_has_no_automatic_precise_coordinate(self):
        rows,result,_=scene();result.update(target_hit='miss',point_precision='coarse')
        with patch('demo15.render.draw_panel') as panel,patch('demo15.render.draw_label'):
            annotate(np.zeros((240,320,3),np.uint8),rows[19],result)
            before='\n'.join(panel.call_args.args[1])
            self.assertIn('识别情况：待识别',before)
            self.assertNotIn('脱靶，已跳过',before)
            annotate(np.zeros((240,320,3),np.uint8),rows[30],result)
            text='\n'.join(panel.call_args.args[1])
            self.assertIn('识别情况：脱靶',text);self.assertIn('跳过精确落点',text)
            self.assertNotIn('[100, 100]',text)


if __name__=='__main__':unittest.main()
