"""Fixed event geometry, confirmation gating and display-only labels."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from demo15.target_plan import project_point, contact_plan_point, draw_plan, draw_ring_labels, ring_score, score_text
from demo15.rings import geometry, RingConfig


class TargetPlanTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[200., 0., 300.], [0., 100., 250.], [.1, .05, 1.]])
        self.model = geometry(self.matrix, [1.]*10, .2, 'test', 10, RingConfig(), [10])
        self.result = dict(status='confirmed_estimate', target_hit='hit', event=dict(
            frame_id=10, confirmation_frame=13, point_px=[300.,250.]))

    def test_projective_mapping_and_invalid_geometry(self):
        np.testing.assert_allclose(project_point([300,250], self.model), [0,0], atol=1e-12)
        from demo15.rings import project
        point=project(self.matrix, [[.4,.3]])[0]
        mapped=project_point(point, self.model)
        self.assertAlmostEqual(np.linalg.norm(mapped), .5)
        for matrix in (np.zeros((3,3)), np.full((3,3), np.nan), np.eye(2)):
            self.assertIsNone(project_point(point, {**self.model, 'homography':matrix}))
        self.assertIsNone(project_point(point, {}))

    def test_fixed_event_geometry_and_temporal_gating(self):
        rows=[dict(frame_id=10,target={'ring_model':self.model}),
              dict(frame_id=20,target={'ring_model':{}})]
        point=contact_plan_point(rows,self.result)
        with patch('demo15.target_plan.draw_target_plan') as draw:
            for frame in (9,10,13,20,9,13):
                draw_plan(np.zeros((720,1280,3),np.uint8),frame,self.result,point)
                self.assertEqual(draw.call_args.args[2], point if frame>=13 else None)
            draw_plan(None,20,self.result,None)
            self.assertIn('暂无靶盘定位',draw.call_args.args[3])
            draw_plan(None,20,{**self.result,'point_precision':'coarse','target_hit':'miss'},point)
            self.assertIsNone(draw.call_args.args[2])
            self.assertIn('脱靶',draw.call_args.args[3])
            draw_plan(None,0,{'event':None},None)
            self.assertEqual(draw.call_args.args[3],'未确认首次触地')

    def test_all_ring_interiors_boundaries_and_outside(self):
        self.assertEqual(ring_score((0,0),'hit'),0)
        for ring in range(1,11):
            radius=ring/10
            for r in (radius-.05, radius-1e-6, radius):
                self.assertEqual(ring_score((r,0),'hit'),(ring-1)*20)
            self.assertEqual(ring_score((radius+1e-6,0),'hit'),min(200,ring*20))
        self.assertEqual(ring_score((3,0),'hit'),200)
        self.assertEqual(ring_score(None,'miss','coarse'),200)
        for p in (None,(float('nan'),0),(float('inf'),0)):
            self.assertIsNone(ring_score(p,'hit'))
        self.assertIsNone(ring_score((.2,0),'unknown'))
        self.assertIsNone(ring_score((.2,0),'hit','coarse'))

    def test_score_confirmation_and_replay(self):
        for frame in (9,10,12,13,20,9,13):
            self.assertEqual(score_text(self.result,frame,(.35,0)),
                             '估计分数：60分' if frame>=13 else None)
        self.assertEqual(score_text(self.result,13,None),'分数待确认')
        self.assertIsNone(score_text({**self.result,'status':'needs_review'},20,(.1,0)))
        self.assertEqual(score_text({**self.result,'target_hit':'miss','point_precision':'coarse'},13,None),'脱靶：200分')

    def test_main_marker_and_plan_share_score(self):
        from demo15.render import annotate, annotated_depth
        row=dict(frame_id=13,timestamp=1.,image_size=[1280,720],feet=[],target={})
        result={**self.result,'mode':'fusion','failure_reason':None}
        result['event']={**result['event'],'timestamp':1.}
        with patch('demo15.render.draw_label') as label, patch('demo15.target_plan.draw_target_plan') as plan:
            for render in (lambda: annotate(np.zeros((720,1280,3),np.uint8),row,result,plan_point=(.35,0)),
                           lambda: annotated_depth(np.ones((72,128)),row,result,plan_point=(.35,0))):
                render()
                self.assertEqual(label.call_args.args[1],'估计分数：60分')
                self.assertEqual(plan.call_args.args[4],'估计分数：60分')

    def test_plan_uses_demo15_metric_scale(self):
        from demo15.target_plan import plan_text
        with patch('demo15.target_plan.draw_label') as label:
            draw_plan(np.zeros((720,1280,3),np.uint8),0,{'event':None},None)
            values=[call.args[1] for call in label.call_args_list if call.args[1].endswith('m')]
            self.assertEqual(sorted(values),[f'{i/5:.1f}m' for i in range(1,11)])

    def test_ten_metric_labels_within_image(self):
        for h,w in ((360,640),(720,1280),(1080,1920)):
            with patch('demo15.target_plan.cv2.putText') as text:
                draw_ring_labels(np.zeros((h,w,3),np.uint8),self.model)
                labels=[call.args[1] for call in text.call_args_list[::2]]
                self.assertEqual(sorted(labels),[f'{i/5:.1f}m' for i in range(1,11)])
                for call in text.call_args_list:
                    x,y=call.args[2]
                    self.assertTrue(0<=x<w and 0<=y<h)

if __name__ == '__main__': unittest.main()
