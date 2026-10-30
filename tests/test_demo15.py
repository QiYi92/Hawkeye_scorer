"""Behavior tests for reference isolation, depth evidence and joint acceptance."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from demo15.core import aligned_depth,background_reference,add_depth_features,estimate,ContactConfig,region
from demo15.references import timestamp,evaluate,load_reference
from demo15.render import preview,ReviewReader,annotated_depth
from run_demo15 import tune


def observations(running=False):
    rows=[]
    for i in range(40):
        y=60+2*i if i<20 else 100
        x=100+(i-20)*2 if running and i>=20 else 100
        f=dict(point=[x,y],bbox=[x-4,y-10,x+5,y+1],ground_valid=True,ground_region=True,
               depth_residual=.25 if i<20 else .02)
        rows.append(dict(frame_id=i,timestamp=i/30,image_size=[320,240],feet=[f],
                         camera_stable=True,camera_transform=np.eye(3).tolist(),target={}))
    return rows


class Demo15Tests(unittest.TestCase):
    def test_depth_resize_and_invalid(self):
        self.assertEqual(aligned_depth(np.ones((4,6)),(8,12)).shape,(8,12))
        with self.assertRaises(ValueError):aligned_depth(np.zeros((4,6)),(8,12))

    def test_background_masks_and_scale_drift(self):
        values=np.stack([np.ones((20,20))*v for v in [1,1.1,.9,1.2,1,1.1]])
        exclusions=np.zeros_like(values,bool);exclusions[:,0,0]=True
        ref,valid,scales=background_reference(values,exclusions,np.ones_like(values,bool),[True]*6)
        self.assertFalse(valid[0,0]);self.assertTrue(valid[5,5]);self.assertAlmostEqual(scales[1]/scales[0],1.1,places=5)

    def test_ground_reference_survives_current_shoe_occlusion(self):
        rows=observations()[:6]
        for row in rows:
            row['image_size']=[320,240]
            row['feet'][0]['point']=[160,120]
            row['feet'][0]['shoe_depth']=2.
        depths=np.full((6,60,80),2.,np.float32)
        grounds=np.ones_like(depths,bool);grounds[5,30,40]=False
        exclusions=np.zeros_like(grounds);exclusions[5,30,40]=True
        add_depth_features(rows,depths,exclusions,grounds)
        self.assertTrue(rows[5]['feet'][0]['ground_valid'])
        self.assertAlmostEqual(rows[5]['feet'][0]['depth_residual'],0)

    def test_earliest_depth_transition_and_backdated_confirmation(self):
        result=estimate(observations(),'depth')
        self.assertEqual(result['status'],'confirmed_estimate')
        self.assertEqual(result['event']['frame_id'],20)
        self.assertGreater(result['event']['confirmation_frame'],20)

    def test_running_is_allowed(self):
        result=estimate(observations(True),'fusion')
        self.assertEqual(result['event']['frame_id'],20)

    def test_missing_ground_does_not_confirm(self):
        rows=observations()
        for o in rows:o['feet'][0]['ground_valid']=False
        self.assertIsNone(estimate(rows,'depth')['event'])

    def test_without_prior_separation_depth_does_not_trigger(self):
        rows=observations()
        for o in rows:o['feet'][0]['depth_residual']=.01
        self.assertIsNone(estimate(rows,'depth')['event'])
        self.assertIsNotNone(estimate(rows,'visual')['event'])

    def test_late_lock_latches_early_review_candidate_without_claiming_contact(self):
        rows=observations()
        for o in rows:
            o['feet'][0]['depth_residual']=.02
            if o['frame_id']<18:o['feet']=[]
        for mode in ('depth','fusion'):
            result=estimate(rows,mode)
            self.assertEqual(result['event']['frame_id'],20)
            self.assertEqual(result['status'],'needs_review')
            self.assertEqual(result['failure_reason'],'missing_prior_depth_separation' if mode=='depth' else 'insufficient_observed_approach')
            self.assertNotIn('depth_separation_to_ground_convergence',result['event']['evidence'])

    def test_later_walking_depth_transition_cannot_replace_early_incomplete_candidate(self):
        rows=observations()
        for o in rows:
            o['feet'][0]['depth_residual']=.25 if o['frame_id'] in (30,31) else .02
            if o['frame_id']<18:o['feet']=[]
        result=estimate(rows,'fusion')
        self.assertEqual(result['event']['frame_id'],20)
        self.assertEqual(result['status'],'needs_review')

    def test_late_lock_needs_depth_and_continuous_motion_support(self):
        rows=observations()
        for o in rows:
            if o['frame_id']<18:o['feet']=[]
            else:o['feet'][0]['depth_residual']=None
        result=estimate(rows,'fusion')
        self.assertEqual(result['status'],'needs_review')
        self.assertEqual(result['failure_reason'],'insufficient_observed_approach')

    def test_shadow_or_depth_outlier_cannot_confirm(self):
        rows=observations();rows[21]['feet'][0]['depth_residual']=.3
        result=estimate(rows,'depth')
        self.assertNotEqual(result.get('event',{}).get('frame_id'),20)

    def test_single_gap_with_real_observations_on_both_sides(self):
        rows=observations();rows[19]['feet']=[]
        result=estimate(rows,'fusion')
        self.assertEqual(result['status'],'confirmed_estimate')
        self.assertIn('bounded_observation_gap',result['event']['evidence'])

    def test_camera_motion_rejects_contact(self):
        rows=observations()
        for o in rows:o['camera_stable']=False
        self.assertIsNone(estimate(rows)['event'])

    def test_propagated_shoes_cannot_confirm(self):
        rows=observations()
        for o in rows:o['parts_source']='propagated'
        self.assertIsNone(estimate(rows)['event'])

    def test_invalid_contact_configuration(self):
        with self.assertRaises(ValueError):ContactConfig(near=.2,separation=.1)

    def test_tuning_skips_invalid_threshold_combinations(self):
        rows=observations();ref=dict(frame_id=20,timestamp=20/30,point_px=[100,100])
        config,trials=tune([(rows,ref),(rows,ref)])
        self.assertTrue(trials);self.assertLess(config.near,config.separation)
        self.assertTrue(all(t['config']['near']<t['config']['separation'] for t in trials))

    def test_timestamp_fallback_is_incremental(self):
        self.assertEqual(timestamp(0.,2.,30.,61),(2.+1/30,'fps_increment_fallback'))
        with self.assertRaises(ValueError):timestamp(float('nan'),2.,0.,1)

    def test_joint_score_and_missing_event(self):
        rows=observations();result=estimate(rows)
        ref=dict(frame_id=20,timestamp=20/30,point_px=[103,104])
        score=evaluate(result,ref,rows)
        self.assertTrue(score['joint_pass']);self.assertAlmostEqual(score['position_error_px'],5)
        ref['point_px']=[111,100]
        self.assertFalse(evaluate(result,ref,rows)['joint_pass'])
        self.assertFalse(evaluate(dict(status='unconfirmed',event=None),ref,rows)['joint_pass'])

    def test_unreliable_registration_cannot_pass_position(self):
        rows=observations();result=estimate(rows);rows[21]['camera_stable']=False
        score=evaluate(result,dict(frame_id=21,timestamp=21/30,point_px=[100,100]),rows)
        self.assertTrue(score['time_pass']);self.assertFalse(score['position_pass']);self.assertEqual(score['raw_position_error_px'],0)

    def test_review_candidate_has_errors_but_cannot_pass(self):
        rows=observations();result=estimate(rows);result['status']='needs_review'
        score=evaluate(result,dict(frame_id=20,timestamp=20/30,point_px=[100,100]),rows)
        self.assertEqual(score['time_error_ms'],0);self.assertEqual(score['position_error_px'],0)
        self.assertFalse(score['joint_pass'])

    def test_region_unknown_on_partial_boundary(self):
        self.assertEqual(region([1,1],dict(outer_boundary=[dict(point=None,source='unknown')])),'unknown')
        samples=[dict(point=p,source='observed') for p in [[10,10],[50,10],[50,50],[10,50]]]
        self.assertEqual(region([70,70],dict(outer_boundary=samples)),'outside')
        self.assertEqual(region([30,30],dict(bullseye_boundary=samples,outer_boundary=[])),'bullseye')

    def test_reconcile_reference_by_image_without_mutating_annotation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);video=root/'source.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(64,48))
            for i in range(6):writer.write(np.full((48,64,3),i*35,np.uint8))
            writer.release();cap=cv2.VideoCapture(str(video))
            for _ in range(4):ok,frame=cap.read()
            cap.release();folder=root/'marks'/video.name;folder.mkdir(parents=True)
            cv2.imwrite(str(folder/'raw.png'),frame)
            data=dict(frame_index_base=0,coordinate_system='original_image_pixels_x_right_y_down',width=64,height=48,frame_count=6,
                      marks=[dict(frame_index=2,timestamp_seconds=.1,point=[20,20],image='raw.png')])
            original=json.dumps(data);(folder/'marks.json').write_text(original)
            ref=load_reference(video,root/'marks')
            self.assertEqual(ref['frame_id'],3);self.assertTrue(ref['frame_reconciled'])
            self.assertEqual((folder/'marks.json').read_text(),original)

    def test_review_navigation_and_manual_result_isolation(self):
        rows=observations();result=estimate(rows);before=json.dumps(result)
        ref=dict(frame_id=18,timestamp=18/30,point_px=[100,100])
        callback=[];seen=[];keys=iter([ord('a'),ord('d'),ord('t'),ord('g'),ord('s'),ord('q')])
        def key(unused):
            k=next(keys)
            if k==ord('s'):callback[0](cv2.EVENT_LBUTTONDOWN,60,60,0,None)
            return k
        def read(index):seen.append(index);return np.zeros((240,320,3),np.uint8)
        with tempfile.TemporaryDirectory() as temp,patch('demo15.render.cv2.namedWindow'),patch('demo15.render.cv2.setMouseCallback',side_effect=lambda name,fn:callback.append(fn)),patch('demo15.render.cv2.imshow'),patch('demo15.render.cv2.waitKey',side_effect=key),patch('demo15.render.cv2.getWindowProperty',return_value=1),patch('demo15.render.cv2.destroyWindow'),patch('demo15.render.ReviewReader') as reader:
            reader.return_value.read.side_effect=read
            preview('unused',temp,rows,result,ref,scale=.75)
            review=json.loads((Path(temp)/'review.json').read_text())
            self.assertEqual(review['frame_id'],20);self.assertEqual(review['point_px'],[80,80])
            self.assertTrue(review['automatic_result_unchanged']);self.assertEqual(json.dumps(result),before)
            self.assertEqual(seen[:5],[0,0,1,18,20])

    def test_preview_autoplays_and_synchronizes_annotated_depth(self):
        rows=observations();result=estimate(rows)
        with tempfile.TemporaryDirectory() as temp:
            cache=Path(temp)/'depth.npz';np.savez(cache,depth=np.ones((40,60,80),np.float32))
            for enabled in (True,False):
                with patch('demo15.render.time.perf_counter',return_value=0.),patch('demo15.render.cv2.namedWindow'),patch('demo15.render.cv2.setMouseCallback'),patch('demo15.render.cv2.imshow') as shown,patch('demo15.render.cv2.waitKey',side_effect=[-1,-1,ord('q')]),patch('demo15.render.cv2.getWindowProperty',return_value=1),patch('demo15.render.cv2.destroyWindow'),patch('demo15.render.ReviewReader') as reader,patch('demo15.render.annotated_depth',wraps=annotated_depth) as depth:
                    reader.return_value.read.return_value=np.zeros((240,320,3),np.uint8)
                    preview('unused',temp,rows,result,None,cache,show_depth=enabled)
                    self.assertEqual([c.args[0] for c in reader.return_value.read.call_args_list],[0,1,2])
                    self.assertEqual([c.args[1]['frame_id'] for c in depth.call_args_list],[0,1,2] if enabled else [])
                    self.assertEqual(len(shown.call_args_list),6 if enabled else 3)
                    reader.return_value.close.assert_called_once()

    def test_preview_keeps_source_time_when_display_is_slow(self):
        rows=observations();result=estimate(rows)
        with tempfile.TemporaryDirectory() as temp,patch('demo15.render.time.perf_counter',side_effect=[0.,.2,.2,.2]),patch('demo15.render.cv2.namedWindow'),patch('demo15.render.cv2.setMouseCallback'),patch('demo15.render.cv2.imshow'),patch('demo15.render.cv2.waitKey',side_effect=[-1,ord('q')]),patch('demo15.render.cv2.getWindowProperty',return_value=1),patch('demo15.render.cv2.destroyWindow'),patch('demo15.render.ReviewReader') as reader:
            reader.return_value.read.return_value=np.zeros((240,320,3),np.uint8)
            preview('unused',temp,rows,result,None)
            self.assertEqual([c.args[0] for c in reader.return_value.read.call_args_list],[0,6])

    def test_review_reader_reuses_decoder_and_rewinds_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            video=Path(temp)/'test.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(64,48))
            for i in range(12):writer.write(np.full((48,64,3),i*18,np.uint8))
            writer.release();capture=cv2.VideoCapture(str(video));expected=[]
            while True:
                ok,frame=capture.read()
                if not ok:break
                expected.append(frame)
            capture.release()
            with patch('demo15.render.cv2.VideoCapture',wraps=cv2.VideoCapture) as opened:
                reader=ReviewReader(video)
                try:
                    for i in range(12):np.testing.assert_array_equal(reader.read(i),expected[i])
                    self.assertEqual(opened.call_count,1)
                    np.testing.assert_array_equal(reader.read(3),expected[3])
                    np.testing.assert_array_equal(reader.read(4),expected[4])
                    self.assertEqual(opened.call_count,2)
                finally:reader.close()

    def test_depth_window_draws_text_and_original_pixel_landing_marker(self):
        rows=observations();result=estimate(rows);row=rows[result['event']['confirmation_frame']]
        row={**row,'image_size':[1280,720]}
        result['event']['point_px']=[638,524]
        with patch('demo15.render.draw_panel') as panel,patch('demo15.render.draw_label') as label:
            view=annotated_depth(np.ones((180,320),np.float32),row,result)
            self.assertIn('深度视图',panel.call_args.args[1][0])
            self.assertTrue(any('638' in line for line in panel.call_args.args[1]))
            label.assert_called()
            np.testing.assert_array_equal(view[524,638],[0,255,0])


if __name__=='__main__':unittest.main()
