"""Known non-elliptical geometry and failure-mode tests for Demo14."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from demo14.tracker import Config,TargetTracker,cyclic_path,samples,enforce_ring_order
from demo14.render import annotate
from run_demo14 import parse_args


def synthetic(offset=(0,0),deform=True,occlude=False,blank_center=False):
    frame=np.full((480,800,3),(45,105,50),np.uint8)
    center=np.array([410.,330.])+offset
    theta=np.linspace(0,2*np.pi,1440,endpoint=False)
    curves={}
    for ring in range(10,0,-1):
        radius=ring/10
        wobble=(.025*np.sin(5*theta)+.014*np.cos(3*theta))*radius if deform else 0
        xy=np.stack([(radius+wobble)*290*np.cos(theta),
                     (radius+wobble)*85*np.sin(theta)],axis=1)+center
        curves[ring]=xy
        points=np.rint(xy).astype(np.int32)
        if ring==10: cv2.fillPoly(frame,[points],(225,230,235))
        elif ring==1: cv2.fillPoly(frame,[points],(40,40,40))
        else: cv2.polylines(frame,[points],True,(65,65,65),1,cv2.LINE_AA)
    if blank_center:
        cv2.ellipse(frame,tuple(center.astype(int)),(35,12),0,0,360,(225,230,235),-1)
    if occlude:
        cv2.rectangle(frame,(455,260),(515,395),(35,35,35),-1)
    return frame,center,curves


class Demo14Tests(unittest.TestCase):
    def track(self,frame,count=7):
        t=TargetTracker()
        for i in range(count): r=t.update(frame,i,i/30)
        return t,r

    def test_nonelliptical_image_supported_geometry(self):
        frame,center,curves=synthetic()
        _,r=self.track(frame)
        self.assertEqual(r['center']['source'],'observed')
        self.assertLess(np.linalg.norm(np.array(r['center']['point'])-center),3)
        errors=[];count=0
        for ring in r['rings']:
            gt=curves[ring['candidate_index']]
            for sample in ring['samples']:
                if sample['source']=='observed':
                    count+=1
                    errors.append(np.linalg.norm(gt-sample['point'],axis=1).min())
        self.assertGreater(count,240*.35*8)
        self.assertLess(np.median(errors),3)
        self.assertEqual(json.loads(json.dumps(r,allow_nan=False))['image_size'],[800,480])

    def test_confirmation_then_recovery_after_cut(self):
        f,_,_=synthetic();t=TargetTracker()
        self.assertIsNone(t.update(f,0,0)['center']['point'])
        for i in range(1,5):t.update(f,i,i/30)
        r=t.update(np.zeros_like(f),5,5/30)
        self.assertEqual(r['status'],'searching')
        self.assertFalse(r['rings'])
        for i in range(6,11):r=t.update(f,i,i/30)
        self.assertEqual(r['center']['source'],'observed')

    def test_no_center_fabricated_from_paper(self):
        f,_,_=synthetic(blank_center=True)
        _,r=self.track(f)
        self.assertIsNone(r['center']['point'])
        self.assertTrue(r['outer_boundary'])

    def test_center_prediction_expires_without_refresh(self):
        f,_,_=synthetic();t,r=self.track(f)
        with patch('demo14.tracker.center_region',return_value=None):
            early=t.update(f,7,7/30)
            self.assertEqual(early['center']['source'],'predicted')
            for i in range(8,26):r=t.update(f,i,i/30)
        self.assertIsNone(r['center']['point'])
        self.assertFalse(r['rings'])

    def test_occlusion_does_not_become_observed_ring(self):
        f,_,_=synthetic();t,_=self.track(f)
        occluded,_,_=synthetic(occlude=True)
        for i in range(7,27):r=t.update(occluded,i,i/30)
        interior=[]
        for ring in r['rings']:
            for p in ring['samples']:
                if p['point'] is not None:
                    x,y=p['point']
                    if 460<x<510 and 265<y<390:interior.append(p)
        self.assertFalse(interior)

    def test_camera_translation(self):
        f,c,_=synthetic();t,_=self.track(f)
        for i in range(7,15):
            moved,center,_=synthetic(offset=(i-6,0))
            r=t.update(moved,i,i/30)
        self.assertIsNotNone(r['center']['point'])
        self.assertLess(np.linalg.norm(np.array(r['center']['point'])-center),3)

    def test_rewind_resets_confirmation_and_dimensions(self):
        f,_,_=synthetic();t,_=self.track(f)
        r=t.update(f,0,0)
        self.assertIsNone(r['center']['point'])
        r=t.update(cv2.resize(f,(400,240)),1,1/30)
        self.assertEqual(r['image_size'],[400,240])
        self.assertIsNone(r['center']['point'])

    def test_cyclic_path_tracks_a_wavy_ridge(self):
        x=np.arange(120);truth=np.rint(12+4*np.sin(x*2*np.pi/120)).astype(int)
        score=np.zeros((120,25),np.float32);score[x,truth]=20
        self.assertLess(np.max(abs(cyclic_path(score)-truth)),2)

    def test_null_samples_and_render_do_not_mutate_input(self):
        self.assertEqual(samples([[np.nan,np.nan]])[0],dict(point=None,confidence=0.,source='unknown'))
        f,_,_=synthetic();_,r=self.track(f);original=f.copy()
        view=annotate(f,r,12)
        np.testing.assert_array_equal(f,original)
        self.assertEqual(view.shape,f.shape)

    def test_offcenter_bullseye_is_not_outer_ellipse_center(self):
        f,center,_=synthetic()
        cv2.ellipse(f,tuple(center.astype(int)),(35,12),0,0,360,(225,230,235),-1)
        moved=center+np.array([15.,-7.])
        cv2.ellipse(f,tuple(moved.astype(int)),(29,9),0,0,360,(40,40,40),-1)
        _,r=self.track(f)
        self.assertIsNotNone(r['center']['point'])
        self.assertLess(np.linalg.norm(np.array(r['center']['point'])-moved),3)

    def test_broad_shadow_and_broken_rings(self):
        f,center,curves=synthetic()
        illumination=np.linspace(.75,1.,f.shape[1])[None,:,None]
        f=np.uint8(f*illumination)
        cv2.rectangle(f,(500,325),(560,345),(185,190,195),-1)
        _,r=self.track(f)
        self.assertEqual(r['center']['source'],'observed')
        errors=[]
        for ring in r['rings']:
            for sample in ring['samples']:
                if sample['source']=='observed':
                    errors.append(np.linalg.norm(curves[ring['candidate_index']]-sample['point'],axis=1).min())
        self.assertGreater(len(errors),200)
        self.assertLess(np.median(errors),3)

    def test_headless_export_frame_sync(self):
        import tempfile
        from run_demo14 import run
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.mp4';output=Path(tmp)/'overlay.mp4';data=Path(tmp)/'points.jsonl'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),30,(800,480))
            self.assertTrue(writer.isOpened())
            for _ in range(6):writer.write(synthetic()[0])
            writer.release()
            run(parse_args(['--video',str(source),'--no-preview','--output-video',str(output),'--output-jsonl',str(data)]))
            rows=[json.loads(line) for line in data.read_text().splitlines()]
            self.assertEqual([r['frame_index'] for r in rows],list(range(6)))
            self.assertAlmostEqual(rows[-1]['timestamp'],5/30,places=5)
            cap=cv2.VideoCapture(str(output));self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),6)
            self.assertEqual((int(cap.get(3)),int(cap.get(4))),(800,480));cap.release()

    def test_glare_disk_dark_rim_center(self):
        f,c,_=synthetic()
        pos=tuple(c.astype(int))
        cv2.ellipse(f,pos,(29,9),0,0,360,(190,190,190),-1)
        cv2.ellipse(f,pos,(29,9),0,5,175,(55,55,55),2)
        cv2.line(f,(int(c[0]),int(c[1]-40)),(int(c[0]),int(c[1]-1)),(60,60,60),1)
        _,r=self.track(f)
        self.assertIsNotNone(r['center']['point'])
        self.assertLess(np.linalg.norm(np.array(r['center']['point'])-c),5)

    def test_pause_step_overlay_restart_output(self):
        import tempfile
        from run_demo14 import run
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.mp4';data=Path(tmp)/'data.jsonl'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),30,(800,480))
            for _ in range(6):writer.write(synthetic()[0])
            writer.release()
            with patch('run_demo14.cv2.imshow') as show, \
                 patch('run_demo14.cv2.waitKey',side_effect=[ord(' '),ord('o'),ord('n'),ord('r'),ord('q')]), \
                 patch('run_demo14.cv2.getWindowProperty',return_value=1), \
                 patch('run_demo14.cv2.destroyAllWindows'):
                result=run(parse_args(['--video',str(source),'--output-jsonl',str(data)]))
            self.assertEqual(show.call_count,5)
            self.assertEqual(result['frames'],1)
            rows=[json.loads(line) for line in data.read_text().splitlines()]
            self.assertEqual([r['frame_index'] for r in rows],[0])

    def test_unprinted_paper_does_not_create_prior_rings(self):
        f,c,curves=synthetic()
        cv2.fillPoly(f,[np.rint(curves[10]).astype(np.int32)],(225,230,235))
        cv2.ellipse(f,tuple(c.astype(int)),(29,9),0,0,360,(40,40,40),-1)
        _,r=self.track(f)
        self.assertEqual(r['center']['source'],'observed')
        self.assertTrue(r['rings'])
        self.assertFalse(any(p['point'] is not None for ring in r['rings'] for p in ring['samples']))

    def test_ring_order_in_exported_samples(self):
        f,c,_=synthetic()
        _,r=self.track(f)
        for inner,outer in zip(r['rings'],r['rings'][1:]):
            for a,b in zip(inner['samples'],outer['samples']):
                if a['point'] is not None and b['point'] is not None:
                    self.assertLess(np.linalg.norm(np.array(a['point'])-c),np.linalg.norm(np.array(b['point'])-c))

    def test_crossed_prediction_discards_confirmation_history(self):
        rings=[dict(candidate_index=2,ring_id=2,samples=samples([[30,0]],sources=['predicted'])),
               dict(candidate_index=3,ring_id=3,samples=samples([[20,0]]))]
        state={k:dict(hits=np.array([9]),seen=np.array([1.]),points=np.array([[float(k),0.]])) for k in (2,3)}
        enforce_ring_order(rings,[0,0],state)
        self.assertTrue(all(r['samples'][0]['point'] is None for r in rings))
        self.assertTrue(all(v['hits'][0]==0 and np.isneginf(v['seen'][0]) for v in state.values()))

    def test_cli_refuses_overwriting_input(self):
        with self.assertRaises(SystemExit):parse_args(['--video','a.mp4','--output-video','a.mp4'])
        with self.assertRaises(ValueError):Config(ring_count=0)


if __name__=='__main__':unittest.main()
