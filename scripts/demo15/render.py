"""Evidence preview, full-frame export and independent manual review."""
from pathlib import Path
import time
import cv2
import numpy as np
from demo13.display_text import draw_panel,draw_label
from .extraction import save_json

STATUS={'confirmed_estimate':'已确认触地估计','needs_review':'需要复核','unconfirmed':'未确认'}
REASON={'missing_prior_depth_separation':'缺少此前离地深度证据，早期候选待复核',
        'preceding_observation_gap':'触地前观测中断，需人工复核',
        'insufficient_observed_approach':'触地前真实鞋部观测不足',
        'no_observed_support_transition':'未检测到可靠的鞋底支撑转折',
        'unreliable_background_registration':'背景配准不可靠',
        'truncated_contact_track':'候选后鞋部观测中断，尚不能确认支撑',
        'weak_support_anchor':'缺少独立的局部支撑证据',
        'no_depth_ground_transition':'未检测到可靠的深度触地变化'}


def depth_view(depth):
    valid=np.isfinite(depth)&(depth>0)
    colored=np.zeros((*depth.shape,3),np.uint8)
    if valid.any():
        lo,hi=np.percentile(depth[valid],[2,98])
        gray=np.zeros(depth.shape,np.uint8)
        gray[valid]=np.clip((depth[valid]-lo)/max(hi-lo,1e-6)*255,0,255).astype(np.uint8)
        colored=cv2.applyColorMap(gray,cv2.COLORMAP_INFERNO);colored[~valid]=0
    return colored


def mark_event(canvas,o,result,label,color):
    event=result.get('event') if result else None
    if not event or o['frame_id']<event['frame_id']:return
    point=np.array([*event['point_px'],1.])
    if event.get('camera_transform') is not None:
        point=np.asarray(o['camera_transform'])@np.linalg.inv(np.asarray(event['camera_transform']))@point
    x,y=map(round,point[:2]/point[2]);h,w=canvas.shape[:2]
    if result.get('target_hit')=='miss':
        pending=o['frame_id']<event.get('confirmation_frame',event['frame_id'])
        if pending:color=(0,200,255)
        cv2.circle(canvas,(x,y),20,color,2)
        draw_label(canvas,'候选区域' if pending else ('脱靶区域（粗略）' if result.get('point_precision')=='coarse' else '脱靶区域'),max(0,min(x+24,w-230)),max(25,min(y-16,h-10)),color,size=18)
        return
    confirmed=result['status']=='confirmed_estimate' and o['frame_id']>=event.get('confirmation_frame',event['frame_id'])
    if not confirmed:
        color=(0,200,255)
        # Pending events are regions, not precise scored contact crosshairs.
        cv2.circle(canvas,(x,y),14,color,2)
        draw_label(canvas,'候选区域（待确认）',max(0,min(x+18,w-230)),max(25,min(y-16,h-10)),color,size=18)
        return
    suffix='估计'
    cv2.circle(canvas,(x,y),9,color,2)
    cv2.drawMarker(canvas,(x,y),color,cv2.MARKER_CROSS,24,2)
    draw_label(canvas,f'{label}（{suffix}）',max(0,min(x+14,w-230)),max(25,min(y-16,h-10)),color,size=18)


def recognition_status(result, frame_id):
    event=result.get('event')
    if not event or frame_id<event['frame_id']:return '待识别'
    if result['status']!='confirmed_estimate' or frame_id<event.get('confirmation_frame',event['frame_id']):
        return '待确认'
    return {'hit':'落靶','miss':'脱靶','unknown':'待确认'}.get(result.get('target_hit'),'待确认')


def annotate(frame,o,result,reference=None,trace=None,depth_result=None,view_title='Demo15'):
    canvas=frame.copy();h,w=canvas.shape[:2]
    coarse_miss=result.get('target_hit')=='miss'
    for f in o['feet']:
        x1,y1,x2,y2=f['bbox'];cv2.rectangle(canvas,(x1,y1),(x2,y2),(220,60,220),1)
        cv2.circle(canvas,tuple(f['point']),3,(220,60,220),-1)
    model=o['target'].get('ring_model',{})
    if model.get('status')=='fitted':
        for ring in reversed(model['rings']):
            color=(0,220,255) if ring['ring_id']==1 else (255,220,0)
            curve=np.rint(np.asarray(ring['points'])*256).astype(np.int32)
            cv2.polylines(canvas,[curve],True,color,1,cv2.LINE_AA,shift=8)
        cv2.drawMarker(canvas,tuple(map(round,model['center'])),(0,220,255),cv2.MARKER_CROSS,12,1,cv2.LINE_AA)
    event=result.get('event')
    depth_event=(depth_result or {}).get('event')
    same_event=bool(event and depth_event and event['frame_id']==depth_event['frame_id'] and event['point_px']==depth_event['point_px'])
    if depth_result and result['mode']!='depth' and not same_event and not coarse_miss:
        mark_event(canvas,o,depth_result,'深度候选',(0,200,255))
    label={'fusion':'深度＋运动落点','depth':'深度落点','visual':'视觉落点'}[result['mode']]
    mark_event(canvas,o,result,label,(0,255,0) if result['status']=='confirmed_estimate' else (0,200,255))
    if reference and not coarse_miss:
        reference_point=np.array([*reference['point_px'],1.])
        if reference.get('camera_transform') is not None:
            reference_point=np.asarray(o['camera_transform'])@np.linalg.inv(np.asarray(reference['camera_transform']))@reference_point
        cv2.drawMarker(canvas,tuple(map(round,reference_point[:2]/reference_point[2])),(0,0,255),cv2.MARKER_CROSS,18,1)
    visible_event=event or (depth_result or {}).get('event')
    point=(visible_event['point_px'] if visible_event else (o['feet'][0]['point'] if o['feet'] else None))
    if point and not coarse_miss:
        x,y=map(int,point);crop=canvas[max(0,y-70):min(h,y+40),max(0,x-80):min(w,x+80)]
        if crop.size:
            cw=min(260,w//3);ch=min(180,h//3)
            canvas[h-ch:h,w-cw:w]=cv2.resize(crop,(cw,ch))
    if trace:
        gw=min(400,w//3);gh=min(100,h//5);left=max(0,w-gw);top=250
        if top+gh<h:
            canvas[top:top+gh,left:w]=25
            points=[]
            for n,value in enumerate(trace):
                if value is not None:
                    points.append((left+round(n*gw/max(1,len(trace)-1)),top+gh-1-round(min(value,.5)/.5*(gh-1))))
                else:
                    if len(points)>1:cv2.polylines(canvas,[np.array(points,np.int32)],False,(0,230,230),1)
                    points=[]
            if len(points)>1:cv2.polylines(canvas,[np.array(points,np.int32)],False,(0,230,230),1)
            x=left+round(o['frame_id']*gw/max(1,len(trace)-1));cv2.line(canvas,(x,top),(x,top+gh),(255,255,255),1)
            near=result.get('configuration',{}).get('near',.08)
            yy=top+gh-1-round(min(near,.5)/.5*(gh-1));cv2.line(canvas,(left,yy),(w-1,yy),(0,160,0),1)
            draw_label(canvas,'鞋与地面深度差',left+4,top+22,(230,230,230),size=15)
    if event:
        # Contact confirmation is independent of target visibility/classification.
        confirmed=(result['status']=='confirmed_estimate' and o['frame_id']>=event.get('confirmation_frame',event['frame_id']))
        text=(f'触地时间 {event["timestamp"]:.3f}秒；落点 {event["point_px"]}' if confirmed else
              (f'候选时间 {event["timestamp"]:.3f}秒；候选区域待确认' if o['frame_id']>=event['frame_id'] else '等待首次触地'))
    else:text=f'原因：{REASON.get(result["failure_reason"],result["failure_reason"])}'
    if coarse_miss:
        status=recognition_status(result,o['frame_id'])
        miss_text='脱靶，已跳过精确落点计算' if result.get('point_precision')=='coarse' else '脱靶，边界落点已复核'
        text=(f'触地时间 {event["timestamp"]:.3f}秒；{miss_text}' if status=='脱靶'
              else ('等待首次触地' if status=='待识别' else '首次触地候选确认中'))
    reference_text=(f'参考：{reference["timestamp"]:.3f}秒，点 {reference["point_px"]}' if reference else '本视频无手动参考，不计算准确率')
    if coarse_miss and reference:reference_text=f'参考时间：{reference["timestamp"]:.3f}秒'
    temporal_status=STATUS[result['status']]
    if event and o['frame_id']<event['frame_id']:temporal_status='等待首次触地'
    elif event and o['frame_id']<event.get('confirmation_frame',event['frame_id']):temporal_status='触地候选待确认'
    lines=[f'{view_title}｜{result["mode"]}｜{temporal_status}',
           f'识别情况：{recognition_status(result,o["frame_id"])}',
           '靶环：10/10（拟合）' if model.get('status')=='fitted' else '靶环待确认',
           f'帧 {o["frame_id"]}｜视频时间 {o["timestamp"]:.3f}秒｜{text}',
           '绿色：已确认估计；黄色：深度候选/待复核；红色：参考；紫色：鞋部',
           reference_text]
    if depth_result and depth_result.get('event') and result['mode']!='depth' and not coarse_miss:
        de=depth_result['event'];lines.append(f'深度候选：{de["timestamp"]:.3f}秒，点 {de["point_px"]}｜{STATUS[depth_result["status"]]}')
    if event and result.get('failure_reason'):
        lines.append(REASON.get(result['failure_reason'],result['failure_reason']))
    comparison=result
    if reference and comparison.get('event') and not coarse_miss:
        ce=comparison['event'];error=comparison.get('evaluation',{}).get('position_error_px')
        position=f'｜位置差 {error:.2f}像素' if error is not None else ''
        lines.append(f'自动点与参考：{ce["frame_id"]-reference["frame_id"]:+d}帧｜{(ce["timestamp"]-reference["timestamp"])*1000:+.0f}毫秒{position}')
    draw_panel(canvas,lines,max_width=min(w,1000),size=18)
    return canvas


def contact_grid(video, path, rows, result, reference=None):
    """Evaluation-only native-pixel crops, after prediction has finished."""
    event=result.get('event')
    centers=[]
    if reference:centers.append(('reference',reference['frame_id'],reference['point_px']))
    if event:centers.append(('prediction',event['frame_id'],event['point_px']))
    if not centers:return
    wanted={max(0,min(len(rows)-1,n+d)) for _,n,_ in centers for d in range(-2,3)}
    capture=cv2.VideoCapture(str(video));frames={}
    try:
        for i in range(max(wanted)+1):
            ok,frame=capture.read()
            if not ok:raise RuntimeError('触地放大图解码失败')
            if i in wanted:frames[i]=frame
    finally:capture.release()
    strips=[]
    for label,n,point in centers:
        tiles=[]
        for d in range(-2,3):
            i=max(0,min(len(rows)-1,n+d));canvas=frames[i].copy()
            def mapped(p,source):
                if rows[i].get('camera_segment',0)!=rows[source].get('camera_segment',0):return None
                value=np.asarray(rows[i]['camera_transform'])@np.linalg.inv(np.asarray(rows[source]['camera_transform']))@np.array([*p,1.])
                return tuple(map(round,value[:2]/value[2]))
            for item,color in ((reference,(0,0,255)),(event,(0,255,0))):
                if item:
                    p=mapped(item['point_px'],item['frame_id'])
                    if p:cv2.drawMarker(canvas,p,color,cv2.MARKER_CROSS,11,1)
            for foot in rows[i]['feet']:
                boundary=foot.get('refined_sole_boundary',foot.get('sole_boundary',[]))
                if boundary:cv2.polylines(canvas,[np.asarray(boundary,np.int32)],False,(220,80,220),1)
            center=mapped(point,n) or tuple(map(round,point))
            x,y=center;h,w=canvas.shape[:2]
            x=max(80,min(w-80,x));y=max(65,min(h-55,y))
            tile=cv2.resize(canvas[y-65:y+55,x-80:x+80],(320,240),interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(tile,(0,0),(320,25),(20,20,20),-1)
            cv2.putText(tile,f'{label} {i} ({d:+d})',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            tiles.append(tile)
        strips.append(np.hstack(tiles))
    cv2.imwrite(str(Path(path)/'contact_grid.png'),np.vstack(strips))


def read_frame(video,index):
    # Sequential decoding avoids the existing marker's seek/off-by-one issue.
    cap=cv2.VideoCapture(str(video));frame=None
    try:
        for _ in range(index+1):
            ok,frame=cap.read()
            if not ok:raise RuntimeError('复核帧解码失败')
    finally:cap.release()
    return frame


class ReviewReader:
    """Keep sequential playback decoding open; rewind only for backward navigation."""
    def __init__(self,video):
        self.video=str(video);self.cap=cv2.VideoCapture(self.video)
        self.index=-1;self.frame=None

    def read(self,index):
        if index<self.index:
            self.cap.release();self.cap=cv2.VideoCapture(self.video);self.index=-1
        while self.index<index:
            ok,self.frame=self.cap.read()
            if not ok:raise RuntimeError('复核帧解码失败')
            self.index+=1
        return self.frame

    def close(self):self.cap.release()


def annotated_depth(depth,o,result,reference=None,depth_result=None):
    w,h=o['image_size']
    canvas=cv2.resize(depth_view(depth),(w,h))
    return annotate(canvas,o,result,reference,depth_result=depth_result,view_title='Demo15 深度视图')


def export(video,path,rows,metadata,result,reference,depth_path=None,depth_result=None):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    w,h=metadata['image_size'];fps=metadata['fps']
    cap=cv2.VideoCapture(str(video));writer=cv2.VideoWriter(str(path/'video_overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():raise RuntimeError('无法创建叠加视频')
    event=result.get('event');clip=None;reference_clip=None
    if reference and reference['frame_id']<len(rows):reference={**reference,'camera_transform':rows[reference['frame_id']]['camera_transform']}
    trace=[min((f['depth_residual'] for f in o['feet'] if f.get('depth_residual') is not None),default=None) for o in rows]
    if event:
        clip=cv2.VideoWriter(str(path/'contact_review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not clip.isOpened():raise RuntimeError('无法创建复核视频')
    elif reference:
        reference_clip=cv2.VideoWriter(str(path/'reference_review.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not reference_clip.isOpened():raise RuntimeError('无法创建参考复核视频')
    count=0
    try:
        for i,o in enumerate(rows):
            ok,frame=cap.read()
            if not ok:raise RuntimeError('导出时视频提前结束')
            view=annotate(frame,o,result,reference,trace=trace,depth_result=depth_result)
            writer.write(view);count+=1
            if event and i==event['frame_id']:
                if not cv2.imwrite(str(path/'first_contact_frame.png'),view):raise RuntimeError('截图写入失败')
            depth_event=(depth_result or {}).get('event')
            if depth_event and i==depth_event['frame_id']:
                if not cv2.imwrite(str(path/'depth_contact_frame.png'),view):raise RuntimeError('深度候选截图写入失败')
            if clip and abs(o['timestamp']-event['timestamp'])<=1.:clip.write(view)
            if reference and i==reference['frame_id']:
                if not cv2.imwrite(str(path/'reference_comparison.png'),view):raise RuntimeError('参考对比截图写入失败')
            if reference_clip and abs(o['timestamp']-reference['timestamp'])<=1.:reference_clip.write(view)
    finally:
        cap.release();writer.release()
        if clip:clip.release()
        if reference_clip:reference_clip.release()
    save_json(path/'first_contact.json',result)
    save_json(path/'export.json',dict(frames=count,fps=fps,audio=False,encoding='nominal_fps_CFR'))


def preview(video,path,rows,result,reference,depth_path=None,scale=.75,show_depth=True,depth_result=None,autoplay=True):
    name='Demo15 - Review';cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    event=result.get('event');initial_event=event or (depth_result or {}).get('event')
    index=0
    if reference and reference['frame_id']<len(rows):reference={**reference,'camera_transform':rows[reference['frame_id']]['camera_transform']}
    paused=not autoplay;point=None;dirty=True;depths=None
    if show_depth and depth_path and Path(depth_path).exists():
        with np.load(depth_path) as cache:depths=cache['depth']
    trace=[min((f['depth_residual'] for f in o['feet'] if f.get('depth_residual') is not None),default=None) for o in rows]
    depth_name='Demo15 - Depth';depth_open=depths is not None
    if depth_open:cv2.namedWindow(depth_name,cv2.WINDOW_NORMAL)
    def click(kind,x,y,flags,unused):
        nonlocal point,dirty,paused
        if kind==cv2.EVENT_LBUTTONDOWN:
            point=[max(0,min(rows[index]['image_size'][0]-1,round(x/scale))),max(0,min(rows[index]['image_size'][1]-1,round(y/scale)))];dirty=True;paused=True
    cv2.setMouseCallback(name,click)
    if depth_open:cv2.setMouseCallback(depth_name,click)
    reader=ReviewReader(video)
    timestamps=np.array([o['timestamp'] for o in rows])
    playback_start=time.perf_counter();playback_time=rows[0]['timestamp']
    try:
        while True:
            if depth_open and cv2.getWindowProperty(depth_name,cv2.WND_PROP_VISIBLE)<1:
                depth_open=False
            if dirty:
                frame=reader.read(index)
                views=[(name,annotate(frame,rows[index],result,reference,trace=trace,depth_result=depth_result))]
                if depth_open:
                    views.append((depth_name,annotated_depth(depths[index],rows[index],result,reference,depth_result)))
                state='播放结束' if paused and index==len(rows)-1 else ('已暂停' if paused else '连续播放中')
                for window,view in views:
                    if point:cv2.drawMarker(view,tuple(point),(255,255,0),cv2.MARKER_CROSS,20,2)
                    draw_panel(view,[f'{state}｜空格播放/暂停｜A/D前后帧｜G预测｜T参考｜R从头播放',
                                     '点击暂停并修正｜S保存修正｜Q/Esc退出'],bottom=True,size=16)
                    cv2.imshow(window,cv2.resize(view,None,fx=scale,fy=scale))
                dirty=False
            next_time=rows[index+1]['timestamp'] if index<len(rows)-1 else rows[index]['timestamp']+1/30
            remaining=next_time-(playback_time+time.perf_counter()-playback_start)
            delay=30 if paused else max(1,round(remaining*1000))
            key=cv2.waitKey(delay)&255
            if key in (ord('q'),27) or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:break
            if key==32:
                paused=not paused;dirty=True
                playback_start=time.perf_counter();playback_time=rows[index]['timestamp']
            elif key in (ord('a'),ord('d')):
                index=max(0,min(len(rows)-1,index+(-1 if key==ord('a') else 1)));paused=True;point=None;dirty=True
            elif key==ord('g') and initial_event:index=initial_event['frame_id'];dirty=True;paused=True;point=None
            elif key==ord('t') and reference:index=reference['frame_id'];dirty=True;paused=True;point=None
            elif key==ord('r'):
                index=0;dirty=True;paused=False;point=None
                playback_start=time.perf_counter();playback_time=rows[0]['timestamp']
            elif key==ord('s') and point:
                save_json(Path(path)/'review.json',dict(source='manual_review',frame_id=index,timestamp=rows[index]['timestamp'],point_px=point,automatic_result_unchanged=True));print('人工复核已单独保存',flush=True)
            if not paused and key not in (32,ord('r')):
                if index<len(rows)-1:
                    elapsed=playback_time+time.perf_counter()-playback_start
                    index=min(len(rows)-1,max(index+1,int(np.searchsorted(timestamps,elapsed,side='right')-1)))
                    dirty=True;point=None
                else:paused=True;dirty=True
    finally:
        reader.close();cv2.destroyWindow(name)
        if depth_open:cv2.destroyWindow(depth_name)
