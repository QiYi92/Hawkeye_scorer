"""逐帧视频点标注：Tkinter 界面，保存原图坐标及源视频时间戳。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}


class VideoMarker:
    def __init__(self, root, video_dir=ROOT / 'training_video', output_dir=ROOT / 'training_video_mark'):
        self.root = root
        self.video_dir, self.output_dir = Path(video_dir), Path(output_dir)
        self.cap = None
        self.frame = None
        self.frame_index = 0
        self.timestamp = 0.0
        self.pending = {}  # 帧号 -> 新点或 None（删除），切换帧仍保留未保存编辑
        self.saved = {}
        self.seek_job = None
        self.render_job = None
        self.setting_slider = False
        self.display_box = None
        self.photo = None
        self.root.title('训练视频触地点标记')
        self.root.geometry('1180x820')
        self.root.minsize(720, 500)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

        top = ttk.Frame(root, padding=10)
        top.pack(fill='x')
        ttk.Label(top, text='视频：').pack(side='left')
        self.video_choice = ttk.Combobox(top, state='readonly', width=28)
        self.video_choice.pack(side='left', padx=5)
        self.video_choice.bind('<<ComboboxSelected>>', self.select_video)
        ttk.Label(top, text='拖动进度条定位 → 点击画面标记 → 点击保存').pack(side='left', padx=15)

        self.canvas = tk.Canvas(root, background='#202124', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=10)
        self.canvas.bind('<Button-1>', self.click)
        self.canvas.bind('<Configure>', self.resize)

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill='x')
        self.info = tk.StringVar(value='请选择视频')
        ttk.Label(controls, textvariable=self.info).pack(anchor='w')
        self.slider = tk.Scale(controls, from_=0, to=0, orient='horizontal',
                               showvalue=False, resolution=1, command=self.seek_slider)
        self.slider.pack(fill='x')
        self.slider.bind('<ButtonRelease-1>', self.seek_release)
        row = ttk.Frame(controls)
        row.pack(fill='x', pady=5)
        ttk.Button(row, text='上一帧 ←', command=lambda: self.step(-1)).pack(side='left')
        ttk.Button(row, text='下一帧 →', command=lambda: self.step(1)).pack(side='left', padx=5)
        ttk.Label(row, text='跳至帧（从 0 开始）：').pack(side='left', padx=(15, 0))
        self.frame_entry = ttk.Entry(row, width=9)
        self.frame_entry.pack(side='left')
        self.frame_entry.bind('<Return>', self.jump)
        ttk.Button(row, text='跳转', command=self.jump).pack(side='left', padx=5)
        ttk.Button(row, text='清除当前点', command=self.clear_point).pack(side='right')
        ttk.Button(row, text='保存当前帧 (Ctrl/⌘+S)', command=self.save).pack(side='right', padx=10)
        self.status = tk.StringVar(value=f'保存目录：{self.output_dir}')
        ttk.Label(controls, textvariable=self.status, wraplength=1100).pack(anchor='w')
        root.bind('<Left>', lambda event: self.key_step(event, -1))
        root.bind('<Right>', lambda event: self.key_step(event, 1))
        root.bind('<Control-s>', lambda event: self.save())
        root.bind('<Command-s>', lambda event: self.save())

        if not self.video_dir.is_dir():
            messagebox.showerror('目录不存在', str(self.video_dir))
            return
        self.videos = sorted((p for p in self.video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS),
                             key=lambda p: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', p.name)])
        self.video_choice['values'] = [p.name for p in self.videos]
        if self.videos:
            self.video_choice.current(0)
            self.select_video()
        else:
            self.status.set(f'未找到视频：{self.video_dir}')

    def cancel_seek(self):
        if self.seek_job is not None:
            self.root.after_cancel(self.seek_job)
            self.seek_job = None

    def discard_ok(self):
        return not self.pending or messagebox.askyesno(
            '有未保存标记', f'还有 {len(self.pending)} 帧编辑未保存。是否放弃这些编辑？\n选择“否”可返回，逐帧点击保存。')

    def select_video(self, event=None):
        selected = self.video_choice.current()
        if self.cap is not None and not self.discard_ok():
            self.video_choice.current(self.selected_video)
            return
        self.cancel_seek()
        if self.cap is not None:
            self.cap.release()
        self.selected_video = selected
        self.path = self.videos[selected]
        self.cap = cv2.VideoCapture(str(self.path))
        self.frame = None
        self.pending, self.saved = {}, {}
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.annotation_path = self.output_dir / self.path.name / 'marks.json'
        if not self.cap.isOpened() or self.total <= 0:
            messagebox.showerror('无法读取视频', str(self.path))
            return
        if self.annotation_path.exists():
            try:
                data = json.loads(self.annotation_path.read_text(encoding='utf-8'))
                if (data['width'], data['height'], data['frame_count']) != (self.width, self.height, self.total):
                    raise ValueError('视频尺寸或帧数已改变，旧标注不能直接复用。')
                self.saved = {m['frame_index']: m for m in data['marks']}
            except (ValueError, KeyError, TypeError) as exc:
                messagebox.showerror('标注读取失败', f'{self.annotation_path}\n{exc}')
                self.cap.release()
                self.cap = None
                return
        self.slider.configure(to=self.total - 1)
        self.show_frame(0)
        self.status.set(f'已载入 {self.path.name}，已有 {len(self.saved)} 帧标注。点击画面设置一个点；再次点击可修改。')

    def seek_slider(self, value):
        if self.setting_slider or self.cap is None:
            return
        self.cancel_seek()
        self.seek_job = self.root.after(65, lambda: self.show_frame(int(float(value))))

    def seek_release(self, event):
        self.show_frame(int(self.slider.get()))

    def show_frame(self, index):
        self.cancel_seek()
        if self.cap is None:
            return
        index = max(0, min(self.total - 1, index))
        # Some MP4 backends seek one decoded frame past the requested index.
        # Sequential decoding gives the displayed frame a verifiable zero-based
        # ordinal and keeps its decoder timestamp attached to that exact image.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = False, None
        for _ in range(index + 1):
            ok, frame = self.cap.read()
            if not ok:
                break
        if not ok:
            self.setting_slider = True
            self.slider.set(self.frame_index)
            self.setting_slider = False
            self.status.set(f'第 {index} 帧读取失败，当前仍显示第 {self.frame_index} 帧。')
            return
        self.frame, self.frame_index = frame, index
        pts = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if pts >= 0 and (index == 0 or pts > 0):
            self.timestamp, self.timestamp_source = pts, 'decoder_pos_msec'
        else:
            self.timestamp = index / self.fps if self.fps > 0 else None
            self.timestamp_source = 'frame_index/fps' if self.fps > 0 else 'unavailable'
        self.setting_slider = True
        self.slider.set(index)
        self.setting_slider = False
        self.render()

    def point(self):
        if self.frame_index in self.pending:
            return self.pending[self.frame_index]
        mark = self.saved.get(self.frame_index)
        return mark['point'] if mark else None

    def resize(self, event):
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
        self.render_job = self.root.after(40, self.render)

    def render(self):
        self.render_job = None
        self.canvas.delete('all')
        if self.frame is None:
            self.display_box = None
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        h, w = self.frame.shape[:2]
        scale = min(cw / w, ch / h)
        dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.display_box = (ox, oy, dw, dh)
        rgb = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(rgb.resize((dw, dh), Image.Resampling.LANCZOS))
        self.canvas.create_image(ox, oy, anchor='nw', image=self.photo)
        point = self.point()
        if point is not None:
            x, y = ox + (point[0] + .5) * dw / w, oy + (point[1] + .5) * dh / h
            self.canvas.create_oval(x-7, y-7, x+7, y+7, outline='#ff3030', width=2)
            self.canvas.create_line(x-12, y, x+12, y, fill='#ff3030', width=2)
            self.canvas.create_line(x, y-12, x, y+12, fill='#ff3030', width=2)
        state = '未保存' if self.frame_index in self.pending else ('已保存' if point else '未标记')
        time_text = f'{self.timestamp:.6f} 秒' if self.timestamp is not None else '时间未知'
        self.info.set(f'{self.path.name}  |  帧 {self.frame_index} / {self.total-1}（从 0 开始）  |  '
                      f'{time_text}  |  {w}×{h}  |  点：{point or "无"}  |  {state}  |  共 {len(self.pending)} 帧未保存')

    def click(self, event):
        if self.display_box is None:
            return
        ox, oy, dw, dh = self.display_box
        if not (ox <= event.x < ox + dw and oy <= event.y < oy + dh):
            return
        h, w = self.frame.shape[:2]
        self.pending[self.frame_index] = [min(w-1, int((event.x-ox)*w/dw)),
                                         min(h-1, int((event.y-oy)*h/dh))]
        self.render()
        self.status.set('已设置当前帧的点，点击“保存当前帧”写入文件。')

    def clear_point(self):
        if self.frame is not None and self.point() is not None:
            self.pending[self.frame_index] = None
            self.render()
            self.status.set('当前点已清除，点击保存使删除生效。')

    def save(self):
        if self.frame is None:
            return
        point = self.point()
        if point is None and self.frame_index not in self.saved:
            self.pending.pop(self.frame_index, None)
            self.render()
            self.status.set('请先点击画面选择标记点。')
            return
        directory = self.annotation_path.parent
        stem = f'frame_{self.frame_index:06d}'
        updated = dict(self.saved)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if point is not None:
                raw_name, preview_name = f'{stem}.png', f'{stem}_marked.png'
                preview = self.frame.copy()
                cv2.drawMarker(preview, tuple(point), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.circle(preview, tuple(point), 7, (0, 0, 255), 2)
                cv2.putText(preview, f'frame={self.frame_index} point={point}', (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                for name, picture in ((raw_name, self.frame), (preview_name, preview)):
                    temp_image = directory / (name.removesuffix('.png') + '.tmp.png')
                    if not cv2.imwrite(str(temp_image), picture):
                        raise OSError(f'图片保存失败：{name}')
                    temp_image.replace(directory / name)
                updated[self.frame_index] = dict(frame_index=self.frame_index, timestamp_seconds=self.timestamp,
                    timestamp_source=self.timestamp_source, point=point, image=raw_name, marked_image=preview_name)
            else:
                updated.pop(self.frame_index, None)
            data = dict(schema_version=1, video=str(self.path.resolve()), frame_index_base=0,
                        coordinate_system='original_image_pixels_x_right_y_down', width=self.width,
                        height=self.height, frame_count=self.total, nominal_fps=self.fps,
                        marks=[updated[i] for i in sorted(updated)])
            temp = directory / 'marks.json.tmp'
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            temp.replace(self.annotation_path)
        except OSError as exc:
            messagebox.showerror('保存失败', str(exc))
            return
        self.saved = updated
        self.pending.pop(self.frame_index, None)
        self.render()
        self.status.set(f'已保存第 {self.frame_index} 帧到 {self.annotation_path}')

    def step(self, delta):
        if self.cap is not None:
            self.show_frame(self.frame_index + delta)

    def key_step(self, event, delta):
        if not isinstance(event.widget, (ttk.Entry, ttk.Combobox, tk.Scale)):
            self.step(delta)
            return 'break'

    def jump(self, event=None):
        try:
            index = int(self.frame_entry.get())
            if self.cap is None or not 0 <= index < self.total:
                raise ValueError
            self.show_frame(index)
        except ValueError:
            self.status.set('请输入视频范围内的整数帧号（从 0 开始）。')

    def close(self):
        if not self.discard_ok():
            return
        self.cancel_seek()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-dir', type=Path, default=ROOT / 'training_video')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'training_video_mark')
    args = parser.parse_args()
    root = tk.Tk()
    VideoMarker(root, args.video_dir, args.output_dir)
    root.mainloop()


if __name__ == '__main__':
    main()
