# 训练视频触地点标记工具

在项目根目录运行（也可在 VS Code 中直接运行根目录的中文入口）：

```bash
.venv/bin/python 00_yolo滑翔伞训练视频标记.py
```

1. 在顶部下拉框选择 `training_video` 中的视频。
2. 拖动进度条定位帧；使用上一帧／下一帧按钮或左右方向键精调，也可以输入帧号跳转。
3. 点击画面设置一个标记点。再次点击会移动该点，画面缩放后仍保存原图像素坐标。
4. 点击“保存当前帧”，或按 Ctrl/Command+S。每帧保留一个点，一个视频可以保存多个帧。
5. 重新打开视频会加载已有标注。修改后保存会更新同一帧；清除点后保存会删除该帧的 JSON 标注（旧截图保留）。

切换帧不会丢失未保存的点，但保存按钮只保存当前帧。切换视频或关闭窗口时，如有未保存编辑会提醒。

输出按完整视频文件名分目录，避免不同扩展名的同名视频发生冲突：

```text
training_video_mark/
└── video7.mp4/
    ├── marks.json                 # 视频信息和全部已保存点
    ├── frame_000118.png           # 无标记的原始帧
    └── frame_000118_marked.png    # 带红色十字的预览帧
```

`marks.json` 中 `frame_index` 从 0 开始；`point=[x,y]` 是原图坐标，左上角为原点，x 向右、y 向下。`timestamp_seconds` 优先采用 OpenCV 解码器提供的视频时间戳，无有效时间戳时按帧号/FPS 换算；`timestamp_source` 记录来源。未提供有效 FPS 时，无法换算的时间戳为 null。工具只记录人工选择，不自动判断是否真实触地。

自定义输入／输出路径：

```bash
.venv/bin/python scripts/mark_training_video.py --video-dir training_video --output-dir training_video_mark
```

依赖项目已有的 OpenCV、Pillow 和 Python Tkinter，无需 YOLO 模型。若 Python 缺少 Tkinter，需要使用包含 Tk 支持的 Python 环境。
