"""YOLO 滑翔伞训练视频标记：选择视频 → 拖动进度条 → 点击触地点 → 保存当前帧。

左右方向键逐帧移动，Ctrl/Command+S 保存。
默认读取 training_video，输出到 training_video_mark。
"""
from scripts.mark_training_video import main


if __name__ == '__main__':
    main()
