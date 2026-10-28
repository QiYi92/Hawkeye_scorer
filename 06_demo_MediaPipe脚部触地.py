"""演示 06：用 YOLO + MediaPipe 细化脚尖/脚跟首次触地判断。

在 VS Code 中打开本文件并点击右上角“运行 Python 文件”。
在“可调试配置”中切换视频或模型；空格暂停，q 或 Esc 退出。
"""

from pathlib import Path
import os
import sys


项目根目录 = Path(__file__).resolve().parent
os.chdir(项目根目录)  # MediaPipe 缓存及视频路径均以项目根目录为基准。

# ===== 可调试配置：直接修改这里即可切换读取文件 =====
输入视频路径 = 项目根目录 / "training_video/video6.mp4"
姿态模型路径 = 项目根目录 / "yolo11n-pose.pt"
# ==================================================

if not 输入视频路径.is_file():
    raise SystemExit(f"找不到输入视频：{输入视频路径}")
if not 姿态模型路径.is_file():
    raise SystemExit(f"找不到姿态模型：{姿态模型路径}")

sys.path.insert(0, str(项目根目录 / "scripts"))
import browse_mediapipe_foot_contact as 演示实现

演示实现.VIDEO_PATH = str(输入视频路径)
演示实现.MODEL_NAME = str(姿态模型路径)
raise SystemExit(演示实现.main())
