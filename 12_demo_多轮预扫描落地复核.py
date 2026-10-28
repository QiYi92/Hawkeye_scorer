"""演示 12：多轮预扫描后，以原始分辨率复核首次脚部触地。

本入口独立于 Demo 08；可直接在 VS Code 中运行并修改以下配置。
"""

from pathlib import Path
import os
import runpy
import sys


项目根目录 = Path(__file__).resolve().parent
os.chdir(项目根目录)

# ===== 可调试配置 =====
输入视频路径 = 项目根目录 / "training_video/video6.mp4"
姿态模型路径 = 项目根目录 / "yolo11n-pose.pt"
预览缩放比例 = 0.75
# =====================

if not 输入视频路径.is_file():
    raise SystemExit(f"找不到输入视频：{输入视频路径}")
if not 姿态模型路径.is_file():
    raise SystemExit(f"找不到姿态模型：{姿态模型路径}")

sys.argv = [
    "browse_multiscan_landing_review.py",
    "--video", str(输入视频路径),
    "--model", str(姿态模型路径),
    "--display-scale", str(预览缩放比例),
]
runpy.run_path(str(项目根目录 / "scripts/browse_multiscan_landing_review.py"), run_name="__main__")
