"""演示 08：组合首次触地识别、靶心追踪和靶盘区域判定。

在 VS Code 中打开本文件并点击右上角“运行 Python 文件”。
在“可调试配置”中切换视频；这是当前组合能力最完整的交互演示。
"""

from pathlib import Path
import os
import runpy
import sys


项目根目录 = Path(__file__).resolve().parent
os.chdir(项目根目录)  # 组合脚本会导入 scripts 下的其他实现，必须固定根目录。

# ===== 可调试配置：直接修改这里即可切换读取文件 =====
输入视频路径 = 项目根目录 / "training_video/video10.mp4"
姿态模型路径 = 项目根目录 / "yolo11n-pose.pt"
预览缩放比例 = 0.75
# ==================================================

if not 输入视频路径.is_file():
    raise SystemExit(f"找不到输入视频：{输入视频路径}")
if not 姿态模型路径.is_file():
    raise SystemExit(f"找不到姿态模型：{姿态模型路径}")

# 组合演示内部复用脚部识别模块，先覆盖它读取的模型路径。
sys.path.insert(0, str(项目根目录 / "scripts"))
import browse_mediapipe_foot_contact as 脚部识别实现

脚部识别实现.MODEL_NAME = str(姿态模型路径)

sys.argv = [
    "browse_target_landing_contact.py",
    "--video", str(输入视频路径),
    "--display-scale", str(预览缩放比例),
]
runpy.run_path(str(项目根目录 / "scripts/browse_target_landing_contact.py"), run_name="__main__")
