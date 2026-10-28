"""演示 07：实时追踪测试视频中的靶盘和靶心。

在 VS Code 中打开本文件并点击右上角“运行 Python 文件”。
在“可调试配置”中切换视频；按 q 或 Esc 关闭预览。
"""

from pathlib import Path
import os
import runpy
import sys


项目根目录 = Path(__file__).resolve().parent
os.chdir(项目根目录)  # 确保默认测试视频可以被找到。

# ===== 可调试配置：直接修改这里即可切换读取文件 =====
输入视频路径 = 项目根目录 / "training_video/video6.mp4"
预览缩放比例 = 0.75
# ==================================================

if not 输入视频路径.is_file():
    raise SystemExit(f"找不到输入视频：{输入视频路径}")

sys.argv = [
    "browse_target_center_video.py",
    "--video", str(输入视频路径),
    "--display-scale", str(预览缩放比例),
]
runpy.run_path(str(项目根目录 / "scripts/browse_target_center_video.py"), run_name="__main__")
