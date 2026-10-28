"""演示 09：离线分析测试视频，并导出标注视频和 JSON 结果。

在 VS Code 中打开本文件并点击右上角“运行 Python 文件”。
输入、模型和输出路径都可在“可调试配置”中修改；默认不打开预览窗。
"""

from pathlib import Path
import os
import runpy
import sys


项目根目录 = Path(__file__).resolve().parent
os.chdir(项目根目录)  # 输出目录和模型文件都按项目根目录解析。

# ===== 可调试配置：直接修改这里即可切换读取和输出文件 =====
输入视频路径 = 项目根目录 / "training_video/video3.mp4"
姿态模型路径 = 项目根目录 / "yolo11n-pose.pt"
输出视频路径 = 项目根目录 / "output/demo_09_landing.mp4"
输出结果路径 = 项目根目录 / "output/demo_09_landing.json"
是否显示预览 = False
# ========================================================

if not 输入视频路径.is_file():
    raise SystemExit(f"找不到输入视频：{输入视频路径}")
if not 姿态模型路径.is_file():
    raise SystemExit(f"找不到姿态模型：{姿态模型路径}")

# 将入口配置转换成原分析脚本的命令行参数。
sys.argv = [
    "analyze_landing_video.py",
    "--video", str(输入视频路径),
    "--output-video", str(输出视频路径),
    "--output-json", str(输出结果路径),
    "--model", str(姿态模型路径),
]
if 是否显示预览:
    sys.argv.append("--preview")
runpy.run_path(str(项目根目录 / "scripts/analyze_landing_video.py"), run_name="__main__")
