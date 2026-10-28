"""演示 04：点击画面中的测试点并估算它到临时靶心的距离。

在 VS Code 中打开本文件并点击右上角“运行 Python 文件”。
鼠标左键选点，按 c 清点、r 重置、q 或 Esc 退出；运行前需连接相机。
"""

from pathlib import Path
import os
import runpy


项目根目录 = Path(__file__).resolve().parent
os.chdir(项目根目录)  # 避免 VS Code 的当前目录影响脚本运行。
runpy.run_path(str(项目根目录 / "scripts/oak_click_distance_demo.py"), run_name="__main__")
