"""演示 03：读取画面中心临时靶心的深度。

在 VS Code 中打开本文件并点击右上角“运行 Python 文件”。
预览窗口中按 q 或 Esc 退出；结果是相机深度，不是正式计分距离。
"""

from pathlib import Path
import os
import runpy


项目根目录 = Path(__file__).resolve().parent
os.chdir(项目根目录)  # 保证模型、输入和输出的相对路径保持一致。
runpy.run_path(str(项目根目录 / "scripts/oak_target_depth_demo.py"), run_name="__main__")
