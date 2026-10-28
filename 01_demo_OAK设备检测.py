"""演示 01：检测 OAK/DepthAI 设备。

在 VS Code 中打开本文件，点击右上角“运行 Python 文件”即可启动。
运行前请连接 OAK-D Lite，并确认 VS Code 使用项目的 .venv 解释器。
"""

from pathlib import Path
import os
import runpy


项目根目录 = Path(__file__).resolve().parent
os.chdir(项目根目录)  # 固定工作目录，避免从 VS Code 启动时找错相对路径。
runpy.run_path(str(项目根目录 / "scripts/oak_probe.py"), run_name="__main__")
