"""演示 02：预览 OAK-D Lite 的彩色画面与深度画面。

在 VS Code 中打开本文件并点击右上角“运行 Python 文件”。
预览窗口中按 q 或 Esc 退出；运行前需要连接 OAK-D Lite。
"""

from pathlib import Path
import os
import runpy


项目根目录 = Path(__file__).resolve().parent
os.chdir(项目根目录)  # 保证所有相对路径都从项目根目录解析。
runpy.run_path(str(项目根目录 / "scripts/oak_capture_demo.py"), run_name="__main__")
