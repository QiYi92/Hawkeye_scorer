"""演示 11：OAK-D Lite 红色小球实时双目测距与首次落地记录。

在 VS Code 中打开本文件，点击右上角“运行 Python 文件”。
按 r 开始下一次抛球，g 重新标定地面，q 或 Esc 退出。
"""

from pathlib import Path
import os
import runpy
import sys


项目根目录 = Path(__file__).resolve().parent
os.chdir(项目根目录)

# ===== 实时采集与相机参数 =====
画面宽度 = 640
画面高度 = 360
采集帧率 = 30
摄像头高度米 = 0.90
同步时间阈值毫秒 = 25

# ===== 红球识别（OpenCV HSV） =====
红色区间1下限 = (0, 90, 60)
红色区间1上限 = (12, 255, 255)
红色区间2下限 = (168, 90, 60)
红色区间2上限 = (179, 255, 255)
最小轮廓面积 = 100
最低圆度 = 0.58
最大丢失帧数 = 8
是否必须确认脱手 = True  # 手持阶段只测距，不允许触发落地。
脱手确认连续帧数 = 3
球周围肤色比例阈值 = 0.06
手部检查圈半径倍数 = 1.75

# ===== 深度、地面与落地判定 =====
最小有效深度毫米 = 250
最大有效深度毫米 = 8000
深度采样内缩比例 = 0.45
最小深度有效比例 = 0.25
最大单帧深度跳变米 = 0.80
轨迹平滑帧数 = 3
地面采样起始高度比例 = 0.48
地面拟合距离容差米 = 0.025
地面最低内点比例 = 0.28
摄像头高度校验容差米 = 0.25
最低明确离地高度米 = 0.08
接触高度容差米 = 0.025
连续接触确认帧数 = 2
最小下落速度米每秒 = 0.12
接触时最大下落速度米每秒 = 0.10

# ===== 输出 =====
输出视频路径 = 项目根目录 / "output/demo_11_red_ball_landing.mp4"
输出结果路径 = 项目根目录 / "output/demo_11_red_ball_landing.json"
是否保存逐帧轨迹 = True
# ========================================================

sys.argv = [
    "oak_red_ball_landing_demo.py",
    "--width", str(画面宽度), "--height", str(画面高度), "--fps", str(采集帧率),
    "--camera-height-m", str(摄像头高度米), "--sync-threshold-ms", str(同步时间阈值毫秒),
    "--red1-low", *map(str, 红色区间1下限), "--red1-high", *map(str, 红色区间1上限),
    "--red2-low", *map(str, 红色区间2下限), "--red2-high", *map(str, 红色区间2上限),
    "--min-area", str(最小轮廓面积), "--min-circularity", str(最低圆度),
    "--max-missing-frames", str(最大丢失帧数),
    "--release-frames", str(脱手确认连续帧数), "--skin-ratio-threshold", str(球周围肤色比例阈值),
    "--hand-ring-scale", str(手部检查圈半径倍数),
    "--min-depth-mm", str(最小有效深度毫米), "--max-depth-mm", str(最大有效深度毫米),
    "--depth-inner-ratio", str(深度采样内缩比例), "--min-depth-valid-ratio", str(最小深度有效比例),
    "--max-depth-jump-m", str(最大单帧深度跳变米), "--smoothing-frames", str(轨迹平滑帧数),
    "--ground-start-ratio", str(地面采样起始高度比例), "--ground-distance-m", str(地面拟合距离容差米),
    "--ground-min-inlier-ratio", str(地面最低内点比例), "--height-tolerance-m", str(摄像头高度校验容差米),
    "--airborne-height-m", str(最低明确离地高度米), "--contact-height-m", str(接触高度容差米),
    "--contact-frames", str(连续接触确认帧数), "--min-down-speed-mps", str(最小下落速度米每秒),
    "--max-contact-down-speed-mps", str(接触时最大下落速度米每秒),
    "--output-video", str(输出视频路径), "--output-json", str(输出结果路径),
]
if 是否必须确认脱手:
    sys.argv.append("--require-hand-release")
if 是否保存逐帧轨迹:
    sys.argv.append("--save-trajectory")

runpy.run_path(str(项目根目录 / "scripts/oak_red_ball_landing_demo.py"), run_name="__main__")
