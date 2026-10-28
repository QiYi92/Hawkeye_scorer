"""演示 10：使用 OAK-D Lite 实时跟踪主要人体并测量左右脚距离。

左右脚各自动选择一个脚尖或脚跟点，实时显示相机到该点的三维直线距离。
打开本文件并点击 VS Code 右上角“运行 Python 文件”即可启动。
"""

from pathlib import Path
import os
import runpy
import sys


项目根目录 = Path(__file__).resolve().parent
os.chdir(项目根目录)

# ===== 模型与实时采集参数 =====
YOLO姿态模型路径 = 项目根目录 / "yolo11n-pose.pt"
MediaPipe完整姿态模型路径 = 项目根目录 / ".cache/mediapipe/pose_landmarker_lite.task"
画面宽度 = 640
画面高度 = 360
采集帧率 = 30  # USB 3.x（UsbSpeed.SUPER）连接下的默认输入帧率。
人体检测置信度 = 0.35
关键点最低可见度 = 0.45
完整人体框最小高宽比 = 0.85  # 截图中的手部框约为 0.74；调大更严格，张开双臂时可调小。

# ===== 双目脚部测距参数 =====
摄像头高度米 = 0.90  # 预留给后续地面距离功能；当前三维直线距离不使用此参数。
深度采样框像素 = 9  # 必须是奇数；在脚部点附近取区域深度中位数。
最小有效深度毫米 = 200
最大有效深度毫米 = 8000
最小深度有效比例 = 0.30
距离平滑帧数 = 5
最大单帧距离跳变米 = 0.60
主要目标丢失帧数 = 15  # 超过此帧数后，重新选择面积最大的完整人体。
同步时间阈值毫秒 = 25

# ===== 第一次跳跃落地检测参数 =====
起跳高度占人体框比例 = 0.05  # 脚部中心上移超过人体框高度的 5%，判为离地。
落地容差占人体框比例 = 0.025  # 下降至起跳前脚部基线附近，判为落地候选。
连续离地帧数 = 2
连续落地确认帧数 = 2
站立基线学习帧数 = 12  # 启动后先保持双脚可见且站立约 0.4 秒。

# ===== 第一次落地事件录像 =====
启用落地录像 = True
落地前录像帧数 = 300
落地后录像帧数 = 60  # 默认与落地前保持 5:1。
落地录像文件夹 = 项目根目录 / "output/demo10_landing_clips"
# ========================================================

if not YOLO姿态模型路径.is_file():
    raise SystemExit(f"找不到 YOLO 姿态模型：{YOLO姿态模型路径}")
# MediaPipe 模型不存在时，实现脚本会自动下载到上面指定的位置。

sys.argv = [
    "oak_live_full_pose_tracking_demo.py",
    "--yolo-model", str(YOLO姿态模型路径),
    "--mediapipe-model", str(MediaPipe完整姿态模型路径),
    "--width", str(画面宽度),
    "--height", str(画面高度),
    "--fps", str(采集帧率),
    "--confidence", str(人体检测置信度),
    "--landmark-visibility", str(关键点最低可见度),
    "--min-person-aspect", str(完整人体框最小高宽比),
    "--camera-height-m", str(摄像头高度米),
    "--depth-box-size", str(深度采样框像素),
    "--min-depth-mm", str(最小有效深度毫米),
    "--max-depth-mm", str(最大有效深度毫米),
    "--min-depth-valid-ratio", str(最小深度有效比例),
    "--smoothing-frames", str(距离平滑帧数),
    "--max-distance-jump", str(最大单帧距离跳变米),
    "--target-lost-frames", str(主要目标丢失帧数),
    "--sync-threshold-ms", str(同步时间阈值毫秒),
    "--jump-takeoff-ratio", str(起跳高度占人体框比例),
    "--jump-landing-ratio", str(落地容差占人体框比例),
    "--jump-takeoff-frames", str(连续离地帧数),
    "--jump-landing-frames", str(连续落地确认帧数),
    "--jump-baseline-frames", str(站立基线学习帧数),
    "--record-before-frames", str(落地前录像帧数),
    "--record-after-frames", str(落地后录像帧数),
    "--record-output-dir", str(落地录像文件夹),
]
if 启用落地录像:
    sys.argv.append("--record-landing")
else:
    sys.argv.append("--no-record-landing")
runpy.run_path(str(项目根目录 / "scripts/oak_live_full_pose_tracking_demo.py"), run_name="__main__")
