"""演示 13：自动分割运动员轮廓、跨帧跟踪并导出；不依赖人工框选或骨骼。"""
from pathlib import Path
import sys

项目根目录 = Path(__file__).resolve().parent

# ===== VS Code：修改配置后直接运行 =====
输入视频路径 = 项目根目录 / "training_video/video6.mp4"
分割模型路径 = 项目根目录 / "yolo26n-seg.pt"  # 首次自动下载；可替换为专用 best.pt
模型模式 = "person"  # 专用模型改为 athlete，必须为唯一 class 0 = paraglider_athlete
跟踪器 = "botsort"  # 或 bytetrack
主目标选择方式 = "descending"  # 等待从上方进入并持续下降的运动员，只显示／导出该目标
入场框顶高度比例 = 0.45
下降确认距离比例 = 0.04  # 相对画面高度；还会检查下降速度和方向一致性
推理设备 = ""  # 自动选择；Mac 可尝试 mps
预览缩放比例 = 0.75
腿脚模式 = "schp"  # none / schp / torchscript
腿脚权重路径 = 项目根目录 / "models/exp-schp-201908261155-lip.pth"
SCHP仓库路径 = 项目根目录 / "third_party/Self-Correction-Human-Parsing"
腿脚推理设备 = "auto"  # Mac 自动使用 MPS；也可改为 cpu
腿脚推理间隔 = 3  # 每 3 帧推理一次，其余帧用光流传播
首次触地判断 = True  # True 开启，False 关闭；开启时确认前逐帧分割腿脚，会降低速度
触地连续确认帧数 = 4
触地稳定容差 = 6.0  # 以 720p 为基准的像素容差，随视频高度缩放
触地地面参考线 = None  # 默认自动检测靶盘区域；可设 (x1, y1, x2, y2) 局部地面线，原图像素
# ====================================


if __name__ == "__main__":
    sys.path.insert(0, str(项目根目录 / "scripts"))
    from run_demo13 import main
    参数 = ["--video", str(输入视频路径), "--model", str(分割模型路径), "--mode", 模型模式,
            "--tracker", 跟踪器, "--device", 推理设备, "--preview",
            "--display-scale", str(预览缩放比例), "--parts", 腿脚模式,
            "--selection", 主目标选择方式, "--entry-top", str(入场框顶高度比例),
            "--descent-distance", str(下降确认距离比例),
            "--parts-device", 腿脚推理设备, "--parts-interval", str(腿脚推理间隔)]
    参数 += ["--first-contact" if 首次触地判断 else "--no-first-contact",
             "--contact-confirm-frames", str(触地连续确认帧数),
             "--contact-tolerance", str(触地稳定容差)]
    if 触地地面参考线 is not None:
        参数 += ["--contact-ground-line", *map(str, 触地地面参考线)]
    if 腿脚权重路径:
        参数 += ["--parts-weights", str(腿脚权重路径)]
    if SCHP仓库路径:
        参数 += ["--schp-repo", str(SCHP仓库路径)]
    raise SystemExit(main(参数))
