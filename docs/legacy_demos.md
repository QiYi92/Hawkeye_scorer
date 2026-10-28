# 早期 Demo 与开发记录

以下保留原项目说明，作为 Demo 01–14 的历史参考。当前推荐入口和新环境安装步骤以根目录 README 为准；文中“本机已准备”不表示仓库包含视频或权重。

# 滑翔伞鹰眼计分器

滑翔伞鹰眼计分器是一个面向滑翔伞定点比赛的视觉辅助计分项目。项目目标是利用双目相机和计算机视觉技术，辅助测量运动员第一落点到靶心的距离，并根据比赛规则输出成绩。

本项目优先作为技术验证系统，不替代正式电子靶和裁判判定。

## Demo 快速入口

根目录提供了 `01_demo_*.py` 至 `14_demo_*.py` 十四个编号入口，并按两类排列：

- `01`～`04`、`10`～`11`：OAK-D 设备、采集、测距与实时追踪演示，需要连接 OAK-D Lite
- `05`～`09`、`12`～`14`：本地测试视频识别演示，不需要连接相机

独立训练视频标记工具 `00_yolo滑翔伞训练视频标记.py` 读取 `training_video`，支持拖动进度条、前后逐帧移动、点击标记点和保存。输出到 `training_video_mark`，包含原始帧、标记预览图和帧号／时间／原图坐标 JSON；重新打开可继续编辑。[使用说明](video_mark.md)。

```bash
.venv/bin/python 00_yolo滑翔伞训练视频标记.py
```

Demo 14 全自动追踪视频中实际变形的靶环与靶心，支持原图曲线叠加、局部放大、暂停单步、重播，以及视频 / JSONL 导出。实测曲线画实线，短时预测画橙色虚线，未知部分留空；仅输出二维像素坐标。[运行说明与边界](demo14.md)。

```bash
.venv/bin/python 14_demo_视频变形靶环标定.py
.venv/bin/python scripts/run_demo14.py --video training_video/video9.mp4 --no-preview --output-jsonl output/demo14/video9.jsonl
```

Demo 13 使用 **YOLO26 实例分割 + BoT-SORT / ByteTrack**，等待从上方进入并持续下降的运动员，只显示和导出这一主目标，包含像素 Mask、跟踪 ID、轮廓、PCA 倾角及像素速度。落地后继续跟踪，离场后不改选地面人员；短时底层 ID 变化可通过运动与外观约束接回，并保留来源 ID。无需首帧框选或人体关键点。支持 SCHP / 专用 TorchScript 腿脚分割和光流传播，导出覆盖视频、CSV、逐帧 JSONL、PNG / COCO RLE。默认 `person` 模式仍是通用人体分割，完整装备覆盖需专用 `paraglider_athlete` 权重。[运行说明、数据格式及训练规则](demo13.md)。

```bash
.venv/bin/python 13_demo_运动员实例分割追踪.py
# 无窗口分析；结果写入 output/demo13_时间戳/
.venv/bin/python scripts/run_demo13.py --video training_video/video6.mp4 --device mps
```

Demo 12 是独立于 Demo 08 的两阶段首次落地复核演示。它先以三种分辨率和抽帧策略预扫描同一视频，融合运动员轨迹与候选落地窗口；再以原始分辨率及 MediaPipe 脚尖/脚跟关键点确认首次接触，并判定黑色靶心、靶盘内或靶盘外。精细复核按脚部“下落后稳定/回弹”的时间轨迹判定，不会因自动地面线偏移而直接跳过触地；脚部刚在靶盘上出现时，还会以靶盘内稳定点和运动员下降趋势作为受限兜底。预扫描只提供候选，最终结果始终以精细复核为准；无共识或复核失败时会自动全片精扫。复核窗口中的“未确认”诊断会说明主要阻塞原因，按 `r` 只会重播，不会重新分析。

在 VS Code 中打开任一入口文件，点击编辑器右上角的“运行 Python 文件”即可启动。项目已配置默认使用 `.venv/bin/python`。各入口对应的实现、默认输入和验证状态见 `开发验证记录.md`。

Demo 10 使用 OAK-D Lite 同步采集 RGB 与对齐深度，通过 YOLO Pose + ByteTrack 锁定画面中面积最大的完整人体，再用 MediaPipe 33 点 Pose Landmarker 获取脚尖和脚跟。程序为左右脚各自动选择一个深度质量更好的点，并实时显示相机光心到该点的三维直线距离。

Demo 11 使用 HSV 颜色与圆度追踪红色小球，结合对齐深度计算球心三维位置。启动时会自动拟合地面，默认用 `0.90m` 摄像头高度校验；程序要先看到球被手握住，再连续确认球已脱手，手持阶段只显示测距而不允许触发落地。检测到明确下落后的首次接触时，保存时间、落地三维/地面坐标、轨迹 JSON 和标注视频。

运行 `python 11_demo_OAK红球首次落地追踪.py`，并让画面下半部看到足够的有纹理地面。界面显示 `GROUND OK` 后再抛球；按 `r` 清除本次结果，按 `g` 重新拟合地面，按 `q` 或 `Esc` 退出。

## 项目目标

- 测量电子靶范围外的落点距离，重点覆盖约 0.16m 至 5m 区域
- 识别或辅助确认运动员第一落点
- 自动计算落点到靶心的实际距离
- 输出比赛成绩并保存测量记录
- 支持后续回放、复核和测试数据分析

## 第一版 MVP

第一版先实现可验证的最小系统：

1. 读取相机画面、图片或测试视频
2. 加载相机标定参数
3. 标定靶心坐标
4. 人工点选落点
5. 自动计算落点到靶心距离
6. 根据距离输出成绩
7. 保存测量结果

自动识别第一落点将在后续阶段逐步加入。

## 首次脚部落地识别 MVP

当前新增一个独立的单目视频分析流程，用于验证“运动员脚部第一次接触地面”的自动识别能力。该流程只输出像素坐标和视频时间戳，不做双目测距、真实距离计算或比赛计分。

处理能力：

- 读取本地 MP4 视频
- 使用 Ultralytics YOLO Pose 检测多个人体，并按连续向下运动趋势锁定滑翔伞运动员
- 提取 COCO pose 的 `left_ankle` 和 `right_ankle`
- 使用 MediaPipe Pose Landmarker 对锁定运动员做二级脚部关键点估计，辅助判断脚尖或脚后跟触地
- 使用 Hough Line 估计地平线/远景边界和落地区地面线，地平线只用于过滤误判
- 当脚点到地面线距离小于阈值并连续多帧成立时，确认首次触地
- 输出标注视频和 JSON 结果

安装依赖：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

如果本机 Python 证书链导致 PyPI SSL 校验失败，可按本项目开发记录中的方式使用 pip `--trusted-host` 参数安装。

运行示例：

```bash
source .venv/bin/activate
python scripts/analyze_landing_video.py \
  --video training_video/video1.mp4 \
  --output-video output/result.mp4 \
  --output-json output/result.json \
  --model yolo11n-pose.pt \
  --contact-threshold 5 \
  --contact-frames 3
```

首次运行 Ultralytics 模型时会自动下载模型权重。也可以把 `--model` 改为本地 `.pt` 文件路径，或使用 `yolov8n-pose.pt`。

可选参数：

- `--conf`：人体检测置信度阈值，默认 `0.25`
- `--keypoint-conf`：脚踝关键点置信度阈值，默认 `0.20`
- `--contact-threshold`：触地像素阈值，默认 `5`
- `--contact-frames`：连续触地帧数，默认 `3`
- `--preview`：处理时显示预览窗口，按 `q` 或 `Esc` 退出
- `--log-level`：日志级别，默认 `INFO`

JSON 输出示例：

```json
{
  "video_name": "video1.mp4",
  "first_contact_frame": 123,
  "timestamp": 4.1,
  "contact_x": 532,
  "contact_y": 704,
  "contact_foot": "right",
  "contact_part": "heel"
}
```

如果未检测到首次触地，`first_contact_frame`、`timestamp`、`contact_x`、`contact_y`、`contact_foot` 和 `contact_part` 会输出为 `null`。如果脚尖/脚跟关键点置信度不足，`contact_part` 会输出为 `unknown`，坐标会回退到脚踝点。识别到 `toe` 或 `heel` 时，`contact_x/contact_y` 使用对应脚尖或脚跟坐标。

当前限制：

- 默认画面中只有一个主要运动员
- 地面线估计依赖画面下部存在较明显的水平边缘
- 复杂草地、遮挡、阴影、脚部关键点缺失会影响结果
- 脚尖/脚跟判断来自 MediaPipe 二级关键点，当前用于工程验证，不作为正式裁判判定
- 地面线只作为辅助约束和可视化；单目草地画面中没有明确边界时，会使用稳定线或 fallback
- 本阶段结果用于技术验证，不作为正式比赛判罚依据

## 技术路线

- Python 3.8+
- OpenCV
- NumPy
- 双目视觉标定与立体测距
- YOLOv8 或其他轻量目标检测模型
- Streamlit 或本地 Web 界面
- Raspberry Pi 4B 或普通笔记本作为验证平台

## 推荐硬件

- 双目摄像头
- 树莓派 4B 8GB 或笔记本电脑
- 三脚架与固定支架
- 标定板
- 激光测距仪或卷尺
- 测试用靶心标记

## 计划目录结构

```text
Hawkeye_scorer/
├── README.md
├── docs/
├── configs/
├── data/
├── src/
│   ├── camera/
│   ├── vision/
│   ├── scoring/
│   └── ui/
└── tests/
```

## 本地开发

当前实验硬件为 OAK-D Lite，开发机为 MacBook，通过 USB-C 连接。建议使用 Python 3.12 创建虚拟环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

检测 OAK-D Lite 是否能被 DepthAI 访问：

```bash
python scripts/oak_probe.py
```

运行最小 RGB + 深度采集预览演示：

```bash
python scripts/oak_capture_demo.py
```

在预览窗口中按 `q` 或 `Esc` 退出。

如果只想验证 RGB 和深度流是否稳定出帧，可以运行无窗口短测：

```bash
python scripts/oak_capture_demo.py --no-preview --duration 8 --width 320 --height 180 --fps 10
```

如果设备检测输出 `UsbSpeed.HIGH`，说明当前是 USB 2.0 级别连接，RGB + 深度流可以运行，但帧率会明显受限。建议优先检查 USB-C 线缆、转接器和接口，尽量连接到 USB 3.x。

也可以单独验证“画面中心靶心”的深度读数：

```bash
python scripts/oak_target_depth_demo.py
```

该演示将画面中心设为靶心，显示中心十字线和采样框，并实时输出中心区域的深度中位数。无窗口短测命令：

```bash
python scripts/oak_target_depth_demo.py --no-preview --duration 8
```

也可以运行点击测距演示：画面中心固定为临时靶心，鼠标左键点击一个测试点，程序会读取 OAK 真实相机内参，并显示像素距离、两点深度、估算三维直线距离和估算地面距离。估算地面距离基于手动输入的相机高度和俯角，不是正式比赛计分结果。

```bash
python scripts/oak_click_distance_demo.py
```

默认保持低带宽采集尺寸，但窗口按 2 倍放大显示。如果需要更大窗口，可以调整：

```bash
python scripts/oak_click_distance_demo.py --display-scale 3
```

快捷键：

- `c`：清除点击点
- `r`：重置
- `q` 或 `Esc`：退出

底部参数区默认值：

- 相机高度：`1.5m`
- 相机俯角：`15度`

点击输入框后输入实测值，再点击 `确认参数`。相机移动或角度变化后，需要重新确认参数。

## 开发阶段

### Phase 0：项目骨架

- 初始化 Python 项目结构
- 建立配置文件目录
- 编写基础 README 和开发记录
- 准备测试图片、视频和标定数据

### Phase 1：相机采集与标定

- 接入单目或双目相机
- 完成棋盘格标定流程
- 保存相机内参、畸变参数和双目外参
- 验证画面采集稳定性

### Phase 2：距离测量

- 建立靶心坐标
- 支持人工点选落点
- 将像素坐标转换为实际坐标
- 计算落点到靶心的距离
- 与卷尺或激光测距仪结果对比

### Phase 3：落点识别

- 检测运动员身体关键区域
- 分析连续帧中的接触动作
- 初步判断第一落点
- 提供人工复核和修正能力

### Phase 4：计分与界面

- 根据距离输出成绩
- 展示当前选手、落点距离和成绩
- 支持比赛记录保存
- 支持结果导出

### Phase 5：现场测试

- 室内静态精度测试
- 模拟落地动态测试
- 户外光照测试
- 汇总误差、成功率和处理时间

## 验收标准

- 能稳定读取测试图像或视频
- 能加载标定参数
- 能完成靶心和落点距离计算
- 0.16m 至 5m 范围内具备可重复测量结果
- 每次测量结果可保存和复核
- 自动识别结果必须允许人工确认

## 风险与限制

- 双目测距精度高度依赖相机标定质量
- 户外强光、阴影、雨雾和复杂地面会影响识别效果
- 第一落点自动识别是高风险模块，应分阶段实现
- 当前项目定位为辅助测量系统，不作为正式比赛唯一判罚依据

## Demo15：深度辅助首次脚部触地实验

使用独立 `.venv-demo15` 运行 `15_demo_深度辅助首次触地复核.py`，对比 YOLO26n-depth、运动转折与深度辅助、纯视觉三种模式。默认 fusion 支持背景运动补偿、双脚轨迹关联和按需鞋底细化。左上角显示“识别情况：落靶／脱靶”；明确脱靶时跳过局部精修和靶环计算，外边线压线算落靶。用 `training_video_mark` 的用户触地标记评估 **±2 帧、10 原图像素**联合目标；video18–video23 单独作为优化回归集，实际成绩见实测报告。

运行与复核说明见 [Demo15 文档](demo15.md)。

## 参考资料

- OpenCV 官方文档
- Ultralytics YOLOv8 文档
- 双目相机 SDK 文档
- 原始技术方案：`滑翔伞鹰眼计分器(1)-utf8.txt`
