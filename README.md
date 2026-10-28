# Hawkeye Scorer · 滑翔伞鹰眼计分器

面向滑翔伞定点降落视频的视觉分析项目：跟踪下降中的运动员，估计脚部首次触地帧与落点，判断落靶或脱靶，并提供逐帧复核与结果导出。

当前推荐 **Demo15**。它采用 YOLO 人体实例分割、SCHP 鞋部分割、背景运动补偿与脚部运动转折分析，结合单目深度作为辅助证据。项目也保留 OAK-D 硬件实验和早期姿态、靶盘追踪 Demo。

> 当前结果属于视觉估计。单目视频输出的是原图像素坐标，不是厘米级距离，也不直接等同于正式比赛成绩。

## 功能

- 自动选择持续下降的运动员，关联短暂漏检，分别跟踪两只脚。
- 在背景补偿坐标系中识别“下降 → 支撑”的首次触地转折，保存接触帧和确认帧。
- 使用实际靶纸外轮廓判断 **落靶／脱靶**；压外边线算落靶，证据不足时显示待确认。
- 明确脱靶后跳过鞋底局部精修和完整靶环计算，保留粗略区域，不展示精确坐标或环号。
- 对比 `fusion`、`visual`、`depth` 三种模式；默认 `fusion`。
- 预览、逐帧回放、人工修正、叠加视频、触地放大图和 JSON/CSV 报告。

## 快速开始：Demo15

已在 macOS、Python 3.12、MPS 环境下验证。没有 MPS 的设备可通过 `--device cpu` 运行。以下命令在仓库根目录执行。

### 1. 克隆及初始化第三方源码

```bash
git clone --recurse-submodules https://github.com/QiYi92/Hawkeye_scorer.git
cd Hawkeye_scorer
```

如果克隆时没有初始化子模块：

```bash
git submodule update --init --recursive
```

SCHP 子模块固定到已验证的上游提交，项目通过本地兼容层执行推理，不需要编译其旧版 CUDA 扩展。

### 2. 创建独立环境

```bash
python3.12 -m venv .venv-demo15
.venv-demo15/bin/python -m pip install --upgrade pip
.venv-demo15/bin/python -m pip install -r requirements-demo15.lock.txt
```

Demo15 使用独立环境，避免与早期 Demo 的 Ultralytics 版本混用。早期 Demo 的依赖入口是 `requirements.txt`，运行方式见对应文档。

### 3. 下载并校验模型权重

```bash
.venv-demo15/bin/python scripts/download_demo15_models.py
```

该脚本按 `models/*SOURCE.json` 下载以下文件，检查大小和 SHA256；匹配的本地文件直接复用，下载中断会保留断点。支持通过 `--proxy` 指定代理。权重不随 Git 仓库分发。

| 文件 | 用途 | 大小约 |
|---|---|---:|
| `yolo26n-seg.pt` | 人体实例分割与跟踪 | 6.4 MiB |
| `models/yolo26n-depth.pt` | 单目深度辅助 | 12.4 MiB |
| `models/exp-schp-201908261155-lip.pth` | SCHP 腿脚／鞋部分割 | 255.1 MiB |

权重来源为 Ultralytics 官方发布及项目记录的 SCHP LIP 公开镜像；具体地址和校验值见 [models](models)。如出现 `No such file or directory: ...exp-schp-...pth`，先执行上述准备命令。

### 4. 准备视频并运行

训练视频及原始标注不随仓库分发。请准备你有权使用的视频，例如放到 `training_video/input.mp4`：

```bash
mkdir -p training_video

# 放入视频后执行；分析结束后打开复核窗口
.venv-demo15/bin/python scripts/run_demo15.py \
  --video training_video/input.mp4 --device mps
```

CPU 或无窗口运行：

```bash
.venv-demo15/bin/python scripts/run_demo15.py \
  --video training_video/input.mp4 --device cpu --no-preview
```

也可编辑根目录 [15_demo_深度辅助首次触地复核.py](15_demo_深度辅助首次触地复核.py) 中的 `输入视频`、`推理设备` 等参数后运行。该入口在 macOS/Linux 上会自动切换到项目的 `.venv-demo15/bin/python`。

## 常用选项

| 选项 | 说明 |
|---|---|
| `--mode fusion` | 默认模式：运动转折、地面证据与深度辅助 |
| `--no-depth` | 跳过深度推理，仅运行视觉模式 |
| `--no-depth-view` | 只隐藏深度窗口，仍计算深度 |
| `--no-preview` | 不打开 GUI，仍生成结果 |
| `--no-export` | 跳过视频编码，保留数据、报告与触地放大图 |
| `--videos a.mp4 b.mp4` | 完整分析指定的多段视频 |
| `--video-dir training_video` | 完整分析该目录内所有 MP4 |
| `--output output/run_001` | 指定新目录，避免覆盖历史运行 |
| `--reuse-extraction output/previous_run` | 复用通过输入、模型与代码校验的基础证据 |
| `--no-miss-shortcut` | 关闭脱靶计算短路，用于耗时对照 |
| `--max-frames 60` | 短测；结果不计为完整视频验收通过 |

预览按键：空格播放/暂停，A/D 前后逐帧，G 跳预测帧，T 跳参考帧，R 重播，点击画面修正位置，S 保存独立 `review.json`，Q/Esc 退出。人工修正不会覆盖自动成绩。

## 输出与数据约定

默认输出到 `output/demo15_<时间戳>/`，每次使用新目录。

| 输出 | 内容 |
|---|---|
| `report.md`、`evaluation.json/csv` | 三种模式、帧差与坐标误差、分类结果 |
| `run.json`、`contact_config.json` | 参数、运行源码校验值、冻结配置 |
| `<视频名>/first_contact.json` | 所选模式的自动识别结果 |
| `<视频名>/fusion_result.json` 等 | 各模式的事件、证据与诊断 |
| `<视频名>/video_overlay.mp4` | 带识别信息和落点叠加的视频 |
| `<视频名>/contact_grid.png` | 接触帧前后各两帧的放大图 |
| `<视频名>/processing_timing.json` | 精修、靶环耗时与调用次数 |
| `<视频名>/extraction/` | 可复用的基础观测与模型证据 |

帧号从 **0** 开始，坐标为**原图像素，x 向右、y 向下**。`target_hit` 为 `hit/miss/unknown`；`point_precision` 为 `coarse/refined`。脱靶短路留下的粗坐标仅作诊断，不参加精确落点验收。

有人工参考时，精确坐标验收采用 **±2 帧、10 原图像素**联合标准。没有标注的视频仍可分析，但不计算准确率。标注工具见 [视频标注说明](docs/video_mark.md)。

## 验证结果与边界

在本地 video18–video23 开发／回归样例中，默认 `fusion` 的落靶分类为 **6/6**：video19 脱靶，其余五段落靶；五段落靶视频均通过上述精确坐标标准。video19 短路后的局部精修与完整靶环调用均为零。

video19 的单次完整运行对照为约 **66.9 → 61.1 秒**。固定同一基础证据、三次配对重复测量，后处理中位数约 **4.83 → 0.78 秒**。完整耗时也受基础模型推理和设备负载影响，不能将一次墙钟差异全部归因于短路。

这些样例参与了算法开发，不能视为独立泛化成绩。历史 video5–video10 在之前的严格回归中仅 1/6 通过，仍存在误差和退化；复杂遮挡、靶纸变形、低分辨率或目标身份错误也可能导致失败。详细历史说明见 [Demo15 文档](docs/demo15.md)。原始视频和本地运行产物未上传，因此上述数值是已记录的本地实验结果。

运行不依赖真实视频或模型权重的行为测试：

```bash
.venv-demo15/bin/python -m unittest discover -s tests -p 'test_demo15*.py' -v
.venv-demo15/bin/python -m unittest discover -s tests -p 'test_demo1[34]*.py' -v
```

## 项目结构

```text
Hawkeye_scorer/
├── 00_*.py                  # 视频标注入口
├── 01_demo_*.py … 15_demo_*.py
├── scripts/                 # 命令行工具、各 Demo 的实现
│   ├── demo13/              # 实例分割、运动员跟踪与 SCHP 兼容层
│   ├── demo14/              # 变形靶纸、靶心与靶环追踪
│   └── demo15/              # 首次触地、落靶分类与按需精修
├── src/landing_detection/   # 早期落地识别流程
├── configs/                 # 统一参数及训练配置示例
├── models/                  # 权重来源记录；二进制权重本地准备
├── third_party/             # SCHP 固定版本子模块
├── tests/                   # 行为与回归测试
└── docs/                    # 各 Demo 的详细说明
```

## 其他 Demo 与文档

| 入口 | 内容 | 文档 |
|---|---|---|
| `00` | 人工触地帧／坐标标记 | [视频标注](docs/video_mark.md) |
| `01`–`04`、`10`–`11` | OAK-D 设备、RGB/深度、测距与实时实验 | [早期 Demo](docs/legacy_demos.md) |
| `05`–`09`、`12` | 视频浏览、姿态、早期首次触地实验 | [早期 Demo](docs/legacy_demos.md) |
| `13` | 下降运动员实例分割跟踪 | [Demo13](docs/demo13.md) |
| `14` | 实际变形的靶盘与靶环追踪 | [Demo14](docs/demo14.md) |
| `15` | 深度辅助首次触地、落靶／脱靶及复核 | [Demo15](docs/demo15.md) |

`training_video/`、`training_video_mark/`、`output/`、模型权重、缓存及虚拟环境由 `.gitignore` 排除，上传仓库不会删除本地文件。需要在其他机器复现本地样例时，请另外准备对应视频与标注。

## 第三方来源

- [Self-Correction Human Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)：固定提交 `eb84c432cc697f494d99662a05f2335eb2f26095`，原始 MIT 许可证保留在子模块中。
- [Ultralytics](https://github.com/ultralytics/ultralytics)：YOLO 跟踪、实例分割和深度模型；各依赖及权重按其上游许可使用。

本仓库尚未为项目自身指定开源许可证。硬件实验、像素定位与单目深度估计均不应直接当作正式裁判系统的测量结论。
