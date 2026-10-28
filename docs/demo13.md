# Demo13：运动员实例分割与跟踪

实现 Phase 1 / 2 的自动视频流程，并提供 Phase 3 SCHP 适配器和 Phase 4 专用腿脚模型推理接口。不做落地事件分类，不使用 17 点或其他骨骼模型，不需要人工首帧框选。默认只读本地视频；逐帧推理、边处理边预览，实际是否达到源视频帧率取决于设备、分辨率与模型。

## 快速运行

预览画面、窗口标题、目标编号／置信度、跟踪状态、腿脚图例、触地诊断及终端进度已汉化。使用 Pillow 和系统中文字体绘制，避免 OpenCV 默认字体显示乱码；长提示会按画面宽度换行。Mac 自动查找系统黑体，其他环境可通过 `DEMO13_FONT` 指定中文 `.ttf/.ttc` 字体路径。JSON / CSV 保留稳定的机器字段名和状态码。

在项目根目录运行：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 13_demo_运动员实例分割追踪.py
```

VS Code 可直接运行根目录入口，并修改其中的视频、模型、跟踪器、设备和腿脚配置。空格暂停／继续，`q` / Esc 或关闭窗口结束；退出仍保存已处理部分。默认全片分析 `training_video/video6.mp4`。首次运行下载官方 `yolo26n-seg.pt`。

无窗口命令：

```bash
.venv/bin/python scripts/run_demo13.py \
  --video training_video/video6.mp4 --model yolo26n-seg.pt \
  --tracker botsort --device mps --mask-format png

# 切换 ByteTrack；RLE 避免大量 PNG 小文件
.venv/bin/python scripts/run_demo13.py \
  --video training_video/video6.mp4 --tracker bytetrack --mask-format rle

# 有专用权重后启用完整目标类别校验
.venv/bin/python scripts/run_demo13.py \
  --video training_video/video6.mp4 --model models/athlete_best.pt --mode athlete
```

`--device` 留空交给 Ultralytics；Mac 可测试 `mps`，CPU 使用 `cpu`，CUDA 使用 `0`。`--imgsz 960` 可测试远距离识别，代价是计算量增加。`--max-frames 60` 用于短测，0 代表全片。`--preview --display-scale .75` 打开预览。`--output` 必须指向新目录或空目录，防止覆盖历史结果；省略则创建时间戳目录。`--no-bbox` 只隐藏框，Mask 和轮廓仍显示。

`--conf` 默认 .25；可测试 `.1`，让低置信度检测参与底层 tracker 的二次关联，但也要同时检查误检。仅增大输入尺寸或降低阈值不能修复类别语义错误。

## 模型边界与目标选择

默认 `--mode person` 使用通用 COCO `person` 类，覆盖视频和 JSON 会明确标记其为人体基线。**通用人体分割不能保证将大伞包、安全带和装备完整识别为一个目标，也不能仅凭 person 类可靠区别选手和旁观者。** 没有专用训练权重时，不把 person 输出改名为已经识别出的 `paraglider_athlete`。

`--mode athlete` 强制检查唯一类别 `class 0 = paraglider_athlete`；使用自有标注训练的实例分割模型后，该实例代表身体、头盔、伞包、安全带与固定装备的整体。检测或姿态权重会被拒绝。程序依赖模型的语义分割结果，不用框或下半身矩形伪造轮廓。

YOLO 每帧执行，`persist=True` 接 BoT-SORT（默认）或 ByteTrack。默认 `--selection descending`：先在内部观察候选，等待从画面上方进入、持续向下运动的目标；初始框顶须在 `--entry-top .45` 的图像高度以上，至少观察 `max(3, confirm_frames)` 次并跨越约 .2 秒，累计下移超过 `max(画面高度×--descent-distance, 身高×.25)`，至少 65% 的相邻位移向下，平均下降速度不少于每秒画面高度的 .1。默认下降距离比例 .04。静止地面人员及缓慢走动者通常不符合此条件；候选同样可信时继续等待。

锁定后即使运动员落地停止下降，也继续跟踪该目标。短时丢失不切换地面人员；`--lost-buffer`（默认 30 帧）内允许底层 ID 改变后重新关联，但必须通过运动预测、预测框重叠、尺度与外观约束，而且候选不能存在歧义。锁定时已在旁边的其他轨迹被排除。超时后保留 lost，**本次视频不再自动选一个新目标**，避免运动员离场后把收拢伞翼或其他人接进来；需要重新启动分析才能开始新一轮选择。

预览和所有导出仅保留主目标，每帧 `tracks` 最多 1 个实例。背景候选不绘制、不写入实例 Mask/CSV；JSON 只额外保存 `candidate_count` 和 `selection_reason` 供排查。确认前为 `searching`，确认后为 `tracked`，丢失为 `lost`；未确认或丢失时不输出旧 Mask。腿脚模型也只处理主目标。

`track_id` 是本次运动员的逻辑 ID，初值沿用被选中的底层 ID；`source_track_id` 记录当前 Ultralytics ID。满足约束的短时重关联保持逻辑 ID，并通过 `tracker_reassociations` 计数和逐帧来源 ID 留痕。不能把逻辑 ID 未变化等同于身份判断必然正确。

这是图像平面运动启发式，适用于以固定机位观察下降入场的单次降落片段；俯仰／平移摄像机、运动员一直位于图像中心、已经落地才开始的视频或多个同时降落目标可能无法确认。不会在超时后退回“最大人体”而默默选错；`--selection largest` 仅保留作旧逻辑对比，该模式也只显示／导出一个目标。要解决通用模型本身的类别误判，仍需要专用整体目标权重。

## 输出与坐标定义

```text
output/demo13_时间戳/
├── video_overlay.mp4   # 保持原图尺寸和源标称 FPS，无音轨
├── tracks.csv          # 每帧主目标一行；未确认／丢失时为 searching/lost 行
├── frames.jsonl        # 每个已处理视频帧一行，含完整轮廓与可选 RLE
├── metrics.json        # 覆盖率、丢失帧、ID 变化、实际处理速度与阶段耗时
├── config.json         # 本次配置
├── masks/              # 原图尺寸 uint8 PNG，0 背景 / 255 目标
├── legs/               # 仅启用腿脚且主目标可见时创建
└── feet/
```

PNG 文件名为 `000328_id1.png`；选择 RLE 时 Mask 嵌入 JSONL，不创建这些 PNG 目录。RLE 为 COCO **未压缩**格式：`size=[height,width]`，`counts` 按列展开，从背景游程开始。原始 Mask 保留洞和多个连通区域；轮廓以 `RETR_TREE` 输出全部边界，`contour_hierarchy` 保留父子关系。不能把只含最外层 polygon 的面积当作真实 Mask 面积。

帧号从 0 开始；优先使用解码器的单调 `POS_MSEC`，无效时按源 FPS 递增。可变帧率视频的 JSON 保留可用源时间戳，但覆盖 MP4 按源标称 FPS 编码。每帧 `tracks` 可为空；此时没有真实检测 Mask，不能理解为空间中已知的空目标。

`bbox` 是 YOLO 原图坐标浮点框；`geometry.bbox` 是 Mask 非零区域的整数紧包围框。二者均采用 `[x1,y1,x2,y2)`，右下界不包含；`bottom_y` 是最后一个前景像素 y，因而等于 `y2-1`。质心来自前景像素矩，不是框中心。

CSV 保存原始 bbox、质心、Mask 面积、宽高、宽高比、整体轴角、x/y 像素速度以及腿脚面积和脚底 y。完整 JSONL 还保存腿脚 bbox / centroid / top_y / bottom_y，原始轮廓与 Mask 路径或 RLE。

PCA 主轴角相对图像 +x 轴顺时针，范围 `[0,180)`，90° 为竖直；这是无方向的整体形状轴，不能自行判断前扑方向。近似圆形／无主轴区域返回 null。速度单位为 `pixel/s`，x 向右、y 向下为正，包含相机运动；没有相机标定不转换为 m/s。首次检测及检测空档后的首帧速度为空，避免把空档位移误标成逐帧速度。

`smoothed` 单独保存 bbox、centroid、area、width、height、orientation 的 EMA（`--ema-alpha .65`；越接近 1 越保留快速变化）。角度按模 180° 平滑；不会在 179°→1° 时错误过渡至 90°。**原始证据 Mask 和 CSV 不被平滑改写。** 腿脚区域的时序处理为有限间隔的光流传播，每个推理周期重新纠正；不根据遮挡猜测不可见腿部。

未启用／不可用腿脚时字段为空，和“模型已执行但无该类像素”的面积 0 区别开。JSON 中 `parts_source=inferred/propagated/disabled/not_primary` 标记来源；`parts_age_frames` 表示距上次模型推理的帧数。腿脚区域裁剪到当前主目标 Mask 内，因此整体 Mask 漏掉的鞋部不会凭空补上。

## 可选 SCHP / LIP

本机已准备 SCHP 源码和 LIP 权重，根目录入口已配置 `腿脚模式 = "schp"`、权重路径和源码路径，直接运行即可。`腿脚推理设备 = "auto"` 在 Mac 优先使用 MPS，否则使用 CPU；`腿脚推理间隔 = 3` 控制模型执行频率。新环境需要先准备同样的资产；仅修改模式开关不会自动下载。

```bash
.venv/bin/python scripts/run_demo13.py \
  --video training_video/video7.mp4 --parts schp \
  --schp-repo third_party/Self-Correction-Human-Parsing \
  --parts-weights models/exp-schp-201908261155-lip.pth \
  --parts-device auto --parts-interval 3
```

适配器复用上游仿射坐标恢复，遵循其 BGR 预处理。LIP 的 16/17 映射为腿，18/19 映射为脚；同时将 Pants=9、Socks=8 并入可见腿区，以免穿长裤时只剩裸露小腿；连体服不整体视为腿。不强制左右分区。[上游推理实现](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing/blob/master/simple_extractor.py)。

已加入 `scripts/demo13/schp_compat.py` 推理兼容层，不编译原项目的旧版 InPlaceABN / CUDA 扩展。它保留 checkpoint 的参数名称及原扩展的 `abs(weight)+eps` 有效缩放系数，支持 CPU / MPS / CUDA 推理；不支持训练。加载时只替换网络模块中的旧扩展导入，不修改第三方源码，不污染进程的 `sys.path` 或通用 `modules/utils/networks` 模块名。

源码来自 [SCHP 官方仓库](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)，本机版本 `eb84c432cc697f494d99662a05f2335eb2f26095`，原 MIT 许可保留在第三方目录。LIP checkpoint 来自 [OOTDiffusion 的公开 SCHP 权重镜像](https://huggingface.co/levihsu/OOTDiffusion/blob/main/checkpoints/humanparsing/exp-schp-201908261155-lip.pth)，网络下载使用 hf-mirror 镜像；大小 267449349 字节，已核对 SHA256 `24fa3254ceeb74c8435458994a64b522fb439a3635b7b86ff470457e0413da00`。来源记录见 `models/SCHP_LIP_SOURCE.json`。

如果再次看到“腿脚模式配置不完整”，检查入口中的 `腿脚权重路径` 是否指向真实 `.pth` 文件、`SCHP仓库路径` 是否包含 `networks/AugmentCE2P.py` 和 `utils/transforms.py`；整体分割模型 `yolo26n-seg.pt` 不能代替 LIP 权重。若改用 `torchscript`，需同时换成符合下一节契约的专用权重，不能复用 LIP `.pth`。

SCHP ResNet101 较重，开启后帧率明显下降；这是模型推理耗时，不是程序卡死。可提高腿脚推理间隔降低负载，但光流传播时间越长越容易漂移。模型能运行与滑翔伞腿脚精度达标是不同指标；遮挡和远距离时腿脚可能为空。

历史验证（主目标选择修复前）：`video7.mp4` 的旧逻辑锁定了地面人员，163 帧腿部像素和无脚部输出的统计仅能说明模型可执行，**不能用于评价降落运动员的腿脚识别**。旧输出为 `output/demo13_schp_video7_verified/`。另对 `video6.mp4` 第 140 帧运动员 ROI 做 CPU 推理，输出腿和脚类别，确认 CPU 回退可运行。

当前下降主目标验证结果：`output/demo13_descending_video7_final/video_overlay.mp4` 及同目录的 `legs/`、`feet/`、`frames.jsonl`。video7 全片 219 帧，第 87 帧确认空中下降的运动员，主 ID 保持 5；底层 ID 5→14 在第 145 帧接回，至第 169 帧仍能跟踪，随后离场保持 lost。77 个有效跟踪帧均有腿部和脚部像素，包括下降、接近地面和落地后跑动；显示橙色腿区和粉色脚区，暂不区分左右腿。目视检查示例帧未标注地面人员；像素非空不代表边界精度达标，遮挡与衣着仍会影响结果。

## 专用 Legs / Feet 模型

`--parts torchscript --parts-weights models/legs_feet.ts` 接入用户训练的轻量语义分割模型。约定：

- 输入：ROI 拉伸到 `--parts-size`（默认 256）方形，RGB，float32 `[1,3,S,S]`，除以 255，再按 ImageNet mean `[.485,.456,.406]` / std `[.229,.224,.225]` 归一化。
- 输出：单个 Tensor logits `[1,3,H,W]`，类别 `0=background, 1=legs, 2=feet`。不接受 dict、tuple 或类别图，错误形状会明确报错。
- 输出 argmax 后以最近邻恢复到原始 ROI 尺寸，再粘贴回全帧。模型训练／导出须使用同样预处理。
- `--parts-interval 3` 每三帧执行一次模型，中间帧用反向 Farneback 光流映射前帧标签，再与当前整体 Mask 相交；主目标变化或整体检测空档立即清缓存、重新推理。快速运动或遮挡时可设为 1。

已用小型 TorchScript 测试模型验证张量契约、缩放和像素标签，未提供经过滑翔伞数据训练的模型。生产训练优先 `background / legs / feet`，只标可见部位，不推测遮挡后的腿。

## 训练整体目标

数据配置模板见 [configs/demo13_athlete_dataset.yaml](../configs/demo13_athlete_dataset.yaml)。按 **视频／运动员分组** 划分 train / val / test，不能把同一连续片段的相邻帧分散到不同集合。第一轮建议覆盖多个视频的 300～500 帧，再扩展至 1500～3000 帧；应覆盖悬挂、伸腿、屈膝、跑动、跌倒、翻滚和不同设备、角度、天气、远近、背景。

标注类别固定 `0=paraglider_athlete`，Polygon 包含身体、头盔、伞包、安全带及贴身固定装备；不包含远离身体的伞绳、伞翼、背景。多人场景中，每个运动员单独实例，不把其他人并到主目标中。实例分割标注为归一化 polygon；原始精细 Mask 建议保留为后续评测真值。

```bash
.venv/bin/yolo segment train model=yolo26n-seg.pt \
  data=configs/demo13_athlete_dataset.yaml imgsz=960 epochs=100
```

先修改模板的数据绝对路径。训练参数是起点，需在独立视频上评估轮廓 IoU、装备覆盖、主目标选择正确率、ID switch、丢失帧和腿脚边界；仅有处理 FPS 不代表精度达标。不自动执行没有标注数据的训练，不伪造训练权重。

## 验证与性能对比

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_demo13*.py' -v
.venv/bin/python scripts/benchmark_demo13.py \
  --video training_video/video6.mp4 --device mps
```

性能对比按相同视频和输入尺寸串行执行两种 tracker，各自输出独立目录，并汇总 `benchmark.json`。计时包含解码、分割、腿脚、特征与导出，不含加载模型／暂停预览；小视频初次设备预热会拉低均值。指标中的 `primary_id_changes` 是观察到的主 ID 变化次数，不是基于人工真值计算的 ID-switch 率。结果目录中的 `stop_reason` 区分正常结束、短测、提前解码结束、用户退出和异常。

代码分为 `scripts/demo13/{tracking,geometry,parsing,output}.py`，入口编排为 `scripts/run_demo13.py`；与既有 Demo 05～12 的姿态链路独立。官方接口依据：[Ultralytics Tracking](https://docs.ultralytics.com/modes/track/)、[Segmentation](https://docs.ultralytics.com/tasks/segment/)。

## 本机验证（2026-09-16）

- Python 3.12，Ultralytics 8.4.62，PyTorch 2.12.0，OpenCV 4.13.0，Mac MPS；已下载官方 `yolo26n-seg.pt`。
- `unittest discover -s tests -v`：48 项通过，其中 Demo13 新增 9 项，覆盖 RLE 洞／空 Mask、真实像素几何、PCA、跨 180° 平滑、检测空档速度、主目标连续确认／丢失保持、光流方向、腿脚推理间隔与模型输入输出契约、视频／CSV／JSON／Mask 联动及拒绝覆盖。
- `video6.mp4` 全片 413 帧，BoT-SORT / 640 / conf .25 / PNG：约 **17.7 FPS**，221 帧有被跟踪实例，187 帧主目标可见，主 ID 变化 2 次。输出 `output/demo13_video6_botsort/`。
- 同片 ByteTrack / 640 / conf .25 / RLE：约 **24.1 FPS**，218 帧有被跟踪实例，主 ID 变化 2 次。输出 `output/demo13_video6_bytetrack/`。两次保存格式不同，不能把速度差全部归因于跟踪器；严格对比使用 benchmark 脚本统一参数。
- 已逐帧解码两份覆盖视频并核对均为 413 帧；两份 JSON 时间戳严格递增；共 439 份实例 Mask 的 PNG / RLE 解码尺寸和像素面积均与 JSON 一致。已目视检查原图与覆盖帧对齐。
- 全片覆盖率的分母包括开头无人和运动员出画的帧，不能直接解释为检测召回率。通用模型在部分悬挂姿态漏检，在伞翼收拢时误报 person，在后段部分漏掉伞包。**这些误差尚未解决；需要专用整体目标标注与训练。** 提高到 960、conf .1 的附加试验仍有伞翼误检和 ID 变化，不能通过阈值调整宣称完成稳定识别。
- 在 2026-09-16 首次验收时没有真实 SCHP 或滑翔伞专用腿脚权重，只验证了接口与传播流程。2026-09-18 已补齐真实 SCHP LIP 权重并完成 Mac MPS 推理验证；仍未完成专用腿脚模型训练或带真值的精度评估，也不保证完整装备 Mask、零 ID 切换或 30 FPS。

## 首次脚部触地判断开关与方案

根目录 `13_demo_运动员实例分割追踪.py` 的配置区新增：

```python
首次触地判断 = False       # True 开启，False 关闭，默认关闭
触地连续确认帧数 = 4
触地稳定容差 = 6.0         # 720p 基准像素，按图像高度缩放
触地地面参考线 = None      # 自动检测靶盘；也可提供 (x1, y1, x2, y2) 局部地面线
```

开启要求同时启用 `schp` 或 `torchscript` 腿脚分割。命令行使用 `--first-contact` / `--no-first-contact`；对应参数为 `--contact-confirm-frames`、`--contact-tolerance`、`--contact-ground-line X1 Y1 X2 Y2`。关闭时不运行地面分析或触地判定，也不强制增加腿脚推理频率。

流程：

1. 只分析已经锁定的下降运动员；从脚部 Mask 的有效连通区域提取脚底边缘点，不把双脚之间的中点当作触地点。忽略极小噪声和画面边缘被截断的鞋部。
2. 地面参考独立于脚部轨迹。默认复用靶盘检测，检查脚点投影是否在可靠靶盘区域；盘外接触需要独立配置局部地面线。靶盘只提供地面区域约束，不是高度测量，也不把靶盘边缘当作整片地面的接触线。
3. 使用背景特征光流和 RANSAC 检查机位稳定性，排除运动员区域。明显镜头移动后停止固定点位判定；背景证据不足时不确认。
4. 要求此前有足够的脚部下降，再观察脚底是否在地面参考区域附近连续稳定至少 4 帧。确认前强制每帧重新运行腿脚模型，光流传播结果不能作为接触确认的证据。
5. 在后续帧完成复核后，回溯保存候选支撑首帧、时间戳及像素点，而不是把确认帧当作触地时刻。确认结果只锁存一次，后续跑动、回弹不会覆盖首次估计。候选阶段发生脚部遮挡或证据中断时保持未确认，避免把后续落脚误报为首次接触。

开启后增加 `first_contact.json`，在逐帧 JSON 和 CSV 中写入触地状态。确认时保存 `first_contact_frame.jpg`（候选首帧截图），视频从复核完成时起标记估计点。结果包含 `frame_id`、`timestamp`、`point`、`confirmation_frame`、证据来源及观察空档信息。`confirmed_estimate` 表示算法确认了一个**触地估计**，不代表已得到真实接触真值；置信度为定性说明，像素容差不是经过标定的统计误差区间。未确认时 `event=null` 并输出原因。

当前验证：66 项测试通过。video7 全片 219 帧开启触地判断后，第 118 帧（3.933 秒）得到候选点 `(648, 522)`，第 121 帧完成复核；此前存在目标检测空档，因此结果置信度标为低，需结合原视频核验。结果位于 `output/demo13_first_contact_video7_verified/`。没有独立人工真值，尚不能声明落地帧准确率；只输出像素点，不输出地面米制坐标或正式计分。中文画面示例位于 `output/demo13_中文显示预览/`。
