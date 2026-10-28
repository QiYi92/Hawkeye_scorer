# Demo15：首次脚部触地识别与复核

默认 `fusion` 使用同一只脚的下降→支撑转折定位首次接触，深度提供辅助证据。保留 `depth`（深度分离→收敛）与 `visual`（仅视觉地面证据）作为对照。所有模式共用模型观测，正式推理不读取人工标注，也不按视频名称选择参数。

## 运行

使用现有独立 `.venv-demo15`，依赖仍由 `requirements-demo15.lock.txt` 锁定，无需安装新模型。中文入口 `15_demo_深度辅助首次触地复核.py` 自动切换解释器；保留其中的视频选择和预览设置。

```bash
# 单段完整分析及交互复核
.venv-demo15/bin/python scripts/run_demo15.py --video training_video/video21.mp4

# 本次六段完整验收，生成叠加视频、结果和逐帧放大图
.venv-demo15/bin/python scripts/run_demo15.py \
  --videos training_video/video18.mp4 training_video/video19.mp4 \
           training_video/video20.mp4 training_video/video21.mp4 \
           training_video/video22.mp4 training_video/video23.mp4 \
  --no-preview --output output/demo15_new_validation

# 已有同版本证据可复用；输出必须使用新目录
.venv-demo15/bin/python scripts/run_demo15.py \
  --video training_video/video20.mp4 \
  --reuse-extraction output/demo15_optimized_full \
  --no-preview --output output/demo15_video20_recheck

# 关闭深度计算，仅运行视觉模式
.venv-demo15/bin/python scripts/run_demo15.py \
  --video training_video/video20.mp4 --no-depth --no-preview
```

相对路径以项目根目录为基准。默认 MPS，可用 `--device cpu`。`--no-export` 跳过视频编码，但仍保存 JSON、评估与接触放大图；`--max-frames` 是短测，不能计为完整验收成功。`--video-dir` 批量处理该目录全部 MP4，`--videos` 只处理明确列出的文件，批量默认不弹出预览。

默认读取 `configs/demo15_contact.json`，`--config` 可选择其他冻结参数。`--tune` 仅在包含 video18–video23 的完整批量输入上调参，搜索统一的下降幅度、支撑抖动容差和支撑时长；其他视频不参与选参。本次六段已用于算法开发，因此其成绩属于开发/回归成绩。

## 算法与证据

1. **离线运动员轨迹**：640 尺寸的人体分割跟踪与独立 960 尺寸检测互补。下降运动员身份确认后，使用真实检测、运动、区域重叠和外观约束向前后补齐轨迹。缺失或身份不明确的帧保留缺失，不生成虚构的人体框。
2. **相机补偿**：排除人物后，对背景做双向光流检查与 RANSAC 配准。相机发生移动不会永久禁用判定；配准失败会开启新的坐标段，禁止跨未知变换做位置验收或地面深度融合。
3. **鞋部观测**：SCHP 在扩展人体区域推理，允许鞋尖/鞋跟超出粗人体 mask。保存原图鞋底轮廓。首次触地确认及落靶粗分类之后，按需对接触帧做局部放大分割，与完整人体分割取一致区域；不同模式共享同一帧的精修。最终落点采用最低连续鞋底轮廓段的实际像素，排除单个突出的噪声像素。
4. **双脚关联**：使用速度预测、形状尺度与区域重叠做一对一关联；允许短暂缺失，但不强制把失踪的支撑脚关联到另一只脚的分割碎片。
5. **首次接触**：在背景补偿坐标中拟合局部下降和支撑轨迹，限制支撑期间的垂直抖动与水平位移。之后仍继续下降的空中停顿被排除。相邻候选按轨迹分段拟合误差确定起始帧，确认帧保留实际使用的后续观测时间。坐标始终来自起始帧的真实鞋底，不能用后续跑动脚步代替。

`fusion` 不再要求每个支撑帧都通过固定单目深度阈值；深度缺失或异常时，充分的鞋部运动与地面证据仍可确认。缺少足够的下降观测时保留 `needs_review`。`depth` 仍保留严格的深度证据要求，方便对比其局限。

地面来自可见靶纸及邻近地面的外观与深度参考，不使用人工落点限定搜索窗口。草地、阴影、靶盘形变、严重遮挡或身份跟踪错误仍可能造成失败；单目深度残差不是物理离地高度。

## 验收与输出

本次统一验收：**帧差绝对值≤2，配准后距离≤10 原图像素**。精确坐标验收只接受完整视频的 `confirmed_estimate` 和精修坐标；未识别、待复核、不可配准和部分视频运行都不能计为联合成功。明确脱靶的粗坐标另列为诊断，不计入精确坐标验收的分母。还需通过放大图确认对应首次触地鞋底，而非后续脚步。

标注帧按顺序解码与保存原图核对，原始 `marks.json` 不修改。video9 的历史截图匹配第193帧，报告同时保留其原标注192。坐标比较使用参考帧坐标系，并额外报告未经配准的距离。

- video18–video23：本次优化开发/回归集，单独报告 6 段的成绩。
- 其他已有标注视频：额外回归集，含历史调参视频，不宣称为独立泛化测试。
- 未标注视频：输出识别结果，不计算准确率。

每次运行写入新目录，包括 `run.json`、冻结配置、`evaluation.json/csv`、`report.md`，以及每段视频的三模式结果、`first_contact.json`、`contact_grid.png` 和叠加视频。放大图上红色是人工参考、绿色是自动预测、紫色是鞋底轮廓；每行显示中心帧前后各2帧。预测与参考可能有±1/2帧差，两行分别展示，不能把参考图定位诊断当作预测成功。

证据缓存 schema 为 v5，含脚轨迹、基础鞋底轮廓、配准质量、坐标段、多帧靶纸外轮廓和真实运动员掩膜（`athlete_masks.npz`）。基础缓存不含局部精修结果，精修后另存 `refinement.json`。缓存校验输入视频、提取配置、提取及依赖代码、深度特征代码和模型权重。旧版本或基础提取代码变化会在新目录重新提取，保留历史缓存。单纯调整事件判定或落靶分类规则不需要重跑基础模型；每次运行的精修器独立，不跨运行复用精修。

优化前后报告使用同一严格标准重新评估历史结果：

```bash
.venv-demo15/bin/python scripts/benchmark_demo15.py \
  --current output/demo15_optimized_verified \
  --additional-current output/demo15_regression_verified \
  --baseline-manifest output/demo15_optimization/baseline_manifest.json \
  --output output/demo15_comparison_new
```

基线清单为每段视频指定历史 `fusion_result.json` 和 `extraction` 目录。比较输出包含逐视频帧差、位置误差、状态、联合通过结果及标注校验值。

## 预览与测试

预览分析完成后从第0帧开始播放。空格暂停/播放，A/D 前后逐帧，G 跳预测帧，T 跳参考帧，R 从头播放，点击暂停并修正点，S 保存独立 `review.json`，Q/Esc 退出。修正不会覆盖自动结果。`--no-depth-view` 只隐藏独立深度窗口，`--no-depth` 才关闭深度推理。

```bash
.venv-demo15/bin/python -m unittest discover -s tests -p 'test_demo15*.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_demo1[34]*.py'
```

测试覆盖严格验收边界、相机移动/配准失败、坐标段隔离、双脚交叉和碎片误关联、短暂漏检、空中停顿、落地跑动、深度异常、候选回溯、标注隔离及旧缓存失效。真实视频精度以完整实测报告为准。


## 落靶／脱靶与计算短路

左上角新增“识别情况”：触地前为“待识别”，候选确认中或证据不足为“待确认”，确认后为“落靶／脱靶”。主窗口、深度窗口和叠加视频使用同一状态逻辑。外边线属于靶盘，压线算落靶。

分类使用实际靶纸外轮廓，不依赖靶心或环号。对触地前0.75秒内有效背景配准的轮廓分组，至少三帧的轮廓面积重叠率达到85%，并排除图像截断或人物明显遮挡的轮廓。选择一致组中较完整的三个真实轮廓，减少脚部遮挡形成的轮廓缺口；不按椭圆补造靶盘。

只有首次触地已确认、三个轮廓都支持盘外，且整个候选鞋部框与各轮廓并集保持按720p缩放的10像素安全距离时，才直接判脱靶。该分支跳过局部鞋底精修和完整靶环复核，保留触地帧与粗略区域，界面隐藏自动坐标和环号。边界附近继续精修；精修后确认脱靶也可跳过靶环计算。轮廓缺失、严重遮挡或配准失败不触发提前脱靶。

JSON 新增字段：`target_hit` 为 `hit/miss/unknown`，`point_precision` 为 `coarse/refined`，`shortcut_reason` 记录短路原因。`event.point_px` 保持兼容：`coarse` 表示诊断用粗位置，不是精确成绩。每段视频另保存 `processing_timing.json`，记录精修/靶环耗时、调用次数和短路模式数。

默认开启短路。中文入口可设置 `脱靶短路=False`，命令行使用 `--no-miss-shortcut` 关闭，以进行对照。此开关不取消落靶分类。

```bash
# 相同基础证据下对比开关；每次均重新做所需的精修，不使用已有 refinement.json
.venv-demo15/bin/python scripts/benchmark_demo15_hit.py \
  --extraction output/demo15_hit_full \
  --output output/demo15_hit_timing_new
```

耗时报告对 video19 做三次交替顺序的配对测量并取中位数，其余视频各一组。直接实测分类、精修和靶环后处理；总分析耗时用同一次真实完整基础提取耗时加后处理耗时重建，排除视频编码和预览。不能把复用基础证据的运行时长当成全流程提速。人体跟踪与基础鞋部分割仍执行，因此总耗时的降幅会小于后处理降幅。
