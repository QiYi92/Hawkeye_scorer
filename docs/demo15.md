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
3. **鞋部观测**：SCHP 在扩展人体区域推理，允许鞋尖/鞋跟超出粗人体 mask。短暂整脚漏检时，对预测脚部附近的下半身原图补检，最多覆盖连续缺失两帧；返回的鞋部必须有当前RGB分割、人体邻域和外观支持。补检像素不等于已确认轨迹，需前后普通观测共同校验，避免误认另一只脚后改变原轨迹。首次触地确认及落靶粗分类之后，另按需对接触帧做精确分割；不同模式共享同一帧的精修。最终落点采用起始帧最低连续鞋底轮廓段的实际像素。
4. **双脚关联**：使用速度预测、近期可靠尺度的中位数、区域重叠和鞋部外观做一对一关联；短暂缺失的预测不确定度有界增加。轨迹关联与触地判定均允许最多连续缺失两帧，不能把失踪的支撑脚强制接到另一只脚或分割碎片。补检观测后接入，不能改变普通观测的身份和速度。
5. **首次接触**：在背景补偿坐标中拟合下降→接触候选→支撑，结合鞋底局部稳定、人体相对支撑脚的运动及有效深度接近证据。默认在候选后0.35秒内寻找继续下降的反证；可信重关联后的继续下降也能淘汰空中候选。轨迹截断不能视为反证通过，仅有平坦鞋底轨迹但缺少独立支撑且后续观察不足时保留待复核。早期伪候选明确排除后才允许后续事件成立，不能用后续跑动替代尚未排除的早期真实接触。相邻候选按轨迹分段拟合误差回溯起始帧，确认帧保留实际使用的后续观测时间。

`fusion` 不再要求每个支撑帧都通过固定单目深度阈值；深度缺失或异常时，充分的鞋部运动与地面证据仍可确认。缺少足够的下降观测时保留 `needs_review`。`depth` 仍保留严格的深度证据要求，方便对比其局限。

地面来自可见靶纸及邻近地面的外观与深度参考，不使用人工落点限定搜索窗口。草地、阴影、靶盘形变、严重遮挡或身份跟踪错误仍可能造成失败；单目深度残差不是物理离地高度。

草地掩膜只作场景门控；`ground_valid` 表示该位置有可用背景深度参考，两者均不独立证明触地。`counter_evidence_seconds` 控制反证窗口，默认0.35秒。`candidate_diagnostics` 记录候选的 `contact_candidate/support_confirmed/rejected_airborne` 状态、原因及真实证据帧；事件内另存 `support_anchors`、反证覆盖时间、观察是否完整和人体相对运动诊断。

## 验收与输出

本次统一验收：**帧差绝对值≤2，配准后距离≤10 原图像素**。精确坐标验收只接受完整视频的 `confirmed_estimate` 和精修坐标；未识别、待复核、不可配准和部分视频运行都不能计为联合成功。明确脱靶的粗坐标另列为诊断，不计入精确坐标验收的分母。还需通过放大图确认对应首次触地鞋底，而非后续脚步。

标注帧按顺序解码与保存原图核对，原始 `marks.json` 不修改。video9 的历史截图匹配第193帧，报告同时保留其原标注192。坐标比较使用参考帧坐标系，并额外报告未经配准的距离。

- video18–video23：本次优化开发/回归集，单独报告 6 段的成绩。
- 其他已有标注视频：额外回归集，含历史调参视频，不宣称为独立泛化测试。
- 未标注视频：输出识别结果，不计算准确率。

每次运行写入新目录，包括 `run.json`、冻结配置、`evaluation.json/csv`、`report.md`，以及每段视频的三模式结果、`first_contact.json`、`contact_grid.png` 和叠加视频。放大图上红色是人工参考、绿色是自动预测、紫色是鞋底轮廓；每行显示中心帧前后各2帧。预测与参考可能有±1/2帧差，两行分别展示，不能把参考图定位诊断当作预测成功。

证据缓存 schema 为 v7，含脚轨迹、鞋部外观、补检来源、关联质量、基础鞋底轮廓、配准质量、坐标段、多帧靶纸外轮廓、逐帧十环几何模型和真实运动员掩膜（`athlete_masks.npz`）。基础缓存不含精确精修结果，精修后另存 `refinement.json`。缓存校验输入视频、提取配置、提取及依赖代码、补检代码、靶环模型代码、深度特征代码和模型权重。旧版本或基础提取代码变化会在新目录重新提取，保留历史缓存。单纯调整事件判定或落靶分类规则不需要重跑基础模型；每次运行的精修器独立，不跨运行复用精修。

`metadata.json` 的 `observation_recovery` 记录额外RGB补检调用、模型推理次数和恢复的鞋部观测数量，`stage_seconds.observation_recovery` 单独计时，包含补检条件检查。每帧 `shoe_recovery` 和鞋部 `observation_source/association` 区分普通观测、补检观测和是否可信接入轨迹。补检属于确认触地所需的基础证据；明确脱靶后的精确精修、完整靶环复核仍单独计数并执行短路。

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

触地前显示等待识别；候选出现但尚未确认时只显示黄色候选区域和候选时间，隐藏正式落点坐标与十字标记。到达确认帧后才显示已确认触地和原始接触位置。主窗口、深度窗口及导出共用相同显示逻辑。

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

## 平滑完整十环（Demo15 独立）

统一使用十个等距半径圆的共同透视模型，输出原图中的平滑闭合曲线。先用靶纸轮廓与实际靶心建立透视初值，再从各环附近提取暗线或靶心边界，排除人物、彩色伞布和图像截断边缘，以鲁棒距离残差联合拟合。靶纸的白边只用于初始化，不强制等同于最外环。四方向边缘候选和更多可见弧段共同提供诊断；缺失方向由几何模型补全并记录来源。

预览默认每五帧重新取证拟合，中间帧通过可靠背景配准投影历史模型；历史有效期为0.5秒，配准失败、坐标段变化或时间中断会清除历史。靶心或环线证据不足时显示“靶环待确认”。拟合成立后显示“靶环：10/10（拟合）”，主窗口、深度窗口和导出视频均以抗锯齿实线显示完整十环。布面褶皱不符合平面模型时，曲线表示整体平滑近似。

统一配置是 `configs/demo15_rings.json`：环数固定10，预览96个方向、完整复核192个方向，每环361个绘图点。至少四个环且八个方向扇区中的六个具有一致支持，才接受模型。默认允许的中位拟合残差为按720p缩放的3像素；这是提取边缘的贴合度，不代表人工标注误差。模型代码或配置变化会使旧证据缓存失效。

每帧 `target.ring_model` 包含 `homography`、`center`、`rings`、`four_points`、`fit_error_px`、`evidence_frames` 和 `source`。每环的 `points` 都标为 `fitted`，`observed_fraction` 单独记录当前帧真实边缘支持比例；使用历史的帧不会冒充当前观测。`edge_keypoints` 在重新拟合帧记录四方向的 `observed/fitted` 来源。触地完整复核的模型保存在 `event.target_evidence.ring_model`，相同触地帧的各模式共用复核结果。

落靶／脱靶仍使用真实靶纸轮廓，完整绘图不会改变其判定。明确脱靶仍不调用鞋底精修或完整环复核；轻量预览拟合用于全视频显示，耗时单列于 `metadata.json` 的 `stage_seconds.ring_preview`。精确环区只有在当前图像支持足够、且落点远离相关环边界时才输出；几何补全本身不能认证环号。

完整运行后生成复核报告和三时刻的新旧绘图对照（输出使用新目录）：

```bash
.venv-demo15/bin/python scripts/evaluate_demo15_rings.py \
  --run output/demo15_rings_verified_20261029 \
  --baseline output/demo15_hit_verified \
  --output output/demo15_rings_review_new
```

报告包括逐视频拟合覆盖帧数、未知片段、首次触地误差、预览拟合耗时及完整复核调用数。各视频目录输出 `rings_main.png`、`rings_depth.png`、`rings_before_after_*.jpg` 和逐环诊断。视频和报告存放在本机 `output/`，不随代码仓库分发。

本次七段视频的汇总结果见 [2026-10-29 实测报告](demo15_rings_validation.md)。

## 无人工标注的首次触地回归

video33 等未标注视频直接运行原入口或 `--video`，无需提供正确帧、坐标或落地区域。人工标注只用于已有样例的结果评估，不参与事件选择。video33 已用于本次算法开发与原图复核，不是独立泛化测试。批量完整运行后可生成候选淘汰、鞋部逐帧放大、主/深度窗口和耗时报告：

```bash
.venv-demo15/bin/python scripts/evaluate_demo15_contact.py \
  --run output/demo15_contact_v7_final \
  --baseline output/demo15_rings_final_20261029 output/demo15_20261029_102757_694028 \
  --output output/demo15_contact_review_new
```

脚部序列放大图由自动预测的触地帧与坐标选取，紫色为普通鞋底观测，橙色为原图补检观测。无人工真值的视频只做原图复核，不宣称已满足正式10像素验收。新增算法实测见 [v7触地优化报告](demo15_contact_v7_validation.md)。
