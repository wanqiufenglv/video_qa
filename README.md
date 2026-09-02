# Video QA 数据生成

该目录提供视频元信息扫描、分段、关键帧抽取、大模型调用、JSONL 流式写入、断点续跑以及 ShareGPT 格式转换。

## 运行

```bash
python main.py --input-dir data --output-dir outputs --max-workers 3
```

`--max-workers` 表示同时处理的视频进程数量。主进程负责扫描视频、分配视频任务和汇总进度；每个子进程负责一个视频内的全部分段、抽帧、模型调用和结果写入。

任务支持分阶段断点续跑。每完成视频截断、片段质量检查、关键帧抽取、模型调用、结果写入或整个片段，都会把断点追加到 `outputs/progress/checkpoints.jsonl`。再次使用相同输入目录和输出目录启动时，会校验对应的视频段和关键帧文件；文件仍然存在时从最近阶段继续，文件缺失时自动回退到需要重做的阶段。已经写入 `raw_results.jsonl` 的片段不会再次调用模型，缺失的普通任务文件和 ShareGPT 文件会从原始结果自动补建，并按稳定标识去重。

## 后台查看进度

运行过程中会持续更新：

- `outputs/progress/progress.json`：当前总视频数、完成视频数、失败视频数、总片段数、完成片段数、失败片段数、跳过片段数、待处理数、进度百分比、进程并发数和更新时间。
- `outputs/progress/checkpoints.jsonl`：片段级分阶段断点；用于中断后跳过已经成功且产物仍存在的阶段。
- `outputs/logs/process.log`：文本日志，适合后台运行时直接查看主进程进度和报错。
- `outputs/logs/process_log.jsonl`：结构化过程日志，包含子进程写入的视频、片段、抽帧、模型请求和错误事件。
- `outputs/logs/errors.jsonl`：失败样本和异常上下文。

PowerShell 查看方式：

```powershell
Get-Content video_qa/outputs/logs/process.log -Wait
Get-Content video_qa/outputs/progress/progress.json
Get-Content video_qa/outputs/logs/process_log.jsonl -Wait
```


## 生成逻辑

系统会先扫描视频时长。短视频整体作为一个片段；超过阈值的视频会按配置的分段时长截断为多个实际视频段，保存到 `outputs/02_segments/clips/`，并根据截断后时长写入长度分类。系统会对每个截断视频段做质量检查，过滤大量空白画面或画面变化不足的片段；合格视频段才会作为独立任务生成 `video_description`。默认关闭异常检测任务，只生成 `video_caption`、`video_vqa` 和 `temporal_reasoning`。开启 `enable_anomaly_detection_tasks` 后，系统才会基于描述对异常事件识别、设备异常检测、飞机螺丝及结构件异常检测三类任务打分；最高分达到阈值时只生成对应检测任务，否则仍生成三个基础任务。问题由大模型参考 `video_description` 生成，要求角度丰富、多样、具体。最终 ShareGPT 文件只包含 `conversations` 和 `videos`，其中 `videos` 指向截断后的视频段；答案不写时间范围，也不使用“未发现异常事件”“未发现明显异常”“无明显异常”等否定异常模板。

## 关键配置参数

视频截断、长中短分类、三阶段抽帧和质量阈值集中在 `video_qa/config.py` 的 `VideoConfig` 中。常用修改项如下：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `short_video_threshold_seconds` | `30.0` | 短视频判定阈值，单位秒；视频总时长小于等于该值时，整体作为一个视频段，不再切成多段。 |
| `default_segment_seconds` | `300.0` | 长视频物理截断时长，单位秒；视频超过短视频阈值后，按该时长切成多个视频段。 |
| `clip_length_categories` | `(("short", 60.0), ("medium", 180.0), ("long", 300.0))` | 截断后视频段的长度分类阈值。 |
| `clip_length_overflow_category` | `"extra_long"` | 截断后视频段超过最大分类阈值时使用的分类名称。 |
| `keyframe_sampling_profiles` | short: 2s/4-8, medium: 5s/8-16, long: 10s/12-32, extra_long: 18s/12-36 | 第一阶段：按截断后视频段长度分类设置基础抽帧间隔、最少帧数和最多帧数。 |
| `frame_interval_seconds` | `10.0` | 全局兜底抽帧间隔；当片段未命中 `keyframe_sampling_profiles` 时使用。 |
| `max_keyframes_per_segment` | `24` | 全局兜底最大关键帧数量。 |
| `keyframe_low_density_scene_change_threshold` | `4.0` | 第二阶段：低画面变化密度阈值，低于该值时放大抽帧间隔。 |
| `keyframe_high_density_scene_change_threshold` | `12.0` | 第二阶段：高画面变化密度阈值，高于该值时缩小抽帧间隔。 |
| `keyframe_high_static_ratio_threshold` | `0.75` | 第二阶段：静态画面比例阈值，高于该值时放大抽帧间隔。 |
| `enable_local_density_keyframe_boost` | `True` | 第三阶段：是否启用局部高变化区域补帧。 |
| `keyframe_density_sample_interval_seconds` | `2.0` | 第三阶段：局部密度分析的低分辨率采样间隔。 |
| `keyframe_density_max_sample_frames` | `120` | 第三阶段：局部密度分析最多采样帧数。 |
| `keyframe_local_change_multiplier` | `1.8` | 第三阶段：局部变化量需要高于平均变化量的倍数。 |
| `keyframe_local_min_scene_change` | `8.0` | 第三阶段：局部补帧的最低变化量阈值。 |
| `clip_quality_sample_interval_seconds` | `2.0` | 截断视频质量检测的采样时间间隔，单位秒；用于检测大量空白画面或画面变化不足的片段。 |

任务开关位于 `video_qa/config.py` 的 `PipelineConfig`：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `enable_anomaly_detection_tasks` | `False` | `False` 时只执行视频摘要、视频问答、时序推理；`True` 时额外启用异常事件识别、设备异常检测、飞机螺丝及结构件异常检测的评分与路由。 |
| `task_relevance_threshold` | `0.9` | 仅在异常检测任务开关开启时生效；三类检测任务的最高相关度达到该值后，路由到对应检测任务。 |

开启方式：

```python
class PipelineConfig:
    enable_anomaly_detection_tasks: bool = True
```

任务模式会写入原始结果和断点完成标识。沿用同一 `--output-dir` 切换模式时，切片和关键帧等通用产物会继续复用，但模型任务会按新模式重新执行；之前模式生成的任务文件不会被自动删除，如需完全隔离两种模式的结果，建议使用不同的输出目录。

三阶段抽帧策略：

1. 先根据 `clip_length_category` 使用 `keyframe_sampling_profiles` 得到基础抽帧间隔和帧数上下限。
2. 再复用 `clip_quality` 中的 `average_scene_change` 与 `static_frame_ratio`，对基础间隔做整体放大或缩小。
3. 最后用低分辨率灰度采样分析局部画面变化，在变化显著高于平均值的位置补充关键帧，但不超过该分类的最大帧数。

注意：`default_segment_seconds` / `short_video_threshold_seconds` 控制“怎么截断视频”，`clip_length_categories` 控制“截断后的视频段归为 short、medium、long 哪一类”，`keyframe_sampling_profiles` 控制“各类片段如何抽帧”。

## 输出文件

```text
outputs/
  01_metadata/metadata.jsonl
  02_segments/segments.jsonl
  02_segments/clips/
  03_keyframes/
  04_model_raw/raw_results.jsonl
  05_task_records/
    video_caption.jsonl
    video_vqa.jsonl
    temporal_reasoning.jsonl
    abnormal_event_detection.jsonl
    equipment_anomaly_detection.jsonl
    aircraft_structure_anomaly_detection.jsonl
  06_sharegpt/
    video_caption_sharegpt.jsonl
    video_vqa_sharegpt.jsonl
    temporal_reasoning_sharegpt.jsonl
    abnormal_event_detection_sharegpt.jsonl
    equipment_anomaly_detection_sharegpt.jsonl
    aircraft_structure_anomaly_detection_sharegpt.jsonl
  logs/
    errors.jsonl
    process_log.jsonl
    process.log
  progress/progress.json
  progress/checkpoints.jsonl
```

开关关闭时只会新增三个基础任务对应的任务文件和 ShareGPT 文件；三类异常检测文件仅在开关开启且路由选中对应任务时写入。若输出目录中已有旧模式文件，程序会保留它们。

## 依赖

Python 代码仅使用标准库；视频扫描、视频截断和抽帧依赖本机 `ffprobe` 与 `ffmpeg`，需要先安装 FFmpeg 并加入 `PATH`。

大模型接口、提示词、分段时长、抽帧数量、抽帧时间间隔、截断视频质量阈值、输出文件名、视频进程并发数量等可变配置均集中在 `video_qa/config.py`。
