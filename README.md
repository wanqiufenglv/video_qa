# Video QA 数据生成

该目录提供视频元信息扫描、分段、关键帧抽取、大模型调用、JSONL 流式写入、断点续跑以及 ShareGPT 格式转换。

## 运行

```bash
python main.py --input-dir data --output-dir outputs --max-workers 3
```

`--max-workers` 表示同时处理的视频进程数量。主进程负责扫描视频、分配视频任务和汇总进度；每个子进程负责一个视频内的全部分段、抽帧、模型调用和结果写入。

## 后台查看进度

运行过程中会持续更新：

- `outputs/progress/progress.json`：当前总视频数、完成视频数、失败视频数、总片段数、完成片段数、失败片段数、跳过片段数、待处理数、进度百分比、进程并发数和更新时间。
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

系统会先扫描视频时长。短视频整体作为一个片段；超过阈值的视频会按配置的分段时长截断为多个实际视频段，保存到 `outputs/02_segments/clips/`，并根据截断后时长写入长度分类。系统会对每个截断视频段做质量检查，过滤大量空白画面或画面变化不足的片段；合格视频段才会作为独立任务生成 `video_description`，再基于描述对异常事件识别、设备异常检测、飞机螺丝及结构件异常检测三类任务打分；最高分达到阈值时只生成对应检测任务，否则生成 `video_caption`、`video_vqa` 和 `temporal_reasoning`。问题由大模型参考 `video_description` 生成，要求角度丰富、多样、具体。最终 ShareGPT 文件只包含 `conversations` 和 `videos`，其中 `videos` 指向截断后的视频段；答案不写时间范围，也不使用“未发现异常事件”“未发现明显异常”“无明显异常”等否定异常模板。

## 关键配置参数

视频截断、长中短分类和抽帧参数集中在 `video_qa/config.py` 的 `VideoConfig` 中。常用修改项如下：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `short_video_threshold_seconds` | `180.0` | 短视频判定阈值，单位秒；视频总时长小于等于该值时，整体作为一个视频段，不再切成多段。 |
| `default_segment_seconds` | `180.0` | 长视频物理截断时长，单位秒；视频超过短视频阈值后，按该时长切成多个视频段。 |
| `clip_length_categories` | `(("short", 30.0), ("medium", 180.0), ("long", 600.0))` | 截断后视频段的长/中/短分类阈值，单位秒；当前表示 `<=30` 为 short，`<=180` 为 medium，`<=600` 为 long。 |
| `clip_length_overflow_category` | `"extra_long"` | 截断后视频段超过 `clip_length_categories` 最大阈值时使用的分类名称。 |
| `frame_interval_seconds` | `0.0` | 抽帧时间间隔，单位秒；大于 0 时启用按时间间隔抽帧，例如 `2.0` 表示每 2 秒抽一帧。 |
| `keyframes_per_segment` | `5` | 当 `frame_interval_seconds <= 0` 时启用固定数量抽帧模式，每个视频段均匀抽取的关键帧数量。 |
| `max_keyframes_per_segment` | `12` | 按时间间隔抽帧时，每个视频段最多抽取的关键帧数量。 |
| `clip_quality_sample_interval_seconds` | `2.0` | 截断视频质量检测的采样时间间隔，单位秒；用于检测大量空白画面或画面变化不足的片段。 |

注意：`default_segment_seconds` / `short_video_threshold_seconds` 控制“怎么截断视频”，`clip_length_categories` 控制“截断后的视频段归为 short、medium、long 哪一类”，两者用途不同。

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
```

## 依赖

Python 代码仅使用标准库；视频扫描、视频截断和抽帧依赖本机 `ffprobe` 与 `ffmpeg`，需要先安装 FFmpeg 并加入 `PATH`。

大模型接口、提示词、分段时长、抽帧数量、抽帧时间间隔、截断视频质量阈值、输出文件名、视频进程并发数量等可变配置均集中在 `video_qa/config.py`。
