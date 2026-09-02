# 视频多模态数据处理提示词

本文档用于指导视频处理、视频问答生成、异常识别与 LLaMA-Factory 训练数据转换相关代码。目标是将输入视频处理为可溯源、可复查、可流式保存的中文多模态训练数据，仅支持 ShareGPT 格式。

## 角色设定

你是一个多模态视频数据处理专家，负责将视频中真实可见的内容转换为高质量中文训练样本。你需要基于视频画面、关键帧、时间片段和中间分析结果，生成视频摘要、视频问答、时序推理、异常事件识别、设备异常检测、飞机螺丝及结构件异常检测等数据。

必须遵守：

- 只描述视频中能够观察到的内容，不得编造不可见信息。
- 不确定内容必须使用“疑似”“可能”等审慎表述。
- 所有问题、答案、异常判断都必须能追溯到视频片段、时间戳或关键帧。
- 输出使用中文，表达客观、准确、自然。
- 不做身份推断、意图推断、背景推断或不可验证结论。
- 对异常事件、军事活动、设备故障、飞机结构异常等场景，只做客观视觉分析和分类说明，不输出操作性建议。
- 每条数据必须包含 `video_id`、`segment_id`、`time_range`、`keyframes`、`evidence`、`confidence` 等溯源字段。

## 整体处理流程

### 1. 视频扫描与元信息提取

对输入目录中的视频进行扫描，生成视频级元信息。

需要提取：

- 视频文件名、绝对路径或相对路径。
- 视频时长。
- 分辨率。
- 帧率。
- 编码格式。
- 文件大小。
- `video_id`。

推荐元信息格式：

```json
{
  "video_id": "video_001",
  "video_path": "data/videos/video_001.mp4",
  "duration": 32.5,
  "fps": 25,
  "resolution": "1920x1080",
  "codec": "h264",
  "file_size": 12345678
}
```

### 2. 视频截断、分段与关键帧抽取

对视频进行时长判断、物理截断、分段记录和抽帧。系统需要先根据视频总时长计算段数，再对每个视频段生成独立任务。

推荐策略：

- 短视频可以整体作为一个片段。
- 当视频超过设定阈值时，按固定时长截断，例如默认每 600 秒一段。
- 段数计算方式为 `ceil(video_duration / segment_seconds)`，最后一段允许短于固定时长。
- 每个片段生成唯一 `segment_id`，并生成实际可用的视频段文件。
- 视频段文件保存到 `outputs/02_segments/clips/<video_id>/<segment_id>.<ext>`。
- 每个任务以视频段为单位生成；一个视频被截成 N 段，就生成 N 个独立的视频任务。
- 根据截断后的视频长度写入 `clip_duration` 和 `clip_length_category`，分类包括 short、medium、long、extra_long。
- 对截断后的视频段进行质量检查，过滤大量空白画面或画面变化不足的片段，并记录 `blank_frame_ratio`、`static_frame_ratio`、`average_scene_change` 等指标。
- 每个关键帧生成唯一 `frame_id`，保存关键帧图像路径、原视频绝对时间戳、所属片段、原始视频路径和截断后的视频段路径。
- 关键帧应尽量覆盖片段前、中、后段；当触发最大帧数上限时，关键帧需要在整个片段范围内均匀分布，不能只集中在开头。

三阶段关键帧抽取规则：

1. 第一阶段：按片段长度分类设置基础抽帧间隔和帧数上下限。
   - short：`<= 60s`，基础间隔 `2s`，最少 `4` 帧，最多 `12` 帧。
   - medium：`60s-180s`，基础间隔 `5s`，最少 `8` 帧，最多 `24` 帧。
   - long：`180s-600s`，基础间隔 `10s`，最少 `12` 帧，最多 `36` 帧。
   - extra_long：`> 600s`，基础间隔 `18s`，最少 `16` 帧，最多 `48` 帧。
2. 第二阶段：根据片段整体画面信息密度调整基础间隔。
   - 当 `static_frame_ratio >= 0.75` 或 `average_scene_change <= 4.0` 时，认为片段整体变化较少，将抽帧间隔放大到基础间隔的 `1.4` 倍。
   - 当 `average_scene_change >= 12.0` 时，认为片段整体变化较多，将抽帧间隔缩小到基础间隔的 `0.6` 倍。
   - 其他情况保持基础间隔不变。
3. 第三阶段：对局部高变化区域补充关键帧。
   - 使用低分辨率灰度帧按约 `2s` 间隔做局部采样，比较相邻采样帧的平均绝对差异。
   - 当某个局部变化量同时高于全片段平均变化量的 `1.8` 倍且不低于 `8.0` 时，可在该局部时间点附近补充关键帧。
   - 补帧需要遵守该分类的最大帧数限制，并与已有关键帧保持至少 `0.5s` 的间隔。
   - 如果局部密度分析失败或采样帧不足，保留第一、第二阶段生成的关键帧，不中断整体处理。

关键帧使用要求：

- `keyframes` 是视频理解和问答生成的重要视觉依据，描述和问答都应优先基于这些关键帧与视频片段本身。
- 如果关键帧显示局部突变、主体快速移动、烟雾火光出现、设备状态变化、结构件细节变化等，应在 `video_description` 和 `evidence` 中体现。
- 如果关键帧较少或画面变化不明显，应保持保守描述，不得补充不可见信息。

推荐片段元信息格式：

```json
{
  "video_id": "video_001",
  "source_video_path": "data/videos/video_001.mp4",
  "video_path": "outputs/02_segments/clips/video_001/video_001_seg_0001.mp4",
  "segment_id": "video_001_seg_0001",
  "time_range": [0.0, 180.0],
  "clip_duration": 180.0,
  "clip_length_category": "medium",
  "clip_quality": {
    "is_valid": true,
    "reason": "ok",
    "blank_frame_ratio": 0.0,
    "static_frame_ratio": 0.18,
    "average_scene_change": 9.6
  },
  "keyframes": [
    {
      "frame_id": "video_001_seg_0001_frame_0001",
      "timestamp": 12.0,
      "frame_path": "outputs/03_keyframes/video_001/video_001_seg_0001/000001.jpg"
    },
    {
      "frame_id": "video_001_seg_0001_frame_0002",
      "timestamp": 24.0,
      "frame_path": "outputs/03_keyframes/video_001/video_001_seg_0001/000002.jpg"
    }
  ]
}
```

### 3. 视频内容理解

对每个视频片段及其关键帧进行内容理解，提取场景、对象、动作、空间关系、时序变化和异常线索。

需要关注：

- 场景类型：室外、室内、航拍、遥感、监控、工业巡检、飞机检修、道路、机场、工厂等。
- 主要对象：人、车辆、飞机、设备、建筑物、烟雾、火光、零部件等。
- 动作行为：移动、停留、聚集、爆炸、冒烟、维修、检查、旋转、泄漏等。
- 空间关系：左侧、右侧、中央、前景、背景、上方、下方、靠近、远离等。
- 时间变化：对象状态如何随时间变化。
- 异常线索：烟雾、火光、泄漏、变形、缺失、松动、异常聚集、越界移动等。

### 4. 视频描述与任务路由

任务模式由 `config.py` 中的 `PipelineConfig.enable_anomaly_detection_tasks` 控制，默认值为 `False`。

- `False`（默认）：只执行 `video_caption`、`video_vqa` 和 `temporal_reasoning`，不要生成 `task_relevance`，也不要生成任何异常检测任务字段。
- `True`：启用三类专业检测任务的评分与路由。

任务模式会写入 `task_mode` 和 `task_run_id`，用于区分两种模式的断点。切换开关后可以复用切片和关键帧，但必须按新模式重新执行模型任务。

开关关闭时只写入三个基础任务对应的中间文件和 ShareGPT 文件。程序不会删除同一输出目录内由旧模式产生的异常检测文件。

每个视频片段始终必须先生成一段尽量完整的 `video_description`。仅当开关为 `True` 时，才基于该描述对三类专业检测任务打 0 到 1 的相关度分数：

- `abnormal_event_detection`：重点事件与可见现象识别。
- `equipment_anomaly_detection`：设备异常检测。
- `aircraft_structure_anomaly_detection`：飞机螺丝及结构件异常检测。

开关开启时，路由阈值以配置 `task_relevance_threshold` 为准，当前默认值为 `0.9`。如果三类检测任务中最高分达到阈值，只选择最高分对应的一类任务，并根据该任务提示词生成问题与答案；如果三类检测任务均未达到阈值，则分别生成三个基础任务。开关关闭时不进行检测评分，`selected_task_types` 必须固定为 `["video_caption", "video_vqa", "temporal_reasoning"]`。所有问题都要参考 `video_description` 生成，提问角度要丰富、多样、具体。未被选中的任务不要生成“不适用”问答。

最终用于 ShareGPT 的答案不要出现时间范围，不要使用“未发现异常事件”“未发现明显异常”“无明显异常”等否定异常模板。

默认关闭状态下的推荐路由字段：

```json
{
  "video_description": "先生成的完整视频描述",
  "selected_task_types": ["video_caption", "video_vqa", "temporal_reasoning"]
}
```

开关开启后才添加 `task_relevance`：

```json
{
  "task_relevance": {
    "abnormal_event_detection": 0.12,
    "equipment_anomaly_detection": 0.08,
    "aircraft_structure_anomaly_detection": 0.03
  }
}
```

## 模块一：短视频 Caption / 视频摘要

### 任务目标

根据视频整体内容生成中文视频 caption 或摘要，适合训练模型进行长时序内容理解。

### 提示词

请根据输入视频片段及关键帧内容，生成一段客观、准确、完整的中文视频描述。描述应覆盖场景、主要对象、动作变化、空间关系和可见线索。随后参考该 `video_description` 生成一个丰富、具体、自然的问题，不要固定使用模板问题。不要添加视频中不可见的信息。

要求：

- 描述视频发生的整体场景。
- 说明主要对象及其位置。
- 描述对象随时间发生的变化。
- 如果存在可见异常线索，需要客观指出。
- 如果视频内容简单，也要保持描述完整。
- 字数建议 80 到 512 字。
- 输出必须包含可溯源字段。

输出格式：

```json
{
  "task_type": "video_caption",
  "video_id": "video_001",
  "segment_id": "video_001_seg_0001",
  "time_range": [0.0, 5.0],
  "question": "画面中的车辆活动和烟雾变化说明了什么场景信息？",
  "caption": "视频显示一处航拍视角下的开阔区域，画面中央可见多辆车辆停放，部分车辆在后半段开始缓慢移动。画面右侧出现灰白色烟雾，并随时间向周围扩散。整体来看，该片段表现出车辆活动和疑似烟雾线索。",
  "evidence": {
    "keyframes": ["frame_0001", "frame_0003", "frame_0005"],
    "timestamps": [1.0, 3.0, 5.0]
  },
  "confidence": 0.86
}
```

## 模块二：视频 VQA/时序问答

### 任务目标

基于视频内容生成视频问答和时序问答数据，适合训练模型理解视频中的可见事实、空间关系、先后顺序、状态变化、细节对比和场景信息。

### 提示词

请参考 `video_description` 默认生成 3 条高质量中文视频 VQA/时序问答数据。问题应覆盖观察事实、空间定位、对象状态、先后顺序、细节对比、场景理解和依据说明，提问方式要丰富、多样、具体。答案必须基于视频可见内容，不能凭空推测，且不要出现时间范围或否定异常模板。

问题类型包括：

- 事实问答：视频中出现了什么。
- 空间问答：某个对象位于画面什么位置。
- 时序问答：先发生了什么，后发生了什么。
- 状态变化问答：对象状态是否发生变化。
- 可见线索问答：画面中有哪些值得注意的变化或现象。
- 多选式问答：从多个选项中判断正确答案。
- 综合理解问答：结合多个画面线索概括场景信息。

生成要求：

- 每个视频片段默认生成 3 条问答。
- 答案应简洁，但必须包含视觉依据，不要写具体秒数或“时间范围”。
- 时序类问题可以描述先后变化，但答案不要写具体时间范围。
- 避免生成“是否没有异常”“是否未发现异常”这类否定异常问题。
- 避免多个问题重复询问同一事实。

## 模块三：时序推理

### 任务目标

基于视频片段中的多帧信息生成需要推理才能回答的问题，突出动作先后、状态转移、事件发展、变化依据和多线索综合判断。

### 提示词

请参考 `video_description` 默认生成 3 条高质量中文时序推理问答数据。问题应要求模型结合视频中前后画面、主体动作变化、空间关系变化或多个可见线索进行推理，不要只问单帧事实。答案必须给出可见依据，但不要出现具体时间范围或否定异常模板。

问题类型包括：

- 先后推理：根据画面变化判断动作顺序。
- 状态转移：说明对象、设备或场景状态如何变化。
- 过程归纳：概括视频中某个现象的发展过程。
- 多帧对比：比较前后画面中主体、位置或状态的差异。
- 依据推理：说明为什么可以得出某个视觉判断。
- 关系推理：结合空间位置和动作变化判断对象之间的关系。

生成要求：

- 每个视频片段默认生成 3 条时序推理问答。
- 问题要具体、自然、丰富，避免与视频 VQA/时序问答重复。
- 答案必须基于视频可见内容，不能补充不可验证的背景信息。
- 答案不要写具体秒数、时间范围或否定异常模板。

## 模块四：重点事件与可见现象识别

### 任务目标

用于航拍、遥感、监控或其他观察画面中的重点事件与可见风险线索识别，例如人员聚集、车辆聚集、爆炸、烟雾、火光、越界移动、不寻常停留等。生成问题和回答时不要刻意强调“异常”，应直接围绕具体可见现象提问和作答。

现象类型候选：

- 人员聚集。
- 车辆聚集。
- 不寻常移动。
- 越界移动。
- 烟雾。
- 火光。
- 爆炸。
- 建筑损毁。
- 大面积扬尘。
- 疑似军事活动。
- 正常状态描述。

输出要求：

- 如果画面存在重点现象，说明具体现象类别。
- 描述该现象在画面中的空间位置。
- 给出可见证据。
- 给出置信度。
- 问题避免使用“是否异常”“有没有异常情况”等问法，回答避免使用“异常事件”“异常情况”“有异常”等表述。
- 如果证据不足，应改为描述可见线索，不要生成否定异常模板。

## 模块五：设备异常检测

### 任务目标

用于工业、运维、巡检、检修或维护场景中的设备运行异常识别，如发热、泄漏、结构变形、部件缺失、破损、松动、腐蚀等。

异常类型候选：

- 发热。
- 泄漏。
- 烟雾。
- 火花。
- 结构变形。
- 部件缺失。
- 表面破损。
- 锈蚀。
- 松动。
- 异常振动。
- 异常积液。
- 正常状态描述。

## 模块六：飞机螺丝及结构件异常检测

### 任务目标

用于飞机维修、检修、巡检或结构检查视频中的细粒度异常识别，重点关注螺丝、铆钉、盖板、蒙皮、连接件、边缘结构等。

异常类型候选：

- 螺丝缺失。
- 螺丝松动。
- 螺丝锈蚀。
- 螺丝滑丝。
- 铆钉异常。
- 结构裂纹。
- 表面划痕。
- 蒙皮变形。
- 盖板错位。
- 异物残留。
- 污渍或腐蚀。
- 正常状态描述。

## ShareGPT 格式转换提示词

### 任务目标

将视频分析结果转换为 LLaMA-Factory 支持的 ShareGPT 格式。每条样本只包含视频路径、用户问题和助手回答，不包含 `metadata` 等元数据字段。回答中需要保留视觉依据和结论，但不要出现时间范围，表达应自然，不要像纯 JSON 报告。

### 提示词

请将视频分析结果转换为 LLaMA-Factory ShareGPT 格式。`human` 字段中使用 `<video>` 标记表示视频输入；`gpt` 字段中给出中文回答。回答必须基于已有分析结果，不能新增未验证信息。最终 ShareGPT 样本不要包含 `metadata`、`video_id`、`segment_id`、`time_range`、`keyframes`、`evidence`、`confidence` 等元数据字段。

输出格式：

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<video>请描述这段视频的主要内容。"
    },
    {
      "from": "gpt",
      "value": "这段视频展示了一处航拍视角下的开阔区域，画面中央有多辆车辆，部分车辆在后半段开始移动。画面右侧出现灰白色烟雾，并逐渐扩散。结合画面中的车辆分布和烟雾变化，可以判断该片段存在疑似烟雾线索，但烟雾来源不能仅凭画面确定。"
    }
  ],
  "videos": [
    "data/videos/video_001.mp4"
  ]
}
```

## 多进程与流式处理要求

后续生成代码时，应优先支持多进程、流式读取和流式保存。

处理要求：

- 主进程负责扫描视频文件、生成任务队列。
- 子进程负责视频解码、分段、抽帧和关键帧保存。
- 大模型调用模块以截断后的视频段为单位处理，避免一次性加载全部视频。
- 每完成一个视频段，立即保存段视频、质量检查结果、关键帧和中间结果。
- 每完成一条 caption、QA 或异常检测结果，立即追加写入 JSONL 文件。
- 中间任务记录使用 `video_id + segment_id + frame_id + question_id` 保证数据可追溯。
- 最终 ShareGPT 训练文件只保留 `conversations` 和 `videos`，不要写入 `metadata` 或其他元数据字段。
- 失败任务写入错误日志，不能中断整体处理。
- 支持断点续跑，已完成的 `segment_id` 不重复处理。
- 保存原始视频路径、截断后的视频段路径、截断视频长度分类、质量检查结果、原始分析结果、清洗后结果和最终训练格式文件。
- JSONL 写入时建议使用追加写入或原子写入策略，降低异常退出导致文件损坏的风险。
- 大模型调用需要支持动态加载多个模型接口；单个接口请求失败、超时或返回不可解析结果时，应在完成该接口重试后自动切换到下一个可用接口。
- 大模型调用需要增加重试、超时、并发限制和速率限制。

推荐输出目录：

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
  progress/
```

## 可溯源字段规范

每条中间结果和最终训练样本都应尽量包含以下字段：

```json
{
  "sample_id": "video_001_seg_0001_qa_0001",
  "video_id": "video_001",
  "source_video_path": "data/videos/video_001.mp4",
  "video_path": "outputs/02_segments/clips/video_001/video_001_seg_0001.mp4",
  "segment_id": "video_001_seg_0001",
  "time_range": [0.0, 5.0],
  "keyframes": [
    {
      "frame_id": "video_001_seg_0001_frame_0001",
      "timestamp": 1.0,
      "frame_path": "outputs/03_keyframes/video_001/seg_0001/000001.jpg"
    }
  ],
  "evidence": {
    "timestamps": [1.0, 3.0],
    "keyframes": ["video_001_seg_0001_frame_0001", "video_001_seg_0001_frame_0002"],
    "reason": "画面右侧可见灰白色烟雾，并在后续帧中扩散。"
  },
  "model_info": {
    "model_name": "fx-q3-235",
    "prompt_version": "video_vqa_prompt_v1",
    "generated_at": "2026-06-14T00:00:00"
  },
  "confidence": 0.84
}
```

## 质量检查提示词

请对生成的数据进行质量检查，判断是否满足训练要求。

检查标准：

- 问题是否清晰。
- 答案是否准确。
- 答案是否基于视频可见证据。
- 是否包含时间戳或关键帧依据。
- 是否存在幻觉或过度推断。
- 是否适合用于视频多模态模型训练。
- 是否符合 ShareGPT 格式。
- 中文表达是否自然。
- 中间任务记录中的 `video_id`、`segment_id`、`time_range`、`evidence` 是否完整；最终 ShareGPT 样本只检查 `conversations` 与 `videos`。

输出格式：

```json
{
  "sample_id": "video_001_seg_0001_qa_0001",
  "is_valid": true,
  "issues": [],
  "quality_score": 0.91,
  "revision_suggestion": "无需修改"
}
```

## 推荐主控提示词

后续调用大模型时，可以使用下面这段作为主控 prompt：

```text
你是一个专业的多模态视频数据处理专家。现在给你一个视频片段及其关键帧信息，请你基于视频中真实可见内容，生成可用于大模型训练的中文视频理解数据。

你需要完成以下任务：
1. 先生成尽量丰富的 video_description。
2. 读取 enable_anomaly_detection_tasks：若为 false，只生成 video_caption、video_vqa 和 temporal_reasoning，不进行检测评分；若为 true，才对 abnormal_event_detection、equipment_anomaly_detection、aircraft_structure_anomaly_detection 分别打 0 到 1 的相关度分数。
3. 仅在开关开启时：如果最高检测分达到配置阈值，只选择最高分对应的一类检测任务，并根据该任务提示词生成问题与答案。
4. 开关关闭或开关开启但三类检测分均低于阈值时，只生成 video_caption、video_vqa 和 temporal_reasoning 三类任务的问题与答案。
5. 将选中任务转换为 ShareGPT 格式；最终 ShareGPT 样本只包含 conversations 和 videos，不包含 metadata 或其他元数据字段。
6. 中间任务数据必须包含 video_id、segment_id、time_range、keyframes、evidence、confidence，保证可溯源。

请严格遵守：
- 不得编造视频中不可见的内容。
- 不确定的信息必须使用“疑似”“可能”等审慎表述。
- 不得输出无依据的背景推测。
- 判断必须给出视觉证据。
- 答案不要出现时间范围，不要使用“未发现异常事件”“未发现明显异常”“无明显异常”等否定异常模板。
- 回答使用中文，客观、准确、详细。

请输出 JSONL 兼容的 JSON 对象。
```

## 后续代码生成建议

建议拆分为以下模块：

- `video_scanner`：扫描视频并提取元信息。
- `video_segmenter`：按时间或镜头切分视频。
- `frame_extractor`：抽取关键帧并保存路径。
- `prompt_builder`：根据任务类型拼接提示词。
- `vlm_client`：动态加载多个多模态大模型接口，按配置顺序调用并在接口不可用时自动切换。
- `stream_writer`：流式写入 JSONL。
- `resume_manager`：管理断点续跑和已完成片段。
- `format_converter`：转换 ShareGPT 格式。
- `quality_checker`：执行质量检查和低质量样本过滤。
- `error_logger`：记录失败样本、错误原因和重试状态。

## 当前代码输出约定

本项目运行后按处理阶段保存数据，目录结构如下：

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
  progress/
```

模型需要先根据视频内容生成尽量丰富的 `video_description`，再根据描述对三类专业检测任务打分并按阈值路由。若最高检测分达到阈值，只生成对应检测任务问答；若均未达到阈值，则生成短视频 Caption / 视频摘要、视频 VQA/时序问答和时序推理三类任务问答。所有问题都应参考 `video_description` 生成，提问角度要丰富、多样、具体，符合普通用户提问习惯，但不要固定套用模板问题。

六类任务分别写入六个 ShareGPT 文件：

- `video_caption`：短视频 Caption / 视频摘要。
- `video_vqa`：视频 VQA/时序问答。
- `temporal_reasoning`：时序推理。
- `abnormal_event_detection`：重点事件与可见现象识别。
- `equipment_anomaly_detection`：设备异常检测。
- `aircraft_structure_anomaly_detection`：飞机螺丝及结构件异常检测。

如果视频内容不涉及三类专业检测任务，不得编造异常或生成“不适用”检测问答，应改为生成 `video_caption`、`video_vqa` 与 `temporal_reasoning`。所有 ShareGPT 样本必须包含 `<video>` 和 `videos`，且不要包含 `metadata` 或其他元数据字段。
