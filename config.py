"""Central configuration for video QA data generation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LLMEndpointConfig:
    name: str
    url: str
    model: str
    api_key: str = ""
    stream: bool = False
    temperature: float = 0.6
    max_tokens: int = 4096


@dataclass(frozen=True)
class LLMConfig:
    api_url: str = "https://mcc-pre.3xmt.com/gateway/ai-service/v1/chat/completions"
    api_key: str = os.getenv("VIDEO_QA_API_KEY", "sk-2h1RwMjqYcdh6F5Fs5")
    model: str = "fx-q3-235"
    max_tokens: int = 4096
    temperature: float = 0.6
    stream: bool = False
    timeout_seconds: int = 120
    retry_times: int = 3
    retry_interval_seconds: float = 2.0
    http_method: str = "POST"
    endpoints: tuple[LLMEndpointConfig, ...] = field(
        default_factory=lambda: (
            LLMEndpointConfig(
                name="InternVL3_5-38B",
                url="http://192.168.78.35:8879/intern-vl/v1/chat/completions",
                model="InternVL3_5-38B",
                api_key="",
                stream=False,
                temperature=0.6,
                max_tokens=4096,
            ),
            LLMEndpointConfig(
                name="Qwen3.6-27B",
                url="http://192.168.78.36:3012/v1/chat/completions",
                model="Qwen3.6-27B",
                api_key="sk-bveYeVn6NAdRRElTWCqhtyJbkTL5XwweedczV9FJ05kDqhqX",
                stream=False,
                temperature=0.6,
                max_tokens=4096,
            ),
        )
    )


@dataclass(frozen=True)
class VideoConfig:
    video_extensions: tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv")
    # 超过 short_video_threshold_seconds 的视频会按该时长进行物理截断，单位：秒；默认每 300 秒生成一个视频段。
    default_segment_seconds: float = 300.0
    # 短视频判定阈值，单位：秒；视频总时长小于等于该值时不再切成多段，而是整体作为一个视频段。
    short_video_threshold_seconds: float = 30.0
    # 当 frame_interval_seconds <= 0 且未命中分类抽帧配置时，启用固定数量抽帧模式。
    keyframes_per_segment: int = 3
    # 全局兜底抽帧时间间隔，单位：秒；分类抽帧配置优先级更高。
    frame_interval_seconds: float = 10.0
    # 全局兜底最大关键帧数量，避免长视频段产生过多帧。
    max_keyframes_per_segment: int = 24
    # 截断后视频段的长/中/短分类阈值，单位：秒；<=30 为 short，<=180 为 medium，<=600 为 long。
    clip_length_categories: tuple[tuple[str, float], ...] = (("short", 60.0), ("medium", 180.0), ("long", 300.0))
    # 截断后视频段时长超过 clip_length_categories 最大阈值时使用的分类名称。
    clip_length_overflow_category: str = "extra_long"
    # 第一阶段：按截断后视频段长/中/短分类设置基础抽帧间隔、最少帧数和最多帧数。
    # 格式：(clip_length_category, interval_seconds, min_frames, max_frames)
    keyframe_sampling_profiles: tuple[tuple[str, float, int, int], ...] = (
        ("short", 2.0, 4, 8),
        ("medium", 5.0, 8, 16),
        ("long", 10.0, 12, 32),
        ("extra_long", 18.0, 12, 36),
    )
    # 第二阶段：根据片段整体画面变化密度微调基础抽帧间隔。
    keyframe_low_density_scene_change_threshold: float = 4.0
    keyframe_high_density_scene_change_threshold: float = 12.0
    keyframe_high_static_ratio_threshold: float = 0.75
    keyframe_low_density_interval_multiplier: float = 1.4
    keyframe_high_density_interval_multiplier: float = 0.6
    # 第三阶段：对局部高变化区域补充关键帧，避免固定间隔漏掉短时事件。
    enable_local_density_keyframe_boost: bool = True
    keyframe_density_sample_interval_seconds: float = 2.0
    keyframe_density_max_sample_frames: int = 120
    keyframe_density_frame_width: int = 64
    keyframe_density_frame_height: int = 64
    keyframe_local_change_multiplier: float = 1.8
    keyframe_local_min_scene_change: float = 8.0
    keyframe_min_gap_seconds: float = 0.5
    # 截断视频质量检测的采样时间间隔，单位：秒；用于抽样判断空白画面和静态画面比例。
    clip_quality_sample_interval_seconds: float = 2.0
    clip_quality_max_sample_frames: int = 12
    clip_quality_frame_width: int = 64
    clip_quality_frame_height: int = 64
    clip_blank_luma_low_threshold: float = 8.0
    clip_blank_luma_high_threshold: float = 247.0
    clip_blank_variance_threshold: float = 4.0
    clip_max_blank_frame_ratio: float = 0.6
    clip_static_diff_threshold: float = 2.0
    clip_max_static_frame_ratio: float = 0.85
    jpeg_quality: int = 95
    ffprobe_command: str = "ffprobe"
    ffmpeg_command: str = "ffmpeg"
    ffprobe_output_format: str = "json"
    ffprobe_error_level: str = "error"
    ffprobe_show_entries: str = "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,duration"
    ffmpeg_overwrite_flag: str = "-y"
    ffmpeg_seek_flag: str = "-ss"
    ffmpeg_input_flag: str = "-i"
    ffmpeg_frame_count_flag: str = "-frames:v"
    ffmpeg_quality_flag: str = "-q:v"
    ffmpeg_duration_flag: str = "-t"
    ffmpeg_codec_flag: str = "-c"
    ffmpeg_copy_codec: str = "copy"
    ffmpeg_avoid_negative_ts_flag: str = "-avoid_negative_ts"
    ffmpeg_make_zero_value: str = "make_zero"
    single_frame_count: str = "1"
    video_stream_type: str = "video"
    unknown_value: str = "unknown"
    missing_ffprobe_message: str = "ffprobe is required to scan video metadata. Please install FFmpeg and add it to PATH."
    missing_ffmpeg_message: str = "ffmpeg is required to clip videos and extract keyframes. Please install FFmpeg and add it to PATH."
    include_keyframe_images_in_prompt: bool = True
    keyframe_image_mime_type: str = "image/jpeg"
    image_url_type: str = "image_url"
    text_type: str = "text"
    image_detail: str = "low"


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path = Path("outputs")
    metadata_file: str = "01_metadata/metadata.jsonl"
    segments_file: str = "02_segments/segments.jsonl"
    segment_clips_dir_name: str = "02_segments/clips"
    frames_dir_name: str = "03_keyframes"
    raw_results_file: str = "04_model_raw/raw_results.jsonl"
    captions_file: str = "05_task_records/video_caption.jsonl"
    vqa_file: str = "05_task_records/video_vqa.jsonl"
    temporal_reasoning_file: str = "05_task_records/temporal_reasoning.jsonl"
    abnormal_events_file: str = "05_task_records/abnormal_event_detection.jsonl"
    equipment_anomalies_file: str = "05_task_records/equipment_anomaly_detection.jsonl"
    aircraft_structure_anomalies_file: str = "05_task_records/aircraft_structure_anomaly_detection.jsonl"
    sharegpt_dir_name: str = "06_sharegpt"
    sharegpt_caption_file: str = "06_sharegpt/video_caption_sharegpt.jsonl"
    sharegpt_vqa_file: str = "06_sharegpt/video_vqa_sharegpt.jsonl"
    sharegpt_temporal_reasoning_file: str = "06_sharegpt/temporal_reasoning_sharegpt.jsonl"
    sharegpt_abnormal_event_file: str = "06_sharegpt/abnormal_event_detection_sharegpt.jsonl"
    sharegpt_equipment_anomaly_file: str = "06_sharegpt/equipment_anomaly_detection_sharegpt.jsonl"
    sharegpt_aircraft_structure_file: str = "06_sharegpt/aircraft_structure_anomaly_detection_sharegpt.jsonl"
    errors_file: str = "logs/errors.jsonl"
    process_log_file: str = "logs/process_log.jsonl"
    runtime_log_file: str = "logs/process.log"
    progress_file: str = "progress/progress.json"
    checkpoints_file: str = "progress/checkpoints.jsonl"

    @property
    def sharegpt_task_files(self) -> dict[str, str]:
        return {
            "video_caption": self.sharegpt_caption_file,
            "video_vqa": self.sharegpt_vqa_file,
            "temporal_reasoning": self.sharegpt_temporal_reasoning_file,
            "abnormal_event_detection": self.sharegpt_abnormal_event_file,
            "equipment_anomaly_detection": self.sharegpt_equipment_anomaly_file,
            "aircraft_structure_anomaly_detection": self.sharegpt_aircraft_structure_file,
        }


@dataclass(frozen=True)
class PromptConfig:
    prompt_file_name: str = "video_vqa_prompt.md"
    prompt_version: str = "video_vqa_prompt_v1"
    system_prompt: str = (
        "你是专业的多模态视频数据处理专家。请只基于视频片段、关键帧和已给元信息中"
        "真实可见的内容生成中文训练数据。不得编造不可见信息；不确定内容必须使用"
        "“疑似”“可能”等审慎表述；最终回答不要写时间范围，不要使用“未发现异常事件”、"
        "“未发现明显异常”“无明显异常”等否定异常模板。"
    )
    task_prompt: str = (
        "请根据下面的视频片段信息生成一个 JSON 对象。必须先生成 video_description，"
        "再只基于 video_description 对 abnormal_event_detection、equipment_anomaly_detection、"
        "aircraft_structure_anomaly_detection 三类任务分别给出 0 到 1 的相关度分数，写入 task_relevance。"
        "若三类检测任务中最高相关度达到阈值，请只选择最高的一类检测任务生成问题与答案；"
        "若三类检测任务均未达到阈值，请只生成 video_caption、video_vqa 和 temporal_reasoning 三类任务的问题与答案。"
        "最终选择写入 selected_task_types。"
        "如果画面不涉及设备或飞机结构，不要为对应检测任务生成不适用问答。"
        "问题必须参考 video_description 由大模型生成，提问角度要丰富、多样、具体。"
        "答案不要出现时间范围，不要使用“未发现异常事件”“未发现明显异常”“无明显异常”等字眼。"
        "所有中间任务结果必须包含 video_id、segment_id、time_range、keyframes、evidence、confidence。"
        "回答必须是合法 JSON，不要输出 Markdown。"
    )
    video_caption_prompt: str = (
        "任务 video_caption：生成短视频 Caption / 视频摘要。"
        "描述要尽量丰富，覆盖场景、主体、动作、空间位置、时间变化和可见线索；"
        "必须先参考 video_description 生成一个具体、多样、自然的问题，不要固定使用模板问题；"
        "回答应是完整中文段落，不要出现时间范围，也不要使用否定异常模板。"
    )
    video_vqa_prompt: str = (
        "任务 video_vqa：生成视频 VQA/时序问答数据。"
        "参考 video_description 默认生成 3 条丰富、多样、自然的问题，符合普通人提问习惯；"
        "覆盖观察事实、空间定位、先后顺序、状态变化、细节对比和场景理解等角度。"
        "答案必须基于视觉依据，但不要出现时间范围或否定异常模板。"
    )
    temporal_reasoning_prompt: str = (
        "任务 temporal_reasoning：生成时序推理数据。"
        "参考 video_description 默认生成 3 条需要结合先后变化、状态转移、动作顺序、因果依据或多帧对比才能回答的问题；"
        "问题要丰富、具体、自然，避免与 video_vqa 重复。"
        "答案必须基于视觉依据，但不要出现时间范围或否定异常模板。"
    )
    abnormal_event_detection_prompt: str = (
        "任务 abnormal_event_detection：识别画面中的重点事件和可见风险线索。"
        "仅在路由选中该任务时使用，关注人员或车辆聚集、不寻常移动、越界移动、烟雾、火光、爆炸、建筑损毁、大面积扬尘等具体可见现象；"
        "问题必须参考 video_description 生成，角度要具体丰富，避免直接询问是否异常；答案直接描述具体事件、画面位置、依据和置信度，不要刻意使用异常相关表述，不要出现时间范围或否定异常模板。"
    )
    equipment_anomaly_detection_prompt: str = (
        "任务 equipment_anomaly_detection：检测设备异常。"
        "仅在路由选中该任务时使用，关注发热、泄漏、烟雾、火花、结构变形、部件缺失、表面破损、锈蚀、松动、异常振动、异常积液等；"
        "问题必须参考 video_description 生成，角度要具体丰富；答案说明异常结论、位置、依据和置信度，不要出现时间范围或否定异常模板。"
    )
    aircraft_structure_anomaly_detection_prompt: str = (
        "任务 aircraft_structure_anomaly_detection：检测飞机螺丝及结构件异常。"
        "仅在路由选中该任务时使用，关注螺丝缺失、螺丝松动、螺丝锈蚀、滑丝、铆钉异常、结构裂纹、表面划痕、蒙皮变形、盖板错位、异物残留、污渍或腐蚀等；"
        "问题必须参考 video_description 生成，角度要具体丰富；答案说明异常结论、位置、依据和置信度，不要出现时间范围或否定异常模板。"
    )
    sharegpt_video_token: str = "<video>"
    generated_at_format: str = "%Y-%m-%dT%H:%M:%S"
    json_fence_prefix: str = "```json"
    fence_marker: str = "```"

    @property
    def task_prompts(self) -> dict[str, str]:
        return {
            "video_caption": self.video_caption_prompt,
            "video_vqa": self.video_vqa_prompt,
            "temporal_reasoning": self.temporal_reasoning_prompt,
            "abnormal_event_detection": self.abnormal_event_detection_prompt,
            "equipment_anomaly_detection": self.equipment_anomaly_detection_prompt,
            "aircraft_structure_anomaly_detection": self.aircraft_structure_anomaly_detection_prompt,
        }


@dataclass(frozen=True)
class FieldConfig:
    choices: str = "choices"
    message: str = "message"
    content: str = "content"
    role: str = "role"
    system_role: str = "system"
    user_role: str = "user"
    authorization_header: str = "Authorization"
    content_type_header: str = "Content-Type"
    bearer_prefix: str = "Bearer"
    json_content_type: str = "application/json"
    video_id: str = "video_id"
    video_path: str = "video_path"
    source_video_path: str = "source_video_path"
    clip_duration: str = "clip_duration"
    clip_length_category: str = "clip_length_category"
    clip_quality: str = "clip_quality"
    segment_id: str = "segment_id"
    time_range: str = "time_range"
    keyframes: str = "keyframes"
    confidence: str = "confidence"
    task_type: str = "task_type"
    metadata: str = "metadata"
    videos: str = "videos"


@dataclass(frozen=True)
class TaskConfig:
    video_description: str = "video_description"
    task_relevance: str = "task_relevance"
    selected_task_types: str = "selected_task_types"
    caption: str = "caption"
    qa_pairs: str = "qa_pairs"
    temporal_reasoning_pairs: str = "temporal_reasoning_pairs"
    sharegpt_samples: str = "sharegpt_samples"
    video_caption: str = "video_caption"
    video_vqa: str = "video_vqa"
    temporal_reasoning: str = "temporal_reasoning"
    abnormal_event_detection: str = "abnormal_event_detection"
    equipment_anomaly_detection: str = "equipment_anomaly_detection"
    aircraft_structure_anomaly_detection: str = "aircraft_structure_anomaly_detection"


@dataclass(frozen=True)
class CliConfig:
    description: str = "Generate video QA training data."
    input_dir_arg: str = "--input-dir"
    output_dir_arg: str = "--output-dir"
    max_workers_arg: str = "--max-workers"
    input_dir_default: str = "data"


@dataclass(frozen=True)
class PipelineConfig:
    completed_segment_key: str = "segment_id"
    checkpoint_payload_key: str = "payload"
    checkpoint_status_key: str = "status"
    checkpoint_status_completed: str = "completed"
    checkpoint_version: int = 1
    sample_id_key: str = "sample_id"
    # 是否启用异常事件、设备异常、飞机螺丝及结构件异常检测任务；默认关闭。
    enable_anomaly_detection_tasks: bool = False
    task_mode_key: str = "task_mode"
    task_run_id_key: str = "task_run_id"
    basic_task_mode: str = "basic_only"
    anomaly_task_mode: str = "anomaly_detection_enabled"
    minimum_quality_score: float = 0.7
    task_relevance_threshold: float = 0.9
    default_confidence: float = 0.0
    json_indent: int | None = None
    default_max_workers: int = 2
    minimum_max_workers: int = 1
    stage_start: str = "start"
    stage_dependency_check: str = "dependency_check"
    stage_scan: str = "scan"
    stage_video: str = "video"
    stage_segment: str = "segment"
    stage_clip_video: str = "clip_video"
    stage_clip_quality: str = "clip_quality"
    stage_extract_frames: str = "extract_frames"
    stage_model_request: str = "model_request"
    stage_write_outputs: str = "write_outputs"
    stage_error: str = "error"
    event_started: str = "started"
    event_finished: str = "finished"
    event_skipped: str = "skipped"
    event_failed: str = "failed"
    event_progress: str = "progress"
    progress_total_key: str = "total_segments"
    progress_completed_key: str = "completed_segments"
    progress_failed_key: str = "failed_segments"
    progress_skipped_key: str = "skipped_segments"
    progress_pending_key: str = "pending_segments"
    progress_percent_key: str = "progress_percent"
    progress_total_videos_key: str = "total_videos"
    progress_completed_videos_key: str = "completed_videos"
    progress_failed_videos_key: str = "failed_videos"
    progress_pending_videos_key: str = "pending_videos"
    progress_status_key: str = "status"
    progress_error_key: str = "error"
    progress_status_running: str = "running"
    progress_status_finished: str = "finished"
    progress_status_failed: str = "failed"


@dataclass(frozen=True)
class LoggingConfig:
    logger_name: str = "video_qa"
    log_format: str = "%(asctime)s [%(levelname)s] %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_encoding: str = "utf-8"


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    fields: FieldConfig = field(default_factory=FieldConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    cli: CliConfig = field(default_factory=CliConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


CONFIG = AppConfig()
