"""Process worker for one video."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from video_qa.config import CONFIG
from video_qa.converters.format_converter import FormatConverter
from video_qa.io_utils.error_logger import ErrorLogger
from video_qa.io_utils.resume_manager import CheckpointManager, current_task_mode, task_run_id
from video_qa.io_utils.stream_writer import JsonlStreamWriter
from video_qa.llm.client import VLMClient
from video_qa.llm.prompt_builder import PromptBuilder
from video_qa.models.schemas import KeyFrame, VideoSegment
from video_qa.quality.quality_checker import QualityChecker
from video_qa.video.clip_extractor import VideoClipExtractor
from video_qa.video.clip_quality_checker import VideoClipQualityChecker
from video_qa.video.frame_extractor import FrameExtractor


def process_video_task(task: dict[str, Any]) -> dict[str, Any]:
    worker = VideoProcessWorker(output_dir=Path(task["output_dir"]))
    return worker.process_video(
        video_id=task["video_id"],
        video_path=task["video_path"],
        segments=[_segment_from_dict(segment) for segment in task["segments"]],
        checkpoint_states=task.get("checkpoint_states", {}),
    )


def _segment_from_dict(item: dict[str, Any]) -> VideoSegment:
    keyframes = []
    for index, frame in enumerate(item.get(CONFIG.fields.keyframes, []), start=1):
        if not isinstance(frame, dict):
            continue
        keyframes.append(
            KeyFrame(
                frame_id=str(frame.get("frame_id") or f"{item['segment_id']}_frame_{index:04d}"),
                timestamp=float(frame.get("timestamp") or 0.0),
                frame_path=str(frame.get("frame_path") or ""),
            )
        )
    return VideoSegment(
        video_id=item["video_id"],
        video_path=item["video_path"],
        segment_id=item["segment_id"],
        time_range=item["time_range"],
        keyframes=keyframes,
        source_video_path=item.get(CONFIG.fields.source_video_path, item["video_path"]),
        clip_duration=float(item.get(CONFIG.fields.clip_duration) or 0.0),
        clip_length_category=str(item.get(CONFIG.fields.clip_length_category) or ""),
        clip_quality=(
            item.get(CONFIG.fields.clip_quality)
            if isinstance(item.get(CONFIG.fields.clip_quality), dict)
            else {}
        ),
    )


class VideoProcessWorker:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.clip_extractor = VideoClipExtractor(output_dir)
        self.clip_quality_checker = VideoClipQualityChecker()
        self.extractor = FrameExtractor(output_dir)
        self.prompt_builder = PromptBuilder()
        self.client = VLMClient()
        self.converter = FormatConverter()
        self.quality_checker = QualityChecker()
        self.process_log_writer = self._writer(CONFIG.output.process_log_file)
        self.error_logger = ErrorLogger(self._writer(CONFIG.output.errors_file))
        self.checkpoint_manager = CheckpointManager(self.output_dir / CONFIG.output.checkpoints_file)

    def process_video(
        self,
        video_id: str,
        video_path: str,
        segments: list[VideoSegment],
        checkpoint_states: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        result = {
            "video_id": video_id,
            "video_path": video_path,
            "total_segments": len(segments),
            "completed_segments": 0,
            "failed_segments": 0,
        }
        self._log_event(CONFIG.pipeline.stage_video, CONFIG.pipeline.event_started, video_id=video_id)
        checkpoint_states = checkpoint_states or {}
        for segment in segments:
            if self._process_segment(segment, checkpoint_states.get(segment.segment_id, {})):
                result["completed_segments"] += 1
            else:
                result["failed_segments"] += 1
        self._log_event(CONFIG.pipeline.stage_video, CONFIG.pipeline.event_finished, **result)
        return result

    def _process_segment(
        self,
        segment: VideoSegment,
        checkpoint_state: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        checkpoint_state = checkpoint_state or {}
        try:
            self._log_event(
                CONFIG.pipeline.stage_segment,
                CONFIG.pipeline.event_started,
                segment_id=segment.segment_id,
                video_id=segment.video_id,
                time_range=segment.time_range,
            )
            segment_with_clip = self._restore_segment(checkpoint_state, CONFIG.pipeline.stage_clip_video)
            if segment_with_clip is None:
                self._log_event(
                    CONFIG.pipeline.stage_clip_video,
                    CONFIG.pipeline.event_started,
                    segment_id=segment.segment_id,
                    video_id=segment.video_id,
                    time_range=segment.time_range,
                )
                segment_with_clip = self.clip_extractor.extract(segment)
                self._save_checkpoint(CONFIG.pipeline.stage_clip_video, segment_with_clip)
                self._log_event(
                    CONFIG.pipeline.stage_clip_video,
                    CONFIG.pipeline.event_finished,
                    segment_id=segment.segment_id,
                    video_id=segment.video_id,
                    clip_path=segment_with_clip.video_path,
                    clip_duration=segment_with_clip.clip_duration,
                    clip_length_category=segment_with_clip.clip_length_category,
                )
            else:
                self._log_resumed_stage(CONFIG.pipeline.stage_clip_video, segment)

            quality_payload = checkpoint_state.get(CONFIG.pipeline.stage_clip_quality, {})
            quality_segment = self._segment_from_payload(quality_payload)
            if quality_segment is not None and self._segment_artifacts_exist(quality_segment):
                segment_with_clip = quality_segment
                is_valid_clip = bool(quality_payload.get("is_valid_clip"))
                clip_quality = segment_with_clip.clip_quality
                self._log_resumed_stage(CONFIG.pipeline.stage_clip_quality, segment)
            else:
                self._log_event(
                    CONFIG.pipeline.stage_clip_quality,
                    CONFIG.pipeline.event_started,
                    segment_id=segment.segment_id,
                    video_id=segment.video_id,
                )
                is_valid_clip, clip_quality = self.clip_quality_checker.check(segment_with_clip)
                segment_with_clip = VideoSegment(
                    video_id=segment_with_clip.video_id,
                    video_path=segment_with_clip.video_path,
                    segment_id=segment_with_clip.segment_id,
                    time_range=segment_with_clip.time_range,
                    keyframes=segment_with_clip.keyframes,
                    source_video_path=segment_with_clip.source_video_path,
                    clip_duration=segment_with_clip.clip_duration,
                    clip_length_category=segment_with_clip.clip_length_category,
                    clip_quality=clip_quality,
                )
                self._save_checkpoint(
                    CONFIG.pipeline.stage_clip_quality,
                    segment_with_clip,
                    is_valid_clip=is_valid_clip,
                )
                self._log_event(
                    CONFIG.pipeline.stage_clip_quality,
                    CONFIG.pipeline.event_finished if is_valid_clip else CONFIG.pipeline.event_skipped,
                    segment_id=segment.segment_id,
                    video_id=segment.video_id,
                    **clip_quality,
                )
            if not is_valid_clip:
                self._writer(CONFIG.output.segments_file).append_unique(
                    segment_with_clip.to_dict(), CONFIG.fields.segment_id, durable=True
                )
                self._writer(CONFIG.output.raw_results_file).append_unique(
                    self._filtered_raw_result(segment_with_clip, clip_quality),
                    CONFIG.pipeline.task_run_id_key,
                    durable=True,
                )
                self._save_checkpoint(CONFIG.pipeline.stage_write_outputs, segment_with_clip, filtered=True)
                self._save_checkpoint(CONFIG.pipeline.stage_segment, segment_with_clip, filtered=True)
                return True

            segment_with_frames = self._restore_segment(
                checkpoint_state,
                CONFIG.pipeline.stage_extract_frames,
                require_keyframes=True,
            )
            if segment_with_frames is None:
                self._log_event(
                    CONFIG.pipeline.stage_extract_frames,
                    CONFIG.pipeline.event_started,
                    segment_id=segment.segment_id,
                    video_id=segment.video_id,
                )
                segment_with_frames = self.extractor.extract(segment_with_clip)
                self._writer(CONFIG.output.segments_file).append_unique(
                    segment_with_frames.to_dict(), CONFIG.fields.segment_id, durable=True
                )
                self._save_checkpoint(CONFIG.pipeline.stage_extract_frames, segment_with_frames)
                self._log_event(
                    CONFIG.pipeline.stage_extract_frames,
                    CONFIG.pipeline.event_finished,
                    segment_id=segment.segment_id,
                    video_id=segment.video_id,
                    frame_count=len(segment_with_frames.keyframes),
                )
            else:
                self._writer(CONFIG.output.segments_file).append_unique(
                    segment_with_frames.to_dict(), CONFIG.fields.segment_id, durable=True
                )
                self._log_resumed_stage(CONFIG.pipeline.stage_extract_frames, segment)
            messages = self.prompt_builder.build_messages(segment_with_frames)
            self._log_event(
                CONFIG.pipeline.stage_model_request,
                CONFIG.pipeline.event_started,
                segment_id=segment.segment_id,
                video_id=segment.video_id,
            )
            raw_result = self.client.complete(messages)
            self._log_event(
                CONFIG.pipeline.stage_model_request,
                CONFIG.pipeline.event_finished,
                segment_id=segment.segment_id,
                video_id=segment.video_id,
            )
            self._add_raw_trace(raw_result, segment_with_frames)
            self._writer(CONFIG.output.raw_results_file).append_unique(
                raw_result, CONFIG.pipeline.task_run_id_key, durable=True
            )
            self._save_checkpoint(CONFIG.pipeline.stage_model_request, segment_with_frames)
            self._write_converted_records(raw_result, segment_with_frames)
            self._save_checkpoint(CONFIG.pipeline.stage_write_outputs, segment_with_frames)
            self._save_checkpoint(CONFIG.pipeline.stage_segment, segment_with_frames)
            self._log_event(
                CONFIG.pipeline.stage_segment,
                CONFIG.pipeline.event_finished,
                segment_id=segment.segment_id,
                video_id=segment.video_id,
            )
            return True
        except Exception as error:
            self.error_logger.log(
                stage=CONFIG.pipeline.stage_segment,
                message=str(error),
                context=segment.to_dict(),
            )
            self._log_event(
                CONFIG.pipeline.stage_error,
                CONFIG.pipeline.event_failed,
                segment_id=segment.segment_id,
                video_id=segment.video_id,
                error=str(error),
            )
            return False

    def _restore_segment(
        self,
        checkpoint_state: dict[str, dict[str, Any]],
        stage: str,
        require_keyframes: bool = False,
    ) -> VideoSegment | None:
        restored = self._segment_from_payload(checkpoint_state.get(stage, {}))
        if restored is None or not self._segment_artifacts_exist(restored, require_keyframes=require_keyframes):
            return None
        return restored

    def _segment_from_payload(self, payload: dict[str, Any]) -> VideoSegment | None:
        item = payload.get("segment") if isinstance(payload, dict) else None
        if not isinstance(item, dict):
            return None
        try:
            return _segment_from_dict(item)
        except (KeyError, TypeError, ValueError):
            return None

    def _segment_artifacts_exist(self, segment: VideoSegment, require_keyframes: bool = False) -> bool:
        if not segment.video_path or not Path(segment.video_path).is_file():
            return False
        if not require_keyframes:
            return True
        return bool(segment.keyframes) and all(
            frame.frame_path and Path(frame.frame_path).is_file() for frame in segment.keyframes
        )

    def _save_checkpoint(self, stage: str, segment: VideoSegment, **extra: Any) -> None:
        payload = {"segment": segment.to_dict(), **extra}
        self.checkpoint_manager.save(segment.segment_id, stage, payload)

    def _log_resumed_stage(self, stage: str, segment: VideoSegment) -> None:
        self._log_event(
            stage,
            CONFIG.pipeline.event_skipped,
            segment_id=segment.segment_id,
            video_id=segment.video_id,
            resumed_from_checkpoint=True,
        )

    def _add_raw_trace(self, raw_result: dict[str, Any], segment: VideoSegment) -> None:
        raw_result.setdefault(CONFIG.pipeline.completed_segment_key, segment.segment_id)
        raw_result.setdefault(CONFIG.pipeline.task_mode_key, current_task_mode())
        raw_result.setdefault(CONFIG.pipeline.task_run_id_key, task_run_id(segment.segment_id))
        raw_result.setdefault(CONFIG.fields.video_id, segment.video_id)
        raw_result.setdefault(CONFIG.fields.video_path, segment.video_path)
        raw_result.setdefault(CONFIG.fields.source_video_path, segment.source_video_path)
        raw_result.setdefault(CONFIG.fields.segment_id, segment.segment_id)
        raw_result.setdefault(CONFIG.fields.time_range, segment.time_range)
        raw_result.setdefault(CONFIG.fields.keyframes, [frame.to_dict() for frame in segment.keyframes])
        raw_result.setdefault(CONFIG.fields.clip_duration, segment.clip_duration)
        raw_result.setdefault(CONFIG.fields.clip_length_category, segment.clip_length_category)
        raw_result.setdefault(CONFIG.fields.clip_quality, segment.clip_quality)

    def _filtered_raw_result(self, segment: VideoSegment, clip_quality: dict[str, Any]) -> dict[str, Any]:
        return {
            CONFIG.pipeline.completed_segment_key: segment.segment_id,
            CONFIG.pipeline.task_mode_key: current_task_mode(),
            CONFIG.pipeline.task_run_id_key: task_run_id(segment.segment_id),
            CONFIG.fields.video_id: segment.video_id,
            CONFIG.fields.video_path: segment.video_path,
            CONFIG.fields.source_video_path: segment.source_video_path,
            CONFIG.fields.segment_id: segment.segment_id,
            CONFIG.fields.time_range: segment.time_range,
            CONFIG.fields.clip_duration: segment.clip_duration,
            CONFIG.fields.clip_length_category: segment.clip_length_category,
            CONFIG.fields.clip_quality: clip_quality,
            "filtered": True,
            "filter_reason": clip_quality.get("reason", "clip_quality_failed"),
        }

    def _write_converted_records(self, raw_result: dict[str, Any], segment: VideoSegment) -> None:
        caption = self.converter.caption_record(raw_result, segment)
        if caption:
            self._append_if_valid(CONFIG.output.captions_file, caption, segment)

        for record in self.converter.vqa_records(raw_result, segment):
            self._append_if_valid(CONFIG.output.vqa_file, record, segment)

        for record in self.converter.temporal_reasoning_records(raw_result, segment):
            self._append_if_valid(CONFIG.output.temporal_reasoning_file, record, segment)

        task_to_file = {
            CONFIG.tasks.abnormal_event_detection: CONFIG.output.abnormal_events_file,
            CONFIG.tasks.equipment_anomaly_detection: CONFIG.output.equipment_anomalies_file,
            CONFIG.tasks.aircraft_structure_anomaly_detection: CONFIG.output.aircraft_structure_anomalies_file,
        }
        for task_key, file_name in task_to_file.items():
            record = self.converter.task_record(raw_result, segment, task_key)
            if record:
                self._append_if_valid(file_name, record, segment)

        for task_type, records in self.converter.sharegpt_records_by_task(raw_result, segment).items():
            file_name = CONFIG.output.sharegpt_task_files[task_type]
            for record in records:
                self._append_if_valid(file_name, record, segment)

    def _append_if_valid(self, file_name: str, record: dict[str, Any], segment: VideoSegment) -> None:
        if self.quality_checker.is_valid(record):
            self._writer(file_name).append(record)
            return
        self.error_logger.log(
            stage=CONFIG.pipeline.stage_write_outputs,
            message="Generated record failed local quality checks.",
            context={
                CONFIG.fields.video_id: segment.video_id,
                CONFIG.fields.segment_id: segment.segment_id,
                "target_file": file_name,
                "record": record,
            },
        )

    def _writer(self, file_name: str) -> JsonlStreamWriter:
        return JsonlStreamWriter(self.output_dir / file_name)

    def _log_event(self, stage: str, event: str, **context: Any) -> None:
        self.process_log_writer.append(
            {
                "stage": stage,
                "event": event,
                "context": context,
                "created_at": datetime.now().strftime(CONFIG.prompt.generated_at_format),
            }
        )
