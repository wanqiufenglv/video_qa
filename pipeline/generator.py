"""End-to-end video QA generation pipeline."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
from threading import Lock
from typing import Any

from video_qa.config import CONFIG
from video_qa.converters.format_converter import FormatConverter
from video_qa.io_utils.error_logger import ErrorLogger
from video_qa.io_utils.logging_config import setup_logging
from video_qa.io_utils.progress_writer import ProgressWriter
from video_qa.io_utils.resume_manager import ResumeManager
from video_qa.io_utils.stream_writer import JsonlStreamWriter
from video_qa.models.schemas import KeyFrame, VideoSegment
from video_qa.pipeline.video_worker import process_video_task
from video_qa.quality.quality_checker import QualityChecker
from video_qa.video.dependencies import VideoDependencyChecker
from video_qa.video.scanner import VideoScanner
from video_qa.video.segmenter import VideoSegmenter


class VideoQAGenerator:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir or CONFIG.output.output_dir
        self.max_workers = max(max_workers or CONFIG.pipeline.default_max_workers, CONFIG.pipeline.minimum_max_workers)
        self.scanner = VideoScanner(input_dir)
        self.segmenter = VideoSegmenter()
        self.logger = setup_logging(self.output_dir)
        self.process_log_writer = self._writer(CONFIG.output.process_log_file)
        self.progress_writer = ProgressWriter(self.output_dir / CONFIG.output.progress_file)
        self.error_logger = ErrorLogger(self._writer(CONFIG.output.errors_file), self.logger)
        self.dependency_checker = VideoDependencyChecker()
        self._progress_lock = Lock()
        self._progress = self._empty_progress()

    def run(self) -> None:
        self._log_event(
            CONFIG.pipeline.stage_start,
            CONFIG.pipeline.event_started,
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            max_workers=self.max_workers,
        )
        self._write_progress()
        self._log_event(CONFIG.pipeline.stage_dependency_check, CONFIG.pipeline.event_started)
        try:
            self.dependency_checker.validate()
        except Exception as error:
            self.logger.exception("%s failed: %s", CONFIG.pipeline.stage_dependency_check, error)
            self._set_progress_failed(str(error))
            self._log_event(
                CONFIG.pipeline.stage_dependency_check,
                CONFIG.pipeline.event_failed,
                error=str(error),
            )
            raise
        self._log_event(CONFIG.pipeline.stage_dependency_check, CONFIG.pipeline.event_finished)
        completed_segments = ResumeManager(
            self.output_dir / CONFIG.output.raw_results_file,
            CONFIG.pipeline.completed_segment_key,
        ).load_completed_ids()
        self._log_event(CONFIG.pipeline.stage_scan, CONFIG.pipeline.event_started)
        videos = self.scanner.scan()
        self._log_event(CONFIG.pipeline.stage_scan, CONFIG.pipeline.event_finished, video_count=len(videos))

        video_tasks: list[dict[str, Any]] = []
        skipped_count = 0
        total_pending_segments = 0
        for metadata in videos:
            self._log_event(
                CONFIG.pipeline.stage_video,
                CONFIG.pipeline.event_progress,
                video_id=metadata.video_id,
                video_path=metadata.video_path,
            )
            self._writer(CONFIG.output.metadata_file).append(metadata.to_dict())
            pending_segments: list[VideoSegment] = []
            for segment in self.segmenter.build_segments(metadata):
                if segment.segment_id in completed_segments:
                    skipped_count += 1
                    self._log_event(
                        CONFIG.pipeline.stage_segment,
                        CONFIG.pipeline.event_skipped,
                        segment_id=segment.segment_id,
                    )
                    continue
                pending_segments.append(segment)
            if pending_segments:
                total_pending_segments += len(pending_segments)
                video_tasks.append(
                    {
                        "video_id": metadata.video_id,
                        "video_path": metadata.video_path,
                        "output_dir": str(self.output_dir),
                        "segments": [segment.to_dict() for segment in pending_segments],
                    }
                )

        self._initialize_progress(
            total_segments=total_pending_segments,
            skipped_segments=skipped_count,
            total_videos=len(video_tasks),
        )
        self._log_event(
            CONFIG.pipeline.stage_segment,
            CONFIG.pipeline.event_progress,
            total_segments=total_pending_segments,
            total_videos=len(video_tasks),
            skipped_segments=skipped_count,
            max_workers=self.max_workers,
        )
        self._run_video_tasks(video_tasks)
        self._rebuild_missing_sharegpt_records()
        self._set_progress_finished()
        self._log_event(CONFIG.pipeline.stage_start, CONFIG.pipeline.event_finished)

    def _run_video_tasks(self, video_tasks: list[dict[str, Any]]) -> None:
        if not video_tasks:
            self._write_progress()
            return

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(process_video_task, task): task for task in video_tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "video_id": task["video_id"],
                        "video_path": task["video_path"],
                        "total_segments": len(task["segments"]),
                        "completed_segments": 0,
                        "failed_segments": len(task["segments"]),
                    }
                    self.error_logger.log(
                        stage=CONFIG.pipeline.stage_video,
                        message=str(error),
                        context=task,
                    )
                self._mark_video_done(result)

    def _rebuild_missing_sharegpt_records(self) -> None:
        raw_path = self.output_dir / CONFIG.output.raw_results_file
        if not raw_path.exists():
            return

        existing_segment_ids_by_task = self._load_sharegpt_segment_ids_by_task()
        existing_signatures_by_task = self._load_sharegpt_signatures_by_task()
        metadata_by_video_id = self._load_metadata_video_paths()
        segments_by_id = self._load_segments_by_id(metadata_by_video_id)
        converter = FormatConverter()
        checker = QualityChecker()
        rebuilt_records = 0
        rebuilt_segments: set[str] = set()

        for raw_result in self._iter_jsonl(raw_path):
            segment_id = self._raw_segment_id(raw_result)
            if not segment_id:
                continue

            segment = self._segment_from_raw_result(raw_result, metadata_by_video_id, segments_by_id)
            if segment is None:
                self.error_logger.log(
                    stage=CONFIG.pipeline.stage_write_outputs,
                    message="Cannot rebuild ShareGPT record because raw result lacks trace fields.",
                    context=raw_result,
                )
                continue

            records_by_task = converter.sharegpt_records_by_task(raw_result, segment)
            for task_type, records in records_by_task.items():
                if segment_id in existing_segment_ids_by_task.get(task_type, set()):
                    continue
                writer = self._writer(CONFIG.output.sharegpt_task_files[task_type])
                task_records = 0
                for record in records:
                    signature = self._sharegpt_signature(record)
                    if signature and signature in existing_signatures_by_task.get(task_type, set()):
                        continue
                    if checker.is_valid(record):
                        writer.append(record)
                        if signature:
                            existing_signatures_by_task.setdefault(task_type, set()).add(signature)
                        rebuilt_records += 1
                        task_records += 1
                if task_records:
                    existing_segment_ids_by_task.setdefault(task_type, set()).add(segment_id)
                    rebuilt_segments.add(segment_id)

        if rebuilt_segments:
            self._log_event(
                CONFIG.pipeline.stage_write_outputs,
                CONFIG.pipeline.event_finished,
                rebuilt_segments=len(rebuilt_segments),
                rebuilt_sharegpt_records=rebuilt_records,
            )

    def _load_sharegpt_segment_ids_by_task(self) -> dict[str, set[str]]:
        segment_ids_by_task: dict[str, set[str]] = {}
        for task_type, file_name in CONFIG.output.sharegpt_task_files.items():
            segment_ids: set[str] = set()
            sharegpt_path = self.output_dir / file_name
            for item in self._iter_jsonl(sharegpt_path):
                segment_id = item.get(CONFIG.fields.segment_id)
                metadata = item.get(CONFIG.fields.metadata)
                if not segment_id and isinstance(metadata, dict):
                    segment_id = metadata.get(CONFIG.fields.segment_id)
                if segment_id:
                    segment_ids.add(str(segment_id))
            segment_ids_by_task[task_type] = segment_ids
        return segment_ids_by_task

    def _load_sharegpt_signatures_by_task(self) -> dict[str, set[str]]:
        signatures_by_task: dict[str, set[str]] = {}
        for task_type, file_name in CONFIG.output.sharegpt_task_files.items():
            signatures: set[str] = set()
            sharegpt_path = self.output_dir / file_name
            for item in self._iter_jsonl(sharegpt_path):
                signature = self._sharegpt_signature(item)
                if signature:
                    signatures.add(signature)
            signatures_by_task[task_type] = signatures
        return signatures_by_task

    def _sharegpt_signature(self, item: dict[str, Any]) -> str:
        conversations = item.get("conversations")
        videos = item.get(CONFIG.fields.videos)
        if not isinstance(conversations, list) or not isinstance(videos, list):
            return ""
        return json.dumps(
            {"conversations": conversations, CONFIG.fields.videos: videos},
            ensure_ascii=False,
            sort_keys=True,
        )

    def _load_metadata_video_paths(self) -> dict[str, str]:
        video_paths: dict[str, str] = {}
        metadata_path = self.output_dir / CONFIG.output.metadata_file
        for item in self._iter_jsonl(metadata_path):
            video_id = item.get(CONFIG.fields.video_id)
            video_path = item.get(CONFIG.fields.video_path)
            if video_id and video_path:
                video_paths[str(video_id)] = str(video_path)
        return video_paths

    def _load_segments_by_id(self, metadata_by_video_id: dict[str, str]) -> dict[str, VideoSegment]:
        segments: dict[str, VideoSegment] = {}
        segments_path = self.output_dir / CONFIG.output.segments_file
        for item in self._iter_jsonl(segments_path):
            segment = self._segment_from_record(item, metadata_by_video_id)
            if segment is not None:
                segments[segment.segment_id] = segment
        return segments

    def _segment_from_record(
        self,
        item: dict[str, Any],
        metadata_by_video_id: dict[str, str],
    ) -> VideoSegment | None:
        video_id = item.get(CONFIG.fields.video_id)
        segment_id = item.get(CONFIG.fields.segment_id)
        time_range = item.get(CONFIG.fields.time_range)
        if not video_id or not segment_id or not isinstance(time_range, list) or len(time_range) != 2:
            return None
        video_path = item.get(CONFIG.fields.video_path) or metadata_by_video_id.get(str(video_id))
        if not video_path:
            return None
        return VideoSegment(
            video_id=str(video_id),
            video_path=str(video_path),
            segment_id=str(segment_id),
            time_range=[float(time_range[0]), float(time_range[1])],
            keyframes=self._keyframes_from_raw(item.get(CONFIG.fields.keyframes, []), str(segment_id)),
            source_video_path=str(item.get(CONFIG.fields.source_video_path) or metadata_by_video_id.get(str(video_id)) or video_path),
            clip_duration=float(item.get(CONFIG.fields.clip_duration) or 0.0),
            clip_length_category=str(item.get(CONFIG.fields.clip_length_category) or ""),
            clip_quality=item.get(CONFIG.fields.clip_quality) if isinstance(item.get(CONFIG.fields.clip_quality), dict) else {},
        )

    def _iter_jsonl(self, file_path: Path):
        if not file_path.exists():
            return
        with file_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    yield item

    def _raw_segment_id(self, raw_result: dict[str, Any]) -> str:
        return str(raw_result.get(CONFIG.pipeline.completed_segment_key) or raw_result.get(CONFIG.fields.segment_id) or "")

    def _segment_from_raw_result(
        self,
        raw_result: dict[str, Any],
        metadata_by_video_id: dict[str, str],
        segments_by_id: dict[str, VideoSegment],
    ) -> VideoSegment | None:
        segment_id = self._raw_segment_id(raw_result)
        if segment_id in segments_by_id:
            return segments_by_id[segment_id]

        video_id = raw_result.get(CONFIG.fields.video_id) or self._video_id_from_segment_id(segment_id)
        time_range = raw_result.get(CONFIG.fields.time_range)
        if not video_id or not segment_id or not isinstance(time_range, list) or len(time_range) != 2:
            return None

        video_path = (
            raw_result.get(CONFIG.fields.video_path)
            or self._video_path_from_sharegpt_samples(raw_result)
            or metadata_by_video_id.get(str(video_id))
        )
        if not video_path:
            return None

        return VideoSegment(
            video_id=str(video_id),
            video_path=str(video_path),
            segment_id=segment_id,
            time_range=[float(time_range[0]), float(time_range[1])],
            keyframes=self._keyframes_from_raw(raw_result.get(CONFIG.fields.keyframes, []), segment_id),
            source_video_path=str(raw_result.get(CONFIG.fields.source_video_path) or metadata_by_video_id.get(str(video_id)) or video_path),
            clip_duration=float(raw_result.get(CONFIG.fields.clip_duration) or 0.0),
            clip_length_category=str(raw_result.get(CONFIG.fields.clip_length_category) or ""),
            clip_quality=raw_result.get(CONFIG.fields.clip_quality) if isinstance(raw_result.get(CONFIG.fields.clip_quality), dict) else {},
        )

    def _keyframes_from_raw(self, raw_keyframes: Any, segment_id: str) -> list[KeyFrame]:
        keyframes: list[KeyFrame] = []
        if not isinstance(raw_keyframes, list):
            return keyframes
        for index, frame in enumerate(raw_keyframes, start=1):
            if isinstance(frame, dict):
                keyframes.append(
                    KeyFrame(
                        frame_id=str(frame.get("frame_id") or f"{segment_id}_frame_{index:04d}"),
                        timestamp=float(frame.get("timestamp") or 0.0),
                        frame_path=str(frame.get("frame_path") or ""),
                    )
                )
            elif isinstance(frame, str):
                keyframes.append(KeyFrame(frame_id=frame, timestamp=0.0, frame_path=""))
        return keyframes

    def _video_id_from_segment_id(self, segment_id: str) -> str:
        marker = "_seg_"
        if marker not in segment_id:
            return ""
        return segment_id.split(marker, 1)[0]

    def _video_path_from_sharegpt_samples(self, raw_result: dict[str, Any]) -> str:
        samples = raw_result.get(CONFIG.tasks.sharegpt_samples)
        if not isinstance(samples, list):
            return ""
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            videos = sample.get(CONFIG.fields.videos)
            if isinstance(videos, list) and videos:
                return str(videos[0])
        return ""

    def _writer(self, file_name: str) -> JsonlStreamWriter:
        return JsonlStreamWriter(self.output_dir / file_name)

    def _log_event(self, stage: str, event: str, **context: Any) -> None:
        record = {
            "stage": stage,
            "event": event,
            "context": context,
            "created_at": datetime.now().strftime(CONFIG.prompt.generated_at_format),
        }
        self.process_log_writer.append(record)
        context_text = " ".join(f"{key}={value}" for key, value in context.items())
        message = f"{stage} {event}"
        if context_text:
            message = f"{message} {context_text}"
        self.logger.info(message)

    def _empty_progress(self) -> dict[str, Any]:
        return {
            CONFIG.pipeline.progress_total_key: 0,
            CONFIG.pipeline.progress_completed_key: 0,
            CONFIG.pipeline.progress_failed_key: 0,
            CONFIG.pipeline.progress_skipped_key: 0,
            CONFIG.pipeline.progress_pending_key: 0,
            CONFIG.pipeline.progress_percent_key: 0.0,
            CONFIG.pipeline.progress_total_videos_key: 0,
            CONFIG.pipeline.progress_completed_videos_key: 0,
            CONFIG.pipeline.progress_failed_videos_key: 0,
            CONFIG.pipeline.progress_pending_videos_key: 0,
            CONFIG.pipeline.progress_status_key: CONFIG.pipeline.progress_status_running,
            CONFIG.pipeline.progress_error_key: "",
            "max_workers": self.max_workers,
            "updated_at": datetime.now().strftime(CONFIG.prompt.generated_at_format),
        }

    def _initialize_progress(self, total_segments: int, skipped_segments: int, total_videos: int) -> None:
        with self._progress_lock:
            self._progress = self._empty_progress()
            self._progress[CONFIG.pipeline.progress_total_key] = total_segments
            self._progress[CONFIG.pipeline.progress_skipped_key] = skipped_segments
            self._progress[CONFIG.pipeline.progress_pending_key] = total_segments
            self._progress[CONFIG.pipeline.progress_total_videos_key] = total_videos
            self._progress[CONFIG.pipeline.progress_pending_videos_key] = total_videos
            self._refresh_progress_locked()
            self.progress_writer.write(self._progress)

    def _mark_video_done(self, result: dict[str, Any]) -> None:
        with self._progress_lock:
            completed_segments = result["completed_segments"]
            failed_segments = result["failed_segments"]
            self._progress[CONFIG.pipeline.progress_completed_key] += completed_segments
            self._progress[CONFIG.pipeline.progress_failed_key] += failed_segments
            if failed_segments:
                self._progress[CONFIG.pipeline.progress_failed_videos_key] += 1
            else:
                self._progress[CONFIG.pipeline.progress_completed_videos_key] += 1
            self._refresh_progress_locked()
            self.progress_writer.write(self._progress)
            log_context = dict(result)
            log_context.update(self._progress)
            self._log_event(CONFIG.pipeline.stage_video, CONFIG.pipeline.event_progress, **log_context)

    def _write_progress(self) -> None:
        with self._progress_lock:
            self._refresh_progress_locked()
            self.progress_writer.write(self._progress)

    def _refresh_progress_locked(self) -> None:
        total = self._progress[CONFIG.pipeline.progress_total_key]
        completed = self._progress[CONFIG.pipeline.progress_completed_key]
        failed = self._progress[CONFIG.pipeline.progress_failed_key]
        pending = max(total - completed - failed, 0)
        percent = ((completed + failed) / total * 100) if total else 100.0
        total_videos = self._progress[CONFIG.pipeline.progress_total_videos_key]
        completed_videos = self._progress[CONFIG.pipeline.progress_completed_videos_key]
        failed_videos = self._progress[CONFIG.pipeline.progress_failed_videos_key]
        self._progress[CONFIG.pipeline.progress_pending_key] = pending
        self._progress[CONFIG.pipeline.progress_pending_videos_key] = max(total_videos - completed_videos - failed_videos, 0)
        self._progress[CONFIG.pipeline.progress_percent_key] = round(percent, 2)
        self._progress["updated_at"] = datetime.now().strftime(CONFIG.prompt.generated_at_format)

    def _set_progress_failed(self, error: str) -> None:
        with self._progress_lock:
            self._progress[CONFIG.pipeline.progress_status_key] = CONFIG.pipeline.progress_status_failed
            self._progress[CONFIG.pipeline.progress_error_key] = error
            self._refresh_progress_locked()
            self.progress_writer.write(self._progress)

    def _set_progress_finished(self) -> None:
        with self._progress_lock:
            failed = self._progress[CONFIG.pipeline.progress_failed_key]
            self._progress[CONFIG.pipeline.progress_status_key] = (
                CONFIG.pipeline.progress_status_failed if failed else CONFIG.pipeline.progress_status_finished
            )
            self._refresh_progress_locked()
            self.progress_writer.write(self._progress)
