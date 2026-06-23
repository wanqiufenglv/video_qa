"""Create physical video clips for time-based segments."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from video_qa.config import CONFIG
from video_qa.models.schemas import VideoSegment


class VideoClipExtractor:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def extract(self, segment: VideoSegment) -> VideoSegment:
        source_video_path = segment.source_video_path or segment.video_path
        source_path = Path(source_video_path)
        clip_dir = self.output_root / CONFIG.output.segment_clips_dir_name / segment.video_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip_path = clip_dir / f"{segment.segment_id}{self._clip_suffix(source_path)}"
        clip_duration = self._duration(segment.time_range)
        self._extract_one(source_video_path, segment.time_range, clip_path)
        return VideoSegment(
            video_id=segment.video_id,
            video_path=str(clip_path),
            segment_id=segment.segment_id,
            time_range=segment.time_range,
            keyframes=segment.keyframes,
            source_video_path=source_video_path,
            clip_duration=clip_duration,
            clip_length_category=self._length_category(clip_duration),
            clip_quality=segment.clip_quality,
        )

    def _extract_one(self, source_video_path: str, time_range: list[float], clip_path: Path) -> None:
        if shutil.which(CONFIG.video.ffmpeg_command) is None:
            raise RuntimeError(CONFIG.video.missing_ffmpeg_message)
        start_time, end_time = time_range
        duration = self._duration(time_range)
        command = [
            CONFIG.video.ffmpeg_command,
            CONFIG.video.ffmpeg_overwrite_flag,
            CONFIG.video.ffmpeg_seek_flag,
            f"{start_time:.3f}",
            CONFIG.video.ffmpeg_input_flag,
            source_video_path,
            CONFIG.video.ffmpeg_duration_flag,
            f"{duration:.3f}",
            CONFIG.video.ffmpeg_codec_flag,
            CONFIG.video.ffmpeg_copy_codec,
            CONFIG.video.ffmpeg_avoid_negative_ts_flag,
            CONFIG.video.ffmpeg_make_zero_value,
            str(clip_path),
        ]
        subprocess.run(command, capture_output=True, text=True, check=True)

    def _duration(self, time_range: list[float]) -> float:
        start_time, end_time = time_range
        return round(max(end_time - start_time, 0.001), 3)

    def _length_category(self, duration: float) -> str:
        for category, max_seconds in CONFIG.video.clip_length_categories:
            if duration <= max_seconds:
                return category
        return CONFIG.video.clip_length_overflow_category

    def _clip_suffix(self, source_path: Path) -> str:
        suffix = source_path.suffix.lower()
        if suffix in CONFIG.video.video_extensions:
            return suffix
        return ".mp4"
