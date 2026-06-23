"""Extract keyframes for each segment."""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

from video_qa.config import CONFIG
from video_qa.models.schemas import KeyFrame, VideoSegment


class FrameExtractor:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def extract(self, segment: VideoSegment) -> VideoSegment:
        frame_dir = self.output_root / CONFIG.output.frames_dir_name / segment.video_id / segment.segment_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        relative_timestamps = self._relative_timestamps(segment.time_range)
        keyframes = []
        absolute_start = segment.time_range[0]

        for index, relative_timestamp in enumerate(relative_timestamps, start=1):
            frame_id = f"{segment.segment_id}_frame_{index:04d}"
            frame_path = frame_dir / f"{index:06d}.jpg"
            self._extract_one(segment.video_path, relative_timestamp, frame_path)
            keyframes.append(
                KeyFrame(
                    frame_id=frame_id,
                    timestamp=round(absolute_start + relative_timestamp, 3),
                    frame_path=str(frame_path),
                )
            )

        return VideoSegment(
            video_id=segment.video_id,
            video_path=segment.video_path,
            segment_id=segment.segment_id,
            time_range=segment.time_range,
            keyframes=keyframes,
            source_video_path=segment.source_video_path,
            clip_duration=segment.clip_duration,
            clip_length_category=segment.clip_length_category,
            clip_quality=segment.clip_quality,
        )

    def _relative_timestamps(self, time_range: list[float]) -> list[float]:
        start_time, end_time = time_range
        duration = max(end_time - start_time, 0.001)
        if CONFIG.video.frame_interval_seconds > 0:
            return self._interval_timestamps(duration)
        frame_count = max(CONFIG.video.keyframes_per_segment, 1)
        step = duration / (frame_count + 1)
        return [step * index for index in range(1, frame_count + 1)]

    def _interval_timestamps(self, duration: float) -> list[float]:
        interval = max(CONFIG.video.frame_interval_seconds, 0.001)
        max_frames = max(CONFIG.video.max_keyframes_per_segment, 1)
        timestamps = []
        current = interval
        while current < duration and len(timestamps) < max_frames:
            timestamps.append(current)
            current += interval
        if not timestamps:
            timestamps.append(duration / 2)
        return timestamps

    def _extract_one(self, video_path: str, timestamp: float, frame_path: Path) -> None:
        if shutil.which(CONFIG.video.ffmpeg_command) is None:
            raise RuntimeError(CONFIG.video.missing_ffmpeg_message)
        command = [
            CONFIG.video.ffmpeg_command,
            CONFIG.video.ffmpeg_overwrite_flag,
            CONFIG.video.ffmpeg_seek_flag,
            f"{timestamp:.3f}",
            CONFIG.video.ffmpeg_input_flag,
            video_path,
            CONFIG.video.ffmpeg_frame_count_flag,
            CONFIG.video.single_frame_count,
            CONFIG.video.ffmpeg_quality_flag,
            str(CONFIG.video.jpeg_quality),
            str(frame_path),
        ]
        subprocess.run(command, capture_output=True, text=True, check=True)
