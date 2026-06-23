"""Scan videos and extract metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from video_qa.config import CONFIG
from video_qa.models.schemas import VideoMetadata, build_video_id


class VideoScanner:
    def __init__(self, input_dir: Path) -> None:
        self.input_dir = input_dir

    def scan(self) -> list[VideoMetadata]:
        videos = []
        for video_path in sorted(self.input_dir.rglob("*")):
            if video_path.suffix.lower() in CONFIG.video.video_extensions:
                videos.append(self.extract_metadata(video_path))
        return videos

    def extract_metadata(self, video_path: Path) -> VideoMetadata:
        probe = self._run_ffprobe(video_path)
        stream = self._find_video_stream(probe)
        duration = self._read_duration(probe, stream)
        fps = self._read_fps(stream)
        width = stream.get("width", CONFIG.video.unknown_value)
        height = stream.get("height", CONFIG.video.unknown_value)

        return VideoMetadata(
            video_id=build_video_id(video_path),
            video_path=str(video_path),
            duration=duration,
            fps=fps,
            resolution=f"{width}x{height}",
            codec=str(stream.get("codec_name", CONFIG.video.unknown_value)),
            file_size=video_path.stat().st_size,
        )

    def _run_ffprobe(self, video_path: Path) -> dict:
        if shutil.which(CONFIG.video.ffprobe_command) is None:
            raise RuntimeError(CONFIG.video.missing_ffprobe_message)
        command = [
            CONFIG.video.ffprobe_command,
            "-v",
            CONFIG.video.ffprobe_error_level,
            "-show_entries",
            CONFIG.video.ffprobe_show_entries,
            "-of",
            CONFIG.video.ffprobe_output_format,
            str(video_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        return json.loads(completed.stdout)

    def _find_video_stream(self, probe: dict) -> dict:
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == CONFIG.video.video_stream_type:
                return stream
        return {}

    def _read_duration(self, probe: dict, stream: dict) -> float:
        duration = stream.get("duration") or probe.get("format", {}).get("duration")
        return round(float(duration or 0.0), 3)

    def _read_fps(self, stream: dict) -> float:
        rate = stream.get("r_frame_rate", "0/1")
        numerator, denominator = rate.split("/")
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 0.0
        return round(float(numerator) / denominator_value, 3)
