"""Quality checks for physical video clips."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from video_qa.config import CONFIG
from video_qa.models.schemas import VideoSegment


class VideoClipQualityChecker:
    def check(self, segment: VideoSegment) -> tuple[bool, dict[str, Any]]:
        frames = self._sample_gray_frames(segment.video_path)
        if not frames:
            return False, self._result(segment, "no_sample_frames", 0, 1.0, 1.0, 0.0)

        variances = [self._variance(frame) for frame in frames]
        means = [sum(frame) / len(frame) for frame in frames]
        blank_flags = [
            variance <= CONFIG.video.clip_blank_variance_threshold
            or mean <= CONFIG.video.clip_blank_luma_low_threshold
            or mean >= CONFIG.video.clip_blank_luma_high_threshold
            for mean, variance in zip(means, variances)
        ]
        blank_ratio = sum(blank_flags) / len(blank_flags)

        diffs = [self._mean_abs_diff(frames[index - 1], frames[index]) for index in range(1, len(frames))]
        static_flags = [diff <= CONFIG.video.clip_static_diff_threshold for diff in diffs]
        static_ratio = (sum(static_flags) / len(static_flags)) if static_flags else 0.0
        average_diff = (sum(diffs) / len(diffs)) if diffs else 0.0

        if blank_ratio >= CONFIG.video.clip_max_blank_frame_ratio:
            return False, self._result(segment, "too_many_blank_frames", len(frames), blank_ratio, static_ratio, average_diff)
        if diffs and static_ratio >= CONFIG.video.clip_max_static_frame_ratio:
            return False, self._result(segment, "scene_static", len(frames), blank_ratio, static_ratio, average_diff)
        return True, self._result(segment, "ok", len(frames), blank_ratio, static_ratio, average_diff)

    def _sample_gray_frames(self, video_path: str) -> list[bytes]:
        if shutil.which(CONFIG.video.ffmpeg_command) is None:
            raise RuntimeError(CONFIG.video.missing_ffmpeg_message)
        width = CONFIG.video.clip_quality_frame_width
        height = CONFIG.video.clip_quality_frame_height
        frame_size = width * height
        fps = 1.0 / max(CONFIG.video.clip_quality_sample_interval_seconds, 0.001)
        command = [
            CONFIG.video.ffmpeg_command,
            "-v",
            CONFIG.video.ffprobe_error_level,
            CONFIG.video.ffmpeg_input_flag,
            video_path,
            "-vf",
            f"fps={fps:.6f},scale={width}:{height},format=gray",
            CONFIG.video.ffmpeg_frame_count_flag,
            str(CONFIG.video.clip_quality_max_sample_frames),
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        completed = subprocess.run(command, capture_output=True, check=True)
        data = completed.stdout
        return [data[index : index + frame_size] for index in range(0, len(data), frame_size) if len(data[index : index + frame_size]) == frame_size]

    def _variance(self, frame: bytes) -> float:
        mean = sum(frame) / len(frame)
        return sum((value - mean) ** 2 for value in frame) / len(frame)

    def _mean_abs_diff(self, left: bytes, right: bytes) -> float:
        return sum(abs(a - b) for a, b in zip(left, right)) / min(len(left), len(right))

    def _result(
        self,
        segment: VideoSegment,
        reason: str,
        sample_count: int,
        blank_ratio: float,
        static_ratio: float,
        average_scene_change: float,
    ) -> dict[str, Any]:
        return {
            "is_valid": reason == "ok",
            "reason": reason,
            CONFIG.fields.clip_duration: segment.clip_duration,
            CONFIG.fields.clip_length_category: segment.clip_length_category,
            "sample_count": sample_count,
            "blank_frame_ratio": round(blank_ratio, 4),
            "static_frame_ratio": round(static_ratio, 4),
            "average_scene_change": round(average_scene_change, 4),
        }
