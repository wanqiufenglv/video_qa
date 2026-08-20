"""Extract keyframes for each segment."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from video_qa.config import CONFIG
from video_qa.models.schemas import KeyFrame, VideoSegment


class FrameExtractor:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def extract(self, segment: VideoSegment) -> VideoSegment:
        frame_dir = self.output_root / CONFIG.output.frames_dir_name / segment.video_id / segment.segment_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        relative_timestamps = self._relative_timestamps(segment)
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

    def _relative_timestamps(self, segment: VideoSegment) -> list[float]:
        start_time, end_time = segment.time_range
        duration = max(segment.clip_duration or (end_time - start_time), 0.001)
        profile = self._sampling_profile(segment.clip_length_category)

        if profile:
            interval, min_frames, max_frames = profile
            interval = self._density_adjusted_interval(interval, segment.clip_quality)
            timestamps = self._interval_timestamps(duration, interval, min_frames, max_frames)
            return self._local_density_boost(segment.video_path, duration, timestamps, max_frames)

        if CONFIG.video.frame_interval_seconds > 0:
            timestamps = self._interval_timestamps(
                duration,
                CONFIG.video.frame_interval_seconds,
                1,
                CONFIG.video.max_keyframes_per_segment,
            )
            return self._local_density_boost(
                segment.video_path,
                duration,
                timestamps,
                CONFIG.video.max_keyframes_per_segment,
            )

        frame_count = max(CONFIG.video.keyframes_per_segment, 1)
        return self._even_timestamps(duration, frame_count)

    def _sampling_profile(self, category: str) -> tuple[float, int, int] | None:
        for profile_category, interval, min_frames, max_frames in CONFIG.video.keyframe_sampling_profiles:
            if profile_category == category:
                return max(interval, 0.001), max(min_frames, 1), max(max_frames, 1)
        return None

    def _density_adjusted_interval(self, interval: float, clip_quality: dict[str, object]) -> float:
        if not clip_quality:
            return interval

        scene_change = self._float_value(clip_quality.get("average_scene_change"))
        static_ratio = self._float_value(clip_quality.get("static_frame_ratio"))

        if (
            static_ratio >= CONFIG.video.keyframe_high_static_ratio_threshold
            or scene_change <= CONFIG.video.keyframe_low_density_scene_change_threshold
        ):
            return interval * max(CONFIG.video.keyframe_low_density_interval_multiplier, 0.001)
        if scene_change >= CONFIG.video.keyframe_high_density_scene_change_threshold:
            return interval * max(CONFIG.video.keyframe_high_density_interval_multiplier, 0.001)
        return interval

    def _interval_timestamps(self, duration: float, interval: float, min_frames: int, max_frames: int) -> list[float]:
        interval = max(interval, 0.001)
        max_frames = max(max_frames, 1)
        min_frames = min(max(min_frames, 1), max_frames)
        timestamps = []
        current = interval

        while current < duration:
            timestamps.append(self._clamp_timestamp(current, duration))
            current += interval

        if len(timestamps) > max_frames:
            return self._rounded_sorted(self._even_timestamps(duration, max_frames))

        if len(timestamps) < min_frames:
            timestamps = self._merge_timestamps(
                duration,
                timestamps,
                self._even_timestamps(duration, min_frames),
                max_frames,
            )
        if not timestamps:
            timestamps.append(self._clamp_timestamp(duration / 2, duration))
        return self._rounded_sorted(timestamps)

    def _even_timestamps(self, duration: float, frame_count: int) -> list[float]:
        frame_count = max(frame_count, 1)
        step = duration / (frame_count + 1)
        return [self._clamp_timestamp(step * index, duration) for index in range(1, frame_count + 1)]

    def _local_density_boost(
        self,
        video_path: str,
        duration: float,
        base_timestamps: list[float],
        max_frames: int,
    ) -> list[float]:
        if not CONFIG.video.enable_local_density_keyframe_boost or len(base_timestamps) >= max_frames:
            return self._rounded_sorted(base_timestamps)

        samples = self._sample_density_frames(video_path, duration)
        if len(samples) < 2:
            return self._rounded_sorted(base_timestamps)

        changes = []
        for index in range(1, len(samples)):
            previous_time, previous_frame = samples[index - 1]
            current_time, current_frame = samples[index]
            scene_change = self._mean_abs_diff(previous_frame, current_frame)
            timestamp = (previous_time + current_time) / 2
            changes.append((scene_change, timestamp))

        if not changes:
            return self._rounded_sorted(base_timestamps)

        average_change = sum(change for change, _ in changes) / len(changes)
        threshold = max(
            average_change * CONFIG.video.keyframe_local_change_multiplier,
            CONFIG.video.keyframe_local_min_scene_change,
        )
        candidates = sorted(
            ((change, timestamp) for change, timestamp in changes if change >= threshold),
            reverse=True,
        )
        boosted = list(base_timestamps)
        for _, timestamp in candidates:
            if len(boosted) >= max_frames:
                break
            timestamp = self._clamp_timestamp(timestamp, duration)
            if self._is_far_enough(timestamp, boosted):
                boosted.append(timestamp)
        return self._rounded_sorted(boosted)

    def _sample_density_frames(self, video_path: str, duration: float) -> list[tuple[float, bytes]]:
        if shutil.which(CONFIG.video.ffmpeg_command) is None:
            raise RuntimeError(CONFIG.video.missing_ffmpeg_message)

        width = CONFIG.video.keyframe_density_frame_width
        height = CONFIG.video.keyframe_density_frame_height
        frame_size = width * height
        max_frames = max(CONFIG.video.keyframe_density_max_sample_frames, 2)
        sample_interval = max(
            CONFIG.video.keyframe_density_sample_interval_seconds,
            duration / max_frames,
            0.001,
        )
        command = [
            CONFIG.video.ffmpeg_command,
            "-v",
            CONFIG.video.ffprobe_error_level,
            CONFIG.video.ffmpeg_input_flag,
            video_path,
            "-vf",
            f"fps={1.0 / sample_interval:.6f},scale={width}:{height},format=gray",
            CONFIG.video.ffmpeg_frame_count_flag,
            str(max_frames),
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, check=True)
        except subprocess.CalledProcessError:
            return []

        data = completed.stdout
        frames = [
            data[index : index + frame_size]
            for index in range(0, len(data), frame_size)
            if len(data[index : index + frame_size]) == frame_size
        ]
        return [(min(index * sample_interval, duration), frame) for index, frame in enumerate(frames)]

    def _merge_timestamps(
        self,
        duration: float,
        timestamps: list[float],
        candidates: list[float],
        max_frames: int,
    ) -> list[float]:
        merged = list(timestamps)
        for candidate in candidates:
            if len(merged) >= max_frames:
                break
            candidate = self._clamp_timestamp(candidate, duration)
            if self._is_far_enough(candidate, merged):
                merged.append(candidate)
        return merged

    def _is_far_enough(self, timestamp: float, timestamps: list[float]) -> bool:
        min_gap = max(CONFIG.video.keyframe_min_gap_seconds, 0.0)
        return all(abs(timestamp - existing) >= min_gap for existing in timestamps)

    def _clamp_timestamp(self, timestamp: float, duration: float) -> float:
        if duration <= 0.002:
            return max(duration / 2, 0.001)
        return min(max(timestamp, 0.001), duration - 0.001)

    def _rounded_sorted(self, timestamps: list[float]) -> list[float]:
        return sorted({round(timestamp, 3) for timestamp in timestamps})

    def _float_value(self, value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _mean_abs_diff(self, left: bytes, right: bytes) -> float:
        return sum(abs(a - b) for a, b in zip(left, right)) / min(len(left), len(right))

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
