"""Split video metadata into time-based segments."""

from __future__ import annotations

from video_qa.config import CONFIG
from video_qa.models.schemas import VideoMetadata, VideoSegment


class VideoSegmenter:
    def build_segments(self, metadata: VideoMetadata) -> list[VideoSegment]:
        segment_length = self._segment_length(metadata.duration)
        segments = []
        start_time = 0.0
        index = 1

        while start_time < metadata.duration:
            end_time = min(start_time + segment_length, metadata.duration)
            segment_id = f"{metadata.video_id}_seg_{index:04d}"
            segments.append(
                VideoSegment(
                    video_id=metadata.video_id,
                    video_path=metadata.video_path,
                    segment_id=segment_id,
                    time_range=[round(start_time, 3), round(end_time, 3)],
                    keyframes=[],
                    source_video_path=metadata.video_path,
                )
            )
            start_time = end_time
            index += 1
        return segments

    def _segment_length(self, duration: float) -> float:
        if duration <= CONFIG.video.short_video_threshold_seconds:
            return max(duration, 0.001)
        return CONFIG.video.default_segment_seconds

