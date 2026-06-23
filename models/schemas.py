"""Typed schemas used by the generation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    video_path: str
    duration: float
    fps: float
    resolution: str
    codec: str
    file_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KeyFrame:
    frame_id: str
    timestamp: float
    frame_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoSegment:
    video_id: str
    video_path: str
    segment_id: str
    time_range: list[float]
    keyframes: list[KeyFrame]
    source_video_path: str = ""
    clip_duration: float = 0.0
    clip_length_category: str = ""
    clip_quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["keyframes"] = [frame.to_dict() for frame in self.keyframes]
        return data


def build_video_id(video_path: Path) -> str:
    return video_path.stem.replace(" ", "_")

