"""Lightweight generated-data quality checks."""

from __future__ import annotations

from typing import Any

from video_qa.config import CONFIG


class QualityChecker:
    def is_valid(self, item: dict[str, Any]) -> bool:
        if self._is_sharegpt_sample(item):
            return self._valid_sharegpt(item)
        return self._has_trace(item)

    def _valid_sharegpt(self, item: dict[str, Any]) -> bool:
        videos = item.get(CONFIG.fields.videos)
        if not isinstance(videos, list) or not videos or not all(isinstance(video, str) and video for video in videos):
            return False
        conversations = item.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            return False
        human_turns = [turn for turn in conversations if isinstance(turn, dict) and turn.get("from") == "human"]
        gpt_turns = [turn for turn in conversations if isinstance(turn, dict) and turn.get("from") == "gpt"]
        if not human_turns or not gpt_turns:
            return False
        if not any(CONFIG.prompt.sharegpt_video_token in str(turn.get("value", "")) for turn in human_turns):
            return False
        if not all(str(turn.get("value", "")).strip() for turn in human_turns + gpt_turns):
            return False
        return self._confidence_ok(item)

    def _has_trace(self, item: dict[str, Any]) -> bool:
        metadata = item.get(CONFIG.fields.metadata) if isinstance(item.get(CONFIG.fields.metadata), dict) else {}
        required_keys = (
            CONFIG.fields.video_id,
            CONFIG.fields.segment_id,
            CONFIG.fields.time_range,
        )
        for key in required_keys:
            if key not in item and key not in metadata:
                return False
        time_range = item.get(CONFIG.fields.time_range, metadata.get(CONFIG.fields.time_range))
        return isinstance(time_range, list) and len(time_range) == 2

    def _confidence_ok(self, item: dict[str, Any]) -> bool:
        confidence = item.get(CONFIG.fields.confidence)
        metadata = item.get(CONFIG.fields.metadata)
        if confidence is None and isinstance(metadata, dict):
            confidence = metadata.get(CONFIG.fields.confidence)
        if confidence is None:
            return True
        score = self._confidence_score(confidence)
        if score is None:
            return True
        return score >= CONFIG.pipeline.minimum_quality_score

    def _confidence_score(self, confidence: Any) -> float | None:
        if isinstance(confidence, int | float):
            return float(confidence)
        if isinstance(confidence, str):
            normalized = confidence.strip().lower()
            mapping = {
                "low": 0.3,
                "medium": 0.65,
                "high": 0.9,
                "低": 0.3,
                "中": 0.65,
                "高": 0.9,
            }
            if normalized in mapping:
                return mapping[normalized]
            try:
                return float(normalized)
            except ValueError:
                return None
        return None

    def _is_sharegpt_sample(self, item: dict[str, Any]) -> bool:
        return "conversations" in item or CONFIG.fields.videos in item
