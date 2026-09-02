"""Resume support based on completed JSONL records."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from video_qa.config import CONFIG
from video_qa.io_utils.stream_writer import JsonlStreamWriter


def current_task_mode() -> str:
    if CONFIG.pipeline.enable_anomaly_detection_tasks:
        return CONFIG.pipeline.anomaly_task_mode
    return CONFIG.pipeline.basic_task_mode


def task_run_id(segment_id: str) -> str:
    return f"{segment_id}::{current_task_mode()}"


class ResumeManager:
    def __init__(self, file_path: Path, key_name: str) -> None:
        self.file_path = file_path
        self.key_name = key_name

    def load_completed_ids(self) -> set[str]:
        if not self.file_path.exists():
            return set()

        completed_ids: set[str] = set()
        with self.file_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = item.get(self.key_name)
                if value:
                    completed_ids.add(str(value))
        return completed_ids


class CheckpointManager:
    """Durable, append-only stage checkpoints for segment-level resume."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.writer = JsonlStreamWriter(file_path)

    def save(self, segment_id: str, stage: str, payload: dict[str, Any] | None = None) -> None:
        self.writer.append(
            {
                "version": CONFIG.pipeline.checkpoint_version,
                CONFIG.fields.segment_id: segment_id,
                "stage": stage,
                CONFIG.pipeline.checkpoint_status_key: CONFIG.pipeline.checkpoint_status_completed,
                CONFIG.pipeline.checkpoint_payload_key: payload or {},
                "created_at": datetime.now().strftime(CONFIG.prompt.generated_at_format),
            },
            durable=True,
        )

    def load_states(self) -> dict[str, dict[str, dict[str, Any]]]:
        states: dict[str, dict[str, dict[str, Any]]] = {}
        if not self.file_path.exists():
            return states

        with self.file_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                segment_id = item.get(CONFIG.fields.segment_id)
                stage = item.get("stage")
                status = item.get(CONFIG.pipeline.checkpoint_status_key)
                payload = item.get(CONFIG.pipeline.checkpoint_payload_key, {})
                if not segment_id or not stage or status != CONFIG.pipeline.checkpoint_status_completed:
                    continue
                states.setdefault(str(segment_id), {})[str(stage)] = payload if isinstance(payload, dict) else {}
        return states
