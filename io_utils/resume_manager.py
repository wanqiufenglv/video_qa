"""Resume support based on completed JSONL records."""

from __future__ import annotations

import json
from pathlib import Path


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

