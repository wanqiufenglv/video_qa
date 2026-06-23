"""Atomic progress file writer."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class ProgressWriter:
    def __init__(self, file_path: Path, ensure_ascii: bool = False) -> None:
        self.file_path = file_path
        self.ensure_ascii = ensure_ascii
        self._lock = Lock()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, progress: dict[str, Any]) -> None:
        temp_path = self.file_path.with_suffix(f"{self.file_path.suffix}.tmp")
        content = json.dumps(progress, ensure_ascii=self.ensure_ascii, indent=2)
        with self._lock:
            temp_path.write_text(content, encoding="utf-8")
            temp_path.replace(self.file_path)
