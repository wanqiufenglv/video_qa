"""Append-only JSONL writer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


class InterProcessFileLock:
    """Small cross-process lock based on a sidecar lock file."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_obj = None

    def __enter__(self) -> "InterProcessFileLock":
        self._file_obj = self.lock_path.open("a+b")
        if os.name == "nt":
            import msvcrt

            self._file_obj.seek(0)
            if not self._file_obj.read(1):
                self._file_obj.seek(0)
                self._file_obj.write(b"0")
                self._file_obj.flush()
            self._file_obj.seek(0)
            msvcrt.locking(self._file_obj.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self._file_obj.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._file_obj is None:
            return
        if os.name == "nt":
            import msvcrt

            self._file_obj.seek(0)
            msvcrt.locking(self._file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._file_obj.fileno(), fcntl.LOCK_UN)
        self._file_obj.close()
        self._file_obj = None


class JsonlStreamWriter:
    """Write JSON objects one line at a time."""

    _locks: dict[Path, Lock] = {}
    _locks_guard = Lock()

    def __init__(self, file_path: Path, ensure_ascii: bool = False) -> None:
        self.file_path = file_path
        self.ensure_ascii = ensure_ascii
        self._lock = self._get_lock(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, item: dict[str, Any], durable: bool = False) -> None:
        line = json.dumps(item, ensure_ascii=self.ensure_ascii)
        with self._lock:
            with InterProcessFileLock(self.file_path.with_suffix(f"{self.file_path.suffix}.lock")):
                with self.file_path.open("a", encoding="utf-8") as file_obj:
                    file_obj.write(line)
                    file_obj.write("\n")
                    if durable:
                        file_obj.flush()
                        os.fsync(file_obj.fileno())

    def append_unique(self, item: dict[str, Any], key_name: str, durable: bool = False) -> bool:
        """Append an item unless a valid record with the same key already exists."""
        key_value = item.get(key_name)
        if key_value is None:
            self.append(item, durable=durable)
            return True

        line = json.dumps(item, ensure_ascii=self.ensure_ascii)
        with self._lock:
            with InterProcessFileLock(self.file_path.with_suffix(f"{self.file_path.suffix}.lock")):
                if self.file_path.exists():
                    with self.file_path.open("r", encoding="utf-8") as file_obj:
                        for existing_line in file_obj:
                            try:
                                existing = json.loads(existing_line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(existing, dict) and existing.get(key_name) == key_value:
                                return False
                with self.file_path.open("a", encoding="utf-8") as file_obj:
                    file_obj.write(line)
                    file_obj.write("\n")
                    if durable:
                        file_obj.flush()
                        os.fsync(file_obj.fileno())
        return True

    @classmethod
    def _get_lock(cls, file_path: Path) -> Lock:
        resolved_path = file_path.resolve()
        with cls._locks_guard:
            if resolved_path not in cls._locks:
                cls._locks[resolved_path] = Lock()
            return cls._locks[resolved_path]
