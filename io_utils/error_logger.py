"""Structured error logging."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any

from video_qa.config import CONFIG
from video_qa.io_utils.stream_writer import JsonlStreamWriter


class ErrorLogger:
    def __init__(self, writer: JsonlStreamWriter, logger: logging.Logger | None = None) -> None:
        self.writer = writer
        self.logger = logger

    def log(self, stage: str, message: str, context: dict[str, Any]) -> None:
        if self.logger:
            self.logger.exception("%s failed: %s", stage, message)
        self.writer.append(
            {
                "stage": stage,
                "message": message,
                "context": context,
                "created_at": datetime.now().strftime(CONFIG.prompt.generated_at_format),
            }
        )
