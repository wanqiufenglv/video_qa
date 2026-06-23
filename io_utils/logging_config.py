"""Runtime logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from video_qa.config import CONFIG


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(CONFIG.logging.logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt=CONFIG.logging.log_format,
        datefmt=CONFIG.logging.date_format,
    )

    log_path = output_dir / CONFIG.output.runtime_log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_path,
        encoding=CONFIG.logging.file_encoding,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
