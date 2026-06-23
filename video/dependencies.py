"""External video tool checks."""

from __future__ import annotations

import shutil

from video_qa.config import CONFIG


class VideoDependencyChecker:
    def validate(self) -> None:
        if shutil.which(CONFIG.video.ffprobe_command) is None:
            raise RuntimeError(CONFIG.video.missing_ffprobe_message)
        if shutil.which(CONFIG.video.ffmpeg_command) is None:
            raise RuntimeError(CONFIG.video.missing_ffmpeg_message)
