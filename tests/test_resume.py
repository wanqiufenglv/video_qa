from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_qa.config import CONFIG
from video_qa.converters.format_converter import FormatConverter
from video_qa.io_utils.error_logger import ErrorLogger
from video_qa.io_utils.resume_manager import CheckpointManager
from video_qa.io_utils.stream_writer import JsonlStreamWriter
from video_qa.models.schemas import KeyFrame, VideoSegment
from video_qa.pipeline.generator import VideoQAGenerator
from video_qa.pipeline.video_worker import VideoProcessWorker
from video_qa.quality.quality_checker import QualityChecker


class CheckpointManagerTests(unittest.TestCase):
    def test_loads_latest_completed_stage_and_ignores_broken_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoints.jsonl"
            manager = CheckpointManager(checkpoint_path)
            manager.save("video_seg_0001", "clip_video", {"attempt": 1})
            with checkpoint_path.open("a", encoding="utf-8") as file_obj:
                file_obj.write("{broken\n")
                file_obj.write(
                    json.dumps(
                        {
                            "segment_id": "video_seg_0001",
                            "stage": "extract_frames",
                            "status": "failed",
                            "payload": {"attempt": 2},
                        }
                    )
                    + "\n"
                )
            manager.save("video_seg_0001", "clip_video", {"attempt": 3})

            states = manager.load_states()

            self.assertEqual(states["video_seg_0001"]["clip_video"], {"attempt": 3})
            self.assertNotIn("extract_frames", states["video_seg_0001"])


class JsonlStreamWriterTests(unittest.TestCase):
    def test_append_unique_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "items.jsonl"
            writer = JsonlStreamWriter(output_path)

            self.assertTrue(writer.append_unique({"segment_id": "seg-1", "value": 1}, "segment_id"))
            self.assertFalse(writer.append_unique({"segment_id": "seg-1", "value": 2}, "segment_id"))

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["value"], 1)


class OutputReconciliationTests(unittest.TestCase):
    def test_rebuilds_all_missing_outputs_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            generator = VideoQAGenerator.__new__(VideoQAGenerator)
            generator.output_dir = output_dir
            generator.checkpoint_manager = CheckpointManager(output_dir / CONFIG.output.checkpoints_file)
            generator.error_logger = _FailOnErrorLogger()
            generator._log_event = lambda *args, **kwargs: None

            raw_result = {
                "segment_id": "video_seg_0001",
                "task_mode": CONFIG.pipeline.basic_task_mode,
                "task_run_id": f"video_seg_0001::{CONFIG.pipeline.basic_task_mode}",
                "video_id": "video",
                "video_path": "clip.mp4",
                "source_video_path": "source.mp4",
                "time_range": [0.0, 10.0],
                "clip_duration": 10.0,
                "clip_length_category": "short",
                "clip_quality": {"reason": "ok"},
                "keyframes": [],
                "selected_task_types": ["video_caption"],
                "video_description": "画面展示一架飞机停放在机坪上。",
                "caption": {
                    "question": "这段视频展示了什么场景？",
                    "caption": "画面展示一架飞机停放在机坪上。",
                },
            }
            JsonlStreamWriter(output_dir / CONFIG.output.raw_results_file).append(raw_result)

            generator._rebuild_missing_output_records()
            generator._rebuild_missing_output_records()

            task_lines = (output_dir / CONFIG.output.captions_file).read_text(encoding="utf-8").splitlines()
            sharegpt_lines = (output_dir / CONFIG.output.sharegpt_caption_file).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(task_lines), 1)
            self.assertEqual(len(sharegpt_lines), 1)
            states = generator.checkpoint_manager.load_states()
            self.assertIn(CONFIG.pipeline.stage_write_outputs, states["video_seg_0001"])
            self.assertIn(CONFIG.pipeline.stage_segment, states["video_seg_0001"])


class WorkerResumeTests(unittest.TestCase):
    def test_resumes_clip_quality_and_frames_when_artifacts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            clip_path = output_dir / "clip.mp4"
            frame_path = output_dir / "frame.jpg"
            clip_path.write_bytes(b"clip")
            frame_path.write_bytes(b"frame")
            segment = VideoSegment(
                video_id="video",
                video_path=str(clip_path),
                source_video_path="source.mp4",
                segment_id="video_seg_0001",
                time_range=[0.0, 10.0],
                keyframes=[KeyFrame("frame-1", 5.0, str(frame_path))],
                clip_duration=10.0,
                clip_length_category="short",
                clip_quality={"reason": "ok"},
            )
            payload = {"segment": segment.to_dict()}
            state = {
                CONFIG.pipeline.stage_clip_video: payload,
                CONFIG.pipeline.stage_clip_quality: {**payload, "is_valid_clip": True},
                CONFIG.pipeline.stage_extract_frames: payload,
            }
            worker = _resume_worker(output_dir)
            original = VideoSegment(
                video_id="video",
                video_path="source.mp4",
                source_video_path="source.mp4",
                segment_id="video_seg_0001",
                time_range=[0.0, 10.0],
                keyframes=[],
            )

            self.assertTrue(worker._process_segment(original, state))
            self.assertEqual(worker.clip_extractor.calls, 0)
            self.assertEqual(worker.clip_quality_checker.calls, 0)
            self.assertEqual(worker.extractor.calls, 0)
            completed = (output_dir / CONFIG.output.raw_results_file).read_text(encoding="utf-8")
            self.assertIn("video_seg_0001", completed)

    def test_missing_frame_invalidates_frame_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            clip_path = output_dir / "clip.mp4"
            clip_path.write_bytes(b"clip")
            segment = VideoSegment(
                video_id="video",
                video_path=str(clip_path),
                source_video_path="source.mp4",
                segment_id="video_seg_0001",
                time_range=[0.0, 10.0],
                keyframes=[KeyFrame("frame-1", 5.0, str(output_dir / "missing.jpg"))],
            )
            worker = _resume_worker(output_dir)

            restored = worker._restore_segment(
                {CONFIG.pipeline.stage_extract_frames: {"segment": segment.to_dict()}},
                CONFIG.pipeline.stage_extract_frames,
                require_keyframes=True,
            )

            self.assertIsNone(restored)


class _FailOnErrorLogger:
    def log(self, stage: str, message: str, context: dict) -> None:
        raise AssertionError(f"Unexpected error log: {stage}: {message}: {context}")


class _FailIfCalled:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("A completed stage should not be repeated")

    def check(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("A completed stage should not be repeated")


class _PromptBuilder:
    def build_messages(self, segment: VideoSegment) -> list:
        return []


class _Client:
    def complete(self, messages: list) -> dict:
        return {"selected_task_types": []}


def _resume_worker(output_dir: Path) -> VideoProcessWorker:
    worker = VideoProcessWorker.__new__(VideoProcessWorker)
    worker.output_dir = output_dir
    worker.clip_extractor = _FailIfCalled()
    worker.clip_quality_checker = _FailIfCalled()
    worker.extractor = _FailIfCalled()
    worker.prompt_builder = _PromptBuilder()
    worker.client = _Client()
    worker.converter = FormatConverter()
    worker.quality_checker = QualityChecker()
    worker.process_log_writer = JsonlStreamWriter(output_dir / CONFIG.output.process_log_file)
    worker.error_logger = ErrorLogger(JsonlStreamWriter(output_dir / CONFIG.output.errors_file))
    worker.checkpoint_manager = CheckpointManager(output_dir / CONFIG.output.checkpoints_file)
    return worker


if __name__ == "__main__":
    unittest.main()
