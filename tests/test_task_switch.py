from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import video_qa.converters.format_converter as converter_module
import video_qa.io_utils.resume_manager as resume_module
import video_qa.llm.prompt_builder as prompt_builder_module
from video_qa.config import CONFIG
from video_qa.models.schemas import VideoSegment


class AnomalyTaskSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segment = VideoSegment(
            video_id="video",
            video_path="clip.mp4",
            source_video_path="source.mp4",
            segment_id="video_seg_0001",
            time_range=[0.0, 10.0],
            keyframes=[],
        )
        self.raw_result = {
            "selected_task_types": ["abnormal_event_detection"],
            "task_relevance": {"abnormal_event_detection": 0.99},
            "abnormal_event_detection": {
                "question": "画面中出现了什么情况？",
                "answer": "设备附近出现烟雾。",
                "detected": True,
                "confidence": 0.99,
            },
            "video_description": "设备附近出现烟雾。",
            "caption": {"caption": "设备附近出现烟雾。"},
            "qa_pairs": [{"question": "画面中有什么？", "answer": "画面中有一台设备。"}],
            "temporal_reasoning_pairs": [{"question": "画面如何变化？", "answer": "烟雾逐渐增多。"}],
        }

    def test_switch_is_disabled_by_default(self) -> None:
        self.assertFalse(CONFIG.pipeline.enable_anomaly_detection_tasks)

    def test_disabled_switch_forces_three_basic_tasks(self) -> None:
        task_config = replace(CONFIG, pipeline=replace(CONFIG.pipeline, enable_anomaly_detection_tasks=False))
        with patch.object(converter_module, "CONFIG", task_config):
            selected = converter_module.FormatConverter().selected_output_task_types(self.raw_result)

        self.assertEqual(selected, ["video_caption", "video_vqa", "temporal_reasoning"])

    def test_enabled_switch_allows_detection_routing(self) -> None:
        task_config = replace(CONFIG, pipeline=replace(CONFIG.pipeline, enable_anomaly_detection_tasks=True))
        with patch.object(converter_module, "CONFIG", task_config):
            selected = converter_module.FormatConverter().selected_output_task_types(self.raw_result)

        self.assertEqual(selected, ["abnormal_event_detection"])

    def test_task_run_id_changes_with_switch(self) -> None:
        disabled_config = replace(CONFIG, pipeline=replace(CONFIG.pipeline, enable_anomaly_detection_tasks=False))
        enabled_config = replace(CONFIG, pipeline=replace(CONFIG.pipeline, enable_anomaly_detection_tasks=True))
        with patch.object(resume_module, "CONFIG", disabled_config):
            disabled_id = resume_module.task_run_id(self.segment.segment_id)
        with patch.object(resume_module, "CONFIG", enabled_config):
            enabled_id = resume_module.task_run_id(self.segment.segment_id)

        self.assertNotEqual(disabled_id, enabled_id)
        self.assertTrue(disabled_id.endswith(CONFIG.pipeline.basic_task_mode))
        self.assertTrue(enabled_id.endswith(CONFIG.pipeline.anomaly_task_mode))

    def test_disabled_prompt_excludes_detection_schema_and_prompts(self) -> None:
        task_config = replace(CONFIG, pipeline=replace(CONFIG.pipeline, enable_anomaly_detection_tasks=False))
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_prompt = Path(temp_dir) / "prompt.md"
            empty_prompt.write_text("", encoding="utf-8")
            with patch.object(prompt_builder_module, "CONFIG", task_config):
                builder = prompt_builder_module.PromptBuilder(empty_prompt)
                routing = builder._routing_policy_text()
                task_prompts = builder._task_prompts_text()
                schema = builder._output_schema_text()

        self.assertIn("异常检测任务开关已关闭", routing)
        self.assertNotIn("abnormal_event_detection:", task_prompts)
        self.assertNotIn('"abnormal_event_detection": {', schema)
        self.assertNotIn('"task_relevance":', schema)


if __name__ == "__main__":
    unittest.main()
