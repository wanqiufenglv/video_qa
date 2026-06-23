"""Build model prompts from segment metadata and keyframe images."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from video_qa.config import CONFIG
from video_qa.models.schemas import KeyFrame, VideoSegment


class PromptBuilder:
    def __init__(self, prompt_file: Path | None = None) -> None:
        self.prompt_file = prompt_file or Path(__file__).resolve().parents[1] / CONFIG.prompt.prompt_file_name
        self.prompt_document = self._load_prompt_document()

    def build_messages(self, segment: VideoSegment) -> list[dict[str, Any]]:
        segment_payload = {
            CONFIG.fields.video_id: segment.video_id,
            CONFIG.fields.video_path: segment.video_path,
            CONFIG.fields.source_video_path: segment.source_video_path or segment.video_path,
            CONFIG.fields.clip_duration: segment.clip_duration,
            CONFIG.fields.clip_length_category: segment.clip_length_category,
            CONFIG.fields.clip_quality: segment.clip_quality,
            CONFIG.fields.segment_id: segment.segment_id,
            CONFIG.fields.time_range: segment.time_range,
            CONFIG.fields.keyframes: [frame.to_dict() for frame in segment.keyframes],
        }
        user_text = "\n\n".join(
            part
            for part in (
                self.prompt_document,
                CONFIG.prompt.task_prompt,
                self._routing_policy_text(),
                self._task_prompts_text(),
                self._output_schema_text(),
                "视频片段元信息与关键帧列表：",
                json.dumps(segment_payload, ensure_ascii=False),
            )
            if part
        )
        user_content = self._build_user_content(user_text, segment.keyframes)
        return [
            {CONFIG.fields.role: CONFIG.fields.system_role, CONFIG.fields.content: CONFIG.prompt.system_prompt},
            {CONFIG.fields.role: CONFIG.fields.user_role, CONFIG.fields.content: user_content},
        ]

    def _load_prompt_document(self) -> str:
        if not self.prompt_file.exists():
            return ""
        return self.prompt_file.read_text(encoding="utf-8").strip()

    def _routing_policy_text(self) -> str:
        detection_tasks = ", ".join(
            (
                CONFIG.tasks.abnormal_event_detection,
                CONFIG.tasks.equipment_anomaly_detection,
                CONFIG.tasks.aircraft_structure_anomaly_detection,
            )
        )
        return (
            "任务路由规则：\n"
            "1. 先生成 video_description，描述必须完整覆盖场景、主体、动作、空间关系、时序变化和可见异常线索。\n"
            f"2. 只基于 video_description 给三类检测任务打分：{detection_tasks}。"
            f"分数写入 task_relevance，范围 0 到 1，阈值为 {CONFIG.pipeline.task_relevance_threshold}。\n"
            "3. 如果三类检测任务最高分达到阈值，只选择最高分对应的一类任务，并只为该任务生成问答。\n"
            "4. 如果三类检测任务均低于阈值，只选择 video_caption、video_vqa 与 temporal_reasoning 三类任务，分别生成短视频 Caption / 视频摘要、视频 VQA/时序问答和时序推理问答。\n"
            "5. 所有问题都必须参考 video_description 生成，提问角度要丰富、多样、具体。\n"
            "6. 所有答案不要出现时间范围，不要使用‘未发现异常事件’、‘未发现明显异常’、‘无明显异常’等否定异常模板。\n"
            "7. selected_task_types 必须与实际生成的任务字段一致，不要输出未选任务的不适用问答。"
        )

    def _task_prompts_text(self) -> str:
        lines = ["各任务类型提示词："]
        for task_type, prompt in CONFIG.prompt.task_prompts.items():
            lines.append(f"- {task_type}: {prompt}")
        return "\n".join(lines)

    def _output_schema_text(self) -> str:
        return (
            "输出 JSON 结构要求：\n"
            "{\n"
            '  "video_description": "先生成的完整视频描述",\n'
            '  "task_relevance": {"abnormal_event_detection": 0.0, "equipment_anomaly_detection": 0.0, "aircraft_structure_anomaly_detection": 0.0},\n'
            '  "selected_task_types": ["video_caption", "video_vqa", "temporal_reasoning"],\n'
            '  "caption": {"question": "基于 video_description 生成的丰富问题", "caption": "不含时间范围和否定异常模板的回答", "evidence": "...", "confidence": 0.0},\n'
            '  "qa_pairs": [{"question": "第1条视频 VQA/时序问答", "answer": "不含时间范围和否定异常模板的回答", "evidence": "...", "confidence": 0.0}, {"question": "第2条视频 VQA/时序问答", "answer": "不含时间范围和否定异常模板的回答", "evidence": "...", "confidence": 0.0}, {"question": "第3条视频 VQA/时序问答", "answer": "不含时间范围和否定异常模板的回答", "evidence": "...", "confidence": 0.0}],\n'
            '  "temporal_reasoning_pairs": [{"question": "第1条时序推理问题", "answer": "不含时间范围和否定异常模板的回答", "evidence": "...", "confidence": 0.0}, {"question": "第2条时序推理问题", "answer": "不含时间范围和否定异常模板的回答", "evidence": "...", "confidence": 0.0}, {"question": "第3条时序推理问题", "answer": "不含时间范围和否定异常模板的回答", "evidence": "...", "confidence": 0.0}],\n'
            '  "abnormal_event_detection": {"question": "基于 video_description 生成的丰富问题", "answer": "不含时间范围和否定异常模板的回答", "detected": true, "description": "...", "evidence": "...", "confidence": 0.0},\n'
            '  "equipment_anomaly_detection": {"question": "基于 video_description 生成的丰富问题", "answer": "不含时间范围和否定异常模板的回答", "detected": true, "description": "...", "evidence": "...", "confidence": 0.0},\n'
            '  "aircraft_structure_anomaly_detection": {"question": "基于 video_description 生成的丰富问题", "answer": "不含时间范围和否定异常模板的回答", "detected": true, "description": "...", "evidence": "...", "confidence": 0.0}\n'
            "}\n"
            "未被 selected_task_types 选中的任务字段可以省略。"
        )

    def _build_user_content(self, user_text: str, keyframes: list[KeyFrame]) -> str | list[dict[str, Any]]:
        if not CONFIG.video.include_keyframe_images_in_prompt:
            return user_text

        content: list[dict[str, Any]] = [{"type": CONFIG.video.text_type, "text": user_text}]
        for frame in keyframes:
            image_url = self._frame_data_url(frame)
            if image_url:
                content.append(
                    {
                        "type": CONFIG.video.image_url_type,
                        CONFIG.video.image_url_type: {
                            "url": image_url,
                            "detail": CONFIG.video.image_detail,
                        },
                    }
                )
        return content

    def _frame_data_url(self, frame: KeyFrame) -> str | None:
        frame_path = Path(frame.frame_path)
        if not frame_path.exists():
            return None
        encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        return f"data:{CONFIG.video.keyframe_image_mime_type};base64,{encoded}"
