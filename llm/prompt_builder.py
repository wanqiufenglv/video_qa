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
            "enable_anomaly_detection_tasks": CONFIG.pipeline.enable_anomaly_detection_tasks,
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
                self._task_mode_prompt(),
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
        if not CONFIG.pipeline.enable_anomaly_detection_tasks:
            return (
                "任务路由规则（异常检测任务开关已关闭）：\n"
                "1. 先生成 video_description，描述必须完整覆盖场景、主体、动作、空间关系和时序变化。\n"
                "2. 只选择 video_caption、video_vqa 与 temporal_reasoning 三类任务，并分别生成视频摘要、视频问答和时序推理问答。\n"
                "3. 不要对异常检测任务进行相关度评分，不要选择或生成 abnormal_event_detection、"
                "equipment_anomaly_detection、aircraft_structure_anomaly_detection。\n"
                "4. selected_task_types 必须固定为 [\"video_caption\", \"video_vqa\", \"temporal_reasoning\"]。\n"
                "5. 所有问题都必须参考 video_description 生成，提问角度要丰富、多样、具体。\n"
                "6. 所有答案不要出现时间范围，不要使用‘未发现异常事件’、‘未发现明显异常’、"
                "‘无明显异常’等否定异常模板。"
            )

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

    def _task_mode_prompt(self) -> str:
        if CONFIG.pipeline.enable_anomaly_detection_tasks:
            return CONFIG.prompt.task_prompt
        return (
            "请根据下面的视频片段信息生成一个 JSON 对象。必须先生成 video_description，"
            "然后只生成 video_caption、video_vqa 和 temporal_reasoning 三类任务的问题与答案。"
            "禁止生成异常事件识别、设备异常检测、飞机螺丝及结构件异常检测任务及其相关度评分。"
            "问题必须参考 video_description，角度丰富、多样、具体；答案不得出现时间范围。"
            "所有中间任务结果必须包含 video_id、segment_id、time_range、keyframes、evidence、confidence。"
            "回答必须是合法 JSON，不要输出 Markdown。"
        )

    def _task_prompts_text(self) -> str:
        lines = ["各任务类型提示词："]
        for task_type, prompt in CONFIG.prompt.task_prompts.items():
            if not CONFIG.pipeline.enable_anomaly_detection_tasks and task_type in self._detection_task_types():
                continue
            lines.append(f"- {task_type}: {prompt}")
        return "\n".join(lines)

    def _output_schema_text(self) -> str:
        if not CONFIG.pipeline.enable_anomaly_detection_tasks:
            return (
                "输出 JSON 结构要求（只允许三个基础任务）：\n"
                "{\n"
                '  "video_description": "先生成的完整视频描述",\n'
                '  "selected_task_types": ["video_caption", "video_vqa", "temporal_reasoning"],\n'
                '  "caption": {"question": "基于 video_description 生成的问题", "caption": "视频摘要", "evidence": "...", "confidence": 0.0},\n'
                '  "qa_pairs": [{"question": "视频问答问题", "answer": "回答", "evidence": "...", "confidence": 0.0}],\n'
                '  "temporal_reasoning_pairs": [{"question": "时序推理问题", "answer": "回答", "evidence": "...", "confidence": 0.0}]\n'
                "}\n"
                "不要输出 task_relevance 或任何异常检测任务字段。"
            )
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

    def _detection_task_types(self) -> tuple[str, str, str]:
        return (
            CONFIG.tasks.abnormal_event_detection,
            CONFIG.tasks.equipment_anomaly_detection,
            CONFIG.tasks.aircraft_structure_anomaly_detection,
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
