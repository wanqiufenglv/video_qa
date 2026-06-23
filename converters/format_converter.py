"""Convert raw model output into task records and task-specific ShareGPT files."""

from __future__ import annotations

import json
import re
from typing import Any

from video_qa.config import CONFIG
from video_qa.models.schemas import VideoSegment


class FormatConverter:
    def caption_record(self, raw_result: dict[str, Any], segment: VideoSegment) -> dict[str, Any] | None:
        if CONFIG.tasks.video_caption not in self.selected_output_task_types(raw_result):
            return None
        caption = raw_result.get(CONFIG.tasks.caption) or raw_result.get(CONFIG.tasks.video_description)
        if isinstance(caption, str):
            caption = {CONFIG.tasks.caption: caption}
        if not isinstance(caption, dict):
            return None
        return self._with_trace(caption, segment, CONFIG.tasks.video_caption)

    def vqa_records(self, raw_result: dict[str, Any], segment: VideoSegment) -> list[dict[str, Any]]:
        if CONFIG.tasks.video_vqa not in self.selected_output_task_types(raw_result):
            return []
        qa_pairs = raw_result.get(CONFIG.tasks.qa_pairs, [])
        if not isinstance(qa_pairs, list):
            return []
        records = []
        for index, item in enumerate(qa_pairs, start=1):
            if not isinstance(item, dict):
                continue
            records.append(self._with_trace(item, segment, CONFIG.tasks.video_vqa, index=index))
        return records

    def temporal_reasoning_records(self, raw_result: dict[str, Any], segment: VideoSegment) -> list[dict[str, Any]]:
        if CONFIG.tasks.temporal_reasoning not in self.selected_output_task_types(raw_result):
            return []
        qa_pairs = raw_result.get(CONFIG.tasks.temporal_reasoning_pairs, [])
        if not isinstance(qa_pairs, list):
            return []
        records = []
        for index, item in enumerate(qa_pairs, start=1):
            if not isinstance(item, dict):
                continue
            records.append(self._with_trace(item, segment, CONFIG.tasks.temporal_reasoning, index=index))
        return records

    def task_record(self, raw_result: dict[str, Any], segment: VideoSegment, key: str) -> dict[str, Any] | None:
        if key not in self.selected_output_task_types(raw_result):
            return None
        item = raw_result.get(key)
        if isinstance(item, list):
            if not item:
                return None
            item = {"items": item}
        if not isinstance(item, dict):
            return None
        return self._with_trace(item, segment, key)

    def sharegpt_records(self, raw_result: dict[str, Any], segment: VideoSegment) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for task_records in self.sharegpt_records_by_task(raw_result, segment).values():
            records.extend(task_records)
        return records

    def sharegpt_records_by_task(self, raw_result: dict[str, Any], segment: VideoSegment) -> dict[str, list[dict[str, Any]]]:
        records = {
            CONFIG.tasks.video_caption: [],
            CONFIG.tasks.video_vqa: [],
            CONFIG.tasks.temporal_reasoning: [],
            CONFIG.tasks.abnormal_event_detection: [],
            CONFIG.tasks.equipment_anomaly_detection: [],
            CONFIG.tasks.aircraft_structure_anomaly_detection: [],
        }
        selected_task_types = set(self.selected_output_task_types(raw_result))

        if CONFIG.tasks.video_caption in selected_task_types:
            caption_source = raw_result.get(CONFIG.tasks.caption) or raw_result.get(CONFIG.tasks.video_description)
            caption_text = self._caption_text(caption_source)
            if caption_text:
                records[CONFIG.tasks.video_caption].append(
                    self._build_sharegpt_sample(
                        segment=segment,
                        index=1,
                        task_type=CONFIG.tasks.video_caption,
                        question=self._caption_question(caption_source),
                        answer=self._caption_answer(caption_text, segment),
                        source=self._source_dict(caption_source, caption_text),
                    )
                )

        if CONFIG.tasks.video_vqa in selected_task_types:
            qa_pairs = raw_result.get(CONFIG.tasks.qa_pairs, [])
            if isinstance(qa_pairs, list):
                for qa in qa_pairs:
                    if not isinstance(qa, dict):
                        continue
                    question = self._simple_question(self._as_text(qa.get("question")))
                    answer = self._as_text(qa.get("answer"))
                    if not question or not answer:
                        continue
                    records[CONFIG.tasks.video_vqa].append(
                        self._build_sharegpt_sample(
                            segment=segment,
                            index=len(records[CONFIG.tasks.video_vqa]) + 1,
                            task_type=CONFIG.tasks.video_vqa,
                            question=question,
                            answer=answer,
                            source=qa,
                        )
                    )

        if CONFIG.tasks.temporal_reasoning in selected_task_types:
            qa_pairs = raw_result.get(CONFIG.tasks.temporal_reasoning_pairs, [])
            if isinstance(qa_pairs, list):
                for qa in qa_pairs:
                    if not isinstance(qa, dict):
                        continue
                    question = self._simple_question(self._as_text(qa.get("question")))
                    answer = self._as_text(qa.get("answer"))
                    if not question or not answer:
                        continue
                    records[CONFIG.tasks.temporal_reasoning].append(
                        self._build_sharegpt_sample(
                            segment=segment,
                            index=len(records[CONFIG.tasks.temporal_reasoning]) + 1,
                            task_type=CONFIG.tasks.temporal_reasoning,
                            question=question,
                            answer=answer,
                            source=qa,
                        )
                    )

        detection_questions = {
            CONFIG.tasks.abnormal_event_detection: "画面中有哪些值得关注的现象？",
            CONFIG.tasks.equipment_anomaly_detection: "画面里的设备有没有异常？",
            CONFIG.tasks.aircraft_structure_anomaly_detection: "飞机结构有没有异常？",
        }
        for task_type, question in detection_questions.items():
            if task_type in selected_task_types:
                self._append_detection_sample(
                    records=records,
                    raw_result=raw_result,
                    segment=segment,
                    task_type=task_type,
                    question=question,
                )

        self._merge_model_sharegpt_samples(records, raw_result, segment, selected_task_types)
        return records

    def selected_output_task_types(self, raw_result: dict[str, Any]) -> list[str]:
        detection_task = self._selected_detection_task(raw_result)
        if detection_task:
            return [detection_task]
        return [CONFIG.tasks.video_caption, CONFIG.tasks.video_vqa, CONFIG.tasks.temporal_reasoning]

    def _selected_detection_task(self, raw_result: dict[str, Any]) -> str:
        detection_tasks = self._detection_task_types()
        scores = self._task_relevance_scores(raw_result)
        if scores:
            best_task = max(detection_tasks, key=lambda task_type: scores.get(task_type, 0.0))
            if scores.get(best_task, 0.0) >= CONFIG.pipeline.task_relevance_threshold:
                return best_task
            return ""

        selected_task_types = self._selected_task_types(raw_result)
        candidates = [
            task_type
            for task_type in detection_tasks
            if task_type in selected_task_types and self._detection_item_relevant(raw_result.get(task_type))
        ]
        if not candidates:
            return ""
        return max(candidates, key=lambda task_type: self._item_confidence(raw_result.get(task_type)))

    def _detection_task_types(self) -> tuple[str, str, str]:
        return (
            CONFIG.tasks.abnormal_event_detection,
            CONFIG.tasks.equipment_anomaly_detection,
            CONFIG.tasks.aircraft_structure_anomaly_detection,
        )

    def _task_relevance_scores(self, raw_result: dict[str, Any]) -> dict[str, float]:
        score_source = (
            raw_result.get(CONFIG.tasks.task_relevance)
            or raw_result.get("relevance_scores")
            or raw_result.get("task_scores")
            or raw_result.get("routing_scores")
        )
        if not isinstance(score_source, dict):
            return {}

        scores: dict[str, float] = {}
        aliases = {
            CONFIG.tasks.abnormal_event_detection: ("abnormal_event", "abnormal", "异常事件识别"),
            CONFIG.tasks.equipment_anomaly_detection: ("equipment_anomaly", "equipment", "设备异常检测"),
            CONFIG.tasks.aircraft_structure_anomaly_detection: (
                "aircraft_structure_anomaly",
                "aircraft_structure",
                "飞机螺丝及结构件异常检测",
            ),
        }
        for task_type in self._detection_task_types():
            value = score_source.get(task_type)
            if value is None:
                for alias in aliases[task_type]:
                    if alias in score_source:
                        value = score_source[alias]
                        break
            score = self._confidence_score(value)
            if score is not None:
                scores[task_type] = score
        return scores

    def _selected_task_types(self, raw_result: dict[str, Any]) -> set[str]:
        selected = raw_result.get(CONFIG.tasks.selected_task_types)
        if not isinstance(selected, list):
            return set()
        return {str(task_type) for task_type in selected}

    def _detection_item_relevant(self, item: Any) -> bool:
        if isinstance(item, list):
            return bool(item)
        if not isinstance(item, dict):
            return False

        for key in ("is_relevant", "applicable", "detected", "has_abnormality", "has_anomaly"):
            if key not in item:
                continue
            value = item.get(key)
            if isinstance(value, bool):
                return value
            text = self._as_text(value).lower()
            if text in {"not_applicable", "not applicable", "n/a", "false", "no", "none", "不适用", "无法确认", "无", "否"}:
                return False
            if text in {"true", "yes", "是", "有", "相关", "适用"}:
                return True

        combined = json.dumps(item, ensure_ascii=False).lower()
        negative_markers = ("not_applicable", "不适用", "无法进行", "完全不涉及", "无关", "不足以确认")
        if any(marker in combined for marker in negative_markers):
            return False
        return bool(self._as_text(item.get("answer") or item.get("description") or item.get("summary") or item.get("details")))

    def _item_confidence(self, item: Any) -> float:
        if not isinstance(item, dict):
            return CONFIG.pipeline.default_confidence
        score = self._confidence_score(item.get(CONFIG.fields.confidence))
        if score is None:
            return CONFIG.pipeline.default_confidence
        return score

    def _confidence_score(self, value: Any) -> float | None:
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {
                "low": 0.3,
                "medium": 0.65,
                "high": 0.9,
                "低": 0.3,
                "中": 0.65,
                "高": 0.9,
            }
            if normalized in mapping:
                return mapping[normalized]
            try:
                return float(normalized)
            except ValueError:
                return None
        return None

    def _append_detection_sample(
        self,
        records: dict[str, list[dict[str, Any]]],
        raw_result: dict[str, Any],
        segment: VideoSegment,
        task_type: str,
        question: str,
    ) -> None:
        item = raw_result.get(task_type)
        if isinstance(item, list):
            if not item:
                return
            item = {"items": item}
        if not isinstance(item, dict):
            return
        answer = self._detection_answer(item, task_type)
        if not answer:
            return
        records[task_type].append(
            self._build_sharegpt_sample(
                segment=segment,
                index=len(records[task_type]) + 1,
                task_type=task_type,
                question=self._detection_question(item, question, task_type),
                answer=answer,
                source=item,
            )
        )

    def _merge_model_sharegpt_samples(
        self,
        records: dict[str, list[dict[str, Any]]],
        raw_result: dict[str, Any],
        segment: VideoSegment,
        selected_task_types: set[str],
    ) -> None:
        samples = raw_result.get(CONFIG.tasks.sharegpt_samples, [])
        if isinstance(samples, dict):
            iterable = []
            for task_type, task_samples in samples.items():
                if isinstance(task_samples, list):
                    iterable.extend((str(task_type), item) for item in task_samples)
        elif isinstance(samples, list):
            iterable = [("", item) for item in samples]
        else:
            return

        for hinted_task_type, item in iterable:
            if not isinstance(item, dict):
                continue
            task_type = self._sample_task_type(item, hinted_task_type)
            if task_type not in records or task_type not in selected_task_types:
                continue
            sample = self._normalize_train_sample(item, segment, task_type, len(records[task_type]) + 1)
            if sample:
                records[task_type].append(sample)

    def _with_trace(
        self,
        item: dict[str, Any],
        segment: VideoSegment,
        task_type: str,
        index: int | None = None,
    ) -> dict[str, Any]:
        normalized = dict(item)
        normalized.setdefault(CONFIG.fields.task_type, task_type)
        normalized.setdefault(CONFIG.fields.video_id, segment.video_id)
        normalized.setdefault(CONFIG.fields.video_path, segment.video_path)
        if segment.source_video_path:
            normalized.setdefault(CONFIG.fields.source_video_path, segment.source_video_path)
        normalized.setdefault(CONFIG.fields.clip_duration, segment.clip_duration)
        normalized.setdefault(CONFIG.fields.clip_length_category, segment.clip_length_category)
        normalized.setdefault(CONFIG.fields.clip_quality, segment.clip_quality)
        normalized.setdefault(CONFIG.fields.segment_id, segment.segment_id)
        normalized.setdefault(CONFIG.fields.time_range, segment.time_range)
        normalized.setdefault(CONFIG.fields.keyframes, [frame.to_dict() for frame in segment.keyframes])
        normalized.setdefault(CONFIG.fields.confidence, CONFIG.pipeline.default_confidence)
        if index is not None:
            normalized.setdefault(CONFIG.pipeline.sample_id_key, f"{segment.segment_id}_{task_type}_{index:04d}")
        else:
            normalized.setdefault(CONFIG.pipeline.sample_id_key, f"{segment.segment_id}_{task_type}")
        return normalized

    def _normalize_train_sample(
        self,
        item: dict[str, Any],
        segment: VideoSegment,
        task_type: str,
        index: int,
    ) -> dict[str, Any] | None:
        normalized = dict(item)
        conversations = normalized.get("conversations")
        if not self._valid_conversations(conversations):
            conversations = self._conversations_from_item(normalized)
        if not self._valid_conversations(conversations):
            return None

        return {
            "conversations": self._normalize_conversations(conversations),
            CONFIG.fields.videos: self._normalize_videos(normalized.get(CONFIG.fields.videos), segment),
        }

    def _build_sharegpt_sample(
        self,
        segment: VideoSegment,
        index: int,
        task_type: str,
        question: str,
        answer: str,
        source: dict[str, Any],
    ) -> dict[str, Any]:
        question = self._with_video_token(question)
        return {
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": self._clean_answer(answer)},
            ],
            CONFIG.fields.videos: [segment.video_path],
        }

    def _sample_task_type(self, item: dict[str, Any], hinted_task_type: str) -> str:
        metadata = item.get(CONFIG.fields.metadata) if isinstance(item.get(CONFIG.fields.metadata), dict) else {}
        task_type = item.get(CONFIG.fields.task_type) or metadata.get(CONFIG.fields.task_type) or hinted_task_type
        task_type = str(task_type)
        if task_type in CONFIG.output.sharegpt_task_files:
            return task_type

        text = json.dumps(item.get("conversations", item), ensure_ascii=False)
        if any(keyword in text for keyword in ("螺丝", "铆钉", "蒙皮", "机翼", "飞机结构", "结构件")):
            return CONFIG.tasks.aircraft_structure_anomaly_detection
        if any(keyword in text for keyword in ("设备", "机器", "泄漏", "故障", "运维", "巡检")):
            return CONFIG.tasks.equipment_anomaly_detection
        if any(keyword in text for keyword in ("异常", "烟雾", "火光", "爆炸", "聚集", "越界")):
            return CONFIG.tasks.abnormal_event_detection
        if any(keyword in text for keyword in ("推理", "因果", "先后", "前后", "变化过程", "状态转移", "动作顺序")):
            return CONFIG.tasks.temporal_reasoning
        if any(keyword in text for keyword in ("描述", "主要内容", "场景")):
            return CONFIG.tasks.video_caption
        return CONFIG.tasks.video_vqa

    def _conversations_from_item(self, item: dict[str, Any]) -> list[dict[str, str]] | None:
        question = self._as_text(item.get("question") or item.get("prompt"))
        answer = self._as_text(item.get("answer") or item.get("response"))
        if not question or not answer:
            return None
        return [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]

    def _normalize_conversations(self, conversations: Any) -> list[dict[str, str]]:
        normalized = []
        for turn in conversations:
            normalized_turn = {"from": str(turn.get("from", "")), "value": str(turn.get("value", ""))}
            if normalized_turn["from"] == "human":
                normalized_turn["value"] = self._with_video_token(self._simple_question(normalized_turn["value"]))
            elif normalized_turn["from"] == "gpt":
                normalized_turn["value"] = self._clean_answer(normalized_turn["value"])
            normalized.append(normalized_turn)
        return normalized

    def _valid_conversations(self, conversations: Any) -> bool:
        if not isinstance(conversations, list) or len(conversations) < 2:
            return False
        has_human = any(isinstance(turn, dict) and turn.get("from") == "human" and turn.get("value") for turn in conversations)
        has_gpt = any(isinstance(turn, dict) and turn.get("from") == "gpt" and turn.get("value") for turn in conversations)
        return has_human and has_gpt

    def _normalize_videos(self, videos: Any, segment: VideoSegment) -> list[str]:
        if isinstance(videos, list) and videos:
            return [str(video) for video in videos]
        return [segment.video_path]

    def _caption_text(self, caption: Any) -> str:
        if isinstance(caption, str):
            return caption.strip()
        if isinstance(caption, dict):
            return self._as_text(caption.get(CONFIG.tasks.caption) or caption.get("text") or caption.get("description"))
        return ""

    def _caption_question(self, caption_source: Any) -> str:
        if isinstance(caption_source, dict):
            question = self._as_text(caption_source.get("question") or caption_source.get("prompt"))
            if question:
                return self._simple_question(question)
        return "请结合画面内容，概括这段视频呈现了哪些主要信息？"

    def _caption_answer(self, caption_text: str, segment: VideoSegment) -> str:
        return self._clean_answer(caption_text)

    def _detection_question(self, item: dict[str, Any], fallback: str, task_type: str) -> str:
        question = self._simple_question(self._as_text(item.get("question") or item.get("prompt") or fallback))
        if task_type != CONFIG.tasks.abnormal_event_detection:
            return question
        if any(phrase in question for phrase in ("异常", "异常事件", "异常情况", "有没有异常", "是否异常")):
            return "画面中有哪些值得关注的现象？"
        return question

    def _detection_answer(self, item: dict[str, Any], task_type: str) -> str:
        detected = item.get("detected")
        prefix = ""
        if task_type != CONFIG.tasks.abnormal_event_detection and isinstance(detected, bool) and detected:
            prefix = "有异常。"
        text = self._as_text(item.get("answer") or item.get("description") or item.get("details") or item.get("summary"))
        if text:
            answer = f"{prefix}{text}" if prefix and not text.startswith(prefix) else text
            return self._clean_detection_answer(answer, task_type)
        return self._clean_detection_answer(json.dumps(item, ensure_ascii=False), task_type)

    def _clean_detection_answer(self, answer: str, task_type: str) -> str:
        cleaned = self._clean_answer(answer)
        if task_type != CONFIG.tasks.abnormal_event_detection:
            return cleaned
        replacements = {
            "有异常。": "",
            "有异常，": "",
            "有异常": "",
            "存在异常。": "",
            "存在异常，": "",
            "存在异常": "",
            "异常事件为": "画面中出现",
            "异常情况位于": "相关现象位于",
            "异常类别": "现象类别",
            "异常事件": "画面事件",
            "异常情况": "画面情况",
            "异常现象": "可见现象",
            "异常线索": "可见线索",
            "异常判断": "画面判断",
            "异常移动": "不寻常移动",
        }
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        while "。。" in cleaned:
            cleaned = cleaned.replace("。。", "。")
        return cleaned.strip(" ，,。") or answer.strip()

    def _clean_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        removals = (
            "未发现异常事件",
            "未发现明显异常事件",
            "未发现明显异常",
            "没有发现明显异常",
            "无明显异常事件",
            "无明显异常",
            "没有异常事件",
        )
        for phrase in removals:
            cleaned = cleaned.replace(phrase, "")
        cleaned = cleaned.replace("这段视频的时间范围是", "")
        cleaned = cleaned.replace("时间范围为", "")
        cleaned = cleaned.replace("时间范围：", "")
        cleaned = cleaned.replace("时间范围:", "")
        range_pattern = r"\d+(?:\.\d+)?\s*(?:秒)?\s*(?:到|至|-|—)\s*\d+(?:\.\d+)?\s*秒(?:之间|内)?"
        cleaned = re.sub(rf"根据\s*{range_pattern}的?", "根据", cleaned)
        cleaned = re.sub(range_pattern, "", cleaned)
        cleaned = re.sub(r"约?\d+(?:\.\d+)?\s*秒(?:处|时|左右)?", "", cleaned)
        cleaned = re.sub(r"\d+(?:\.\d+)?\s*(?:到|至|-|—)\s*[，,。]", "", cleaned)
        while "。。" in cleaned:
            cleaned = cleaned.replace("。。", "。")
        return cleaned.strip(" ，,。") or answer.strip()

    def _source_dict(self, value: Any, fallback_text: str) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {CONFIG.tasks.caption: fallback_text, CONFIG.fields.confidence: CONFIG.pipeline.minimum_quality_score}

    def _simple_question(self, question: str) -> str:
        question = question.replace(CONFIG.prompt.sharegpt_video_token, "").strip()
        question = question.replace("\n", " ").strip()
        if not question:
            return question
        if len(question) > 60 and "？" in question:
            question = question.split("？", 1)[0] + "？"
        if not question.endswith(("？", "?")):
            question = f"{question}？"
        return question

    def _with_video_token(self, value: str) -> str:
        value = value.strip()
        if CONFIG.prompt.sharegpt_video_token in value:
            return value
        return f"{CONFIG.prompt.sharegpt_video_token}{value}"

    def _keyframe_ids(self, segment: VideoSegment) -> list[str]:
        return [frame.frame_id for frame in segment.keyframes]

    def _as_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()
