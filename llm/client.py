"""Chat completions client for the configured large model APIs."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from video_qa.config import CONFIG, LLMEndpointConfig


class VLMClient:
    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        last_error: Exception | None = None
        for endpoint in self._endpoints():
            payload = self._payload(messages, endpoint)
            try:
                raw_text = self._post(payload, endpoint)
                return self._parse_response(raw_text)
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
                last_error = error
                continue
        raise RuntimeError(f"All configured model endpoints failed: {last_error}")

    def _endpoints(self) -> tuple[LLMEndpointConfig, ...]:
        if CONFIG.llm.endpoints:
            return CONFIG.llm.endpoints
        return (
            LLMEndpointConfig(
                name=CONFIG.llm.model,
                url=CONFIG.llm.api_url,
                model=CONFIG.llm.model,
                api_key=CONFIG.llm.api_key,
                stream=CONFIG.llm.stream,
                temperature=CONFIG.llm.temperature,
                max_tokens=CONFIG.llm.max_tokens,
            ),
        )

    def _payload(self, messages: list[dict[str, Any]], endpoint: LLMEndpointConfig) -> dict[str, Any]:
        return {
            "max_tokens": endpoint.max_tokens,
            "messages": messages,
            "model": endpoint.model,
            "stream": endpoint.stream,
            "temperature": endpoint.temperature,
        }

    def _post(self, payload: dict[str, Any], endpoint: LLMEndpointConfig) -> str:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {CONFIG.fields.content_type_header: CONFIG.fields.json_content_type}
        if endpoint.api_key:
            headers[CONFIG.fields.authorization_header] = f"{CONFIG.fields.bearer_prefix} {endpoint.api_key}"
        request = urllib.request.Request(
            endpoint.url,
            data=body,
            headers=headers,
            method=CONFIG.llm.http_method,
        )

        last_error: Exception | None = None
        for attempt in range(CONFIG.llm.retry_times):
            try:
                with urllib.request.urlopen(request, timeout=CONFIG.llm.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                if attempt + 1 < CONFIG.llm.retry_times:
                    time.sleep(CONFIG.llm.retry_interval_seconds)

        raise RuntimeError(f"{endpoint.name} request failed: {last_error}")

    def _parse_response(self, raw_text: str) -> dict[str, Any]:
        response = json.loads(raw_text)
        if self._looks_like_generation_result(response):
            return response

        content = response[CONFIG.fields.choices][0][CONFIG.fields.message][CONFIG.fields.content]
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise ValueError("Model response content is not a JSON string or object.")
        return self._loads_json_object(content)

    def _loads_json_object(self, content: str) -> dict[str, Any]:
        stripped = self._strip_json_fence(content)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(stripped[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Model response JSON must be an object.")
        return parsed

    def _looks_like_generation_result(self, response: dict[str, Any]) -> bool:
        generation_keys = {
            CONFIG.tasks.video_description,
            CONFIG.tasks.task_relevance,
            CONFIG.tasks.selected_task_types,
            CONFIG.tasks.caption,
            CONFIG.tasks.qa_pairs,
            CONFIG.tasks.temporal_reasoning_pairs,
            CONFIG.tasks.temporal_reasoning,
            CONFIG.tasks.abnormal_event_detection,
            CONFIG.tasks.equipment_anomaly_detection,
            CONFIG.tasks.aircraft_structure_anomaly_detection,
            CONFIG.tasks.sharegpt_samples,
        }
        return any(key in response for key in generation_keys)

    def _strip_json_fence(self, content: str) -> str:
        stripped = content.strip()
        if stripped.startswith(CONFIG.prompt.json_fence_prefix):
            stripped = stripped.removeprefix(CONFIG.prompt.json_fence_prefix).strip()
        elif stripped.startswith(CONFIG.prompt.fence_marker):
            stripped = stripped.removeprefix(CONFIG.prompt.fence_marker).strip()
        if stripped.endswith(CONFIG.prompt.fence_marker):
            stripped = stripped.removesuffix(CONFIG.prompt.fence_marker).strip()
        return stripped
