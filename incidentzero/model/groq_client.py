from __future__ import annotations

import json
import os
from typing import Any

from groq import Groq

from incidentzero.domain.models import ModelReply, ToolCall
from .errors import PermanentModelError, TransientModelError


class GroqModelClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.1,
        max_temperature: float = 0.9,
        temperature_step: float = 0.35,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_temperature = max_temperature
        self.temperature_step = temperature_step
        # Number of consecutive malformed generations. Used to widen sampling so that
        # a retry is a genuinely different sample rather than the same one again.
        self._generation_failures = 0
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def _sampling_temperature(self) -> float:
        """Temperature for the next request, widened after a malformed generation.

        Retrying a failed tool call at the original near-deterministic temperature
        tends to reproduce the same malformed output, which turns a bounded retry
        into three wasted requests. Each consecutive failure widens sampling so the
        retry explores a different decode.
        """
        return min(
            self.temperature + self.temperature_step * self._generation_failures,
            self.max_temperature,
        )

    #: Provider error codes that arrive as 4xx but describe a failed *generation*
    #: rather than a bad request. The request was accepted and the parameters were
    #: valid; the model simply emitted a malformed tool call, which a fresh sample
    #: usually fixes. Treating these as permanent would abort a run over a single
    #: unlucky decode, so they are classified as retryable.
    RETRYABLE_ERROR_CODES = {"tool_use_failed", "json_validate_failed"}
    RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

    def _translate_error(self, exc: Exception) -> Exception:
        status = getattr(exc, "status_code", None)
        if status in self.RETRYABLE_STATUS:
            return TransientModelError(str(exc))
        if any(code in str(exc) for code in self.RETRYABLE_ERROR_CODES):
            return TransientModelError(str(exc))
        return PermanentModelError(str(exc))

    def decide(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=self._sampling_temperature(),
                reasoning_effort="low",
            )
        except Exception as exc:  # provider-specific classes intentionally kept out of student code
            translated = self._translate_error(exc)
            if isinstance(translated, TransientModelError):
                self._generation_failures += 1
            raise translated from exc
        self._generation_failures = 0
        msg = response.choices[0].message
        calls: list[ToolCall] = []
        for call in (msg.tool_calls or []):
            try:
                args = json.loads(call.function.arguments)
            except Exception:
                args = {"__malformed_arguments__": call.function.arguments}
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=args))
        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
            }
        return ModelReply(
            content=msg.content,
            tool_calls=calls,
            usage=usage,
            finish_reason=response.choices[0].finish_reason,
        )

    def structured(self, messages: list[dict[str, Any]], schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
                temperature=self._sampling_temperature(),
                reasoning_effort="low",
            )
        except Exception as exc:
            translated = self._translate_error(exc)
            if isinstance(translated, TransientModelError):
                self._generation_failures += 1
            raise translated from exc
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise TransientModelError(f"Model returned invalid structured JSON: {content[:160]}") from exc
