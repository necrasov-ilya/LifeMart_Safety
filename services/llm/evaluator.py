from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx

from config.config import settings
from core.types import Action, AnalysisResult, LLMEvaluation
from services.llm.prompt_builder import build_prompt_messages
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class LLMEvaluator:
    """OpenRouter-backed final moderation stage."""

    _BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self) -> None:
        self._api_key = settings.OPENROUTER_API_KEY or ""
        self._enabled = bool(settings.LLM_EVAL_ENABLED and self._api_key)
        self._model = settings.LLM_EVAL_MODEL
        self._temperature = float(settings.LLM_EVAL_TEMPERATURE)
        self._min_confidence = float(settings.LLM_EVAL_MIN_CONFIDENCE)
        self._timeout = float(settings.LLM_EVAL_TIMEOUT_SEC)

        if not self._enabled:
            LOGGER.info("LLM evaluator disabled (flag off or API key missing)")
        else:
            LOGGER.info(
                "LLM evaluator ready: model=%s min_conf=%.2f timeout=%.1fs",
                self._model,
                self._min_confidence,
                self._timeout,
            )

    def is_enabled(self) -> bool:
        return self._enabled

    def should_accept(self, evaluation: Optional[LLMEvaluation]) -> bool:
        if not evaluation:
            return False
        return evaluation.confidence >= self._min_confidence

    async def evaluate(self, text: str, analysis: AnalysisResult) -> Optional[LLMEvaluation]:
        if not self._enabled:
            return None

        messages = build_prompt_messages(text, analysis)
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "LifeMart Safety Bot",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._BASE_URL, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.error("LLM evaluation HTTP error: %s", exc)
            return None

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            LOGGER.error("LLM evaluation: failed to decode JSON response (%s)", exc)
            return None

        evaluation = self._parse_response(data)
        if evaluation:
            LOGGER.info(
                "LLM evaluation -> action=%s confidence=%.2f",
                evaluation.action.value,
                evaluation.confidence,
            )
        return evaluation

    def _parse_response(self, payload: Dict[str, Any]) -> Optional[LLMEvaluation]:
        if not payload:
            LOGGER.error("LLM evaluation: empty response payload")
            return None

        choices = payload.get("choices") or []
        if not choices:
            LOGGER.error("LLM evaluation: no choices in payload")
            return None

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            LOGGER.error("LLM evaluation: empty content field")
            return None

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            LOGGER.error("LLM evaluation: response is not JSON (%s): %s", exc, content[:200])
            return None

        action = self._to_action(parsed.get("action"))
        if action is None:
            LOGGER.error("LLM evaluation: unknown action in response %s", parsed)
            return None

        confidence_raw = parsed.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        except (TypeError, ValueError):
            confidence = 0.0

        reasoning = str(parsed.get("reasoning") or "").strip()

        raw = {
            "parsed": parsed,
            "usage": payload.get("usage"),
            "id": payload.get("id"),
            "model": payload.get("model"),
        }

        return LLMEvaluation(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            raw=raw,
        )

    @staticmethod
    def _to_action(raw: Any) -> Optional[Action]:
        if not raw:
            return None
        value = str(raw).strip().lower()
        mapping = {
            "approve": Action.APPROVE,
            "notify": Action.NOTIFY,
            "delete": Action.DELETE,
            "kick": Action.KICK,
        }
        return mapping.get(value)
