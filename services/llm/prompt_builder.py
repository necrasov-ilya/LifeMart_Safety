from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from core.types import AnalysisResult, FilterResult


def _sanitize(value: Any) -> Any:
    """Convert complex objects into JSON-friendly previews."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        if not value:
            return []

        if all(isinstance(item, (int, float)) for item in value):
            preview = [round(float(item), 4) for item in value[:8]]
            return {"length": len(value), "preview": preview}

        return [_sanitize(item) for item in value[:8]]

    if is_dataclass(value):
        return {k: _sanitize(v) for k, v in asdict(value).items()}

    return str(value)


def _serialize_filter(result: FilterResult | None) -> Dict[str, Any] | None:
    if result is None:
        return None

    payload: Dict[str, Any] = {
        "name": result.filter_name,
        "score": round(float(result.score), 4),
        "confidence": round(float(result.confidence), 4),
    }

    details = result.details or {}
    if details:
        sanitized = {
            key: _sanitize(val)
            for key, val in details.items()
            if not str(key).lower().startswith("embedding")
        }
        if sanitized:
            payload["details"] = sanitized

    return payload


def build_prompt_messages(text: str, analysis: AnalysisResult) -> list[dict[str, str]]:
    """Build OpenRouter chat messages that summarise all signals for the LLM."""
    metadata = asdict(analysis.metadata) if analysis.metadata else None
    filters = {
        "keyword": _serialize_filter(analysis.keyword_result),
        "tfidf": _serialize_filter(analysis.tfidf_result),
        "embedding": _serialize_filter(analysis.embedding_result),
    }

    meta_block: Dict[str, Any] = {}
    if analysis.meta_proba is not None:
        meta_block["probability"] = round(float(analysis.meta_proba), 4)
    if analysis.applied_downweights:
        meta_block["applied_downweights"] = list(analysis.applied_downweights)
    if analysis.meta_debug:
        meta_block["debug"] = _sanitize(analysis.meta_debug)

    context_block: Dict[str, Any] = {
        "message_text": text,
        "context_capsule": analysis.context_capsule,
        "user_capsule": analysis.user_capsule,
    }

    if analysis.embedding_vectors:
        context_block["embedding_vectors"] = {
            "has_msg": analysis.embedding_vectors.E_msg is not None,
            "has_ctx": analysis.embedding_vectors.E_ctx is not None,
            "has_user": analysis.embedding_vectors.E_user is not None,
        }

    payload = {
        "filters": filters,
        "meta": meta_block or None,
        "metadata": metadata,
        "context": context_block,
    }

    user_instructions = (
        "Ты получаешь результаты автоматических фильтров и краткий контекст переписки. "
        "Оцени, относится ли сообщение к легитимной акции/рассылке магазина LifeMart или несёт риски "
        "(спам, мошенничество, нарушение правил сообщества). "
        "Доступные решения: approve — пропустить, notify — отправить модераторам, "
        "delete — удалить без уведомления, kick — заблокировать автора. "
        "Учитывай агрессивность сообщения, признаки мошенничества, whitelist/admin-флаги, предыдущие нарушения "
        "и качество контекста (degraded_ctx). Верни строго JSON с ключами: action, confidence (0-1), "
        "reasoning (краткое объяснение на русском языке)."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — финальный модератор LifeMart, принимающий решение после автоматических фильтров. "
                "Опирайся на их сигналы, но можешь переопределить их, если контекст показывает низкий риск или, наоборот, срочную угрозу. "
                "Избегай ложных срабатываний на законные акции и объявления магазина."
            ),
        },
        {
            "role": "user",
            "content": f"{user_instructions}\n\nSIGNALS_JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}",
        },
    ]

    return messages
