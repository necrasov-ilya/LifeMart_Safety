# LifeMart_Safety/config/config.py
"""
Модуль конфигурации проекта «LifeMart Safety Bot».

▪️ Загружает переменные окружения из файла .env (в корне репозитория).
▪️ Собирает типизированный контейнер Settings, доступный как singleton `settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# ───────────────────────────────
#  Загрузка .env
# ───────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")           # .env должен лежать в корне проекта

# ───────────────────────────────
#  Типизированный контейнер
# ───────────────────────────────
@dataclass(frozen=True, slots=True)
class Settings:
    BOT_TOKEN: str
    MODERATOR_CHAT_ID: int
    WHITELIST_USER_IDS: List[int]
    
    MISTRAL_API_KEY: str | None
    EMBEDDING_MODE: str
    
    POLICY_MODE: str
    AUTO_DELETE_THRESHOLD: float
    AUTO_KICK_THRESHOLD: float
    NOTIFY_THRESHOLD: float
    
    KEYWORD_THRESHOLD: float
    TFIDF_THRESHOLD: float
    EMBEDDING_THRESHOLD: float
    
    RETRAIN_THRESHOLD: int
    ANNOUNCE_BLOCKS: bool
    
    LOG_LEVEL: str = "INFO"


# ───────────────────────────────
#  Парс вспомогательных полей
# ───────────────────────────────
def _parse_int_list(raw: str | None) -> List[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _str_to_bool(raw: str | None, *, default: bool = True) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ───────────────────────────────
#  Сборка Settings
# ───────────────────────────────
def _build_settings() -> Settings:
    try:
        bot_token = os.environ["BOT_TOKEN"]
        mod_chat_id = int(os.environ["MODERATOR_CHAT_ID"])
    except KeyError as miss:
        raise RuntimeError(f"Переменная {miss.args[0]} не задана в .env") from None

    whitelist = _parse_int_list(os.environ.get("WHITELIST_USER_IDS"))
    
    mistral_api_key = os.environ.get("MISTRAL_API_KEY")
    embedding_mode = os.environ.get("EMBEDDING_MODE", "api").lower()
    if embedding_mode not in {"api", "local", "disabled"}:
        raise ValueError("EMBEDDING_MODE должен быть api | local | disabled")
    
    policy_mode = os.environ.get("POLICY_MODE", "semi-auto").lower()
    if policy_mode not in {"manual", "semi-auto", "auto"}:
        raise ValueError("POLICY_MODE должен быть manual | semi-auto | auto")
    
    auto_delete_threshold = float(os.environ.get("AUTO_DELETE_THRESHOLD", "0.85"))
    auto_kick_threshold = float(os.environ.get("AUTO_KICK_THRESHOLD", "0.95"))
    notify_threshold = float(os.environ.get("NOTIFY_THRESHOLD", "0.5"))
    
    keyword_threshold = float(os.environ.get("KEYWORD_THRESHOLD", "0.7"))
    tfidf_threshold = float(os.environ.get("TFIDF_THRESHOLD", "0.6"))
    embedding_threshold = float(os.environ.get("EMBEDDING_THRESHOLD", "0.7"))
    
    retrain_thr = int(os.environ.get("RETRAIN_THRESHOLD", "100"))
    announce_blocks = _str_to_bool(os.environ.get("ANNOUNCE_BLOCKS"), default=True)
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    return Settings(
        BOT_TOKEN=bot_token,
        MODERATOR_CHAT_ID=mod_chat_id,
        WHITELIST_USER_IDS=whitelist,
        MISTRAL_API_KEY=mistral_api_key,
        EMBEDDING_MODE=embedding_mode,
        POLICY_MODE=policy_mode,
        AUTO_DELETE_THRESHOLD=auto_delete_threshold,
        AUTO_KICK_THRESHOLD=auto_kick_threshold,
        NOTIFY_THRESHOLD=notify_threshold,
        KEYWORD_THRESHOLD=keyword_threshold,
        TFIDF_THRESHOLD=tfidf_threshold,
        EMBEDDING_THRESHOLD=embedding_threshold,
        RETRAIN_THRESHOLD=retrain_thr,
        ANNOUNCE_BLOCKS=announce_blocks,
        LOG_LEVEL=log_level,
    )


# singleton
settings: Settings = _build_settings()

__all__ = ["settings", "Settings"]
