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
    RETRAIN_THRESHOLD: int

    # динамически изменяемые во время работы параметры
    SPAM_POLICY: str          # notify | delete | kick
    ANNOUNCE_BLOCKS: bool     # публиковать ли уведомление в чате

    # сервисные опции
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
    # обязательные
    try:
        bot_token = os.environ["BOT_TOKEN"]
        mod_chat_id = int(os.environ["MODERATOR_CHAT_ID"])
    except KeyError as miss:
        raise RuntimeError(f"Переменная {miss.args[0]} не задана в .env") from None

    # необязательные / с умолчаниями
    whitelist = _parse_int_list(os.environ.get("WHITELIST_USER_IDS"))
    retrain_thr = int(os.environ.get("RETRAIN_THRESHOLD", "100"))

    # политика
    spam_policy = os.environ.get("SPAM_POLICY", "notify").lower()
    if spam_policy not in {"notify", "delete", "kick"}:
        raise ValueError("SPAM_POLICY должен быть notify | delete | kick")

    announce_blocks = _str_to_bool(os.environ.get("ANNOUNCE_BLOCKS"), default=True)

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    return Settings(
        BOT_TOKEN=bot_token,
        MODERATOR_CHAT_ID=mod_chat_id,
        WHITELIST_USER_IDS=whitelist,
        RETRAIN_THRESHOLD=retrain_thr,
        SPAM_POLICY=spam_policy,
        ANNOUNCE_BLOCKS=announce_blocks,
        LOG_LEVEL=log_level,
    )


# singleton
settings: Settings = _build_settings()

__all__ = ["settings", "Settings"]
