# telegram-antispam-bot/config/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")        # загружаем .env из корня

# ───────────────────────────────
#  типизированный контейнер
# ───────────────────────────────
@dataclass(frozen=True, slots=True)
class Settings:
    BOT_TOKEN: str
    MODERATOR_CHAT_ID: int
    WHITELIST_USER_IDS: List[int]
    RETRAIN_THRESHOLD: int
    LOG_LEVEL: str = "INFO"           # ← ОБЯЗАТЕЛЬНО

def _parse_int_list(raw: str | None) -> List[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

def _build_settings() -> Settings:
    try:
        bot_token = os.environ["BOT_TOKEN"]
        mod_chat_id = int(os.environ["MODERATOR_CHAT_ID"])
    except KeyError as miss:
        raise RuntimeError(f"Переменная {miss.args[0]} не задана в .env") from None

    whitelist = _parse_int_list(os.environ.get("WHITELIST_USER_IDS"))
    retrain_thr = int(os.environ.get("RETRAIN_THRESHOLD", "100"))
    log_level   = os.environ.get("LOG_LEVEL", "INFO").upper()

    return Settings(
        BOT_TOKEN=bot_token,
        MODERATOR_CHAT_ID=mod_chat_id,
        WHITELIST_USER_IDS=whitelist,
        RETRAIN_THRESHOLD=retrain_thr,
        LOG_LEVEL=log_level,
    )

settings: Settings = _build_settings()
__all__ = ["settings", "Settings"]
