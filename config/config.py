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
    EMBEDDING_MODEL_ID: str | None
    OLLAMA_MODEL: str | None
    OLLAMA_BASE_URL: str | None
    
    POLICY_MODE: str
    
    # NEW: Пороги для мета-классификатора (заменяют старые AUTO_DELETE/KICK/NOTIFY)
    META_NOTIFY: float
    META_DELETE: float
    META_KICK: float
    
    # NEW: Понижающие множители
    META_DOWNWEIGHT_ANNOUNCEMENT: float
    META_DOWNWEIGHT_REPLY_TO_STAFF: float
    META_DOWNWEIGHT_WHITELIST: float
    META_DOWNWEIGHT_BRAND: float
    LEGACY_KEYWORD_THRESHOLD: float
    LEGACY_TFIDF_THRESHOLD: float
    
    # NEW: Настройки контекста и эмбеддингов
    EMBEDDING_TIMEOUT_MS: int
    EMBEDDING_ENABLE_USER: bool
    CONTEXT_HISTORY_N: int
    CONTEXT_MAX_TOKENS: int
    EMBEDDING_CACHE_TTL_MIN: int
    
    # NEW: Пути к артефактам
    CENTROIDS_PATH: str
    PROTOTYPES_PATH: str
    META_MODEL_PATH: str
    META_CALIBRATOR_PATH: str
    
    RETRAIN_THRESHOLD: int
    ANNOUNCE_BLOCKS: bool
    
    LOG_LEVEL: str = "INFO"
    DETAILED_DEBUG_INFO: bool = False


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
    embedding_mode = os.environ.get("EMBEDDING_MODE", "ollama").lower()
    if embedding_mode not in {"api", "ollama", "local", "disabled"}:
        raise ValueError("EMBEDDING_MODE должен быть api | ollama | local | disabled")
    
    default_model_id = "mistral-embed" if embedding_mode == "api" else "qllama/multilingual-e5-small"
    embedding_model_id = os.environ.get("EMBEDDING_MODEL_ID", default_model_id)
    ollama_model = os.environ.get("OLLAMA_MODEL", embedding_model_id)
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    raw_policy_mode = os.environ.get("POLICY_MODE", "semi-auto")
    policy_mode = raw_policy_mode.strip().lower().replace("_", "-")
    if policy_mode not in {"manual", "semi-auto", "auto", "legacy-manual"}:
        raise ValueError("POLICY_MODE должен быть manual | semi-auto | auto | legacy-manual")
    
    # NEW: Пороги мета-классификатора
    meta_notify = float(os.environ.get("META_NOTIFY", "0.65"))
    meta_delete = float(os.environ.get("META_DELETE", "0.85"))
    meta_kick = float(os.environ.get("META_KICK", "0.95"))
    
    # NEW: Понижающие множители
    meta_downweight_announcement = float(os.environ.get("META_DOWNWEIGHT_ANNOUNCEMENT", "0.85"))
    meta_downweight_reply_to_staff = float(os.environ.get("META_DOWNWEIGHT_REPLY_TO_STAFF", "0.90"))
    meta_downweight_whitelist = float(os.environ.get("META_DOWNWEIGHT_WHITELIST", "0.85"))
    meta_downweight_brand = float(os.environ.get("META_DOWNWEIGHT_BRAND", "0.70"))
    legacy_keyword_threshold = float(os.environ.get("LEGACY_KEYWORD_THRESHOLD", "0.60"))
    legacy_tfidf_threshold = float(os.environ.get("LEGACY_TFIDF_THRESHOLD", str(meta_notify)))
    
    # NEW: Настройки контекста и эмбеддингов
    embedding_timeout_ms = int(os.environ.get("EMBEDDING_TIMEOUT_MS", "800"))
    embedding_enable_user = _str_to_bool(os.environ.get("EMBEDDING_ENABLE_USER"), default=False)
    context_history_n = int(os.environ.get("CONTEXT_HISTORY_N", "4"))
    context_max_tokens = int(os.environ.get("CONTEXT_MAX_TOKENS", "512"))
    embedding_cache_ttl_min = int(os.environ.get("EMBEDDING_CACHE_TTL_MIN", "10"))
    
    # NEW: Пути к артефактам
    centroids_path = os.environ.get("CENTROIDS_PATH", "models/centroids.npz")
    prototypes_path = os.environ.get("PROTOTYPES_PATH", "models/prototypes.npz")
    meta_model_path = os.environ.get("META_MODEL_PATH", "models/meta_model.joblib")
    meta_calibrator_path = os.environ.get("META_CALIBRATOR_PATH", "models/meta_calibrator.joblib")
    
    retrain_thr = int(os.environ.get("RETRAIN_THRESHOLD", "100"))
    announce_blocks = _str_to_bool(os.environ.get("ANNOUNCE_BLOCKS"), default=False)
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    detailed_debug_info = _str_to_bool(os.environ.get("DETAILED_DEBUG_INFO"), default=False)

    return Settings(
        BOT_TOKEN=bot_token,
        MODERATOR_CHAT_ID=mod_chat_id,
        WHITELIST_USER_IDS=whitelist,
        MISTRAL_API_KEY=mistral_api_key,
        EMBEDDING_MODE=embedding_mode,
        EMBEDDING_MODEL_ID=embedding_model_id,
        OLLAMA_MODEL=ollama_model,
        OLLAMA_BASE_URL=ollama_base_url,
        POLICY_MODE=policy_mode,
        META_NOTIFY=meta_notify,
        META_DELETE=meta_delete,
        META_KICK=meta_kick,
        META_DOWNWEIGHT_ANNOUNCEMENT=meta_downweight_announcement,
        META_DOWNWEIGHT_REPLY_TO_STAFF=meta_downweight_reply_to_staff,
        META_DOWNWEIGHT_WHITELIST=meta_downweight_whitelist,
        META_DOWNWEIGHT_BRAND=meta_downweight_brand,
        LEGACY_KEYWORD_THRESHOLD=legacy_keyword_threshold,
        LEGACY_TFIDF_THRESHOLD=legacy_tfidf_threshold,
        EMBEDDING_TIMEOUT_MS=embedding_timeout_ms,
        EMBEDDING_ENABLE_USER=embedding_enable_user,
        CONTEXT_HISTORY_N=context_history_n,
        CONTEXT_MAX_TOKENS=context_max_tokens,
        EMBEDDING_CACHE_TTL_MIN=embedding_cache_ttl_min,
        CENTROIDS_PATH=centroids_path,
        PROTOTYPES_PATH=prototypes_path,
        META_MODEL_PATH=meta_model_path,
        META_CALIBRATOR_PATH=meta_calibrator_path,
        RETRAIN_THRESHOLD=retrain_thr,
        ANNOUNCE_BLOCKS=announce_blocks,
        LOG_LEVEL=log_level,
        DETAILED_DEBUG_INFO=detailed_debug_info,
    )


# singleton
settings: Settings = _build_settings()

__all__ = ["settings", "Settings"]
