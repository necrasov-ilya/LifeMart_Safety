"""
utils/logger.py
────────────────────────────────────────────────────────
Централизованная конфигурация logging.

• Настраивает root-логгер ровно один раз.
• Выводит логи в консоль и во вращающийся файл logs/bot.log.
• Уровень берётся из settings.LOG_LEVEL (INFO по умолчанию).
• Экспортирует функцию `get_logger(name)` для получения
  именованных логгеров в других модулях.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.config import settings
from storage import init_storage

# Параметры форматирования
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = getattr(logging, settings.LOG_LEVEL, logging.INFO)
_ROOT_LOGGER_INITIALIZED = False

def _init_root_logger() -> None:
    """????????? root-?????? ?????? ???? ???."""
    global _ROOT_LOGGER_INITIALIZED
    if _ROOT_LOGGER_INITIALIZED:
        return

    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(LOG_LEVEL)

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(LOG_FMT, DATE_FMT))
        root.addHandler(console)

        file_handler = RotatingFileHandler(
            LOG_DIR / "bot.log",
            maxBytes=2_000_000,       # ~2 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(LOG_FMT, DATE_FMT))
        root.addHandler(file_handler)

        try:
            sqlite_handler = _SQLiteLogHandler()
            sqlite_handler.setLevel(logging.WARNING)
            root.addHandler(sqlite_handler)
        except Exception:
            # storage initialisation should not break logging on failure
            pass

    _ROOT_LOGGER_INITIALIZED = True



class _SQLiteLogHandler(logging.Handler):
    """Persist warnings/errors into SQLite for further analytics."""

    def __init__(self) -> None:
        super().__init__()
        self._storage = None

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING:
            return

        try:
            if self._storage is None:
                self._storage = init_storage()

            context: dict[str, str] | None = None
            if record.exc_info:
                context = {"exc_info": self.formatException(record.exc_info)}
            elif record.stack_info:
                context = {"stack": self.formatStack(record.stack_info)}

            self._storage.logs.write(
                level=record.levelname,
                logger=record.name,
                message=record.getMessage(),
                context=context,
            )
        except Exception:
            # Silently ignore storage errors to avoid recursion
            pass


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Возвратить именованный логгер (или root, если name не указан).
    Использование:
        logger = get_logger(__name__)
        logger.info("Hello!")
    """
    _init_root_logger()
    return logging.getLogger(name or "root")


__all__ = ["get_logger"]
