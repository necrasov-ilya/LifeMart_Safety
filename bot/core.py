"""
telegram-antispam-bot/bot/core.py
────────────────────────────────────────────────────────
Инициализирует Telegram-бота:

• Создаёт Application с HTML-parse-mode по умолчанию.
• Регистрирует команды «/start, /help, /status, /retrain».
• Подключает все хэндлеры из bot.handlers.
• Экспортирует функцию run_polling() для старта из main.py
  либо локальный запуск:  python -m bot.core
"""

from __future__ import annotations

import asyncio
import signal
from typing import List

from telegram import BotCommand
from telegram.constants import ParseMode
from telegram.ext import Application, ApplicationBuilder, Defaults

from config.config import settings
from utils.logger import get_logger
from bot.handlers import register_handlers  # регистрирует message/command/callback

LOGGER = get_logger(__name__)

# ───────────────────────────────
# Команды, которые видны в меню Telegram-клиента
# ───────────────────────────────
BOT_COMMANDS: List[BotCommand] = [
    BotCommand("start", "Начать работу с ботом"),
    BotCommand("help", "Краткая справка"),
    BotCommand("status", "Статус модели (whitelist)"),
    BotCommand("retrain", "Ручное переобучение (whitelist)"),
]


# ───────────────────────────────
# Сборка Application
# ───────────────────────────────
def build_application() -> Application:
    """Создать полностью сконфигурированный объект Application."""
    LOGGER.info("Создание Telegram Application…")

    app = (
        ApplicationBuilder()
        .token(settings.BOT_TOKEN)
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .build()
    )

    # Меню-команды
    if BOT_COMMANDS:
        app.bot.set_my_commands(BOT_COMMANDS)

    # Подключаем все хэндлеры (message, callback, commands)
    register_handlers(app)

    LOGGER.info("Application готово.")
    return app


# ───────────────────────────────
# Запуск long-polling (вызывается из main.py)
# ───────────────────────────────
def run_polling() -> None:
    """Запустить бота в режиме polling с graceful-shutdown."""
    LOGGER.info("▶️  Запуск бота (polling)…")
    application = build_application()

    loop = asyncio.get_event_loop()

    def _ask_exit(sig: int, _frame) -> None:  # noqa: D401,N802
        LOGGER.info("Получен сигнал %s — остановка…", sig)
        loop.create_task(application.shutdown())
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _ask_exit)  # type: ignore[arg-type]

    application.run_polling(
        stop_signals=None,               # сигналы ловим сами
        allowed_updates=["message", "callback_query"],
    )
    LOGGER.info("Бот завершил работу.")


# ───────────────────────────────
# Локальный тест:  python -m bot.core
# ───────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    run_polling()
