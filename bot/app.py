from __future__ import annotations

import asyncio
import signal
from typing import List

from telegram import BotCommand
from telegram.constants import ParseMode
from telegram.ext import Application, ApplicationBuilder, Defaults

from config.config import settings
from utils.logger import get_logger
from bot.handlers import register_handlers

LOGGER = get_logger(__name__)

BOT_COMMANDS: List[BotCommand] = [
    BotCommand("start", "Начать работу с ботом"),
    BotCommand("help", "Краткая справка"),
    BotCommand("status", "Статус модели (whitelist)"),
    BotCommand("retrain", "Ручное переобучение (whitelist)"),
]


def build_application() -> Application:
    LOGGER.info("Создание Telegram Application…")

    app = (
        ApplicationBuilder()
        .token(settings.BOT_TOKEN)
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .build()
    )

    if BOT_COMMANDS:
        app.bot.set_my_commands(BOT_COMMANDS)

    register_handlers(app)

    LOGGER.info("Application готово.")
    return app


def run_polling() -> None:
    LOGGER.info("▶️  Запуск бота (polling)…")
    application = build_application()

    loop = asyncio.get_event_loop()

    def _ask_exit(sig: int, _frame) -> None:
        LOGGER.info("Получен сигнал %s — остановка…", sig)
        loop.create_task(application.shutdown())
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _ask_exit)

    application.run_polling(
        stop_signals=None,
        allowed_updates=["message", "callback_query"],
    )
    LOGGER.info("Бот завершил работу.")


if __name__ == "__main__":
    run_polling()
