from __future__ import annotations

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
    BotCommand("debug", "Детали сообщения по ID (whitelist)"),
    BotCommand("meta_info", "Инфо о мета-классификаторе (whitelist)"),  # NEW
]


def run_polling() -> None:
    LOGGER.info("▶️  Запуск бота (polling)…")
    
    app = (
        ApplicationBuilder()
        .token(settings.BOT_TOKEN)
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .build()
    )

    register_handlers(app)

    async def post_init(application: Application) -> None:
        """Выполняется после инициализации приложения"""
        if BOT_COMMANDS:
            await application.bot.set_my_commands(BOT_COMMANDS)
        LOGGER.info("Application готово.")

    app.post_init = post_init

    LOGGER.info("Запуск polling...")
    app.run_polling(
        allowed_updates=["message", "callback_query"],
    )
    LOGGER.info("Бот завершил работу.")


if __name__ == "__main__":
    run_polling()
