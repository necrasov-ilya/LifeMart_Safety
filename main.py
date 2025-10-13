from __future__ import annotations

import sys
from pathlib import Path

from utils.logger import get_logger

LOGGER = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
if (ROOT_DIR / "telegram-antispam-bot").exists():
    LOGGER.error("Запускайте:  python -m telegram-antispam-bot.main")
    sys.exit(1)

from config.config import settings

if not settings.BOT_TOKEN or not settings.MODERATOR_CHAT_ID:
    LOGGER.critical(
        "Не заданы BOT_TOKEN и/или MODERATOR_CHAT_ID в .env — прекращаю работу."
    )
    sys.exit(2)

from bot.app import run_polling

if __name__ == "__main__":
    try:
        run_polling()
    except KeyboardInterrupt:
        LOGGER.info("Получен KeyboardInterrupt — выход.")
