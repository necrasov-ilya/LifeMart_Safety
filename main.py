"""
main.py
────────────────────────────────────────────────────────
Единая точка входа приложения.

• Настраивает централизованный логгер (импортом utils.logger).
• Валидирует наличие критически важных переменных .env.
• Вызывает bot.core.run_polling(), который строит Application
  и запускает long-polling.

Запуск (из корня проекта):
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ───────────────────────────────
# 1) Инициализация логирования (достаточно импорта)
# ───────────────────────────────
from utils.logger import get_logger  # noqa: E402  (сначала импорт логгера)

LOGGER = get_logger(__name__)

# ───────────────────────────────
# 2) Проверка, что проект запущен из корня
# ───────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
if (ROOT_DIR / "telegram-antispam-bot").exists():  # запущен из родительского
    LOGGER.error("Запускайте:  python -m telegram-antispam-bot.main")
    sys.exit(1)

# ───────────────────────────────
# 3) Проверяем обязательные переменные .env
# ───────────────────────────────
from config.config import settings  # noqa: E402  (после проверки пути)

if not settings.BOT_TOKEN or not settings.MODERATOR_CHAT_ID:
    LOGGER.critical(
        "Не заданы BOT_TOKEN и/или MODERATOR_CHAT_ID в .env — прекращаю работу."
    )
    sys.exit(2)

# ───────────────────────────────
# 4) Старт Telegram-бота (polling)
# ───────────────────────────────
from bot.core import run_polling  # noqa: E402

if __name__ == "__main__":
    try:
        run_polling()
    except KeyboardInterrupt:
        LOGGER.info("Получен KeyboardInterrupt — выход.")
