"""
bot/handlers.py   (полный файл)

Анти-спам бот для чатов LifeMart
────────────────────────────────────────────────────────
Переменная .env  SPAM_POLICY  управляет автодействием:

  notify  – сообщение остаётся до решения модератора  (default)
  delete  – бот сразу удаляет подозрительное сообщение
  kick    – бот удаляет сообщение и временно кикает автора (60 сек)

Логика
──────
1.  Новое сообщение → rule-based + ML → «спам?»
2.  Если «спам» → бот отправляет карточку в мод-чат с кнопками
     🚫 Спам / ✅ Не спам
3.  В соответствии с SPAM_POLICY бот *сразу* (или не сразу) удаляет /
    кикает и публикует в основном чате уведомление
    «🚫 Иван Иванов заблокирован автоматически.»
4.  Модератор нажимает кнопку:
    • «🚫» → бот гарантированно удаляет сообщение, публикует
      «…заблокирован по решению модератора», дописывает пример
      в датасет
    • «✅» → сообщение остаётся, в датасет НИЧЕГО не пишется
"""

from __future__ import annotations

import csv
import html
import os
from pathlib import Path
from typing import Dict, Tuple

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config.config import settings
from ml.model import SpamClassifier
from utils.logger import get_logger

# ─────────────────────────────  GLOBALS
LOGGER = get_logger(__name__)

classifier = SpamClassifier(retrain_threshold=settings.RETRAIN_THRESHOLD)

# храним (chat_id, msg_id) -> (text, author_full_name)
PENDING: Dict[Tuple[int, int], Tuple[str, str]] = {}

SPAM_POLICY = os.getenv("SPAM_POLICY", "notify").lower()
assert SPAM_POLICY in {"notify", "delete", "kick"}, "SPAM_POLICY must be notify|delete|kick"


# ─────────────────────────────  Helper-утилиты
def is_whitelisted(uid: int) -> bool:
    return uid in settings.WHITELIST_USER_IDS


def kb_mod(chat_id: int, msg_id: int) -> InlineKeyboardMarkup:
    payload = f"{chat_id}:{msg_id}"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🚫 Спам", callback_data=f"spam:{payload}"),
                InlineKeyboardButton("✅ Не спам", callback_data=f"ham:{payload}"),
            ]
        ]
    )


async def _announce_block(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    offender_name: str,
    by_moderator: bool,
) -> None:
    reason = "по решению модератора" if by_moderator else "автоматически"
    await context.bot.send_message(
        chat_id,
        f"🚫 Сообщение от пользователя <b>{html.escape(offender_name)}</b> было удалено {reason}.",
        parse_mode=ParseMode.HTML,
    )


def _dataset_rows() -> int:
    try:
        with open(classifier.dataset_path, newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.reader(f)) - 1
    except FileNotFoundError:
        return 0


# ─────────────────────────────  NEW MESSAGE
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg: Message = update.effective_message
    if not msg.from_user:
        return

    text = (msg.text or msg.caption or "").strip()
    if (
        not text
        or msg.chat_id == settings.MODERATOR_CHAT_ID
        or is_whitelisted(msg.from_user.id)
    ):
        return

    hot = ("spam")
    pred = 1 if any(w in text.lower() for w in hot) else classifier.predict(text)
    if pred != 1:
        return

    LOGGER.info("✋ SUSPECT %s…", text[:60])

    # ─ карточка модератору
    link = f"https://t.me/c/{abs(msg.chat_id)}/{msg.message_id}"
    preview = html.escape(text[:150] + ("…" if len(text) > 150 else ""))
    card = (
        "<b>Подозрительное сообщение</b>\n"
        f"👤 <i>{html.escape(msg.from_user.full_name)}</i>\n"
        f"🔗 <a href='{link}'>Перейти</a>\n\n{preview}"
    )
    await context.bot.send_message(
        settings.MODERATOR_CHAT_ID,
        card,
        reply_markup=kb_mod(msg.chat_id, msg.message_id),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

    # сохраняем текст и имя для последующей обработки
    PENDING[(msg.chat_id, msg.message_id)] = (text, msg.from_user.full_name)

    # ─ автоматическое действие, если требуется
    auto_deleted = False
    if SPAM_POLICY in {"delete", "kick"}:
        try:
            await msg.delete()
            auto_deleted = True
        except Exception:
            LOGGER.warning("Cannot delete %s", msg.message_id)

    if SPAM_POLICY == "kick":
        try:
            await context.bot.ban_chat_member(msg.chat_id, msg.from_user.id, until_date=60)
            await context.bot.unban_chat_member(msg.chat_id, msg.from_user.id)
        except Exception:
            LOGGER.warning("Cannot kick %s", msg.from_user.id)

    if auto_deleted:  # публикуем уведомление
        await _announce_block(
            context,
            msg.chat_id,
            msg.from_user.full_name,
            by_moderator=False,
        )


# ─────────────────────────────  CALLBACK BUTTONS
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q:
        return
    await q.answer()

    action, payload = q.data.split(":", 1)
    chat_id, msg_id = map(int, payload.split(":", 1))

    stored = PENDING.pop((chat_id, msg_id), None)
    text = stored[0] if stored else None
    offender = stored[1] if stored else "Пользователь"

    if action == "spam":
        # гарантируем удаление
        try:
            await context.bot.delete_message(chat_id, msg_id)
        except Exception:
            pass

        # объявляем в чате
        await _announce_block(
            context,
            chat_id,
            offender,
            by_moderator=True,
        )

        # сохраняем в датасет только истинный спам
        added = text and classifier.update_dataset(text, 1)
        info = "Спам заблокирован."
        if added:
            info += " Новый пример добавлен в датасет 🙂"

    else:  # ham
        info = "OK, это не спам."

    await q.edit_message_reply_markup(reply_markup=None)
    await q.edit_message_text(f"<i>{html.escape(info)}</i>", parse_mode=ParseMode.HTML)


# ─────────────────────────────  ADMIN COMMANDS
async def cmd_status(update: Update, _c):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    ds = Path(classifier.dataset_path)
    size_kb = ds.stat().st_size // 1024 if ds.exists() else 0
    await update.effective_message.reply_html(
        f"<b>Status</b>\n"
        f"Dataset: <code>{ds.name}</code> • <code>{size_kb} KiB</code> • "
        f"<code>{_dataset_rows()} samples</code>\n"
        f"Policy: <code>{SPAM_POLICY}</code>"
    )


async def cmd_retrain(update: Update, _c):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    await update.effective_message.reply_text("⏳ Переобучаю модель…")
    classifier.train()
    await update.effective_message.reply_text("✅ Модель переобучена.")


async def cmd_start(update: Update, _c):
    await update.effective_message.reply_text(
        "Анти-спам бот активен. Подозрительные сообщения скрываются "
        "и отправляются модераторам."
    )


async def cmd_help(update: Update, _c):
    await update.effective_message.reply_text(
        "SPAM_POLICY:\n"
        " • notify – ждать решения модератора\n"
        " • delete – сразу удалить сообщение\n"
        " • kick   – удалить и временно кикнуть автора\n\n"
        "Команды для админов: /status, /retrain"
    )


# ─────────────────────────────  REGISTRATION
def register_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("retrain", cmd_retrain))
    app.add_handler(CallbackQueryHandler(on_callback, pattern="^(spam|ham):"))
    app.add_handler(MessageHandler(filters.TEXT | filters.CaptionRegex(".*"), on_message))
