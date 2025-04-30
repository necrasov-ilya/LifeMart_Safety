# bot/handlers.py
from __future__ import annotations

import csv
import html
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

# (chat_id, msg_id)  ->  (text, author_full_name)
PENDING: Dict[Tuple[int, int], Tuple[str, str]] = {}

SPAM_POLICY: str = settings.SPAM_POLICY          # notify | delete | kick
ANNOUNCE_BLOCKS: bool = settings.ANNOUNCE_BLOCKS


# ─────────────────────────────  HELPERS
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
    """Публикует уведомление в исходном чате (если разрешено)."""
    if not ANNOUNCE_BLOCKS:
        return
    reason = "по решению модератора" if by_moderator else "автоматически"
    await context.bot.send_message(
        chat_id,
        f"🚫 Сообщение от <b>{html.escape(offender_name)}</b> удалено {reason}.",
        parse_mode=ParseMode.HTML,
    )


def _dataset_rows() -> int:
    try:
        with open(classifier.dataset_path, newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.reader(f)) - 1
    except FileNotFoundError:
        return 0


# ─────────────────────────────  MESSAGE HANDLER
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

    # rule-based эвристика + ML-классификация
    hot_words = ("заработ", "удалёнк", "pиши", "$", "работ", "spam")
    pred = 1 if any(w in text.lower() for w in hot_words) else classifier.predict(text)
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

    # сохраняем для последующей обработки
    PENDING[(msg.chat_id, msg.message_id)] = (text, msg.from_user.full_name)

    # ─ автоматические действия согласно политике
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

    if auto_deleted:
        await _announce_block(context, msg.chat_id, msg.from_user.full_name, by_moderator=False)


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

        await _announce_block(context, chat_id, offender, by_moderator=True)

        # добавляем в датасет только подтверждённый спам
        added = text and classifier.update_dataset(text, 1)
        info = "Спам заблокирован."
        if added:
            info += " Новый пример добавлен в датасет 🙂"
    else:  # ham
        info = "OK, это не спам."

    await q.edit_message_reply_markup(reply_markup=None)
    await q.edit_message_text(f"<i>{html.escape(info)}</i>", parse_mode=ParseMode.HTML)


# ─────────────────────────────  ADMIN COMMANDS
async def cmd_status(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    ds = Path(classifier.dataset_path)
    size_kb = ds.stat().st_size // 1024 if ds.exists() else 0
    await update.effective_message.reply_html(
        f"<b>Status</b>\n"
        f"Dataset: <code>{ds.name}</code> • <code>{size_kb} KiB</code> • "
        f"<code>{_dataset_rows()} samples</code>\n"
        f"Policy: <code>{SPAM_POLICY}</code>\n"
        f"Announce: <code>{'ON' if ANNOUNCE_BLOCKS else 'OFF'}</code>"
    )


async def cmd_retrain(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    await update.effective_message.reply_text("⏳ Переобучаю модель…")
    classifier.train()
    await update.effective_message.reply_text("✅ Модель переобучена.")


async def cmd_policy(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    args = update.message.text.split(maxsplit=1)
    global SPAM_POLICY
    if len(args) == 2 and args[1].lower() in {"notify", "delete", "kick"}:
        SPAM_POLICY = args[1].lower()
        await update.message.reply_text(f"✅ SPAM_POLICY = {SPAM_POLICY}")
    else:
        await update.message.reply_text(
            f"Текущий режим: {SPAM_POLICY}\nИспользование: /policy notify|delete|kick"
        )


async def cmd_announce(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    args = update.message.text.split(maxsplit=1)
    global ANNOUNCE_BLOCKS
    if len(args) == 2 and args[1].lower() in {"on", "off"}:
        ANNOUNCE_BLOCKS = args[1].lower() == "on"
        state = "ВКЛ" if ANNOUNCE_BLOCKS else "ВЫКЛ"
        await update.message.reply_text(f"✅ Уведомления: {state}")
    else:
        state = "ВКЛ" if ANNOUNCE_BLOCKS else "ВЫКЛ"
        await update.message.reply_text(
            f"Уведомления сейчас: {state}\nИспользование: /announce on|off"
        )


async def cmd_start(update: Update, _):
    await update.effective_message.reply_text(
        "Анти-спам бот активен. Подозрительные сообщения скрываются "
        "и отправляются модераторам."
    )


async def cmd_help(update: Update, _):
    await update.effective_message.reply_text(
        "SPAM_POLICY:\n"
        " • notify – ждать решения модератора\n"
        " • delete – сразу удалить сообщение\n"
        " • kick   – удалить и временно кикнуть автора\n\n"
        "Команды: /status, /retrain, /policy, /announce"
    )


# ─────────────────────────────  REGISTRATION
def register_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("retrain",  cmd_retrain))
    app.add_handler(CommandHandler("policy",   cmd_policy))
    app.add_handler(CommandHandler("announce", cmd_announce))

    app.add_handler(CallbackQueryHandler(on_callback, pattern="^(spam|ham):"))
    app.add_handler(MessageHandler(filters.TEXT | filters.CaptionRegex(".*"), on_message))
