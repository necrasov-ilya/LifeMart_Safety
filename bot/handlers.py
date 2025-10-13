from __future__ import annotations

import html
import time
from pathlib import Path

from telegram import Message, Update
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
from core.coordinator import FilterCoordinator
from core.types import Action, AnalysisResult
from filters.keyword import KeywordFilter
from filters.tfidf import TfidfFilter
from filters.embedding import EmbeddingFilter
from services.policy import PolicyEngine
from services.dataset import DatasetManager
from bot.keyboards import moderator_keyboard, format_moderator_card
from utils.logger import get_logger

LOGGER = get_logger(__name__)

keyword_filter = KeywordFilter()
tfidf_filter = TfidfFilter()
embedding_filter = EmbeddingFilter(
    mode=settings.EMBEDDING_MODE,
    api_key=settings.MISTRAL_API_KEY
)

coordinator = FilterCoordinator(
    keyword_filter=keyword_filter,
    tfidf_filter=tfidf_filter,
    embedding_filter=embedding_filter
)

policy_engine = PolicyEngine(
    mode=settings.POLICY_MODE,
    auto_delete_threshold=settings.AUTO_DELETE_THRESHOLD,
    auto_kick_threshold=settings.AUTO_KICK_THRESHOLD,
    notify_threshold=settings.NOTIFY_THRESHOLD
)

dataset_manager = DatasetManager(
    Path(__file__).resolve().parents[1] / "data" / "messages.csv"
)

PENDING: dict[tuple[int, int], tuple[str, str, int, AnalysisResult]] = {}


def is_whitelisted(uid: int) -> bool:
    return uid in settings.WHITELIST_USER_IDS


def is_explicit_command(update: Update) -> bool:
    msg = update.effective_message
    if not msg:
        return False

    if msg.chat_id == settings.MODERATOR_CHAT_ID:
        return True

    entities = msg.entities or []
    for ent in entities:
        if ent.type == "bot_command":
            command_text = msg.text[ent.offset : ent.offset + ent.length]
            if "@" in command_text:
                return True
    return False


def get_message_link(msg: Message) -> str:
    if msg.chat.username:
        return f"https://t.me/{msg.chat.username}/{msg.message_id}"
    return f"https://t.me/c/{abs(msg.chat_id)}/{msg.message_id}"


async def announce_block(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    offender_name: str,
    by_moderator: bool,
) -> None:
    if not settings.ANNOUNCE_BLOCKS:
        return
    
    reason = "по решению модератора" if by_moderator else "автоматически"
    await context.bot.send_message(
        chat_id,
        f"🚫 Сообщение от <b>{html.escape(offender_name)}</b> удалено {reason}.",
        parse_mode=ParseMode.HTML,
    )


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg: Message = update.effective_message
    if not msg or not msg.from_user:
        return

    text = (msg.text or msg.caption or "").strip()
    if not text or msg.chat_id == settings.MODERATOR_CHAT_ID:
        return
    
    if is_whitelisted(msg.from_user.id):
        return
    
    analysis = await coordinator.analyze(text)
    action = policy_engine.decide_action(analysis)
    
    LOGGER.info(
        f"Message from {msg.from_user.full_name}: "
        f"avg={analysis.average_score:.2f}, action={action.value}"
    )
    
    if action == Action.APPROVE:
        return
    
    msg_link = get_message_link(msg)
    
    card = format_moderator_card(
        user_name=msg.from_user.full_name,
        text=text,
        msg_link=msg_link,
        analysis=analysis
    )
    
    await context.bot.send_message(
        settings.MODERATOR_CHAT_ID,
        card,
        reply_markup=moderator_keyboard(msg.chat_id, msg.message_id),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    
    PENDING[(msg.chat_id, msg.message_id)] = (
        text,
        msg.from_user.full_name,
        msg.from_user.id,
        analysis
    )
    
    if action in (Action.DELETE, Action.KICK):
        try:
            await msg.delete()
            if action == Action.KICK:
                await context.bot.ban_chat_member(msg.chat_id, msg.from_user.id, until_date=60)
                await context.bot.unban_chat_member(msg.chat_id, msg.from_user.id)
        except Exception as e:
            LOGGER.warning(f"Failed to delete/kick: {e}")
        
        await announce_block(context, msg.chat_id, msg.from_user.full_name, by_moderator=False)
        
        if settings.ANNOUNCE_BLOCKS:
            await context.bot.send_message(
                msg.from_user.id,
                "❌ Ваше сообщение было отправлено на модерацию.\n"
                "Если это ошибка, модератор скоро всё исправит.",
                parse_mode=ParseMode.HTML,
            )


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q:
        return
    await q.answer()

    action, payload = q.data.split(":", 1)
    chat_id, msg_id = map(int, payload.split(":", 1))

    stored = PENDING.pop((chat_id, msg_id), None)
    if stored:
        text, offender, offender_id, analysis = stored
    else:
        text, offender, offender_id, analysis = None, "Пользователь", None, None

    if action == "kick":
        try:
            await context.bot.delete_message(chat_id, msg_id)
        except Exception:
            pass

        try:
            if offender_id and isinstance(offender_id, int) and offender_id > 0:
                await context.bot.ban_chat_member(chat_id, offender_id)
                info = "⛔ Сообщение удалено, пользователь забанен."
            else:
                info = "⛔ Сообщение удалено. (Не удалось найти отправителя для бана.)"
        except Exception as e:
            info = f"Ошибка при бане пользователя: {e}"

        await announce_block(context, chat_id, offender, by_moderator=True)

        if text:
            dataset_manager.add_sample(text, 1)
            info += " Сохранено как СПАМ."

    elif action == "delete":
        try:
            await context.bot.delete_message(chat_id, msg_id)
            info = "❌ Сообщение удалено."
        except Exception as e:
            info = f"Ошибка при удалении: {e}"
        
        await announce_block(context, chat_id, offender, by_moderator=True)
        
        if text:
            dataset_manager.add_sample(text, 1)
            info += " Сохранено как СПАМ."

    elif action == "ham":
        if text:
            dataset_manager.add_sample(text, 0)
            info = "✅ Помечено как НЕ спам. Пример сохранён."
        else:
            info = "✅ Помечено как НЕ спам."

    else:
        info = "Неизвестное действие."

    await q.edit_message_reply_markup(reply_markup=None)
    await q.edit_message_text(f"<i>{html.escape(info)}</i>", parse_mode=ParseMode.HTML)
    
    if tfidf_filter.should_retrain(settings.RETRAIN_THRESHOLD):
        LOGGER.info("Retrain threshold reached, training TF-IDF model...")
        tfidf_filter.train()


async def cmd_status(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    
    dataset_size_kb = dataset_manager.get_size_kb()
    dataset_rows = dataset_manager.get_row_count()
    
    filters_status = []
    if keyword_filter.is_ready():
        filters_status.append("🔤 Keyword: ✅")
    else:
        filters_status.append("🔤 Keyword: ❌")
    
    if tfidf_filter.is_ready():
        filters_status.append("📈 TF-IDF: ✅")
    else:
        filters_status.append("📈 TF-IDF: ❌")
    
    if embedding_filter.is_ready():
        filters_status.append(f"🧠 Embedding: ✅ ({settings.EMBEDDING_MODE})")
    else:
        filters_status.append(f"🧠 Embedding: ❌ ({settings.EMBEDDING_MODE})")
    
    await update.effective_message.reply_html(
        "<b>📊 Статус антиспам-системы</b>\n\n"
        f"<b>📁 Датасет:</b> <code>messages.csv</code>\n"
        f"<b>📦 Размер:</b> <code>{dataset_size_kb} КиБ</code>\n"
        f"<b>🔢 Записей:</b> <code>{dataset_rows}</code>\n\n"
        "<b>🛡️ Фильтры:</b>\n" + "\n".join(filters_status) + "\n\n"
        f"<b>🤖 Режим политики:</b> <code>{settings.POLICY_MODE}</code>\n"
        f"<b>📣 Объявления:</b> <code>{'ВКЛ' if settings.ANNOUNCE_BLOCKS else 'ВЫКЛ'}</code>"
    )


async def cmd_retrain(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    await update.effective_message.reply_text("⏳ Переобучаю TF-IDF модель…")
    tfidf_filter.train()
    time.sleep(3)
    await update.effective_message.reply_text("✅ Модель переобучена.")


async def cmd_start(update: Update, _):
    if not is_explicit_command(update):
        return

    user = update.effective_user
    if not user:
        return

    if is_whitelisted(user.id):
        text = (
            "✅ Анти-спам бот запущен.\n"
            "Вы модератор: подозрительные сообщения будут направляться вам для проверки."
        )
    else:
        text = (
            "👋 Привет! Я слежу за чистотой в чате и фильтрую подозрительные сообщения.\n"
            "Если ваше сообщение пропало — возможно, оно отправлено на модерацию."
        )

    await update.effective_message.reply_text(text)


async def cmd_help(update: Update, _):
    if not is_explicit_command(update):
        return

    user = update.effective_user
    if not user:
        return

    if is_whitelisted(user.id):
        await update.effective_message.reply_html(
            "📖 <b>Помощь по антиспам-боту</b>\n\n"
            "<b>🛡️ Режимы политики:</b>\n"
            " • <code>manual</code> — всё на модератора\n"
            " • <code>semi-auto</code> — авто при высоких оценках\n"
            " • <code>auto</code> — полная автоматизация\n\n"
            "<b>⚙️ Команды:</b>\n"
            " • <b>/status</b> — статус системы\n"
            " • <b>/retrain</b> — переобучить TF-IDF модель\n"
            " • <b>/help</b> — эта справка"
        )
    else:
        await update.effective_message.reply_html(
            "👋 <b>Привет!</b>\n"
            "Я — бот-модератор <b>LifeMart</b>.\n"
            "Помогаю сохранять порядок и фильтрую спам.\n\n"
            "Если ваше сообщение исчезло — возможно, оно отправлено на проверку модератору."
        )


def register_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("retrain", cmd_retrain))

    app.add_handler(CallbackQueryHandler(on_callback, pattern="^(kick|delete|ham):"))
    app.add_handler(MessageHandler(filters.TEXT | filters.CaptionRegex(".*"), on_message))
