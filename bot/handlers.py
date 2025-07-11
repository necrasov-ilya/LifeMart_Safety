# bot/handlers.py
from __future__ import annotations

import time
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
# Импортируем новые системы логирования и настроек
from storage.settings_manager import get_settings
from storage.message_logger import get_message_logger

# ─────────────────────────────  GLOBALS
LOGGER = get_logger(__name__)

classifier = SpamClassifier(retrain_threshold=settings.RETRAIN_THRESHOLD)

# (chat_id, msg_id)  ->  (text, author_full_name)
PENDING: Dict[Tuple[int, int], Tuple[str, str]] = {}

# Заменяем глобальные переменные на менеджер настроек
settings_manager = get_settings()
message_logger = get_message_logger()


# ─────────────────────────────  HELPERS
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

def _msg_link(msg: Message) -> str:

    if msg.chat.username:                                   # публичная группа
        return f"https://t.me/{msg.chat.username}/{msg.message_id}"
    return f"https://t.me/c/{abs(msg.chat_id)}/{msg.message_id}"



async def _announce_block(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    offender_name: str,
    by_moderator: bool,
) -> None:
    """Публикует уведомление в исходном чате (если разрешено)."""
    if not settings_manager.announce_blocks:
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
    hot_words = ("заработ", "удалёнк", "казино", "$", "работ", "spam")
    has_hot_words = any(w in text.lower() for w in hot_words)

    # Получаем вероятность спама от ML-модели
    spam_probability = classifier.predict_proba(text) if hasattr(classifier, 'predict_proba') else 0.5
    pred = 1 if has_hot_words else classifier.predict(text)

    # Определяем метку и действие
    predicted_label = "spam" if pred == 1 else "ham"
    action_taken = "none"

    if pred != 1:
        # Логируем как НЕ спам
        message_logger.log_message_processed(
            message=msg,
            spam_probability=spam_probability,
            predicted_label=predicted_label,
            action_taken=action_taken
        )
        return

    LOGGER.info("✋ SUSPECT %s…", text[:60])

    # Определяем действие согласно политике
    spam_policy = settings_manager.spam_policy
    if spam_policy == "delete":
        action_taken = "delete"
    elif spam_policy == "kick":
        action_taken = "kick"
    else:
        action_taken = "notify"

    # ─ карточка модератору
    if settings_manager.notification_enabled:
        link = _msg_link(msg)
        preview = html.escape(text[:150] + ("…" if len(text) > 150 else ""))
        card = (
            "<b>Подозрительное сообщение</b>\n"
            f"👤 <i>{html.escape(msg.from_user.full_name)}</i>\n"
            f"🔗 <a href='{link}'>Перейти</a>\n"
            f"📊 Вероятность спама: {spam_probability:.2%}\n\n{preview}"
        )
        await context.bot.send_message(
            settings.MODERATOR_CHAT_ID,
            card,
            reply_markup=kb_mod(msg.chat_id, msg.message_id),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

    PENDING[(msg.chat_id, msg.message_id)] = (text, msg.from_user.full_name, msg.from_user.id)

    # ─ автоматические действия согласно политике
    auto_deleted = False
    if spam_policy in {"delete", "kick"}:
        try:
            await msg.delete()
            auto_deleted = True
        except Exception:
            LOGGER.warning("Cannot delete %s", msg.message_id)

    if spam_policy == "kick":
        try:
            await context.bot.ban_chat_member(msg.chat_id, msg.from_user.id, until_date=60)
            await context.bot.unban_chat_member(msg.chat_id, msg.from_user.id)
        except Exception:
            LOGGER.warning("Cannot kick %s", msg.from_user.id)

    if auto_deleted:
        await _announce_block(context, msg.chat_id, msg.from_user.full_name, by_moderator=False)

    # Логируем обработанное сообщение
    message_logger.log_message_processed(
        message=msg,
        spam_probability=spam_probability,
        predicted_label=predicted_label,
        action_taken=action_taken
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
    if stored:
        text, offender, offender_id = stored
    else:
        text, offender, offender_id = None, "Пользователь", None

    # Логируем решение модератора
    moderator_user_id = q.from_user.id if q.from_user else None
    if moderator_user_id:
        message_logger.log_moderator_decision(
            chat_id=chat_id,
            message_id=msg_id,
            decision=action,  # "spam" или "ham"
            moderator_user_id=moderator_user_id
        )

    if action == "spam":
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

        await _announce_block(context, chat_id, offender, by_moderator=True)

        added = text and classifier.update_dataset(text, 1)
        if added:
            info += " Сообщение сохранено как пример СПАМА 🙂"

    elif action == "ham":
        added = text and classifier.update_dataset(text, 0)
        info = "Сообщение помечено как НЕ спам."
        if added:
            info += " Пример сохранён в датасет 🙂"

    else:
        info = "Неизвестное действие."

    await q.edit_message_reply_markup(reply_markup=None)
    await q.edit_message_text(f"<i>{html.escape(info)}</i>", parse_mode=ParseMode.HTML)

async def cmd_start(update: Update, _):
    await update.effective_message.reply_text("Привет! Я антиспам-бот. Напиши /help для получения справки.")

async def cmd_help(update: Update, _):
    if not is_explicit_command(update):
        return

    user = update.effective_user
    if not user:
        return

    if is_whitelisted(user.id):
        await update.effective_message.reply_html(
            "📖 <b>Помощь по антиспам-боту</b>\n\n"
            "<b>🛡️ SPAM_POLICY</b> — как бот реагирует на подозрительные сообщения:\n"
            " • <code>notify</code> — сообщение остаётся, модератор получает уведомление\n"
            " • <code>delete</code> — сообщение удаляется автоматически\n"
            " • <code>kick</code> — сообщение удаляется, автор временно исключается\n\n"
            "<b>⚙️ Основные команды:</b>\n"
            " • <b>/status</b> — показать параметры антиспама и статистику\n"
            " • <b>/retrain</b> — переобучить модель\n"
            " • <b>/policy [тип]</b> — изменить режим (например: <code>/policy delete</code>)\n"
            " • <b>/announce [on/off]</b> — включить/выключить уведомления о блокировках\n"
            " • <b>/notifications [on/off]</b> — включить/выключить уведомления модераторам\n\n"
            "<b>📊 Статистика и логи:</b>\n"
            " • <b>/stats</b> — подробная статистика точности модели\n"
            " • <b>/logs</b> — последние 10 записей логов\n"
            " • <b>/userinfo [user_id]</b> — статистика конкретного пользователя\n\n"
            "<i>💾 Все настройки сохраняются автоматически и восстанавливаются после перезапуска.</i>"
        )
    else:
        await update.effective_message.reply_html(
            "👋 <b>Привет!</b>\n"
            "Я — бот-модератор беседы одного из филиалов <b>Жизнь Март</b>.\n"
            "Помогаю сохранять порядок, фильтрую спам и работаю с модераторами.\n\n"
            "Если ваше сообщение исчезло — возможно, оно было отправлено на проверку."
        )

# ─────────────────────────────  ADMIN COMMANDS
async def cmd_status(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    # Получаем текущие настройки и статистику
    ds = Path(classifier.dataset_path)
    size_kb = ds.stat().st_size // 1024 if ds.exists() else 0

    # Статистика точности модели
    accuracy_stats = message_logger.get_model_accuracy_stats(days=7)

    await update.effective_message.reply_html(
        "<b>📊 Статус антиспам-системы</b>\n\n"
        
        f"<b>📁 Датасет:</b> <code>{ds.name}</code> — название используемого набора данных\n"
        f"<b>📦 Размер:</b> <code>{size_kb} КиБ</code> — объём загруженного датасета\n"
        f"<b>🔢 Кол-во записей:</b> <code>{_dataset_rows()} строк</code> — количество примеров (сообщений) для анализа\n\n"
        
        f"<b>🛡️ Политика блокировки:</b> <code>{settings_manager.spam_policy}</code> — активный метод определения спама\n"
        f"<b>📣 Объявления о блоках:</b> <code>{'ВКЛ' if settings_manager.announce_blocks else 'ВЫКЛ'}</code> — уведомлять ли чат о блокировках\n"
        f"<b>🔔 Уведомления модераторам:</b> <code>{'ВКЛ' if settings_manager.notification_enabled else 'ВЫКЛ'}</code>\n"
        f"<b>🎯 Порог автоудаления:</b> <code>{settings_manager.auto_delete_threshold:.1%}</code>\n\n"
        
        f"<b>📈 Точность модели (7 дней):</b>\n"
        f"• Проверено модераторами: <code>{accuracy_stats['total_reviewed']}</code>\n"
        f"• Точность: <code>{accuracy_stats['accuracy']:.1f}%</code>\n"
        f"• Ложные срабатывания: <code>{accuracy_stats['false_positives']}</code>\n"
        f"• Пропущенный спам: <code>{accuracy_stats['false_negatives']}</code>"
    )

async def cmd_retrain(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    await update.effective_message.reply_text("⏳ Переобучаю модель…")
    classifier.train()
    time.sleep(5)
    await update.effective_message.reply_text("✅ Модель переобучена.")

async def cmd_policy(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    args = update.message.text.split(maxsplit=1)

    if len(args) == 2 and args[1].lower() in {"notify", "delete", "kick"}:
        settings_manager.spam_policy = args[1].lower()
        await update.message.reply_text(f"✅ SPAM_POLICY = {settings_manager.spam_policy}")
    else:
        await update.message.reply_text(
            f"Текущий режим: {settings_manager.spam_policy}\nИспользование: /policy notify|delete|kick"
        )

async def cmd_announce(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    args = update.message.text.split(maxsplit=1)

    if len(args) == 2 and args[1].lower() in {"on", "off"}:
        settings_manager.announce_blocks = args[1].lower() == "on"
        state = "ВКЛ" if settings_manager.announce_blocks else "ВЫКЛ"
        await update.message.reply_text(f"✅ Уведомления: {state}")
    else:
        state = "ВКЛ" if settings_manager.announce_blocks else "ВЫКЛ"
        await update.message.reply_text(
            f"Уведомления сейчас: {state}\nИспользование: /announce on|off"
        )

async def cmd_notifications(update: Update, _):
    """Включить/выключить уведомления модераторам."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    args = update.message.text.split(maxsplit=1)

    if len(args) == 2 and args[1].lower() in {"on", "off"}:
        settings_manager.notification_enabled = args[1].lower() == "on"
        state = "ВКЛ" if settings_manager.notification_enabled else "ВЫКЛ"
        await update.message.reply_text(f"✅ Уведомления модераторам: {state}")
    else:
        state = "ВКЛ" if settings_manager.notification_enabled else "ВЫКЛ"
        await update.message.reply_text(
            f"Уведомления модераторам: {state}\nИспользование: /notifications on|off"
        )

async def cmd_stats(update: Update, _):
    """Показать детальную статистику."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    # Получаем статистику за разные периоды
    stats_7d = message_logger.get_model_accuracy_stats(days=7)
    stats_30d = message_logger.get_model_accuracy_stats(days=30)

    await update.effective_message.reply_html(
        "<b>📈 Подробная статистика</b>\n\n"
        
        "<b>За 7 дней:</b>\n"
        f"• Всего сообщений проверено: <code>{stats_7d['total_reviewed']}</code>\n"
        f"• Точность модели: <code>{stats_7d['accuracy']:.1f}%</code>\n"
        f"• Ложные срабатывания: <code>{stats_7d['false_positives']}</code>\n"
        f"• Пропущенный спам: <code>{stats_7d['false_negatives']}</code>\n\n"
        
        "<b>За 30 дней:</b>\n"
        f"• Всего сообщений проверено: <code>{stats_30d['total_reviewed']}</code>\n"
        f"• Точность модели: <code>{stats_30d['accuracy']:.1f}%</code>\n"
        f"• Ложные срабатывания: <code>{stats_30d['false_positives']}</code>\n"
        f"• Пропущенный спам: <code>{stats_30d['false_negatives']}</code>"
    )

async def cmd_user_info(update: Update, _):
    """Получить информацию о пользователе."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Использование: /userinfo <user_id>")
        return

    try:
        user_id = int(args[1])
        user_stats = message_logger.get_user_stats(user_id, days=30)

        await update.effective_message.reply_html(
            f"<b>👤 Статистика пользователя {user_id}</b>\n\n"
            f"• Всего сообщений: <code>{user_stats['total_messages']}</code>\n"
            f"• Помечено как спам: <code>{user_stats['spam_messages']}</code>\n"
            f"• Средняя вероятность спама: <code>{user_stats['avg_spam_probability']:.1%}</code>\n"
            f"• Последнее сообщение: <code>{user_stats['last_message'] or 'Нет данных'}</code>"
        )
    except ValueError:
        await update.message.reply_text("❌ Неверный ID пользователя")

async def cmd_logs(update: Update, _):
    """Показать последние логи."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    recent_logs = message_logger.get_recent_logs(limit=10)

    if not recent_logs:
        await update.message.reply_text("📝 Логи пусты")
        return

    log_text = "<b>📝 Последние 10 записей:</b>\n\n"

    for log in recent_logs:
        timestamp = log.get('timestamp', 'Неизвестно')[:19]  # обрезаем до секунд
        user_name = log.get('full_name', 'Неизвестно')
        text_preview = log.get('text', '')[:30] + ('...' if len(log.get('text', '')) > 30 else '')
        prediction = log.get('predicted_label', 'unknown')
        action = log.get('action_taken', 'none')
        probability = log.get('spam_probability', 0)

        log_text += (
            f"<code>{timestamp}</code>\n"
            f"👤 {html.escape(user_name)}\n"
            f"📝 {html.escape(text_preview)}\n"
            f"🎯 {prediction} ({probability:.1%}) → {action}\n\n"
        )

    await update.effective_message.reply_html(log_text)

# ─────────────────────────────  REGISTRATION
def register_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("retrain", cmd_retrain))
    app.add_handler(CommandHandler("policy", cmd_policy))
    app.add_handler(CommandHandler("announce", cmd_announce))
    app.add_handler(CommandHandler("notifications", cmd_notifications))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("userinfo", cmd_user_info))
    app.add_handler(CommandHandler("logs", cmd_logs))

    app.add_handler(CallbackQueryHandler(on_callback, pattern="^(spam|ham):"))
    app.add_handler(MessageHandler(filters.TEXT | filters.CaptionRegex(".*"), on_message))
