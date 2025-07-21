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
# Импортируем новый антимат фильтр
from utils.profanity_filter import profanity_filter

# ─────────────────────────────  GLOBALS
LOGGER = get_logger(__name__)

classifier = SpamClassifier(retrain_threshold=settings.RETRAIN_THRESHOLD)

PENDING: Dict[Tuple[int, int], Tuple[str, str, int]] = {}

PENDING_MODERATORS: Dict[int, Tuple[int, str]] = {}

settings_manager = get_settings()
message_logger = get_message_logger()


# ─────────────────────────────  HELPERS
def is_whitelisted(uid: int) -> bool:
    return settings_manager.is_moderator(uid)

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

def kb_admin_menu() -> InlineKeyboardMarkup:
    """Создает главное меню для модераторов."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 Статус", callback_data="menu_status"),
            InlineKeyboardButton("⚙️ Настройки", callback_data="menu_settings"),
        ],
        [
            InlineKeyboardButton("👥 Модераторы", callback_data="menu_moderators"),
            InlineKeyboardButton("🤬 Антимат", callback_data="menu_profanity"),
        ],
        [
            InlineKeyboardButton("📈 Статистика", callback_data="menu_stats"),
            InlineKeyboardButton("📝 Логи", callback_data="menu_logs"),
        ]
    ])

def kb_settings_menu() -> InlineKeyboardMarkup:
    """Создает меню настроек."""
    policy_text = f"Политика: {settings_manager.spam_policy}"
    announce_text = f"Объявления: {'ВКЛ' if settings_manager.announce_blocks else 'ВЫКЛ'}"
    notif_text = f"Уведомления: {'ВКЛ' if settings_manager.notification_enabled else 'ВЫКЛ'}"

    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🛡️ {policy_text}", callback_data="settings_policy")],
        [InlineKeyboardButton(f"📣 {announce_text}", callback_data="settings_announce")],
        [InlineKeyboardButton(f"🔔 {notif_text}", callback_data="settings_notifications")],
        [InlineKeyboardButton("🔧 Переобучить модель", callback_data="settings_retrain")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")]
    ])

def kb_moderators_menu() -> InlineKeyboardMarkup:
    """Создает меню управления модераторами."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Список модераторов", callback_data="mod_list")],
        [InlineKeyboardButton("➕ Добавить модератора", callback_data="mod_add_help")],
        [InlineKeyboardButton("➖ Удалить модератора", callback_data="mod_remove_help")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")]
    ])

def kb_profanity_menu() -> InlineKeyboardMarkup:
    """Создает меню управления антимат фильтром."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔍 Проверить текст", callback_data="prof_check_help")],
        [InlineKeyboardButton("➕ Добавить слово", callback_data="prof_add_help")],
        [InlineKeyboardButton("➖ Удалить слово", callback_data="prof_remove_help")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")]
    ])

def kb_word_not_found(word: str) -> InlineKeyboardMarkup:
    """Создает кнопки для слова, которое не найдено в фильтре."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"➕ Добавить '{word}' в фильтр", callback_data=f"add_word:{word}")],
        [InlineKeyboardButton("❌ Отмена", callback_data="cancel_action")]
    ])


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
        or settings_manager.is_moderator(msg.from_user.id)
    ):
        return

    # ─ ПРОВЕРКА АНТИМАТА
    if profanity_filter.contains_profanity(text):
        LOGGER.info("🤬 PROFANITY detected from %s: %s", msg.from_user.full_name, text[:30])

        # Удаляем сообщение с матом
        try:
            await msg.delete()
            auto_deleted = True
        except Exception:
            LOGGER.warning("Cannot delete profane message %s", msg.message_id)
            auto_deleted = False

        # Уведомляем модераторов о мате
        if settings_manager.notification_enabled:
            found_words = profanity_filter.get_profanity_words(text)
            link = _msg_link(msg)
            preview = html.escape(text[:100] + ("…" if len(text) > 100 else ""))

            card = (
                "<b>🤬 Обнаружен мат</b>\n"
                f"👤 <i>{html.escape(msg.from_user.full_name)}</i>\n"
                f"🔗 <a href='{link}'>Перейти</a>\n"
                f"🚫 Найдено слов: {', '.join(found_words[:3])}\n\n{preview}"
            )

            await context.bot.send_message(
                settings.MODERATOR_CHAT_ID,
                card,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )

        # Объявляем о блокировке
        if auto_deleted:
            await _announce_block(context, msg.chat_id, msg.from_user.full_name, by_moderator=False)

        # Логируем
        message_logger.log_message_processed(
            message=msg,
            spam_probability=1.0,  # Мат = 100% вероятность блокировки
            predicted_label="profanity",
            action_taken="delete"
        )
        return

    # ─ ПРОВЕРКА СПАМА (убираем порог автоудаления)
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

    # Определяем действие согласно политике (без порога автоудаления)
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

    # Обработка главного меню
    if q.data == "menu_main":
        await q.edit_message_text(
            "🏠 <b>Главное меню модератора</b>\n\nВыберите нужный раздел:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_admin_menu()
        )
        return

    # Обработка меню статуса
    if q.data == "menu_status":
        ds = Path(classifier.dataset_path)
        size_kb = ds.stat().st_size // 1024 if ds.exists() else 0
        accuracy_stats = message_logger.get_model_accuracy_stats(days=7)
        profanity_stats = message_logger.get_profanity_stats(days=7)

        await q.edit_message_text(
            f"<b>📊 Статус антиспам-системы</b>\n\n"
            f"<b>📁 Датасет:</b> <code>{ds.name}</code>\n"
            f"<b>📦 Размер:</b> <code>{size_kb} КиБ</code>\n"
            f"<b>🔢 Записей:</b> <code>{_dataset_rows()}</code>\n\n"
            f"<b>🛡️ Политика:</b> <code>{settings_manager.spam_policy}</code>\n"
            f"<b>📣 Объявления:</b> <code>{'ВКЛ' if settings_manager.announce_blocks else 'ВЫКЛ'}</code>\n"
            f"<b>🔔 Уведомления:</b> <code>{'ВКЛ' if settings_manager.notification_enabled else 'ВЫКЛ'}</code>\n\n"
            f"<b>📈 Точность (7 дней):</b> <code>{accuracy_stats['accuracy']:.1f}%</code>\n"
            f"<b>🤬 Найдено мата:</b> <code>{profanity_stats.get('total_profanity', 0)}</code>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")]])
        )
        return

    # Обработка меню настроек
    if q.data == "menu_settings":
        await q.edit_message_text(
            "⚙️ <b>Настройки системы</b>\n\nВыберите параметр для изменения:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_settings_menu()
        )
        return

    # Обработка настройки политики
    if q.data == "settings_policy":
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔔 notify", callback_data="policy_notify")],
            [InlineKeyboardButton("🗑️ delete", callback_data="policy_delete")],
            [InlineKeyboardButton("👢 kick", callback_data="policy_kick")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="menu_settings")]
        ])
        await q.edit_message_text(
            f"🛡️ <b>Политика обработки спама</b>\n\n"
            f"Текущий режим: <code>{settings_manager.spam_policy}</code>\n\n"
            f"• <b>notify</b> — уведомление модераторам\n"
            f"• <b>delete</b> — автоматическое удаление\n"
            f"• <b>kick</b> — удаление + временный бан",
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard
        )
        return

    # Обработка изменения политики
    if q.data.startswith("policy_"):
        policy = q.data.split("_", 1)[1]
        settings_manager.spam_policy = policy
        await q.edit_message_text(
            f"✅ Политика изменена на: <code>{policy}</code>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ К настройкам", callback_data="menu_settings")]])
        )
        return

    # Обработка переключения объявлений
    if q.data == "settings_announce":
        settings_manager.announce_blocks = not settings_manager.announce_blocks
        state = "включены" if settings_manager.announce_blocks else "выключены"
        await q.edit_message_text(
            f"✅ Объявления о блокировках {state}",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ К настройкам", callback_data="menu_settings")]])
        )
        return

    # Обработка переключения уведомлений
    if q.data == "settings_notifications":
        settings_manager.notification_enabled = not settings_manager.notification_enabled
        state = "включены" if settings_manager.notification_enabled else "выключены"
        await q.edit_message_text(
            f"✅ Уведомления модераторам {state}",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ К настройкам", callback_data="menu_settings")]])
        )
        return

    # Обработка переобучения модели
    if q.data == "settings_retrain":
        await q.edit_message_text("⏳ Переобучаю модель...")
        classifier.train()
        await q.edit_message_text(
            "✅ Модель переобучена!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ К настройкам", callback_data="menu_settings")]])
        )
        return

    # Обработка меню модераторов
    if q.data == "menu_moderators":
        await q.edit_message_text(
            "👥 <b>Управление модераторами</b>\n\nВыберите действие:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_moderators_menu()
        )
        return

    # Обработка списка модераторов
    if q.data == "mod_list":
        all_mods = settings_manager.get_all_moderators()
        additional_mods = settings_manager.get_additional_moderators()
        whitelist_mods = [uid for uid in all_mods if uid not in additional_mods]

        text = "<b>👥 Список модераторов</b>\n\n"
        if whitelist_mods:
            text += "<b>🔒 Основные:</b>\n"
            for mod_id in whitelist_mods[:5]:  # показываем только первые 5
                text += f"• <code>{mod_id}</code>\n"
            if len(whitelist_mods) > 5:
                text += f"... и ещё {len(whitelist_mods) - 5}\n"
            text += "\n"

        if additional_mods:
            text += "<b>➕ Дополнительные:</b>\n"
            for mod_id in additional_mods[:5]:
                text += f"• <code>{mod_id}</code>\n"
            if len(additional_mods) > 5:
                text += f"... и ещё {len(additional_mods) - 5}\n"
            text += "\n"

        text += f"<b>📊 Всего:</b> {len(all_mods)}"

        await q.edit_message_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="menu_moderators")]])
        )
        return

    # Обработка помощи по добавлению модератора
    if q.data == "mod_add_help":
        await q.edit_message_text(
            "➕ <b>Добавление модератора</b>\n\n"
            "Используйте команду:\n"
            "• <code>/givemoderator user_id</code>\n"
            "• Или ответьте на сообщение пользователя командой <code>/givemoderator</code>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="menu_moderators")]])
        )
        return

    # Обработка помощи по удалению модератора
    if q.data == "mod_remove_help":
        await q.edit_message_text(
            "➖ <b>Удаление модератора</b>\n\n"
            "Используйте команду:\n"
            "<code>/removemoderator user_id</code>\n\n"
            "⚠️ Нельзя удалить основных модераторов из ENV",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="menu_moderators")]])
        )
        return

    # Обработка меню антимата
    if q.data == "menu_profanity":
        await q.edit_message_text(
            "🤬 <b>Управление антимат фильтром</b>\n\nВыберите действие:",
            parse_mode=ParseMode.HTML,
            reply_markup=kb_profanity_menu()
        )
        return

    # Обработка помощи по проверке слов
    if q.data == "prof_check_help":
        await q.edit_message_text(
            "🔍 <b>Проверка текста на мат</b>\n\n"
            "Используйте команду:\n"
            "<code>/checkword текст для проверки</code>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="menu_profanity")]])
        )
        return

    # Обработка помощи по добавлению слов
    if q.data == "prof_add_help":
        await q.edit_message_text(
            "➕ <b>Добавление слова в фильтр</b>\n\n"
            "Используйте команду:\n"
            "<code>/addword слово</code>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="menu_profanity")]])
        )
        return

    # Обработка помощи по удалению слов
    if q.data == "prof_remove_help":
        await q.edit_message_text(
            "➖ <b>Удаление слова из фильтра</b>\n\n"
            "Используйте команду:\n"
            "<code>/removeword слово</code>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="menu_profanity")]])
        )
        return

    # Обработка меню статистики
    if q.data == "menu_stats":
        stats_7d = message_logger.get_model_accuracy_stats(days=7)
        stats_30d = message_logger.get_model_accuracy_stats(days=30)
        profanity_stats = message_logger.get_profanity_stats(days=7)

        await q.edit_message_text(
            f"<b>📈 Подробная статистика</b>\n\n"
            f"<b>За 7 дней:</b>\n"
            f"• Проверено: <code>{stats_7d['total_reviewed']}</code>\n"
            f"• Точность: <code>{stats_7d['accuracy']:.1f}%</code>\n"
            f"• Ложные срабатывания: <code>{stats_7d['false_positives']}</code>\n"
            f"• Пропущен спам: <code>{stats_7d['false_negatives']}</code>\n\n"
            f"<b>За 30 дней:</b>\n"
            f"• Проверено: <code>{stats_30d['total_reviewed']}</code>\n"
            f"• Точность: <code>{stats_30d['accuracy']:.1f}%</code>\n\n"
            f"<b>🤬 Матвордов за 7 дней:</b> <code>{profanity_stats.get('total_profanity', 0)}</code>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")]])
        )
        return

    # Обработка меню логов
    if q.data == "menu_logs":
        recent_logs = message_logger.get_recent_logs(limit=5)

        if not recent_logs:
            text = "<b>📝 Последние логи</b>\n\n❌ Логи пусты"
        else:
            text = "<b>📝 Последние 5 записей:</b>\n\n"
            for log in recent_logs:
                timestamp = log.get('timestamp', 'Неизвестно')[:16]
                user_name = log.get('full_name', 'Неизвестно')[:20]
                prediction = log.get('predicted_label', 'unknown')
                action = log.get('action_taken', 'none')
                text += f"<code>{timestamp}</code> {html.escape(user_name)}\n{prediction} → {action}\n\n"

        await q.edit_message_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📄 Полные логи", callback_data="logs_full")],
                [InlineKeyboardButton("⬅️ Назад", callback_data="menu_main")]
            ])
        )
        return

    # Обработка полных логов
    if q.data == "logs_full":
        await q.answer("Используйте команду /logs для полного списка")
        return

    # Обработка добавления слова из inline кнопки
    if q.data.startswith("add_word:"):
        word = q.data.split(":", 1)[1]
        if profanity_filter.add_word(word):
            await q.edit_message_text(f"✅ Слово '{word}' добавлено в антимат фильтр.")
        else:
            await q.edit_message_text(f"⚠️ Слово '{word}' уже есть в фильтре.")
        return

    # Обработка отмены действия
    if q.data == "cancel_action":
        await q.edit_message_text("❌ Действие отменено.")
        return

    # Обработка подтверждения добавления модератора
    if q.data.startswith("confirm_mod:"):
        user_id = int(q.data.split(":", 1)[1])
        if q.from_user.id in PENDING_MODERATORS:
            target_user_id, target_username = PENDING_MODERATORS.pop(q.from_user.id)

            if settings_manager.add_moderator(target_user_id):
                await q.edit_message_text(
                    f"✅ Пользователь {target_username} (ID: {target_user_id}) добавлен в список модераторов!",
                    parse_mode=ParseMode.HTML
                )
                LOGGER.info(f"Модератор {q.from_user.id} добавил нового модератора {target_user_id}")
            else:
                await q.edit_message_text(
                    f"⚠️ Пользователь {target_username} уже является модератором.",
                    parse_mode=ParseMode.HTML
                )
        else:
            await q.edit_message_text("❌ Запрос устарел или не найден.")
        return

    # Обработка отмены добавления модератора
    if q.data.startswith("cancel_mod:"):
        user_id = int(q.data.split(":", 1)[1])
        if q.from_user.id in PENDING_MODERATORS:
            target_user_id, target_username = PENDING_MODERATORS.pop(q.from_user.id)
            await q.edit_message_text(
                f"❌ Добавление модератора {target_username} отменено.",
                parse_mode=ParseMode.HTML
            )
        else:
            await q.edit_message_text("❌ Запрос устарел или не найден.")
        return

    # Существующая обработка спам/хам кнопок
    if ":" in q.data:
        action, payload = q.data.split(":", 1)
        if action in ("spam", "ham"):
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
            " • <b>/menu</b> — главное меню с удобными кнопками\n"
            " • <b>/status</b> — показать параметры антиспама и статистику\n"
            " • <b>/retrain</b> — переобучить модель\n"
            " • <b>/policy [тип]</b> — изменить режим (например: <code>/policy delete</code>)\n"
            " • <b>/announce [on/off]</b> — включить/выключить уведомления о блокировках\n"
            " • <b>/notifications [on/off]</b> — включить/выключить уведомления модераторам\n\n"
            "<b>👥 Управление модераторами:</b>\n"
            " • <b>/givemoderator [user_id]</b> — добавить модератора (с подтверждением)\n"
            " • <b>/removemoderator [user_id]</b> — удалить модератора\n"
            " • <b>/moderators</b> — показать список всех модераторов\n\n"
            "<b>🤬 Антимат фильтр:</b>\n"
            " • <b>/addword [слово]</b> — добавить слово в черный список\n"
            " • <b>/removeword [слово]</b> — удалить слово из черного списка\n"
            " • <b>/checkword [текст]</b> — проверить текст на наличие мата\n\n"
            "<b>📊 Статистика и логи:</b>\n"
            " • <b>/stats</b> — подробная статистика точности модели\n"
            " • <b>/logs</b> — последние 10 записей логов\n"
            " • <b>/userinfo [user_id]</b> — статистика конкретного пользователя\n\n"
            "<i>💾 Все настройки сохраняются автоматически и восстанавливаются после перезапуска.</i>\n"
            "<i>🎯 Порог автоудаления полностью убран из системы.</i>",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="menu_main")]])
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
    try:
        accuracy_stats = message_logger.get_model_accuracy_stats(days=7)
        profanity_stats = message_logger.get_profanity_stats(days=7)
    except Exception:
        accuracy_stats = {'total_reviewed': 0, 'accuracy': 0, 'false_positives': 0, 'false_negatives': 0}
        profanity_stats = {'total_profanity': 0}

    # Убираем порог автоудаления из статуса
    await update.effective_message.reply_html(
        "<b>📊 Статус антиспам-системы</b>\n\n"
        
        f"<b>📁 Датасет:</b> <code>{ds.name}</code> — название используемого набора данных\n"
        f"<b>📦 Размер:</b> <code>{size_kb} КиБ</code> — объём загруженного датасета\n"
        f"<b>🔢 Кол-во записей:</b> <code>{_dataset_rows()} строк</code> — количество примеров (сообщений) для анализа\n\n"
        
        f"<b>🛡️ Политика блокировки:</b> <code>{settings_manager.spam_policy}</code> — активный метод определения спама\n"
        f"<b>📣 Объявления о блоках:</b> <code>{'ВКЛ' if settings_manager.announce_blocks else 'ВЫКЛ'}</code> — уведомлять ли чат о блокировках\n"
        f"<b>🔔 Уведомления модераторам:</b> <code>{'ВКЛ' if settings_manager.notification_enabled else 'ВЫКЛ'}</code>\n\n"
        
        f"<b>📈 Точность модели (7 дней):</b>\n"
        f"• Проверено модераторами: <code>{accuracy_stats['total_reviewed']}</code>\n"
        f"• Точность: <code>{accuracy_stats['accuracy']:.1f}%</code>\n"
        f"• Ложные срабатывания: <code>{accuracy_stats['false_positives']}</code>\n"
        f"• Пропущенный спам: <code>{accuracy_stats['false_negatives']}</code>\n\n"
        f"<b>🤬 Найдено матвордов за 7 дней:</b> <code>{profanity_stats.get('total_profanity', 0)}</code>",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="menu_main")]])
    )

async def cmd_retrain(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    await update.effective_message.reply_text("⏳ Переобучаю модель…")
    try:
        classifier.train()
        await update.effective_message.reply_text("✅ Модель переобучена.")
    except Exception as e:
        await update.effective_message.reply_text(f"❌ Ошибка переобучения: {e}")

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

    try:
        # Получаем статистику за разные периоды
        stats_7d = message_logger.get_model_accuracy_stats(days=7)
        stats_30d = message_logger.get_model_accuracy_stats(days=30)
        profanity_stats = message_logger.get_profanity_stats(days=7)

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
            f"• Пропущенный спам: <code>{stats_30d['false_negatives']}</code>\n\n"
            
            f"<b>🤬 Матвордов за 7 дней:</b> <code>{profanity_stats.get('total_profanity', 0)}</code>",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="menu_main")]])
        )
    except Exception as e:
        await update.effective_message.reply_text(f"❌ Ошибка получения статистики: {e}")

async def cmd_user_info(update: Update, _):
    """Получить информацию о пользователе с улучшенной обработкой ошибок."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_html(
            "❌ <b>Неверное использование команды</b>\n\n"
            "Использование: <code>/userinfo &lt;user_id&gt;</code>\n\n"
            "Пример: <code>/userinfo 123456789</code>"
        )
        return

    try:
        user_id = int(args[1])
    except ValueError:
        await update.message.reply_html(
            "❌ <b>Неверный ID пользователя</b>\n\n"
            "ID пользователя должен быть числом.\n"
            "Пример: <code>/userinfo 123456789</code>"
        )
        return

    try:
        user_stats = message_logger.get_user_stats(user_id, days=30)

        if user_stats['total_messages'] == 0:
            await update.effective_message.reply_html(
                f"<b>👤 Статистика пользователя {user_id}</b>\n\n"
                "📝 <i>Нет записей о сообщениях от этого пользователя за последние 30 дней</i>\n\n"
                "Возможные причины:\n"
                "• Пользователь не писал сообщения\n"
                "• Сообщения не попадали под проверку спама\n"
                "• Данные ещё не собраны системой"
            )
        else:
            await update.effective_message.reply_html(
                f"<b>👤 Статистика пользователя {user_id}</b>\n\n"
                f"• Всего сообщений: <code>{user_stats['total_messages']}</code>\n"
                f"• Помечено как спам: <code>{user_stats['spam_messages']}</code>\n"
                f"• Средняя вероятность спама: <code>{user_stats['avg_spam_probability']:.1%}</code>\n"
                f"• Последнее сообщение: <code>{user_stats['last_message'] or 'Нет данных'}</code>"
            )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка получения статистики: {e}")

async def cmd_logs(update: Update, _):
    """Показать последние логи с улучшенной обработкой ошибок."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    try:
        recent_logs = message_logger.get_recent_logs(limit=10)

        if not recent_logs:
            await update.message.reply_html(
                "<b>📝 Логи системы</b>\n\n"
                "❌ <i>Логи пусты</i>\n\n"
                "Возможные причины:\n"
                "• Система недавно запущена\n"
                "• Не было обработанных сообщений\n"
                "• Проблемы с базой данных логов"
            )
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

        await update.effective_message.reply_html(
            log_text,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="menu_main")]])
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка получения логов: {e}")

async def cmd_removemoderator(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Удалить пользователя из списка модераторов с улучшенной обработкой."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_html(
            "❌ <b>Неверное использование команды</b>\n\n"
            "Использование: <code>/removemoderator &lt;user_id&gt;</code>\n\n"
            "Пример: <code>/removemoderator 123456789</code>"
        )
        return

    try:
        target_user_id = int(args[1])
    except ValueError:
        await update.message.reply_html(
            "❌ <b>Неверный ID пользователя</b>\n\n"
            "ID пользователя должен быть числом.\n"
            "Пример: <code>/removemoderator 123456789</code>"
        )
        return

    # Проверяем, является ли пользователь модератором вообще
    if not settings_manager.is_moderator(target_user_id):
        await update.message.reply_text(f"⚠️ Пользователь {target_user_id} не является модератором.")
        return

    # Проверяем, можно ли удалить (не из основного whitelist)
    from config.config import settings as config_settings
    if target_user_id in config_settings.WHITELIST_USER_IDS:
        await update.message.reply_html(
            f"❌ <b>Нельзя удалить основного модератора</b>\n\n"
            f"Пользователь <code>{target_user_id}</code> является основным модератором из конфигурации и не может быть удален через команду.\n\n"
            f"Для удаления измените переменную окружения <code>WHITELIST_USER_IDS</code>."
        )
        return

    # Пытаемся удалить
    if settings_manager.remove_moderator(target_user_id):
        await update.message.reply_text(f"✅ Пользователь {target_user_id} удален из списка модераторов.")
        LOGGER.info(f"Модератор {update.effective_user.id} удалил модератора {target_user_id}")
    else:
        await update.message.reply_text(f"⚠️ Пользователь {target_user_id} не найден среди дополнительных модераторов.")

async def cmd_addword(update: Update, _) -> None:
    """Добавить слово в антимат фильтр с улучшенной обработкой ошибок."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_html(
            "❌ <b>Неверное использование команды</b>\n\n"
            "Использование: <code>/addword &lt;слово&gt;</code>\n\n"
            "Пример: <code>/addword спам</code>\n\n"
            "⚠️ Слово будет добавлено в антимат фильтр"
        )
        return

    word = args[1].strip().lower()

    if len(word) < 2:
        await update.message.reply_text("❌ Слово должно содержать минимум 2 символа.")
        return

    if len(word) > 50:
        await update.message.reply_text("❌ Слово слишком длинное (максимум 50 символов).")
        return

    try:
        if profanity_filter.add_word(word):
            await update.message.reply_text(f"✅ Слово '{word}' добавлено в антимат фильтр.")
            LOGGER.info(f"Модератор {update.effective_user.id} добавил слово '{word}' в фильтр")
        else:
            await update.message.reply_text(f"⚠️ Слово '{word}' уже есть в фильтре.")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка добавления слова: {e}")

async def cmd_removeword(update: Update, _) -> None:
    """Удалить слово из антимат фильтра с улучшенной обработкой ошибок."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_html(
            "❌ <b>Неверное использование команды</b>\n\n"
            "Использование: <code>/removeword &lt;слово&gt;</code>\n\n"
            "Пример: <code>/removeword спам</code>"
        )
        return

    word = args[1].strip().lower()

    try:
        if profanity_filter.remove_word(word):
            await update.message.reply_text(f"✅ Слово '{word}' удалено из антимат фильтра.")
            LOGGER.info(f"Модератор {update.effective_user.id} удалил слово '{word}' из фильтра")
        else:
            await update.message.reply_text(f"⚠️ Слово '{word}' не найдено в фильтре.")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка удаления слова: {e}")

async def cmd_checkword(update: Update, _) -> None:
    """Проверить текст на наличие мата с улучшенной обработкой ошибок и inline кнопками."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_html(
            "❌ <b>Неверное использование команды</b>\n\n"
            "Использование: <code>/checkword &lt;текст для проверки&gt;</code>\n\n"
            "Пример: <code>/checkword проверить этот текст</code>"
        )
        return

    text_to_check = args[1]

    if len(text_to_check) > 500:
        await update.message.reply_text("❌ Текст слишком длинный для проверки (максимум 500 символов).")
        return

    try:
        if profanity_filter.contains_profanity(text_to_check):
            found_words = profanity_filter.get_profanity_words(text_to_check)
            await update.message.reply_html(
                f"🚫 <b>Обнаружен мат!</b>\n"
                f"Найдено слов: <code>{', '.join(found_words)}</code>"
            )
        else:
            # Предлагаем добавить слова из текста, если мат не найден
            words_in_text = [w.strip().lower() for w in text_to_check.split() if len(w.strip()) >= 2]

            if words_in_text:
                # Показываем первое слово как пример для добавления
                first_word = words_in_text[0]
                await update.message.reply_html(
                    "✅ <b>Мат не обнаружен</b>\n\n"
                    f"Если в тексте есть нежелательные слова, вы можете добавить их в фильтр:",
                    reply_markup=kb_word_not_found(first_word)
                )
            else:
                await update.message.reply_text("✅ Мат не обнаружен.")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка проверки: {e}")

async def cmd_givemoderator(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Добавить пользователя в список модераторов с улучшенной обработкой ошибок."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    # Проверяем, есть ли аргумент или reply
    target_user_id = None
    target_username = "Неизвестно"

    # Если команда в ответ на сообщение
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target_user_id = update.message.reply_to_message.from_user.id
        target_username = update.message.reply_to_message.from_user.full_name
    else:
        # Проверяем аргументы команды
        args = update.message.text.split(maxsplit=1)
        if len(args) < 2:
            await update.message.reply_html(
                "❌ <b>Неверное использование команды</b>\n\n"
                "Использование:\n"
                "• <code>/givemoderator &lt;user_id&gt;</code>\n"
                "• Ответьте на сообщение пользователя командой <code>/givemoderator</code>\n\n"
                "Пример: <code>/givemoderator 123456789</code>"
            )
            return

        arg = args[1].strip()

        # Проверяем упоминание пользователя
        if arg.startswith('@'):
            await update.message.reply_text("❌ Добавление по username пока не поддерживается. Используйте user_id или ответьте на сообщение.")
            return

        # Проверяем user_id
        try:
            target_user_id = int(arg)
            target_username = f"ID: {target_user_id}"
        except ValueError:
            await update.message.reply_html(
                "❌ <b>Неверный формат user_id</b>\n\n"
                "ID пользователя должен быть числом.\n"
                "Пример: <code>/givemoderator 123456789</code>"
            )
            return

    if not target_user_id:
        await update.message.reply_text("❌ Не удалось определить пользователя")
        return

    # Проверяем, не модератор ли уже
    if settings_manager.is_moderator(target_user_id):
        await update.message.reply_text(f"⚠️ Пользователь {target_username} уже является модератором.")
        return

    # Сохраняем в pending и показываем подтверждение
    PENDING_MODERATORS[update.effective_user.id] = (target_user_id, target_username)

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Да, добавить", callback_data=f"confirm_mod:{target_user_id}"),
            InlineKeyboardButton("❌ Отмена", callback_data=f"cancel_mod:{target_user_id}")
        ]
    ])

    await update.message.reply_html(
        f"<b>🔐 Подтверждение добавления модератора</b>\n\n"
        f"Вы точно хотите добавить <b>{html.escape(target_username)}</b> (ID: <code>{target_user_id}</code>) в список модераторов?\n\n"
        f"⚠️ Модераторы получают доступ к управлению ботом и могут принимать решения о спаме.",
        reply_markup=keyboard
    )

async def cmd_moderators(update: Update, _) -> None:
    """Показать список всех модераторов с улучшенным интерфейсом."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    try:
        all_mods = settings_manager.get_all_moderators()
        additional_mods = settings_manager.get_additional_moderators()

        text = "<b>👥 Список модераторов</b>\n\n"

        # Основные модераторы (из whitelist)
        whitelist_mods = [uid for uid in all_mods if uid not in additional_mods]
        if whitelist_mods:
            text += "<b>🔒 Основные модераторы (из конфигурации):</b>\n"
            for mod_id in whitelist_mods:
                text += f"• <code>{mod_id}</code>\n"
            text += "\n"

        # Дополнительные модераторы
        if additional_mods:
            text += "<b>➕ Дополнительные модераторы:</b>\n"
            for mod_id in additional_mods:
                text += f"• <code>{mod_id}</code> (можно удалить)\n"
            text += "\n"

        text += f"<b>📊 Всего модераторов:</b> {len(all_mods)}\n\n"
        text += "<i>💡 Основные модераторы нельзя удалить через команду</i>"

        await update.effective_message.reply_html(
            text,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Главное меню", callback_data="menu_main")]])
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка получения списка модераторов: {e}")

# ─────────────────────────────  NEW MENU COMMAND
async def cmd_menu(update: Update, _) -> None:
    """Показать главное меню модератора."""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return

    await update.effective_message.reply_html(
        "🏠 <b>Главное меню модератора</b>\n\nВыберите нужный раздел:",
        reply_markup=kb_admin_menu()
    )

# ─────────────────────────────  REGISTRATION
def register_handlers(app: Application) -> None:
    # Основные команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("menu", cmd_menu))  # Новая команда для главного меню

    # Администрирование системы
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("retrain", cmd_retrain))
    app.add_handler(CommandHandler("policy", cmd_policy))
    app.add_handler(CommandHandler("announce", cmd_announce))
    app.add_handler(CommandHandler("notifications", cmd_notifications))

    # Статистика и логирование
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("userinfo", cmd_user_info))
    app.add_handler(CommandHandler("logs", cmd_logs))

    # Управление модераторами
    app.add_handler(CommandHandler("givemoderator", cmd_givemoderator))
    app.add_handler(CommandHandler("removemoderator", cmd_removemoderator))
    app.add_handler(CommandHandler("moderators", cmd_moderators))

    # Управление антимат фильтром
    app.add_handler(CommandHandler("addword", cmd_addword))
    app.add_handler(CommandHandler("removeword", cmd_removeword))
    app.add_handler(CommandHandler("checkword", cmd_checkword))

    # Обработчики callback-ов и сообщений (расширенные паттерны)
    app.add_handler(CallbackQueryHandler(on_callback, pattern="^(spam|ham|confirm_mod|cancel_mod|menu_|settings_|policy_|mod_|prof_|add_word|cancel_action|logs_full):"))
    app.add_handler(MessageHandler(filters.TEXT | filters.CaptionRegex(".*"), on_message))
