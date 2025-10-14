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
from services.meta_classifier import MetaClassifier  # NEW
from bot.keyboards import moderator_keyboard, format_debug_card, format_notification_card
from utils.logger import get_logger

LOGGER = get_logger(__name__)

keyword_filter = KeywordFilter()
tfidf_filter = TfidfFilter()
embedding_filter = EmbeddingFilter(
    mode=settings.EMBEDDING_MODE,
    api_key=settings.MISTRAL_API_KEY,
    ollama_model=settings.OLLAMA_MODEL,
    ollama_base_url=settings.OLLAMA_BASE_URL
)

coordinator = FilterCoordinator(
    keyword_filter=keyword_filter,
    tfidf_filter=tfidf_filter,
    embedding_filter=embedding_filter
)

policy_engine = PolicyEngine()  # Reads config from runtime_config singleton

# NEW: Мета-классификатор
meta_classifier = MetaClassifier()

# Логируем загруженную конфигурацию политики
from config.runtime import runtime_config
LOGGER.info(f"Policy configuration loaded:")
LOGGER.info(f"  MODE: {runtime_config.policy_mode}")
LOGGER.info(f"  AUTO_DELETE_THRESHOLD: {runtime_config.auto_delete_threshold}")
LOGGER.info(f"  AUTO_KICK_THRESHOLD: {runtime_config.auto_kick_threshold}")
LOGGER.info(f"  NOTIFY_THRESHOLD: {runtime_config.notify_threshold}")
LOGGER.info(f"  USE_META_CLASSIFIER: {runtime_config.use_meta_classifier}")
LOGGER.info(f"  META_CLASSIFIER_READY: {meta_classifier.is_ready()}")

dataset_manager = DatasetManager(
    Path(__file__).resolve().parents[1] / "data" / "messages.csv"
)

# Хранилище для pending модераторских решений
PENDING: dict[tuple[int, int], tuple[str, str, int, AnalysisResult]] = {}

# Хранилище для debug информации (spam_id -> детальная информация)
SPAM_STORAGE: dict[int, dict] = {}
SPAM_ID_COUNTER = 0


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

    # Временное логирование для определения правильного ID чата
    LOGGER.info(f"📍 Получено сообщение из чата ID: {msg.chat_id} (тип: {msg.chat.type})")

    text = (msg.text or msg.caption or "").strip()
    if not text or msg.chat_id == settings.MODERATOR_CHAT_ID:
        return
    
    if is_whitelisted(msg.from_user.id):
        return
    
    # Шаг 1: Анализ фильтрами
    analysis = await coordinator.analyze(text)
    
    # Шаг 2: Мета-классификатор (если включен и готов)
    p_spam = None
    meta_debug = None
    
    if runtime_config.use_meta_classifier and meta_classifier.is_ready():
        try:
            p_spam, meta_debug = await meta_classifier.predict_proba(text, analysis)
            
            if p_spam is not None:
                LOGGER.info(
                    f"MetaClassifier: p_spam={p_spam:.3f}, "
                    f"sim_diff={meta_debug.get('sim_diff', 'N/A')}"
                )
                
                # Создаем новый AnalysisResult с мета-данными
                from dataclasses import replace
                analysis = replace(analysis, meta_proba=p_spam, meta_debug=meta_debug)
        except Exception as e:
            LOGGER.error(f"MetaClassifier failed: {e}")
    
    # Шаг 3: Принятие решения
    action = policy_engine.decide_action(analysis)
    
    # Логирование
    if analysis.meta_proba is not None:
        LOGGER.info(
            f"Message from {msg.from_user.full_name}: "
            f"p_spam={analysis.meta_proba:.2f}, action={action.value}"
        )
    else:
        LOGGER.info(
            f"Message from {msg.from_user.full_name}: "
            f"avg={analysis.average_score:.2f}, action={action.value}"
        )
    
    if action == Action.APPROVE:
        return
    
    # Генерируем уникальный ID для этого спама
    global SPAM_ID_COUNTER
    SPAM_ID_COUNTER += 1
    spam_id = SPAM_ID_COUNTER
    
    # Сохраняем детальную информацию для debug
    msg_link = get_message_link(msg)
    SPAM_STORAGE[spam_id] = {
        "spam_id": spam_id,
        "user_name": msg.from_user.full_name,
        "user_id": msg.from_user.id,
        "chat_id": msg.chat_id,
        "message_id": msg.message_id,
        "text": text,
        "msg_link": msg_link,
        "analysis": analysis,
        "action": action
    }
    
    # Формируем карточку (простую или детальную в зависимости от DETAILED_DEBUG_INFO)
    card = format_notification_card(
        spam_id=spam_id,
        user_name=msg.from_user.full_name,
        user_id=msg.from_user.id,
        text=text,
        msg_link=msg_link,
        analysis=analysis,
        action=action,
        chat_id=msg.chat_id,
        message_id=msg.message_id
    )
    
    # Отправляем карточку модератору
    # Кнопки только для NOTIFY, для DELETE/KICK - без кнопок
    keyboard = moderator_keyboard(msg.chat_id, msg.message_id) if action == Action.NOTIFY else None
    
    await context.bot.send_message(
        settings.MODERATOR_CHAT_ID,
        card,
        reply_markup=keyboard,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    
    # Сохраняем в PENDING только если нужно решение модератора
    if action == Action.NOTIFY:
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
    
    # NEW: Мета-классификатор статус
    if runtime_config.use_meta_classifier:
        if meta_classifier.is_ready():
            filters_status.append("🎯 MetaClassifier: ✅")
        else:
            filters_status.append("🎯 MetaClassifier: ❌ (not trained)")
    else:
        filters_status.append("🎯 MetaClassifier: 🔕 (disabled)")
    
    await update.effective_message.reply_html(
        "<b>📊 Статус антиспам-системы</b>\n\n"
        f"<b>📁 Датасет:</b> <code>messages.csv</code>\n"
        f"<b>📦 Размер:</b> <code>{dataset_size_kb} КиБ</code>\n"
        f"<b>🔢 Записей:</b> <code>{dataset_rows}</code>\n\n"
        "<b>🛡️ Фильтры:</b>\n" + "\n".join(filters_status) + "\n\n"
        f"<b>🤖 Режим политики:</b> <code>{runtime_config.policy_mode}</code>\n"
        f"<b>📣 Объявления:</b> <code>{'ВКЛ' if settings.ANNOUNCE_BLOCKS else 'ВЫКЛ'}</code>"
    )


async def cmd_retrain(update: Update, _):
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    await update.effective_message.reply_text("⏳ Переобучаю TF-IDF модель…")
    tfidf_filter.train()
    time.sleep(3)
    await update.effective_message.reply_text("✅ Модель переобучена.")


async def cmd_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /debug N - показывает детальную информацию о спаме №N"""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    
    if not context.args or len(context.args) != 1:
        await update.effective_message.reply_text(
            "❌ Использование: /debug <номер>\n"
            "Пример: /debug 123"
        )
        return
    
    try:
        spam_id = int(context.args[0])
    except ValueError:
        await update.effective_message.reply_text("❌ Номер должен быть числом")
        return
    
    if spam_id not in SPAM_STORAGE:
        await update.effective_message.reply_text(
            f"❌ Сообщение №{spam_id} не найдено\n"
            f"Доступные ID: {min(SPAM_STORAGE.keys()) if SPAM_STORAGE else 'нет'} - "
            f"{max(SPAM_STORAGE.keys()) if SPAM_STORAGE else 'нет'}"
        )
        return
    
    # Получаем детальную информацию
    data = SPAM_STORAGE[spam_id]
    
    card = format_debug_card(
        spam_id=data["spam_id"],
        user_name=data["user_name"],
        user_id=data["user_id"],
        text=data["text"],
        msg_link=data["msg_link"],
        analysis=data["analysis"],
        action=data["action"],
        chat_id=data["chat_id"],
        message_id=data["message_id"]
    )
    
    await update.effective_message.reply_html(
        card,
        disable_web_page_preview=True
    )


async def cmd_setpolicy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Изменить режим политики в runtime"""
    if not is_explicit_command(update):
        return

    user = update.effective_user
    if not user or not is_whitelisted(user.id):
        await update.effective_message.reply_text("❌ Команда доступна только модераторам")
        return

    if not context.args or len(context.args) != 1:
        await update.effective_message.reply_html(
            "⚙️ <b>Текущий режим политики:</b> <code>{}</code>\n\n"
            "<b>Использование:</b> <code>/setpolicy &lt;режим&gt;</code>\n\n"
            "<b>Доступные режимы:</b>\n"
            " • <code>manual</code> — всё на модератора\n"
            " • <code>semi-auto</code> — авто при высоких оценках\n"
            " • <code>auto</code> — полная автоматизация".format(
                html.escape(runtime_config.policy_mode)
            )
        )
        return

    new_mode = context.args[0].lower()
    old_mode = runtime_config.policy_mode
    
    try:
        runtime_config.set_policy_mode(new_mode)
        LOGGER.info(f"Policy mode changed: {old_mode} → {new_mode} (by user {user.id})")
        
        await update.effective_message.reply_html(
            "✅ <b>Режим политики изменён</b>\n\n"
            " • Было: <code>{}</code>\n"
            " • Стало: <code>{}</code>\n\n"
            "Изменения применены немедленно.".format(
                html.escape(old_mode),
                html.escape(new_mode)
            )
        )
    except ValueError as e:
        await update.effective_message.reply_html(
            f"❌ Ошибка: {html.escape(str(e))}\n\n"
            "Допустимые значения: <code>manual</code>, <code>semi-auto</code>, <code>auto</code>"
        )


async def cmd_setthreshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Изменить порог фильтра в runtime"""
    if not is_explicit_command(update):
        return

    user = update.effective_user
    if not user or not is_whitelisted(user.id):
        await update.effective_message.reply_text("❌ Команда доступна только модераторам")
        return

    if not context.args or len(context.args) != 2:
        # Показываем текущие значения
        overrides = runtime_config.get_overrides()
        overrides_text = "\n".join(
            f" • <code>{k}</code> = <b>{v}</b> ⚠️ (изменено)" 
            for k, v in overrides.items()
        ) if overrides else " <i>Нет изменённых значений</i>"
        
        await update.effective_message.reply_html(
            "⚙️ <b>Текущие пороги:</b>\n\n"
            "<b>Политика:</b>\n"
            f" • <code>auto_delete</code> = {runtime_config.auto_delete_threshold}\n"
            f" • <code>auto_kick</code> = {runtime_config.auto_kick_threshold}\n"
            f" • <code>notify</code> = {runtime_config.notify_threshold}\n\n"
            "<b>Фильтры:</b>\n"
            f" • <code>keyword</code> = {runtime_config.keyword_threshold}\n"
            f" • <code>tfidf</code> = {runtime_config.tfidf_threshold}\n"
            f" • <code>embedding</code> = {runtime_config.embedding_threshold}\n\n"
            "<b>Изменённые значения:</b>\n" + overrides_text + "\n\n"
            "<b>Использование:</b> <code>/setthreshold &lt;имя&gt; &lt;значение&gt;</code>\n"
            "Пример: <code>/setthreshold auto_delete 0.75</code>\n\n"
            "<b>Сброс:</b> <code>/resetconfig</code>"
        )
        return

    threshold_name = context.args[0].lower()
    
    try:
        new_value = float(context.args[1])
    except ValueError:
        await update.effective_message.reply_text(
            f"❌ Некорректное значение: {context.args[1]}\n"
            "Ожидается число от 0.0 до 1.0"
        )
        return

    # Получаем старое значение
    old_value = getattr(runtime_config, threshold_name, None)
    if old_value is None:
        available = [
            "auto_delete", "auto_kick", "notify",
            "keyword", "tfidf", "embedding"
        ]
        await update.effective_message.reply_html(
            f"❌ Неизвестный порог: <code>{html.escape(threshold_name)}</code>\n\n"
            "<b>Доступные пороги:</b>\n" +
            "\n".join(f" • <code>{t}</code>" for t in available)
        )
        return

    try:
        runtime_config.set_threshold(threshold_name, new_value)
        LOGGER.info(
            f"Threshold changed: {threshold_name} {old_value} → {new_value} "
            f"(by user {user.id})"
        )
        
        await update.effective_message.reply_html(
            "✅ <b>Порог изменён</b>\n\n"
            f" • Параметр: <code>{html.escape(threshold_name)}</code>\n"
            f" • Было: <code>{old_value}</code>\n"
            f" • Стало: <code>{new_value}</code>\n\n"
            "Изменения применены немедленно."
        )
    except ValueError as e:
        await update.effective_message.reply_text(f"❌ Ошибка: {e}")


async def cmd_resetconfig(update: Update, _):
    """Сбросить все изменения конфигурации к значениям по умолчанию"""
    if not is_explicit_command(update):
        return

    user = update.effective_user
    if not user or not is_whitelisted(user.id):
        await update.effective_message.reply_text("❌ Команда доступна только модераторам")
        return

    overrides = runtime_config.get_overrides()
    if not overrides:
        await update.effective_message.reply_text(
            "ℹ️ Нет изменённых значений для сброса.\n"
            "Все параметры используют значения по умолчанию из .env"
        )
        return

    runtime_config.reset_overrides()
    LOGGER.info(f"Configuration reset to defaults (by user {user.id})")
    
    overrides_text = "\n".join(
        f" • <code>{k}</code> = {v}"
        for k, v in overrides.items()
    )
    
    await update.effective_message.reply_html(
        "✅ <b>Конфигурация сброшена</b>\n\n"
        "Сброшены следующие параметры:\n" + overrides_text + "\n\n"
        "Теперь используются значения по умолчанию из .env"
    )


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
            "<b>⚙️ Команды мониторинга:</b>\n"
            " • <b>/status</b> — статус системы\n"
            " • <b>/debug N</b> — детали сообщения №N\n"
            " • <b>/meta_info</b> — инфо о мета-классификаторе\n\n"
            "<b>🔧 Команды конфигурации:</b>\n"
            " • <b>/setpolicy &lt;режим&gt;</b> — изменить режим политики\n"
            " • <b>/setthreshold &lt;имя&gt; &lt;значение&gt;</b> — изменить порог\n"
            " • <b>/resetconfig</b> — сбросить к значениям из .env\n\n"
            "<b>🔄 Обслуживание:</b>\n"
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


async def cmd_meta_info(update: Update, _):
    """Информация о мета-классификаторе"""
    if not update.effective_user or not is_whitelisted(update.effective_user.id):
        return
    
    info = meta_classifier.get_info()
    
    status_icon = "✅" if info['ready'] else "❌"
    
    message = (
        f"🎯 <b>Мета-классификатор {status_icon}</b>\n\n"
        f"<b>Статус:</b> {'Готов' if info['ready'] else 'Не обучен'}\n"
        f"<b>Режим:</b> {'Включен' if runtime_config.use_meta_classifier else 'Отключен'}\n\n"
    )
    
    if info['ready']:
        message += (
            f"<b>📊 Пороги решений:</b>\n"
            f" • High (delete/ban): <code>{runtime_config.meta_threshold_high:.2f}</code>\n"
            f" • Medium (notify): <code>{runtime_config.meta_threshold_medium:.2f}</code>\n\n"
            f"<b>🔧 Модель:</b>\n"
            f" • Фичей: <code>{info['num_features']}</code>\n"
            f" • Калибратор: {'✅' if info['calibrator_loaded'] else '❌'}\n"
            f" • Центроиды: {'✅' if info['centroids_loaded'] else '❌'}\n"
        )
        
        if 'logreg_date' in info:
            message += f" • Дата обучения: <code>{info['logreg_date'][:10]}</code>\n"
        
        message += (
            f"\n<b>📁 Путь:</b> <code>{info['models_dir']}</code>\n\n"
            f"<i>Фичи: {', '.join(info['feature_names'][:5])}...</i>"
        )
    else:
        message += (
            "<b>⚠️ Модель не обучена</b>\n\n"
            "Запустите обучение:\n"
            "<code>python scripts/train_meta.py</code>\n\n"
            f"Путь к артефактам: <code>{info['models_dir']}</code>"
        )
    
    await update.effective_message.reply_html(message)


def register_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("retrain", cmd_retrain))
    app.add_handler(CommandHandler("debug", cmd_debug))
    app.add_handler(CommandHandler("meta_info", cmd_meta_info))  # NEW
    app.add_handler(CommandHandler("setpolicy", cmd_setpolicy))
    app.add_handler(CommandHandler("setthreshold", cmd_setthreshold))
    app.add_handler(CommandHandler("resetconfig", cmd_resetconfig))

    app.add_handler(CallbackQueryHandler(on_callback, pattern="^(kick|delete|ham):"))
    app.add_handler(MessageHandler(filters.TEXT | filters.CaptionRegex(".*"), on_message))
