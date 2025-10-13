from __future__ import annotations

import html

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from core.types import Action, AnalysisResult


def moderator_keyboard(chat_id: int, msg_id: int) -> InlineKeyboardMarkup:
    """Клавиатура для модератора (только для NOTIFY)"""
    payload = f"{chat_id}:{msg_id}"
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🚫 Спам/Бан", callback_data=f"kick:{payload}"),
            InlineKeyboardButton("✅ Не спам", callback_data=f"ham:{payload}"),
        ]
    ])


def format_simple_card(
    spam_id: int,
    user_name: str,
    text: str,
    msg_link: str,
    analysis: AnalysisResult,
    action: Action
) -> str:
    """Упрощенная карточка для модератора (по умолчанию)"""
    preview = html.escape(text[:150] + ("…" if len(text) > 150 else ""))
    avg_score = analysis.average_score
    
    # Иконка и статус в зависимости от действия
    if action == Action.KICK:
        icon = "🚫"
        status = "ЗАБАНЕН автоматически"
    elif action == Action.DELETE:
        icon = "🗑️"
        status = "УДАЛЕН автоматически"
    else:
        icon = "⚠️"
        status = "Требует проверки"
    
    card = (
        f"{icon} <b>Подозрительное сообщение (№{spam_id})</b>\n\n"
        f"👤 {html.escape(user_name)}\n"
        f"📊 Оценка: <b>{avg_score:.0%}</b>\n"
        f"🔗 <a href='{msg_link}'>Перейти</a>\n\n"
        f"💬 <i>{preview}</i>\n\n"
        f"🤖 <b>{status}</b>"
    )
    
    if action == Action.NOTIFY:
        card += f"\n\n<i>Используй /debug {spam_id} для деталей</i>"
    else:
        card += f"\n<i>Детали: /debug {spam_id}</i>"
    
    return card


def format_debug_card(
    spam_id: int,
    user_name: str,
    user_id: int,
    text: str,
    msg_link: str,
    analysis: AnalysisResult,
    action: Action,
    chat_id: int,
    message_id: int
) -> str:
    """Детальная карточка с технической информацией"""
    preview = html.escape(text[:200] + ("…" if len(text) > 200 else ""))
    
    keyword = analysis.keyword_result
    tfidf = analysis.tfidf_result
    embedding = analysis.embedding_result
    
    # Статус действия
    if action == Action.KICK:
        action_text = "🚫 <b>KICK</b> (забанен автоматически)"
    elif action == Action.DELETE:
        action_text = "🗑️ <b>DELETE</b> (удален автоматически)"
    elif action == Action.NOTIFY:
        action_text = "⚠️ <b>NOTIFY</b> (ожидает решения)"
    else:
        action_text = "✅ <b>APPROVE</b> (пропущен)"
    
    card = (
        f"� <b>Debug: Подозрительное сообщение №{spam_id}</b>\n\n"
        f"👤 <b>Пользователь:</b> {html.escape(user_name)}\n"
        f"🆔 <b>User ID:</b> <code>{user_id}</code>\n"
        f"� <b>Chat ID:</b> <code>{chat_id}</code>\n"
        f"📨 <b>Message ID:</b> <code>{message_id}</code>\n"
        f"�🔗 <a href='{msg_link}'>Перейти к сообщению</a>\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>📊 АНАЛИЗ ФИЛЬТРОВ</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )
    
    # Embedding filter (приоритет)
    if embedding and embedding.score != 0.5:
        card += f"🧠 <b>Embedding Filter</b> (вес: 50%)\n"
        card += f"   └ Score: <b>{embedding.score:.2%}</b> (confidence: {embedding.confidence:.0%})\n"
        if embedding.details and embedding.details.get("reasoning"):
            reasoning = html.escape(embedding.details["reasoning"])
            card += f"   └ {reasoning}\n"
    else:
        card += f"🧠 <b>Embedding Filter</b>: <i>недоступен</i>\n"
    
    card += "\n"
    
    # Keyword filter
    card += f"🔤 <b>Keyword Filter</b> (вес: 20%)\n"
    card += f"   └ Score: <b>{keyword.score:.2%}</b> (confidence: {keyword.confidence:.0%})\n"
    if keyword.details:
        if keyword.details.get("matched_keywords"):
            keywords = ", ".join(keyword.details["matched_keywords"])
            card += f"   └ Найдено: <code>{keywords}</code>\n"
        if keyword.details.get("matched_patterns"):
            patterns = ", ".join(keyword.details["matched_patterns"])
            card += f"   └ Паттерны: <code>{patterns}</code>\n"
    
    card += "\n"
    
    # TF-IDF filter
    card += f"� <b>TF-IDF Filter</b> (вес: 30%)\n"
    card += f"   └ Score: <b>{tfidf.score:.2%}</b> (confidence: {tfidf.confidence:.0%})\n"
    if tfidf.details and tfidf.details.get("class_probabilities"):
        probs = tfidf.details["class_probabilities"]
        card += f"   └ P(spam): {probs[1]:.3f}, P(ham): {probs[0]:.3f}\n"
    
    card += (
        f"\n━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>🎯 ИТОГОВАЯ ОЦЕНКА</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 Average Score: <b>{analysis.average_score:.2%}</b>\n"
        f"📊 Max Score: <b>{analysis.max_score:.2%}</b>\n"
        f"📊 All Filters High: <b>{'Да' if analysis.all_high else 'Нет'}</b>\n\n"
        f"🤖 <b>Действие:</b> {action_text}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>💬 ТЕКСТ СООБЩЕНИЯ</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{preview}"
    )
    
    return card
