from __future__ import annotations

import html

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from config.config import settings
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
    )
    
    # NEW: Мета-классификатор (если доступен)
    if analysis.meta_proba is not None:
        card += f"🎯 <b>MetaClassifier:</b> <b>{analysis.meta_proba:.2%}</b>\n"
        
        if analysis.meta_debug:
            # Similarity scores
            sim_spam = analysis.meta_debug.get('sim_spam')
            sim_ham = analysis.meta_debug.get('sim_ham')
            sim_diff = analysis.meta_debug.get('sim_diff')
            
            if sim_spam is not None:
                card += f"   └ Sim(spam): {sim_spam:.3f}, Sim(ham): {sim_ham:.3f}, Diff: {sim_diff:.3f}\n"
            
            # Паттерны
            patterns = analysis.meta_debug.get('patterns', {})
            fired_patterns = [k.replace('has_', '') for k, v in patterns.items() 
                            if k.startswith('has_') and v]
            if fired_patterns:
                card += f"   └ Паттерны: <code>{', '.join(fired_patterns)}</code>\n"
            
            if 'obfuscation_ratio' in patterns and patterns['obfuscation_ratio'] > 0:
                card += f"   └ Обфускация: {patterns['obfuscation_ratio']:.1%}\n"
        
        card += "\n"
    
    card += (
        f"📊 Average Score (legacy): <b>{analysis.average_score:.2%}</b>\n"
        f"📊 Max Score: <b>{analysis.max_score:.2%}</b>\n"
        f"📊 All Filters High: <b>{'Да' if analysis.all_high else 'Нет'}</b>\n\n"
        f"🤖 <b>Действие:</b> {action_text}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>💬 ТЕКСТ СООБЩЕНИЯ</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{preview}"
    )
    
    return card


def format_notification_card(
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
    """
    Форматирует карточку для уведомления модератора.
    Выбирает между простым и детальным форматом в зависимости от DETAILED_DEBUG_INFO.
    """
    if settings.DETAILED_DEBUG_INFO:
        # Показываем полную техническую информацию по умолчанию
        return format_debug_card(
            spam_id=spam_id,
            user_name=user_name,
            user_id=user_id,
            text=text,
            msg_link=msg_link,
            analysis=analysis,
            action=action,
            chat_id=chat_id,
            message_id=message_id
        )
    else:
        # Показываем упрощенную версию
        return format_simple_card(
            spam_id=spam_id,
            user_name=user_name,
            text=text,
            msg_link=msg_link,
            analysis=analysis,
            action=action
        )
