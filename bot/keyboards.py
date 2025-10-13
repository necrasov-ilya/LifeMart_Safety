from __future__ import annotations

import html

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from core.types import AnalysisResult


def moderator_keyboard(chat_id: int, msg_id: int) -> InlineKeyboardMarkup:
    payload = f"{chat_id}:{msg_id}"
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🚫 Спам/Бан", callback_data=f"kick:{payload}"),
            InlineKeyboardButton("✅ Не спам", callback_data=f"ham:{payload}"),
        ]
    ])


def format_moderator_card(
    user_name: str,
    text: str,
    msg_link: str,
    analysis: AnalysisResult
) -> str:
    preview = html.escape(text[:200] + ("…" if len(text) > 200 else ""))
    
    keyword = analysis.keyword_result
    tfidf = analysis.tfidf_result
    embedding = analysis.embedding_result
    
    card = (
        "🚨 <b>Подозрительное сообщение</b>\n\n"
        f"👤 <b>Автор:</b> {html.escape(user_name)}\n"
        f"🔗 <a href='{msg_link}'>Перейти к сообщению</a>\n\n"
        f"📊 <b>Оценка фильтров:</b>\n"
    )
    
    # Приоритет на embedding (если доступен)
    if embedding and embedding.score != 0.5:
        card += f"  🧠 <b>Семантика (приоритет):</b> <b>{embedding.score:.0%}</b>\n"
        if embedding.details and embedding.details.get("reasoning"):
            reasoning = embedding.details["reasoning"][:60]
            card += f"     <i>{reasoning}...</i>\n"
    else:
        card += "  🧠 <i>Семантика: недоступна</i>\n"
    
    card += f"  🔤 Ключевые слова: <b>{keyword.score:.0%}</b>\n"
    
    if keyword.details and keyword.details.get("matched_keywords"):
        keywords = ", ".join(keyword.details["matched_keywords"][:3])
        card += f"     <i>Найдено: {keywords}</i>\n"
    
    card += f"  📈 TF-IDF модель: <b>{tfidf.score:.0%}</b>\n"
    
    avg_score = analysis.average_score
    card += f"\n📊 <b>Итоговая оценка: {avg_score:.0%}</b>\n\n"
    
    if avg_score >= 0.85:
        recommendation = "⚡️ <b>Рекомендация:</b> Удалить и забанить"
    elif avg_score >= 0.70:
        recommendation = "⚠️ <b>Рекомендация:</b> Вероятно спам"
    elif avg_score >= 0.50:
        recommendation = "🤔 <b>Рекомендация:</b> Требуется проверка"
    else:
        recommendation = "ℹ️ <b>Рекомендация:</b> Скорее всего не спам"
    
    card += f"{recommendation}\n\n"
    card += f"💬 <b>Сообщение:</b>\n{preview}"
    
    return card
