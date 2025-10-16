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
    action: Action,
    decision_details: dict = None
) -> str:
    """Упрощенная карточка для модератора (по умолчанию)"""
    preview = html.escape(text[:150] + ("…" if len(text) > 150 else ""))
    kw_score = None
    tfidf_score = None
    
    # Используем p_spam если доступен, иначе average_score
    legacy_mode = bool(decision_details and decision_details.get('legacy_mode'))
    if legacy_mode:
        kw_score = float(decision_details.get('legacy_keyword_score', 0.0) or 0.0)
        tfidf_score = float(decision_details.get('legacy_tfidf_score', 0.0) or 0.0)
        score = decision_details.get('legacy_trigger_score')
        if score is None:
            score = max(kw_score, tfidf_score)
        trigger = decision_details.get('legacy_trigger')
        if trigger == "keyword":
            score_label = "Legacy keyword"
        elif trigger == "tfidf":
            score_label = "Legacy TF-IDF"
        else:
            score_label = "Legacy score"
    elif decision_details and 'p_spam_adjusted' in decision_details:
        score = decision_details['p_spam_adjusted']
        score_label = "p_spam"
    else:
        score = analysis.average_score
        score_label = "average"
    
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
        f"📊 {score_label}: <b>{score:.0%}</b>\n"
        f"🔗 <a href='{msg_link}'>Перейти</a>\n\n"
        f"💬 <i>{preview}</i>\n\n"
        f"🤖 <b>{status}</b>"
    )
    if legacy_mode:
        legacy_chunks = []
        if kw_score is not None:
            legacy_chunks.append(f"KW=<b>{kw_score:.0%}</b>")
        if tfidf_score is not None:
            legacy_chunks.append(f"TF-IDF=<b>{tfidf_score:.0%}</b>")
        if legacy_chunks:
            card += "\nLegacy: " + ", ".join(legacy_chunks)
        meta_preview = decision_details.get("p_spam_original") if decision_details else None
        if meta_preview is not None:
            card += f"\nMeta p_spam: <b>{meta_preview:.0%}</b>"
    
    # Режим политики
    if decision_details and 'policy_mode' in decision_details:
        card += f"\n🔧 Режим: <code>{decision_details['policy_mode']}</code>"
    
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
    message_id: int,
    decision_details: dict = None
) -> str:
    """Техническая карточка с метриками мета-классификатора"""
    preview = html.escape(text[:200] + ("…" if len(text) > 200 else ""))
    
    # Статус действия
    action_icons = {
        Action.KICK: "🚫 KICK",
        Action.DELETE: "🗑️ DELETE", 
        Action.NOTIFY: "⚠️ NOTIFY",
        Action.APPROVE: "✅ APPROVE"
    }
    action_text = action_icons.get(action, "UNKNOWN")
    
    card = (
        f"🔍 <b>Debug #{spam_id}</b>\n"
        f"👤 {html.escape(user_name)} (<code>{user_id}</code>)\n"
        f"💬 Chat: <code>{chat_id}</code> | Msg: <code>{message_id}</code>\n"
        f"🔗 <a href='{msg_link}'>Перейти</a>\n\n"
    )
    
    # Policy Decision
    if decision_details:
        mode = decision_details.get('policy_mode', 'unknown')
        legacy_mode = bool(decision_details.get('legacy_mode'))
        card += "Decision summary:\n"
        card += f"- {action_text} | mode={mode}\n"
        if legacy_mode:
            kw_val = decision_details.get('legacy_keyword_score')
            tfidf_val = decision_details.get('legacy_tfidf_score')
            trigger = decision_details.get('legacy_trigger') or "none"
            trigger_score = decision_details.get('legacy_trigger_score')
            legacy_parts = []
            if kw_val is not None:
                legacy_parts.append(f"KW={kw_val:.1%}")
            if tfidf_val is not None:
                legacy_parts.append(f"TF-IDF={tfidf_val:.1%}")
            if legacy_parts:
                card += "- Legacy scores: " + ", ".join(legacy_parts) + "\n"
            if trigger_score is not None:
                card += f"- Trigger: {trigger} ({trigger_score:.1%})\n"
            reason = decision_details.get('action_reason')
            if reason:
                card += f"- Reason: {reason}\n"
            meta_preview = decision_details.get('p_spam_original')
            if meta_preview is not None:
                card += f"- Meta p_spam: {meta_preview:.1%}\n"
        else:
            p_orig = float(decision_details.get('p_spam_original', 0.0))
            p_adj = float(decision_details.get('p_spam_adjusted', p_orig))
            card += f"- p_spam: {p_orig:.1%}"
            if abs(p_orig - p_adj) > 0.001:
                card += f" -> {p_adj:.1%}"
            card += "\n"
            downweights = decision_details.get('applied_downweights', [])
            if downweights:
                dw_str = ", ".join(f"{d['type']}(-{d['multiplier']})" for d in downweights)
                card += f"- Downweights: {dw_str}\n"
            thresholds = decision_details.get('thresholds_used', {})
            if thresholds:
                threshold_text = "- Thresholds: "
                threshold_text += f"N={thresholds.get('notify', 0):.2f}, "
                threshold_text += f"D={thresholds.get('delete', 0):.2f}, "
                threshold_text += f"K={thresholds.get('kick', 0):.2f}\n"
                card += threshold_text
            if decision_details.get('degraded_ctx'):
                card += "- Degraded context: notify+0.05\n"
        card += "\n"
    # Meta Classifier (компактный формат)
    if analysis.meta_proba is not None and analysis.meta_debug:
        card += f"━━━━ <b>META-CLASSIFIER</b> ━━━━\n"
        meta_debug = analysis.meta_debug
        
        # Embeddings (компактно)
        sim_spam_msg = meta_debug.get('sim_spam_msg')
        if sim_spam_msg is not None:
            sim_ham_msg = meta_debug.get('sim_ham_msg')
            delta_msg = meta_debug.get('delta_msg')
            card += f"🧠 E_msg: s={sim_spam_msg:.3f} h={sim_ham_msg:.3f} Δ={delta_msg:.3f}\n"
        
        sim_spam_ctx = meta_debug.get('sim_spam_ctx')
        if sim_spam_ctx is not None:
            sim_ham_ctx = meta_debug.get('sim_ham_ctx')
            delta_ctx = meta_debug.get('delta_ctx')
            card += f"🧠 E_ctx: s={sim_spam_ctx:.3f} h={sim_ham_ctx:.3f} Δ={delta_ctx:.3f}\n"
        
        sim_spam_user = meta_debug.get('sim_spam_user')
        if sim_spam_user is not None:
            sim_ham_user = meta_debug.get('sim_ham_user')
            delta_user = meta_debug.get('delta_user')
            card += f"🧠 E_user: s={sim_spam_user:.3f} h={sim_ham_user:.3f} Δ={delta_user:.3f}\n"
        
        # Top features (top-3 компактно)
        top_features = meta_debug.get('top_features', [])
        if top_features:
            card += f"\n🔝 Top-3: "
            top3 = [f"{fname}({'+' if contrib>0 else ''}{contrib:.2f})" 
                   for fname, contrib in top_features[:3]]
            card += ", ".join(top3) + "\n"
        
        # Whitelist (одной строкой)
        whitelist_hits = meta_debug.get('whitelist_hits', {})
        total_hits = sum(whitelist_hits.values())
        if total_hits > 0:
            card += f"✅ WL: s={whitelist_hits.get('store', 0)} o={whitelist_hits.get('order', 0)} b={whitelist_hits.get('brand', 0)}\n"
        
        # Context flags
        context_flags = meta_debug.get('context_flags')
        if context_flags:
            flags_active = [k for k, v in context_flags.items() if v]
            if flags_active:
                card += f"🚩 {', '.join(flags_active)}\n"
        
        # Patterns (компактно, max 3)
        patterns = meta_debug.get('patterns', {})
        fired_patterns = [k.replace('has_', '') for k, v in patterns.items() 
                         if k.startswith('has_') and v]
        if fired_patterns:
            card += f"🔍 {', '.join(fired_patterns[:3])}\n"
        
        card += "\n"
    
    card += (
        f"━━━━ <b>ТЕКСТ</b> ━━━━\n"
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
    message_id: int,
    decision_details: dict = None
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
            message_id=message_id,
            decision_details=decision_details
        )
    else:
        # Показываем упрощенную версию
        return format_simple_card(
            spam_id=spam_id,
            user_name=user_name,
            text=text,
            msg_link=msg_link,
            analysis=analysis,
            action=action,
            decision_details=decision_details
        )
