from __future__ import annotations

import html

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from config.config import settings
from core.types import Action, AnalysisResult

_ACTION_TITLES: dict[str, str] = {
    Action.APPROVE.value: "Пропустить",
    Action.NOTIFY.value: "Отправить модератору",
    Action.DELETE.value: "Удалить",
    Action.KICK.value: "Заблокировать",
}


def _format_action_title(action: Action | str | None) -> str:
    if action is None:
        return "—"
    value = action.value if isinstance(action, Action) else str(action)
    return _ACTION_TITLES.get(value, value.upper())


def _format_trigger(trigger: str | None) -> str:
    if not trigger:
        return "—"
    mapping = {
        "keyword": "Keyword",
        "tfidf": "TF-IDF",
    }
    return mapping.get(trigger, trigger)


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
    """Формирует краткую карточку для модератора."""
    preview = html.escape(text[:150] + ("…" if len(text) > 150 else ""))

    legacy_mode = bool(decision_details and decision_details.get("legacy_mode"))
    kw_score = None
    tfidf_score = None

    if legacy_mode:
        kw_score = float(decision_details.get("legacy_keyword_score", 0.0) or 0.0)
        tfidf_score = float(decision_details.get("legacy_tfidf_score", 0.0) or 0.0)
        score = decision_details.get("legacy_trigger_score")
        if score is None:
            score = max(kw_score, tfidf_score)
        metric_label = "Оценка старого контура"
    elif decision_details and "p_spam_adjusted" in decision_details:
        score = float(decision_details["p_spam_adjusted"])
        metric_label = "p_spam"
    else:
        score = analysis.average_score
        metric_label = "Average"

    icon, status = {
        Action.KICK: ("🚫", "Автоблокировка"),
        Action.DELETE: ("🗑️", "Автоудаление"),
        Action.NOTIFY: ("📩", "Нужна проверка модератора"),
        Action.APPROVE: ("✅", "Сообщение пропущено"),
    }.get(action, ("ℹ️", "Информация"))

    card_lines: list[str] = [
        f"{icon} <b>Инцидент #{spam_id}</b>",
        f"👤 {html.escape(user_name)}",
        f"📊 {metric_label}: <b>{score:.0%}</b>",
        f"🔗 <a href='{msg_link}'>Открыть сообщение</a>",
        "",
        f"📝 <i>{preview}</i>",
        "",
        f"⚙️ <b>{status}</b>",
    ]

    if legacy_mode:
        trigger = _format_trigger(decision_details.get("legacy_trigger"))
        legacy_action = _format_action_title(decision_details.get("legacy_action", action))
        card_lines.extend([
            "",
            "🕰️ <b>Старый контур</b>",
            f" • Keyword: <b>{kw_score:.0%}</b>",
            f" • TF-IDF: <b>{tfidf_score:.0%}</b>",
            f" • Итог: <b>{legacy_action}</b> (триггер: {trigger})",
        ])

        meta_preview = decision_details.get("meta_preview") if decision_details else None
        if meta_preview:
            meta_action = _format_action_title(meta_preview.get("recommended_action"))
            card_lines.extend([
                "",
                "🚀 <b>Новая система</b>",
                f" • p_spam: <b>{meta_preview.get('p_spam', 0.0):.0%}</b>",
                f" • Рекомендация: <b>{meta_action}</b>",
            ])

    if decision_details and 'policy_mode' in decision_details:
        card_lines.extend([
            "",
            f"🛠️ Режим: <code>{decision_details['policy_mode']}</code>",
        ])

    if action == Action.NOTIFY:
        card_lines.extend([
            "",
            f"ℹ️ /debug {spam_id} — подробности",
        ])
    else:
        card_lines.extend([
            "",
            f"ℹ️ История: /debug {spam_id}",
        ])

    return "\n".join(card_lines)


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
    """Подробная карточка с диагностикой для команды /debug."""
    preview = html.escape(text[:200] + ("…" if len(text) > 200 else ""))

    card_lines: list[str] = [
        f"🧪 <b>Debug #{spam_id}</b>",
        f"👤 {html.escape(user_name)} (<code>{user_id}</code>)",
        f"💬 Chat: <code>{chat_id}</code> | Msg: <code>{message_id}</code>",
        f"🔗 <a href='{msg_link}'>Открыть сообщение</a>",
        "",
    ]

    action_title = _format_action_title(action)

    if decision_details:
        mode = decision_details.get('policy_mode', 'unknown')
        legacy_mode = bool(decision_details.get('legacy_mode'))
        card_lines.extend([
            f"⚙️ Режим политики: <code>{mode}</code>",
            f"🎯 Итоговое действие: <b>{action_title}</b>",
        ])

        if legacy_mode:
            kw_val = float(decision_details.get('legacy_keyword_score', 0.0) or 0.0)
            tfidf_val = float(decision_details.get('legacy_tfidf_score', 0.0) or 0.0)
            trigger = _format_trigger(decision_details.get('legacy_trigger'))
            trigger_score = decision_details.get('legacy_trigger_score')
            legacy_thresholds = decision_details.get('legacy_thresholds', {})
            legacy_action = _format_action_title(decision_details.get('legacy_action'))

            trigger_line = f" • Триггер: {trigger}"
            if trigger_score is not None:
                trigger_line += f" ({trigger_score:.1%})"

            thresholds_line = None
            if legacy_thresholds:
                thresholds_line = (
                    " • Пороги: keyword ≥ {keyword:.2f}, TF-IDF ≥ {tfidf:.2f}".format(
                        keyword=legacy_thresholds.get('keyword', 0.0),
                        tfidf=legacy_thresholds.get('tfidf', 0.0)
                    )
                )

            card_lines.extend([
                "",
                "🕰️ <b>Старый контур</b>",
                f" • Keyword: <b>{kw_val:.1%}</b>",
                f" • TF-IDF: <b>{tfidf_val:.1%}</b>",
                f" • Итог: <b>{legacy_action}</b>",
                trigger_line,
            ])
            if thresholds_line:
                card_lines.append(thresholds_line)

            reason = decision_details.get('action_reason')
            if reason:
                card_lines.append(f" • Пояснение: {reason}")

            meta_preview = decision_details.get('meta_preview')
            if meta_preview:
                meta_action = _format_action_title(meta_preview.get('recommended_action'))
                meta_thresholds = meta_preview.get('thresholds', decision_details.get('thresholds_used', {}))
                card_lines.extend([
                    "",
                    "🚀 <b>Новая система</b>",
                    f" • p_spam: <b>{meta_preview.get('p_spam', 0.0):.1%}</b>",
                    f" • Рекомендация: <b>{meta_action}</b>",
                ])
                if meta_thresholds:
                    card_lines.append(
                        " • Пороги: notify ≥ {notify:.2f}, delete ≥ {delete:.2f}, kick ≥ {kick:.2f}".format(
                            notify=meta_thresholds.get('notify', 0.0),
                            delete=meta_thresholds.get('delete', 0.0),
                            kick=meta_thresholds.get('kick', 0.0)
                        )
                    )
        else:
            p_orig = float(decision_details.get('p_spam_original', 0.0))
            p_adj = float(decision_details.get('p_spam_adjusted', p_orig))
            summary_line = f"📊 p_spam: <b>{p_orig:.1%}</b>"
            if abs(p_orig - p_adj) > 0.001:
                summary_line += f" → <b>{p_adj:.1%}</b>"

            card_lines.extend([
                "",
                summary_line,
            ])

            downweights = decision_details.get('applied_downweights', [])
            if downweights:
                dw_str = ", ".join(f"{d['type']}(-{d['multiplier']})" for d in downweights)
                card_lines.append(f"🔽 Downweights: {dw_str}")

            thresholds = decision_details.get('thresholds_used', {})
            if thresholds:
                card_lines.append(
                    "📏 Thresholds: N={notify:.2f}, D={delete:.2f}, K={kick:.2f}".format(
                        notify=thresholds.get('notify', 0.0),
                        delete=thresholds.get('delete', 0.0),
                        kick=thresholds.get('kick', 0.0)
                    )
                )

            if decision_details.get('degraded_ctx'):
                card_lines.append("⚠️ Деградация: контекст недоступен, notify +0.05")

            reason = decision_details.get('action_reason')
            if reason:
                card_lines.append(f"📝 {reason}")
    else:
        card_lines.append("⚠️ Нет decision_details")

    if analysis.meta_proba is not None and analysis.meta_debug:
        meta_debug = analysis.meta_debug
        card_lines.extend([
            "",
            "🧠 <b>Диагностика мета-классификатора</b>",
        ])

        sim_spam_msg = meta_debug.get('sim_spam_msg')
        if sim_spam_msg is not None:
            sim_ham_msg = meta_debug.get('sim_ham_msg')
            delta_msg = meta_debug.get('delta_msg')
            card_lines.append(
                f" • E_msg: spam={sim_spam_msg:.3f} ham={sim_ham_msg:.3f} Δ={delta_msg:.3f}"
            )

        sim_spam_ctx = meta_debug.get('sim_spam_ctx')
        if sim_spam_ctx is not None:
            sim_ham_ctx = meta_debug.get('sim_ham_ctx')
            delta_ctx = meta_debug.get('delta_ctx')
            card_lines.append(
                f" • E_ctx: spam={sim_spam_ctx:.3f} ham={sim_ham_ctx:.3f} Δ={delta_ctx:.3f}"
            )

        sim_spam_user = meta_debug.get('sim_spam_user')
        if sim_spam_user is not None:
            sim_ham_user = meta_debug.get('sim_ham_user')
            delta_user = meta_debug.get('delta_user')
            card_lines.append(
                f" • E_user: spam={sim_spam_user:.3f} ham={sim_ham_user:.3f} Δ={delta_user:.3f}"
            )

        top_features = meta_debug.get('top_features', [])
        if top_features:
            top_formatted = ", ".join(
                f"{fname}({'+' if contrib > 0 else ''}{contrib:.2f})"
                for fname, contrib in top_features[:3]
            )
            card_lines.append(f" • Top-3: {top_formatted}")

        whitelist_hits = meta_debug.get('whitelist_hits', {})
        if whitelist_hits:
            card_lines.append(
                " • Whitelist: store={store}, order={order}, brand={brand}".format(
                    store=whitelist_hits.get('store', 0),
                    order=whitelist_hits.get('order', 0),
                    brand=whitelist_hits.get('brand', 0)
                )
            )

        context_flags = meta_debug.get('context_flags')
        if context_flags:
            active = [k for k, v in context_flags.items() if v]
            if active:
                card_lines.append(f" • Флаги контекста: {', '.join(active)}")

        patterns = meta_debug.get('patterns', {})
        fired_patterns = [k.replace('has_', '') for k, v in patterns.items() if k.startswith('has_') and v]
        if fired_patterns:
            card_lines.append(f" • Паттерны: {', '.join(fired_patterns[:5])}")

    card_lines.extend([
        "",
        "📝 <b>Текст</b>",
        preview,
    ])

    return "\n".join(card_lines)


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
