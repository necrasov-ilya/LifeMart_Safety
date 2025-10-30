"""
services/policy.py
────────────────────────────────────────────────────────
Policy Engine для контекстного анализа спама с тремя режимами работы.

РЕЖИМЫ:
- manual: только NOTIFY при p_spam ≥ META_NOTIFY, DELETE/KICK запрещены
- semi-auto: NOTIFY + DELETE разрешены, KICK запрещён
- auto: все действия разрешены при соответствующих порогах

ПОНИЖАЮЩИЕ МНОЖИТЕЛИ:
Применяются ПЕРЕД сравнением с порогами:
- is_channel_announcement -> META_DOWNWEIGHT_ANNOUNCEMENT (0.85)
- reply_to_staff -> META_DOWNWEIGHT_REPLY_TO_STAFF (0.90)
- whitelist_hits > 0 -> META_DOWNWEIGHT_WHITELIST (0.85)
 - brand_hits > 0 -> ДОПОЛНИТЕЛЬНО META_DOWNWEIGHT_BRAND (усиливает влияние брендов)

Множители накладываются мультипликативно.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from config.config import settings
from core.types import Action, AnalysisResult
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class PolicyEngine:
    """
    Policy Engine с тремя режимами работы и понижающими множителями.
    
    ВАЖНО:
    - Downweights применяются ПЕРЕД сравнением с порогами
    - Graceful degradation: если degraded_ctx=True, поднимаем META_NOTIFY на +0.05
    - Все примененные downweights записываются в applied_downweights
    """
    
    def __init__(self):
        self.policy_mode = settings.POLICY_MODE
        
        # Пороги для META
        self.meta_notify = settings.META_NOTIFY
        self.meta_delete = settings.META_DELETE
        self.meta_kick = settings.META_KICK
        
        # Понижающие множители
        self.downweight_announcement = settings.META_DOWNWEIGHT_ANNOUNCEMENT
        self.downweight_reply_to_staff = settings.META_DOWNWEIGHT_REPLY_TO_STAFF
        self.downweight_whitelist = settings.META_DOWNWEIGHT_WHITELIST
        self.downweight_brand = settings.META_DOWNWEIGHT_BRAND
        # Legacy thresholds (keyword-first hysteresis)
        self.legacy_keyword_threshold = getattr(settings, "LEGACY_KEYWORD_THRESHOLD", 0.60)
        self.legacy_tfidf_threshold = getattr(settings, "LEGACY_TFIDF_THRESHOLD", self.meta_notify)
        
        LOGGER.info(
            f"PolicyEngine initialized: mode={self.policy_mode}, "
            f"thresholds=(notify={self.meta_notify}, delete={self.meta_delete}, kick={self.meta_kick})"
        )
    
    def decide_action(self, analysis: AnalysisResult) -> Tuple[Action, Dict]:
        """
        Принять решение на основе p_spam и контекстных факторов.
        
        Returns:
            (action, decision_details)
            
        decision_details содержит:
        - p_spam_original: исходная вероятность
        - p_spam_adjusted: скорректированная вероятность после downweights
        - applied_downweights: список примененных множителей
        - thresholds_used: пороги, использованные для решения
        - degraded_ctx: флаг деградации контекста
        - action_reason: текстовое объяснение
        """
        # Проверяем наличие meta_proba
        if self.policy_mode == "legacy-manual":
            return self._decide_legacy_manual(analysis)

        if analysis.meta_proba is None:
            LOGGER.warning("meta_proba is None, falling back to aggregate filter scores")
            return self._decide_without_meta(analysis)
        
        p_spam_original = analysis.meta_proba
        
        # Применяем понижающие множители
        p_spam_adjusted, applied_downweights = self._apply_downweights(analysis)
        
        # Graceful degradation для контекста
        thresholds_adjusted = self._adjust_thresholds_for_degradation(analysis)
        
        # Выбираем действие в зависимости от режима
        action = self._select_action(
            p_spam_adjusted,
            thresholds_adjusted,
            self.policy_mode
        )
        
        # Формируем объяснение
        reason = self._explain_action(
            action,
            p_spam_original,
            p_spam_adjusted,
            applied_downweights,
            thresholds_adjusted
        )
        
        decision_details = {
            'policy_mode': self.policy_mode,
            'p_spam_original': float(p_spam_original),
            'p_spam_adjusted': float(p_spam_adjusted),
            'applied_downweights': applied_downweights,
            'thresholds_used': thresholds_adjusted,
            'degraded_ctx': getattr(analysis, 'degraded_ctx', False),
            'action_reason': reason
        }
        
        LOGGER.info(
            f"Decision: {action.name} | p_spam: {p_spam_original:.3f} -> {p_spam_adjusted:.3f} | "
            f"mode={self.policy_mode} | downweights={len(applied_downweights)}"
        )
        
        return action, decision_details
    
    def _decide_legacy_manual(self, analysis: AnalysisResult) -> Tuple[Action, Dict]:
        """Fallback logic that emulates legacy keyword + TF-IDF flow."""
        keyword_score = analysis.keyword_result.score if analysis.keyword_result else 0.0
        tfidf_score = analysis.tfidf_result.score if analysis.tfidf_result else 0.0

        action = Action.APPROVE
        trigger = None
        trigger_score = 0.0
        trigger_threshold = 0.0

        if keyword_score >= self.legacy_keyword_threshold:
            action = Action.NOTIFY
            trigger = "keyword"
            trigger_score = keyword_score
            trigger_threshold = self.legacy_keyword_threshold
        elif tfidf_score >= self.legacy_tfidf_threshold:
            action = Action.NOTIFY
            trigger = "tfidf"
            trigger_score = tfidf_score
            trigger_threshold = self.legacy_tfidf_threshold

        if trigger:
            reason = f"{trigger} score {trigger_score:.2f} >= {trigger_threshold:.2f}"
        else:
            reason = "legacy thresholds not exceeded"

        meta_preview = None
        meta_thresholds = {
            "notify": self.meta_notify,
            "delete": self.meta_delete,
            "kick": self.meta_kick
        }

        if analysis.meta_proba is not None:
            meta_value = float(analysis.meta_proba)
            meta_action = self._select_action(
                meta_value,
                meta_thresholds,
                "auto"
            )
            meta_preview = {
                "p_spam": meta_value,
                "recommended_action": meta_action.value,
                "thresholds": meta_thresholds
            }

        decision_details = {
            "policy_mode": self.policy_mode,
            "legacy_mode": True,
            "legacy_action": action.value,
            "legacy_keyword_score": float(keyword_score),
            "legacy_tfidf_score": float(tfidf_score),
            "legacy_keyword_threshold": float(self.legacy_keyword_threshold),
            "legacy_tfidf_threshold": float(self.legacy_tfidf_threshold),
            "legacy_trigger": trigger,
            "legacy_trigger_score": float(trigger_score) if trigger else None,
            "legacy_trigger_threshold": float(trigger_threshold) if trigger else None,
            "action_reason": reason,
            "applied_downweights": [],
            "legacy_thresholds": {
                "keyword": float(self.legacy_keyword_threshold),
                "tfidf": float(self.legacy_tfidf_threshold)
            },
            "thresholds_used": meta_thresholds,
        }

        if meta_preview:
            decision_details["meta_preview"] = meta_preview
            decision_details["p_spam_original"] = meta_preview["p_spam"]
            decision_details["p_spam_adjusted"] = meta_preview["p_spam"]

        LOGGER.info(
            "Legacy policy decision: %s | keyword=%.3f | tfidf=%.3f | trigger=%s",
            action.name,
            keyword_score,
            tfidf_score,
            trigger or "none"
        )

        return action, decision_details

    def _decide_without_meta(self, analysis: AnalysisResult) -> Tuple[Action, Dict]:
        """
        Резервное решение, если метаклассификатор недоступен.
        Используем агрегированный скор keyword / TF-IDF / embedding и текущие пороги.
        """
        # Возвращаемся к наследуемой логике, чтобы старый контур видел привычные данные.
        legacy_action, legacy_details = self._decide_legacy_manual(analysis)
        decision_details = dict(legacy_details)
        if self.policy_mode != "legacy-manual":
            decision_details["legacy_mode"] = False
        decision_details["fallback_meta"] = True
        decision_details["degraded_ctx"] = getattr(analysis, "degraded_ctx", False)

        LOGGER.info(
            "Fallback decision without meta score: %s | keyword=%.3f | tfidf=%.3f | mode=%s",
            legacy_action.name,
            analysis.keyword_result.score if analysis.keyword_result else 0.0,
            analysis.tfidf_result.score if analysis.tfidf_result else 0.0,
            self.policy_mode,
        )

        return legacy_action, decision_details

    def _apply_downweights(self, analysis: AnalysisResult) -> Tuple[float, List[Dict]]:
        """
        Применяет понижающие множители к p_spam.
        
        Returns:
            (adjusted_p_spam, list_of_applied_downweights)
        """
        p_spam = analysis.meta_proba
        applied = []
        
        metadata = analysis.metadata
        if metadata is None:
            return p_spam, applied
        
        # 1. Channel announcement (репосты из канала-донора)
        if metadata.is_channel_announcement:
            p_spam *= self.downweight_announcement
            applied.append({
                'type': 'is_channel_announcement',
                'multiplier': self.downweight_announcement,
                'reason': 'Сообщение из подключенного канала'
            })
        
        # 2. Reply to staff (ответ админу/модератору)
        if metadata.reply_to_staff:
            p_spam *= self.downweight_reply_to_staff
            applied.append({
                'type': 'reply_to_staff',
                'multiplier': self.downweight_reply_to_staff,
                'reason': 'Ответ на сообщение персонала'
            })
        
        # 3. Whitelist hits (упоминание терминов магазина/заказа/бренда)
        if hasattr(analysis, 'meta_debug') and analysis.meta_debug:
            whitelist_hits = analysis.meta_debug.get('whitelist_hits', {})
            total_hits = (
                whitelist_hits.get('store', 0) +
                whitelist_hits.get('order', 0) +
                whitelist_hits.get('brand', 0)
            )
            
            if total_hits > 0:
                p_spam *= self.downweight_whitelist
                applied.append({
                    'type': 'whitelist_hits',
                    'multiplier': self.downweight_whitelist,
                    'reason': f'Обнаружено {total_hits} совпадений whitelist-терминов',
                    'hits': whitelist_hits
                })

            # Дополнительное усиление по брендам
            brand_hits = whitelist_hits.get('brand', 0)
            if brand_hits > 0:
                p_spam *= self.downweight_brand
                applied.append({
                    'type': 'brand_hits',
                    'multiplier': self.downweight_brand,
                    'reason': f'Упоминание брендов ({brand_hits})'
                })
        
        return p_spam, applied
    
    def _adjust_thresholds_for_degradation(self, analysis: AnalysisResult) -> Dict:
        """
        Поднимает порог META_NOTIFY на +0.05 если контекст деградирован.
        
        Это снижает чувствительность при отсутствии E_ctx,
        чтобы не шумить при неполной информации.
        """
        thresholds = {
            'notify': self.meta_notify,
            'delete': self.meta_delete,
            'kick': self.meta_kick
        }
        
        if getattr(analysis, 'degraded_ctx', False):
            thresholds['notify'] += 0.05
            LOGGER.debug("Degraded context detected, raising META_NOTIFY by +0.05")
        
        return thresholds
    
    def _select_action(
        self,
        p_spam: float,
        thresholds: Dict,
        mode: str
    ) -> Action:
        """
        Выбирает действие в зависимости от режима и порогов.
        
        РЕЖИМЫ:
        - manual: только NOTIFY, DELETE/KICK запрещены
        - semi-auto: NOTIFY + DELETE разрешены, KICK запрещён
        - auto: все действия разрешены
        """
        # manual: только уведомления
        if mode == "manual":
            if p_spam >= thresholds['notify']:
                return Action.NOTIFY
            return Action.APPROVE
        
        # semi-auto: уведомления + удаление
        elif mode == "semi-auto":
            if p_spam >= thresholds['delete']:
                return Action.DELETE
            elif p_spam >= thresholds['notify']:
                return Action.NOTIFY
            return Action.APPROVE
        
        # auto: все действия
        elif mode == "auto":
            if p_spam >= thresholds['kick']:
                return Action.KICK
            elif p_spam >= thresholds['delete']:
                return Action.DELETE
            elif p_spam >= thresholds['notify']:
                return Action.NOTIFY
            return Action.APPROVE
        
        else:
            LOGGER.error(f"Unknown policy mode: {mode}, defaulting to manual")
            if p_spam >= thresholds['notify']:
                return Action.NOTIFY
            return Action.APPROVE
    
    def _explain_action(
        self,
        action: Action,
        p_spam_original: float,
        p_spam_adjusted: float,
        applied_downweights: List[Dict],
        thresholds: Dict
    ) -> str:
        """Генерирует текстовое объяснение решения."""
        
        # Базовое объяснение по действию
        if action == Action.KICK:
            base = f"Критический спам (p={p_spam_adjusted:.2%}). Автобан."
        elif action == Action.DELETE:
            base = f"Явный спам (p={p_spam_adjusted:.2%}). Автоудаление."
        elif action == Action.NOTIFY:
            base = f"Подозрительное сообщение (p={p_spam_adjusted:.2%}). Требуется проверка."
        else:
            base = f"Легитимное сообщение (p={p_spam_adjusted:.2%})."
        
        # Добавляем инфо о понижении
        if applied_downweights:
            downweight_info = ", ".join([
                f"{dw['type']} (×{dw['multiplier']})"
                for dw in applied_downweights
            ])
            base += f" Применены понижения: {downweight_info}."
        
        return base
    
    def get_thresholds(self) -> Dict:
        """Возвращает текущие пороги (для отладки/UI)."""
        return {
            'mode': self.policy_mode,
            'notify': self.meta_notify,
            'delete': self.meta_delete,
            'kick': self.meta_kick,
            'downweights': {
                'announcement': self.downweight_announcement,
                'reply_to_staff': self.downweight_reply_to_staff,
                'whitelist': self.downweight_whitelist,
                'brand': self.downweight_brand
            }
        }
