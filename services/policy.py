"""
services/policy.py
────────────────────────────────────────────────────────
Policy Engine с поддержкой мета-классификатора.

ОБНОВЛЕНО: Приоритет отдается p_spam от MetaClassifier,
если USE_META_CLASSIFIER=true и артефакты загружены.
Фоллбэк на взвешенную агрегацию если мета-классификатор не готов.
"""

from __future__ import annotations

from config.runtime import runtime_config
from core.types import Action, AnalysisResult
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class PolicyEngine:
    """
    Policy Engine использует runtime_config для динамической конфигурации.
    
    ИЗМЕНЕНО: Теперь приоритет на meta_proba от MetaClassifier.
    """
    
    def decide_action(self, analysis: AnalysisResult) -> Action:
        """
        Принять решение на основе текущей конфигурации.
        
        Если USE_META_CLASSIFIER=true и есть meta_proba:
            - Использует META_THRESHOLD_HIGH/MEDIUM для решения
        Иначе:
            - Фоллбэк на старую взвешенную агрегацию (average_score)
        """
        # Проверяем доступность мета-классификатора
        if runtime_config.use_meta_classifier and analysis.meta_proba is not None:
            LOGGER.debug(f"Using MetaClassifier: p_spam={analysis.meta_proba:.3f}")
            return self._meta_mode(analysis.meta_proba)
        else:
            # Фоллбэк на старую логику
            if not runtime_config.use_meta_classifier:
                LOGGER.debug("USE_META_CLASSIFIER=false, using legacy mode")
            else:
                LOGGER.warning("MetaClassifier not ready, falling back to legacy mode")
            
            avg_score = analysis.average_score
            max_score = analysis.max_score
            all_high = analysis.all_high
            
            mode = runtime_config.policy_mode
            
            if mode == "manual":
                return self._manual_mode(analysis)
            elif mode == "semi-auto":
                return self._semi_auto_mode(avg_score, max_score, all_high)
            else:
                return self._auto_mode(avg_score, max_score, all_high)
    
    def _meta_mode(self, p_spam: float) -> Action:
        """Решение на основе вероятности от MetaClassifier."""
        if p_spam >= runtime_config.meta_threshold_high:
            # Автоматическое удаление/бан
            if p_spam >= 0.95:
                return Action.KICK
            return Action.DELETE
        elif p_spam >= runtime_config.meta_threshold_medium:
            # Отправить модератору
            return Action.NOTIFY
        else:
            # Пропустить
            return Action.APPROVE
    
    def _manual_mode(self, analysis: AnalysisResult) -> Action:
        if analysis.average_score >= runtime_config.notify_threshold:
            return Action.NOTIFY
        return Action.APPROVE
    
    def _semi_auto_mode(self, avg_score: float, max_score: float, all_high: bool) -> Action:
        if all_high and avg_score >= runtime_config.auto_kick_threshold:
            return Action.KICK
        
        if avg_score >= runtime_config.auto_delete_threshold:
            return Action.DELETE
        
        if avg_score >= runtime_config.notify_threshold:
            return Action.NOTIFY
        
        return Action.APPROVE
    
    def _auto_mode(self, avg_score: float, max_score: float, all_high: bool) -> Action:
        if avg_score >= runtime_config.auto_kick_threshold:
            return Action.KICK
        
        if avg_score >= runtime_config.auto_delete_threshold:
            return Action.DELETE
        
        if avg_score >= runtime_config.notify_threshold:
            return Action.NOTIFY
        
        return Action.APPROVE
    
    def explain_decision(self, analysis: AnalysisResult, action: Action) -> str:
        avg = analysis.average_score
        
        if action == Action.KICK:
            return f"Очевидный спам (оценка: {avg:.0%}). Автоматический бан."
        elif action == Action.DELETE:
            return f"Вероятный спам (оценка: {avg:.0%}). Автоматическое удаление."
        elif action == Action.NOTIFY:
            return f"Подозрительное сообщение (оценка: {avg:.0%}). Требуется проверка."
        else:
            return f"Сообщение прошло проверку (оценка: {avg:.0%})."
