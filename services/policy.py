from __future__ import annotations

from config.runtime import runtime_config
from core.types import Action, AnalysisResult
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class PolicyEngine:
    """Policy Engine использует runtime_config для динамической конфигурации"""
    
    def decide_action(self, analysis: AnalysisResult) -> Action:
        """Принять решение на основе текущей конфигурации"""
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
