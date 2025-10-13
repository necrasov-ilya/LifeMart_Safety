from __future__ import annotations

from core.types import Action, AnalysisResult
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class PolicyEngine:
    def __init__(
        self,
        mode: str = "semi-auto",
        auto_delete_threshold: float = 0.85,
        auto_kick_threshold: float = 0.95,
        notify_threshold: float = 0.5
    ):
        self.mode = mode.lower()
        self.auto_delete_threshold = auto_delete_threshold
        self.auto_kick_threshold = auto_kick_threshold
        self.notify_threshold = notify_threshold
        
        if self.mode not in ("manual", "semi-auto", "auto"):
            LOGGER.warning(f"Unknown mode '{mode}', using 'semi-auto'")
            self.mode = "semi-auto"
    
    def decide_action(self, analysis: AnalysisResult) -> Action:
        avg_score = analysis.average_score
        max_score = analysis.max_score
        all_high = analysis.all_high
        
        if self.mode == "manual":
            return self._manual_mode(analysis)
        elif self.mode == "semi-auto":
            return self._semi_auto_mode(avg_score, max_score, all_high)
        else:
            return self._auto_mode(avg_score, max_score, all_high)
    
    def _manual_mode(self, analysis: AnalysisResult) -> Action:
        if analysis.average_score >= self.notify_threshold:
            return Action.NOTIFY
        return Action.APPROVE
    
    def _semi_auto_mode(self, avg_score: float, max_score: float, all_high: bool) -> Action:
        if all_high and avg_score >= self.auto_kick_threshold:
            return Action.KICK
        
        if avg_score >= self.auto_delete_threshold:
            return Action.DELETE
        
        if avg_score >= self.notify_threshold:
            return Action.NOTIFY
        
        return Action.APPROVE
    
    def _auto_mode(self, avg_score: float, max_score: float, all_high: bool) -> Action:
        if avg_score >= self.auto_kick_threshold:
            return Action.KICK
        
        if avg_score >= self.auto_delete_threshold:
            return Action.DELETE
        
        if avg_score >= self.notify_threshold:
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
