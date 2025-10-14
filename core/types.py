from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    APPROVE = "approve"
    NOTIFY = "notify"
    DELETE = "delete"
    KICK = "kick"


@dataclass(frozen=True, slots=True)
class FilterResult:
    filter_name: str
    score: float
    confidence: float = 1.0
    details: dict[str, any] | None = None


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    keyword_result: FilterResult
    tfidf_result: FilterResult
    embedding_result: FilterResult | None
    meta_proba: float | None = None  # Вероятность спама от MetaClassifier
    meta_debug: dict | None = None   # Отладочная информация от MetaClassifier
    
    @property
    def average_score(self) -> float:
        """Взвешенная оценка с ПРИОРИТЕТОМ на embedding модель"""
        # ВАЖНО: Если есть meta_proba - она игнорируется здесь,
        # т.к. используется PolicyEngine напрямую.
        # Это свойство сохранено для фоллбэка на старую логику.
        
        # Проверяем доступность embedding и его уверенность
        use_embedding = (
            self.embedding_result 
            and self.embedding_result.score != 0.5  # не дефолтное значение
            and self.embedding_result.confidence > 0.0  # не ошибка
        )
        
        if use_embedding:
            # EMBEDDING - ГЛАВНЫЙ ФИЛЬТР (50%)
            # Сравнивает с реальными примерами спама, понимает контекст
            # TF-IDF - вспомогательный (30%) - статистика из датасета
            # Keyword - точечный (20%) - явные паттерны
            return (
                self.embedding_result.score * 0.50 +
                self.tfidf_result.score * 0.30 +
                self.keyword_result.score * 0.20
            )
        else:
            # Без embedding: TF-IDF 60%, Keyword 40%
            return (
                self.tfidf_result.score * 0.60 +
                self.keyword_result.score * 0.40
            )
    
    @property
    def max_score(self) -> float:
        scores = [self.keyword_result.score, self.tfidf_result.score]
        if self.embedding_result:
            scores.append(self.embedding_result.score)
        return max(scores)
    
    @property
    def all_high(self) -> bool:
        threshold = 0.7
        high = self.keyword_result.score >= threshold and self.tfidf_result.score >= threshold
        if self.embedding_result:
            high = high and self.embedding_result.score >= threshold
        return high
