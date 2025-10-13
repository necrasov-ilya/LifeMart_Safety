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
    
    @property
    def average_score(self) -> float:
        scores = [self.keyword_result.score, self.tfidf_result.score]
        if self.embedding_result:
            scores.append(self.embedding_result.score)
        return sum(scores) / len(scores)
    
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
