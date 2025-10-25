from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


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
class MessageMetadata:
    """Метаданные сообщения для контекстного анализа"""
    message_id: int
    user_id: int
    user_name: str
    chat_id: int
    timestamp: float
    is_reply: bool = False
    reply_to_user_id: int | None = None
    reply_to_staff: bool = False  # Ответ модератору/админу
    is_forwarded: bool = False
    author_is_admin: bool = False
    is_channel_announcement: bool = False  # Пост из канала


@dataclass(frozen=True, slots=True)
class EmbeddingVectors:
    """Векторы эмбеддингов для разных уровней контекста"""
    E_msg: List[float] | None = None  # Текущее сообщение
    E_ctx: List[float] | None = None  # Контекстная капсула (сообщение + история)
    E_user: List[float] | None = None  # Капсула последних сообщений пользователя


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    keyword_result: FilterResult
    tfidf_result: FilterResult
    embedding_result: FilterResult | None
    meta_proba: float | None = None  # Вероятность спама от MetaClassifier
    meta_debug: dict | None = None   # Отладочная информация от MetaClassifier
    
    # Новые поля для контекстного анализа
    metadata: MessageMetadata | None = None
    context_capsule: str | None = None  # Нормализованная контекстная капсула
    user_capsule: str | None = None     # Капсула истории пользователя
    embedding_vectors: EmbeddingVectors | None = None
    applied_downweights: List[str] = field(default_factory=list)  # Примененные множители
    degraded_ctx: bool = False
    
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
