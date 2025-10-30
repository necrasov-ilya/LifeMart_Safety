from __future__ import annotations

from typing import List, Optional, Dict, Any
from collections import deque

from telegram import Message

from core.types import AnalysisResult, MessageMetadata, EmbeddingVectors
from filters.base import BaseFilter
from filters.embedding import EmbeddingFilter
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class FilterCoordinator:
    """
    Координатор фильтров с поддержкой контекстного анализа.
    
    НОВОЕ:
    - Сбор истории сообщений по чатам
    - Извлечение метаданных из Telegram Update
    - Формирование капсул (context, user)
    - Параллельный расчет эмбеддингов
    """
    
    def __init__(
        self,
        keyword_filter: BaseFilter,
        tfidf_filter: BaseFilter,
        embedding_filter: EmbeddingFilter | None = None,
        context_history_n: int = 4,
        context_max_tokens: int = 512
    ):
        self.keyword_filter = keyword_filter
        self.tfidf_filter = tfidf_filter
        self.embedding_filter = embedding_filter
        
        self.context_history_n = context_history_n
        self.context_max_tokens = context_max_tokens
        
        # Хранилище истории сообщений по чатам
        # {chat_id: deque([(text, user_id, timestamp), ...])}
        self._chat_history: Dict[int, deque] = {}
        
        # Хранилище истории по пользователям
        # {user_id: deque([text, ...])}
        self._user_history: Dict[int, deque] = {}
    
    def _extract_metadata(self, message: Message) -> MessageMetadata:
        """
        Извлекает метаданные из Telegram Message.
        
        Args:
            message: telegram.Message объект
            
        Returns:
            MessageMetadata с контекстными флагами
        """
        # Базовые данные
        metadata = MessageMetadata(
            message_id=message.message_id,
            user_id=message.from_user.id if message.from_user else 0,
            user_name=message.from_user.full_name if message.from_user else "Unknown",
            chat_id=message.chat_id,
            timestamp=message.date.timestamp() if message.date else 0.0,
        )
        
        # Проверка на ответ (reply)
        is_reply = message.reply_to_message is not None
        reply_to_user_id = None
        reply_to_staff = False
        
        if is_reply and message.reply_to_message.from_user:
            reply_to_user_id = message.reply_to_message.from_user.id
            
            # Проверяем, является ли адресат админом/модератором
            # Пока упрощенно - проверяем по whitelist
            from config.config import settings
            reply_to_staff = reply_to_user_id in settings.WHITELIST_USER_IDS
        
        # Проверка на forward
        is_forwarded = message.forward_from is not None or message.forward_from_chat is not None
        
        # Проверка на админа отправителя
        author_is_admin = False
        if message.from_user:
            from config.config import settings
            author_is_admin = message.from_user.id in settings.WHITELIST_USER_IDS
        
        # Проверка на пост из канала
        is_channel_announcement = (
            message.sender_chat is not None and 
            message.sender_chat.type == "channel"
        )
        
        # Собираем всё вместе
        from dataclasses import replace
        metadata = replace(
            metadata,
            is_reply=is_reply,
            reply_to_user_id=reply_to_user_id,
            reply_to_staff=reply_to_staff,
            is_forwarded=is_forwarded,
            author_is_admin=author_is_admin,
            is_channel_announcement=is_channel_announcement
        )
        
        return metadata
    
    def _update_history(
        self,
        chat_id: int,
        user_id: int,
        text: str,
        timestamp: float
    ) -> None:
        """Обновляет историю сообщений для чата и пользователя"""
        # История чата
        if chat_id not in self._chat_history:
            self._chat_history[chat_id] = deque(maxlen=self.context_history_n * 3)  # Буфер
        
        self._chat_history[chat_id].append((text, user_id, timestamp))
        
        # История пользователя
        if user_id not in self._user_history:
            self._user_history[user_id] = deque(maxlen=10)  # Последние 10 сообщений
        
        self._user_history[user_id].append(text)
    
    def _get_chat_context(
        self,
        chat_id: int,
        current_user_id: int,
        n: int
    ) -> List[str]:
        """
        Получает последние N сообщений из чата (исключая текущего пользователя).
        
        Returns:
            Список текстов (от новых к старым)
        """
        if chat_id not in self._chat_history:
            return []
        
        history = list(self._chat_history[chat_id])
        
        # Фильтруем: не включаем сообщения текущего пользователя
        filtered = [
            text for text, uid, _ in reversed(history)
            if uid != current_user_id
        ]
        
        return filtered[:n]
    
    def _get_user_context(self, user_id: int, k: int = 5) -> List[str]:
        """
        Получает последние K сообщений пользователя.
        
        Returns:
            Список текстов (от новых к старым)
        """
        if user_id not in self._user_history:
            return []
        
        history = list(self._user_history[user_id])
        return list(reversed(history))[:k]
    
    async def analyze(
        self,
        text: str,
        message: Optional[Message] = None,
        enable_context: bool = True,
        enable_user_embedding: bool = False
    ) -> AnalysisResult:
        """
        Анализ сообщения с контекстом.
        
        Args:
            text: текст сообщения
            message: telegram.Message для извлечения метаданных (опционально)
            enable_context: использовать контекстную капсулу
            enable_user_embedding: вычислять E_user (может быть медленно)
            
        Returns:
            AnalysisResult с расширенными полями (metadata, capsules, vectors)
        """
        # Шаг 1: Извлекаем метаданные
        metadata = None
        if message:
            metadata = self._extract_metadata(message)
            LOGGER.debug(
                f"Metadata: reply_to_staff={metadata.reply_to_staff}, "
                f"is_forwarded={metadata.is_forwarded}, "
                f"author_is_admin={metadata.author_is_admin}, "
                f"is_channel_announcement={metadata.is_channel_announcement}"
            )
        
        # Шаг 2: Запускаем простые фильтры (Keyword, TF-IDF)
        keyword_result = await self.keyword_filter.analyze(text)
        LOGGER.debug(f"Keyword: {keyword_result.score:.2f}")
        
        tfidf_result = await self.tfidf_filter.analyze(text)
        LOGGER.debug(f"TF-IDF: {tfidf_result.score:.2f}")
        
        # Шаг 3: Формируем капсулы для эмбеддингов
        context_capsule = None
        user_capsule = None
        embedding_vectors = None
        embedding_result = None
        emb_debug: Dict[str, Any] = {}
        degraded_ctx_flag = False
        
        if self.embedding_filter and self.embedding_filter.is_ready() and metadata:
            from utils.textprep import (
                normalize_entities,
                build_context_capsule,
                build_user_capsule
            )
            
            # Капсула текущего сообщения (E_msg)
            message_capsule = f"passage: {normalize_entities(text)}"
            
            # Контекстная капсула (E_ctx)
            if enable_context:
                chat_history = self._get_chat_context(
                    metadata.chat_id,
                    metadata.user_id,
                    self.context_history_n
                )
                
                context_capsule = build_context_capsule(
                    message=text,
                    history=chat_history,
                    metadata={
                        'reply_to_staff': metadata.reply_to_staff,
                        'is_forwarded': metadata.is_forwarded,
                        'author_is_admin': metadata.author_is_admin,
                        'is_channel_announcement': metadata.is_channel_announcement
                    },
                    max_chars=self.context_max_tokens * 4  # ~4 char/token
                )
                
                LOGGER.debug(f"Context capsule: {len(context_capsule)} chars")
            
            # User-капсула (E_user) - опционально
            if enable_user_embedding:
                user_history = self._get_user_context(metadata.user_id, k=5)
                if user_history:
                    user_capsule = build_user_capsule(
                        last_k_msgs=user_history,
                        max_chars=512
                    )
                    LOGGER.debug(f"User capsule: {len(user_capsule)} chars")
            
            # Вычисляем эмбеддинги (пакетно)
            embedding_vectors, emb_debug = await self.embedding_filter.compute_embeddings_multi(
                message_capsule=message_capsule,
                context_capsule=context_capsule,
                user_capsule=user_capsule,
                user_id=metadata.user_id if metadata else None,
                enable_user_cache=True
            )
            degraded_ctx_flag = emb_debug.get('degraded_ctx', False)
            
            LOGGER.debug(
                f"Embeddings: E_msg={'✓' if embedding_vectors.E_msg else '✗'}, "
                f"E_ctx={'✓' if embedding_vectors.E_ctx else '✗'}, "
                f"E_user={'✓' if embedding_vectors.E_user else '✗'}, "
                f"degraded_ctx={emb_debug.get('degraded_ctx', False)}"
            )
            
            # Legacy FilterResult для обратной совместимости
            embedding_result = self.embedding_filter.build_result_from_vectors(
                vectors=embedding_vectors,
                debug_info=emb_debug
            )
        
        # Шаг 4: Обновляем историю (ПОСЛЕ анализа)
        if metadata:
            self._update_history(
                chat_id=metadata.chat_id,
                user_id=metadata.user_id,
                text=text,
                timestamp=metadata.timestamp
            )
        
        # Шаг 5: Собираем результат
        result = AnalysisResult(
            keyword_result=keyword_result,
            tfidf_result=tfidf_result,
            embedding_result=embedding_result,
            metadata=metadata,
            context_capsule=context_capsule,
            user_capsule=user_capsule,
            embedding_vectors=embedding_vectors,
            degraded_ctx=degraded_ctx_flag
        )
        
        LOGGER.info(
            f"Analysis complete: avg={result.average_score:.2f}, "
            f"metadata={'✓' if metadata else '✗'}, "
            f"context={'✓' if context_capsule else '✗'}"
        )
        
        return result
    
    def is_ready(self) -> bool:
        return (
            self.keyword_filter.is_ready() and
            self.tfidf_filter.is_ready()
        )
    
    def get_history_stats(self) -> Dict[str, Any]:
        """Статистика хранилищ истории"""
        return {
            "chat_history_size": len(self._chat_history),
            "user_history_size": len(self._user_history),
            "total_messages": sum(len(h) for h in self._chat_history.values()),
        }
