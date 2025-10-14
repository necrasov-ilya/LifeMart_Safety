from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.types import FilterResult, EmbeddingVectors
from filters.base import BaseFilter
from utils.logger import get_logger

LOGGER = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# LRU КЭШИРОВАНИЕ USER ЭМБЕДДИНГОВ
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CachedEmbedding:
    """Закэшированный эмбеддинг с меткой времени"""
    vector: List[float]
    timestamp: datetime


class EmbeddingCache:
    """LRU-кэш для user эмбеддингов с TTL"""
    
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 10):
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self._cache: OrderedDict[int, CachedEmbedding] = OrderedDict()
    
    def get(self, user_id: int) -> Optional[List[float]]:
        """Получить закэшированный эмбеддинг если он валиден"""
        if user_id not in self._cache:
            return None
        
        cached = self._cache[user_id]
        
        # Проверка TTL
        if datetime.now() - cached.timestamp > self.ttl:
            LOGGER.debug(f"Cache expired for user {user_id}")
            del self._cache[user_id]
            return None
        
        # Обновляем позицию в LRU (перемещаем в конец)
        self._cache.move_to_end(user_id)
        
        LOGGER.debug(f"Cache HIT for user {user_id}")
        return cached.vector
    
    def set(self, user_id: int, vector: List[float]) -> None:
        """Сохранить эмбеддинг в кэш"""
        # Если уже есть - обновляем
        if user_id in self._cache:
            del self._cache[user_id]
        
        # Проверка размера кэша (выбрасываем самый старый)
        if len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            LOGGER.debug(f"Cache EVICT user {oldest_key} (LRU)")
            del self._cache[oldest_key]
        
        self._cache[user_id] = CachedEmbedding(
            vector=vector,
            timestamp=datetime.now()
        )
        LOGGER.debug(f"Cache SET for user {user_id}")
    
    def invalidate(self, user_id: int) -> None:
        """Инвалидировать кэш для пользователя (новое сообщение)"""
        if user_id in self._cache:
            del self._cache[user_id]
            LOGGER.debug(f"Cache INVALIDATE for user {user_id}")
    
    def size(self) -> int:
        """Текущий размер кэша"""
        return len(self._cache)


# ═══════════════════════════════════════════════════════════════════
# ПРОВАЙДЕРЫ ЭМБЕДДИНГОВ
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# ПРОВАЙДЕРЫ ЭМБЕДДИНГОВ
# ═══════════════════════════════════════════════════════════════════

class EmbeddingProvider(ABC):
    """Абстрактный провайдер эмбеддингов с поддержкой пакетного расчета."""
    
    @abstractmethod
    async def get_embeddings_batch(
        self,
        texts: List[str],
        timeout_ms: Optional[int] = None
    ) -> Tuple[List[Optional[List[float]]], str]:
        """
        Получить эмбеддинги для нескольких текстов.
        
        Args:
            texts: список текстов для эмбеддинга
            timeout_ms: таймаут в миллисекундах (опционально)
            
        Returns:
            (embeddings, status_message)
            embeddings - список векторов (или None при ошибке)
            status_message - детали выполнения
        """
        pass


class OllamaProvider(EmbeddingProvider):
    """
    Провайдер для Ollama API с поддержкой пакетного расчета.
    
    НОВОЕ: Batch processing для E_msg/E_ctx/E_user за один раз.
    """
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        LOGGER.info(f"Initialized OllamaProvider with model: {model}")
    
    async def get_embeddings_batch(
        self,
        texts: List[str],
        timeout_ms: Optional[int] = None
    ) -> Tuple[List[Optional[List[float]]], str]:
        """
        Получает эмбеддинги для нескольких текстов параллельно.
        
        Args:
            texts: список подготовленных текстов (с префиксом passage:)
            timeout_ms: таймаут для каждого запроса в миллисекундах
            
        Returns:
            (embeddings_list, status_message)
            embeddings_list - список векторов (None если ошибка для конкретного текста)
        """
        try:
            import httpx
        except ImportError:
            LOGGER.error("httpx not installed")
            return [None] * len(texts), "httpx not installed"
        
        if not texts:
            return [], "No texts provided"
        
        timeout_sec = (timeout_ms / 1000.0) if timeout_ms else 30.0
        start_time = time.time()
        
        # Параллельный запрос для всех текстов
        async def fetch_one(text: str, idx: int) -> Tuple[int, Optional[List[float]], str]:
            """Получить эмбеддинг для одного текста"""
            try:
                async with httpx.AsyncClient(timeout=timeout_sec) as client:
                    response = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text}
                    )
                    
                    if response.status_code != 200:
                        error_msg = f"API error {response.status_code}"
                        LOGGER.error(f"Ollama embedding #{idx}: {error_msg}")
                        return idx, None, error_msg
                    
                    embedding = response.json()["embedding"]
                    LOGGER.debug(f"Ollama embedding #{idx}: {len(embedding)} dims")
                    return idx, embedding, "OK"
            
            except asyncio.TimeoutError:
                LOGGER.error(f"Ollama embedding #{idx}: timeout after {timeout_sec}s")
                return idx, None, "timeout"
            except Exception as e:
                LOGGER.error(f"Ollama embedding #{idx}: {e}")
                return idx, None, str(e)
        
        # Запускаем все запросы параллельно
        tasks = [fetch_one(text, i) for i, text in enumerate(texts)]
        results = await asyncio.gather(*tasks)
        
        # Сортируем результаты по индексу
        sorted_results = sorted(results, key=lambda x: x[0])
        embeddings = [emb for _, emb, _ in sorted_results]
        statuses = [status for _, _, status in sorted_results]
        
        elapsed = time.time() - start_time
        success_count = sum(1 for emb in embeddings if emb is not None)
        
        status_msg = (
            f"Batch: {success_count}/{len(texts)} OK, "
            f"{elapsed:.2f}s total"
        )
        
        LOGGER.info(status_msg)
        
        return embeddings, status_msg
    
    async def get_embedding(self, text: str) -> tuple[list[float] | None, str]:
        """
        LEGACY: Получает один эмбеддинг (обертка над batch).
        
        Args:
            text: подготовленный текст (с префиксом passage:)
            
        Returns:
            (embedding_vector, status_message)
        """
        embeddings, status = await self.get_embeddings_batch([text])
        if embeddings and embeddings[0]:
            return embeddings[0], status
        return None, status
    
    async def get_spam_score(self, text: str) -> tuple[float, str]:
        """DEPRECATED: используйте get_embeddings_batch()"""
        LOGGER.warning("OllamaProvider.get_spam_score() deprecated")
        return 0.5, "Use get_embeddings_batch()"


class LocalModelProvider(EmbeddingProvider):
    """Заглушка для локальных моделей"""
    def __init__(self, model_path: Path):
        self.model_path = model_path
        LOGGER.warning("LocalModelProvider not implemented yet")
    
    async def get_embeddings_batch(
        self,
        texts: List[str],
        timeout_ms: Optional[int] = None
    ) -> Tuple[List[Optional[List[float]]], str]:
        LOGGER.warning("Local model not implemented")
        return [None] * len(texts), "Not implemented"


# ═══════════════════════════════════════════════════════════════════
# EMBEDDING FILTER С МУЛЬТИ-ЭМБЕДДИНГОМ И КЭШЕМ
# ═══════════════════════════════════════════════════════════════════

class EmbeddingFilter(BaseFilter):
    """
    Фильтр эмбеддингов с поддержкой:
    - Пакетного расчета E_msg/E_ctx/E_user
    - LRU-кэша для user-эмбеддингов
    - Graceful degradation при таймаутах
    """
    
    def __init__(
        self,
        mode: str = "ollama",
        api_key: str | None = None,
        model_path: Path | str | None = None,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
        cache_ttl_minutes: int = 10,
        timeout_ms: int = 800
    ):
        super().__init__("embedding")
        self.mode = mode.lower()
        self.provider: EmbeddingProvider | None = None
        self.timeout_ms = timeout_ms
        
        # Кэш для user-эмбеддингов
        self.user_cache = EmbeddingCache(
            max_size=1000,
            ttl_minutes=cache_ttl_minutes
        )
        
        # Инициализация провайдера (только Ollama в новой версии)
        if self.mode == "ollama":
            model = ollama_model or os.getenv("OLLAMA_MODEL", "qllama/multilingual-e5-small")
            base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.provider = OllamaProvider(model, base_url)
            LOGGER.info(f"Initialized OllamaProvider: {model} at {base_url}")
        
        elif self.mode == "local":
            if model_path:
                self.provider = LocalModelProvider(Path(model_path))
                LOGGER.info(f"Initialized LocalModelProvider: {model_path}")
            else:
                LOGGER.warning("Local model path not provided, embedding filter disabled")
        
        elif self.mode == "disabled":
            LOGGER.info("Embedding filter disabled by configuration")
        
        else:
            LOGGER.error(f"Unknown embedding mode: {mode}")
    
    async def compute_embeddings_multi(
        self,
        message_capsule: str,
        context_capsule: Optional[str] = None,
        user_capsule: Optional[str] = None,
        user_id: Optional[int] = None,
        enable_user_cache: bool = True
    ) -> Tuple[EmbeddingVectors, Dict[str, any]]:
        """
        Вычисляет эмбеддинги для E_msg/E_ctx/E_user с кэшем и graceful degradation.
        
        Args:
            message_capsule: нормализованное сообщение (passage: ...)
            context_capsule: контекстная капсула (опционально)
            user_capsule: user-капсула (опционально)
            user_id: ID пользователя для кэша
            enable_user_cache: использовать кэш для E_user
            
        Returns:
            (EmbeddingVectors, debug_info)
            debug_info содержит статусы, флаги деградации, время выполнения
        """
        if not self.provider:
            return (
                EmbeddingVectors(E_msg=None, E_ctx=None, E_user=None),
                {"error": "Provider not initialized"}
            )
        
        debug_info = {
            "degraded_ctx": False,
            "degraded_user": False,
            "cache_hit_user": False,
            "timeout_ms": self.timeout_ms
        }
        
        start_time = time.time()
        
        # Проверяем кэш для E_user
        cached_e_user = None
        if user_capsule and user_id and enable_user_cache:
            cached_e_user = self.user_cache.get(user_id)
            if cached_e_user:
                debug_info["cache_hit_user"] = True
                LOGGER.debug(f"Using cached E_user for user {user_id}")
        
        # Формируем список текстов для batch-запроса
        texts_to_compute = [message_capsule]  # E_msg всегда обязателен
        
        if context_capsule:
            texts_to_compute.append(context_capsule)
        
        if user_capsule and not cached_e_user:
            texts_to_compute.append(user_capsule)
        
        # Пакетный запрос
        try:
            embeddings, status = await self.provider.get_embeddings_batch(
                texts_to_compute,
                timeout_ms=self.timeout_ms
            )
            
            debug_info["status"] = status
            
            # Распаковываем результаты
            e_msg = embeddings[0] if len(embeddings) > 0 else None
            e_ctx = embeddings[1] if len(embeddings) > 1 and context_capsule else None
            e_user = embeddings[2] if len(embeddings) > 2 and user_capsule and not cached_e_user else cached_e_user
            
            # Проверка деградации
            if context_capsule and not e_ctx:
                debug_info["degraded_ctx"] = True
                LOGGER.warning("Context embedding failed (degraded_ctx=True)")
            
            if user_capsule and not e_user:
                debug_info["degraded_user"] = True
                LOGGER.warning("User embedding failed (degraded_user=True)")
            
            # Сохраняем E_user в кэш (если успешно и не из кэша)
            if e_user and user_id and not cached_e_user and enable_user_cache:
                self.user_cache.set(user_id, e_user)
            
            elapsed = time.time() - start_time
            debug_info["elapsed_ms"] = int(elapsed * 1000)
            
            return (
                EmbeddingVectors(E_msg=e_msg, E_ctx=e_ctx, E_user=e_user),
                debug_info
            )
        
        except Exception as e:
            LOGGER.error(f"Embedding batch failed: {e}")
            return (
                EmbeddingVectors(E_msg=None, E_ctx=None, E_user=None),
                {"error": str(e)}
            )
    
    def invalidate_user_cache(self, user_id: int) -> None:
        """Инвалидировать кэш для пользователя (вызывать при новом сообщении)"""
        self.user_cache.invalidate(user_id)
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Статистика кэша"""
        return {
            "cache_size": self.user_cache.size(),
            "cache_max": self.user_cache.max_size,
            "ttl_minutes": self.user_cache.ttl.total_seconds() / 60
        }
    
    async def analyze(self, text: str) -> FilterResult:
        """
        Legacy метод для обратной совместимости.
        Возвращает neutral score (0.5) + эмбеддинг в details.
        """
        if not self.provider:
            return FilterResult(
                filter_name=self.name,
                score=0.5,
                confidence=0.0,
                details={"status": "disabled", "embedding": None}
            )
        
        try:
            from utils.textprep import prepare_for_embedding
            normalized_text = prepare_for_embedding(text)
            embedding, status = await self.provider.get_embedding(normalized_text)
            
            if embedding is None:
                return FilterResult(
                    filter_name=self.name,
                    score=0.5,
                    confidence=0.0,
                    details={"error": status, "embedding": None}
                )
            
            return FilterResult(
                filter_name=self.name,
                score=0.5,  # neutral placeholder
                confidence=1.0,
                details={"embedding": embedding, "status": status}
            )
        except Exception as e:
            LOGGER.error(f"Embedding analysis failed: {e}")
            return FilterResult(
                filter_name=self.name,
                score=0.5,
                confidence=0.0,
                details={"error": str(e), "embedding": None}
            )
    
    def is_ready(self) -> bool:
        return self.provider is not None
