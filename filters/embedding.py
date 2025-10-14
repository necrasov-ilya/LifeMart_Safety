from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from core.types import FilterResult
from filters.base import BaseFilter
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class EmbeddingProvider(ABC):
    @abstractmethod
    async def get_spam_score(self, text: str) -> tuple[float, str]:
        pass


class MistralAPIProvider(EmbeddingProvider):
    # Rate limiter: максимум 1 запрос в секунду
    _last_request_time = 0.0
    _rate_limit_lock = asyncio.Lock()
    
    def __init__(self, api_key: str, model: str = "mistral-embed", use_dataset_examples: bool = True):
        self.api_key = api_key
        self.model = model
        self.use_dataset_examples = use_dataset_examples
        self._spam_examples_cache = None
    
    def _get_spam_examples(self) -> list[str]:
        """Получить примеры спама из датасета или использовать дефолтные"""
        if self._spam_examples_cache is not None:
            return self._spam_examples_cache
        
        if self.use_dataset_examples:
            try:
                from pathlib import Path
                import pandas as pd
                
                dataset_path = Path(__file__).resolve().parents[1] / "data" / "messages.csv"
                if dataset_path.exists():
                    df = pd.read_csv(dataset_path)
                    spam_df = df[df['label'] == 1]
                    
                    # ФИЛЬТРАЦИЯ: только сообщения 50-300 символов
                    # Короткие (<50) - мало контекста
                    # Длинные (>300) - лишние токены + хуже эмбеддинги
                    spam_df['length'] = spam_df['message'].str.len()
                    filtered = spam_df[(spam_df['length'] >= 50) & (spam_df['length'] <= 300)]
                    
                    if len(filtered) >= 12:
                        # Берем случайную выборку из отфильтрованных
                        examples = filtered.sample(n=12, random_state=42)['message'].tolist()
                        LOGGER.info(f"Loaded {len(examples)} spam examples from dataset (50-300 chars)")
                        self._spam_examples_cache = examples
                        return examples
                    elif len(spam_df) >= 12:
                        # Fallback: если мало отфильтрованных, берем любые
                        examples = spam_df.sample(n=12, random_state=42)['message'].tolist()
                        LOGGER.warning(f"Using {len(examples)} examples without length filter")
                        self._spam_examples_cache = examples
                        return examples
            except Exception as e:
                LOGGER.warning(f"Failed to load examples from dataset: {e}, using defaults")
        
        # Дефолтные примеры если датасет недоступен
        default_examples = [
            "Привет! Есть предложение по удаленной работе, заработок от 50 тысяч рублей в месяц. Пиши в личку если интересно!",
            "Ищем людей для работы онлайн. Свободный график, платим от 1500 рублей в день. Пишите + в комментарии",
            "Обменяю криптовалюту USDT на рубли. Выгодный курс, быстрый перевод. Пишите для деталей",
            "Инвестиции в криптовалюту с доходностью 20% в месяц. Проверенная команда, присоединяйтесь!",
            "Топ казино с бездепозитным бонусом! Регистрируйся по ссылке и получи 5000 рублей на счет",
            "Слоты Олимпус с высоким процентом выигрыша. Жми на ссылку и получи фриспины!",
            "Набираем партнеров в команду. Пассивный доход от структуры. Хочешь узнать больше? Пиши!",
            "Присоединяйся к нашему проекту! Первая линия получает 30% от оборота. Успей войти!",
            "Найду любые фото и видео по запросу. Нейросеть работает быстро. Пиши в личку что нужно",
            "Знакомства для интима, девушки вашего города. Переходи по ссылке для связи"
        ]
        self._spam_examples_cache = default_examples
        return default_examples
    
    async def get_spam_score(self, text: str) -> tuple[float, str]:
        try:
            import httpx
        except ImportError:
            LOGGER.error("httpx not installed. Install with: pip install httpx")
            return 0.5, "httpx not installed"
        
        # Получаем примеры спама (из датасета или дефолтные)
        spam_examples = self._get_spam_examples()
        
        # RATE LIMITING: ждем минимум 1 секунду между запросами
        async with self._rate_limit_lock:
            elapsed = time.time() - MistralAPIProvider._last_request_time
            if elapsed < 1.0:
                wait_time = 1.0 - elapsed
                LOGGER.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            
            MistralAPIProvider._last_request_time = time.time()
        
        # Retry logic для 429 ошибок
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Обрезаем тексты до 250 символов для экономии токенов
                    truncated_text = text[:250]
                    truncated_examples = [ex[:250] for ex in spam_examples[:5]]  # Только 5 примеров
                    
                    # Получаем эмбеддинг для текста сообщения и примеров спама
                    response = await client.post(
                        "https://api.mistral.ai/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model,
                            "input": [truncated_text] + truncated_examples
                        }
                    )
                    
                    if response.status_code == 429:
                        # Rate limit exceeded - retry с экспоненциальной задержкой
                        retry_delay = 2 ** attempt  # 1s, 2s, 4s
                        LOGGER.warning(f"Rate limit hit (429), retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    if response.status_code != 200:
                        LOGGER.error(f"Mistral API error: {response.status_code} {response.text}")
                        return 0.5, f"API error: {response.status_code}"
                    
                    data = response.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    
                    message_embedding = embeddings[0]
                    spam_embeddings = embeddings[1:]
                    
                    # Вычисляем максимальную схожесть с примерами спама
                    max_similarity = 0.0
                    most_similar_example = ""
                    
                    for idx, spam_emb in enumerate(spam_embeddings):
                        similarity = self._cosine_similarity(message_embedding, spam_emb)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_example = truncated_examples[idx][:50]
                    
                    # Более строгая нормализация для примеров (не ключевых слов)
                    # Т.к. мы сравниваем с полными предложениями, порог выше
                    if max_similarity < 0.70:
                        # Низкая схожесть с примерами спама
                        normalized_score = 0.0
                    elif max_similarity > 0.90:
                        # Очень похоже на спам
                        normalized_score = 1.0
                    else:
                        # Линейная интерполяция между 0.70 и 0.90
                        normalized_score = (max_similarity - 0.70) / 0.20
                    
                    reasoning = f"Similarity: {max_similarity:.3f} (like: {most_similar_example}...)"
                    
                    return normalized_score, reasoning
            
            except httpx.TimeoutException:
                LOGGER.error(f"Mistral API timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except Exception as e:
                LOGGER.error(f"Mistral API request failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
        
        # Все попытки исчерпаны
        return 0.5, "Max retries exceeded"
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        if len(vec1) != len(vec2):
            return 0.5
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.5
        
        return dot_product / (magnitude1 * magnitude2)


class OllamaProvider(EmbeddingProvider):
    """
    Провайдер для Ollama API.
    ИЗМЕНЕНО: Больше не использует few-shot примеры.
    Возвращает чистый вектор эмбеддинга для мета-классификатора.
    """
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        LOGGER.info(f"Initialized OllamaProvider with model: {model}")
    
    async def get_embedding(self, text: str) -> tuple[list[float] | None, str]:
        """
        Получает эмбеддинг для текста через Ollama API.
        
        Args:
            text: подготовленный текст (с префиксом passage:)
            
        Returns:
            (embedding_vector, status_message)
        """
        try:
            import httpx
        except ImportError:
            LOGGER.error("httpx not installed. Install with: pip install httpx")
            return None, "httpx not installed"
        
        import time
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text}
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code != 200:
                    LOGGER.error(f"Ollama API error: {response.status_code} {response.text}")
                    return None, f"API error: {response.status_code}"
                
                embedding = response.json()["embedding"]
                LOGGER.debug(f"Ollama embedding: {len(embedding)} dims, {elapsed:.2f}s")
                
                return embedding, f"OK ({len(embedding)} dims, {elapsed:.2f}s)"
        
        except Exception as e:
            elapsed = time.time() - start_time
            LOGGER.error(f"Ollama API request failed after {elapsed:.2f}s: {e}")
            return None, f"Error: {str(e)}"
    
    async def get_spam_score(self, text: str) -> tuple[float, str]:
        """
        УСТАРЕВШИЙ МЕТОД для обратной совместимости.
        Теперь просто возвращает 0.5 (neutral).
        Реальная классификация происходит в MetaClassifier.
        """
        LOGGER.warning(
            "OllamaProvider.get_spam_score() is deprecated. "
            "Use get_embedding() + MetaClassifier instead."
        )
        return 0.5, "Use MetaClassifier for scoring"


class LocalModelProvider(EmbeddingProvider):
    def __init__(self, model_path: Path):
        self.model_path = model_path
        LOGGER.warning("LocalModelProvider not implemented yet")
    
    async def get_spam_score(self, text: str) -> tuple[float, str]:
        LOGGER.warning("Local model not implemented, returning neutral score")
        return 0.5, "Local model not implemented"


class EmbeddingFilter(BaseFilter):
    def __init__(
        self,
        mode: str = "api",
        api_key: str | None = None,
        model_path: Path | str | None = None,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None
    ):
        super().__init__("embedding")
        
        self.mode = mode.lower()
        self.provider: EmbeddingProvider | None = None
        
        if self.mode == "api":
            if not api_key:
                api_key = os.getenv("MISTRAL_API_KEY")
            
            if api_key:
                self.provider = MistralAPIProvider(api_key)
                LOGGER.info("Initialized MistralAPIProvider")
            else:
                LOGGER.warning("MISTRAL_API_KEY not set, embedding filter disabled")
        
        elif self.mode == "ollama":
            model = ollama_model or os.getenv("OLLAMA_MODEL", "nomic-embed-text")
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
    
    async def analyze(self, text: str) -> FilterResult:
        """
        ОБНОВЛЕНО: Теперь возвращает эмбеддинг в details для MetaClassifier.
        Нормализация текста через utils.textprep.prepare_for_embedding().
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
            
            # Нормализация текста с префиксом passage:
            normalized_text = prepare_for_embedding(text)
            
            # Для Ollama используем новый метод get_embedding()
            if isinstance(self.provider, OllamaProvider):
                embedding, status = await self.provider.get_embedding(normalized_text)
                
                if embedding is None:
                    LOGGER.warning(f"Failed to get embedding: {status}")
                    return FilterResult(
                        filter_name=self.name,
                        score=0.5,
                        confidence=0.0,
                        details={"error": status, "embedding": None}
                    )
                
                # Возвращаем neutral score (0.5), т.к. реальная классификация в MetaClassifier
                return FilterResult(
                    filter_name=self.name,
                    score=0.5,  # neutral, не используется для решения
                    confidence=1.0,
                    details={
                        "embedding": embedding,
                        "status": status,
                        "normalized_text": normalized_text[:100]  # для отладки
                    }
                )
            else:
                # Для MistralAPIProvider и других (старая логика для обратной совместимости)
                score, reasoning = await self.provider.get_spam_score(text)
                
                return FilterResult(
                    filter_name=self.name,
                    score=score,
                    confidence=0.8,
                    details={"reasoning": reasoning, "embedding": None}
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
