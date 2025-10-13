from __future__ import annotations

import os
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
    def __init__(self, api_key: str, model: str = "mistral-embed"):
        self.api_key = api_key
        self.model = model
    
    async def get_spam_score(self, text: str) -> tuple[float, str]:
        try:
            import httpx
        except ImportError:
            LOGGER.error("httpx not installed. Install with: pip install httpx")
            return 0.5, "httpx not installed"
        
        # Расширенный список спам-индикаторов на основе анализа датасета
        spam_indicators = [
            # Заработок и деньги (самая частая категория)
            "заработок деньги доход прибыль удаленная работа тысяч рублей день",
            "долларов неделю месяц пассивный заработок онлайн формат",
            "дополнительный доход подработка шабашка халтура без опыта",
            
            # Криптовалюта и инвестиции
            "криптовалюта биткоин usdt обмен инвестиции трейдинг форекс биржа",
            "пассивный доход инвестиции криптоактивы цифровые валюты web3",
            
            # Призывы к действию
            "пиши личные сообщения лс плюс старт хочу подробности детали",
            "интересно кому нужно жду ответа обращайтесь свяжитесь",
            
            # MLM и сетевой маркетинг  
            "команда партнеры набор в команду сотрудничество проект взаимовыгодно",
            "новое направление новая тема присоединяйся к команде",
            
            # Казино и азартные игры
            "казино казик слоты ставки выигрыш бонус депозит занос олимпус",
            "лотерея азартные игры телеграм казино топ казино",
            
            # Удаленная работа (часто спам)
            "удаленка удаленная работа дистанционная занятость из дома телефон компьютер",
            "свободный график гибкий график пару часов день",
            
            # Сомнительные услуги
            "автошкола права водительское удостоверение сдача экзамена без экзамена",
            "помощь документы быстро восстановление категория",
            
            # Курьеры и доставка (подозрительные предложения)
            "курьер доставка рейсы зарплата тысяч день личный автомобиль",
            "грузчик разнорабочий подработка платим сразу оплата день",
            
            # Товары и услуги сомнительного качества
            "накрутка подписчики лайки продвижение реклама канал группа телеграм",
            "взлом аккаунт пароль данные конфиденциально",
            
            # Мошеннические схемы
            "помогу вернуть деньги потеряли средства схема обман компенсация",
            "быстрые деньги легкий заработок без вложений гарантия",
            
            # Запрещенные товары (завуалированные)
            "закладки расходник товар район доставка курьер телега",
            "работа серая тема белая тема легально проверено",
            
            # Сексуальный контент
            "интим фото голые видео нейросеть найти девушка контент",
            "знакомства встречи досуг эскорт массаж релакс",
            
            # Недвижимость и ремонт (спамные предложения)
            "ремонт квартира кухня мебель натяжные потолки дизайн проект скидка",
            "бесплатный замер гарантия лет фиксированная стоимость",
            
            # Типичные спам-фразы
            "срочно требуются человека возраст лет пишите личку",
            "последние места ограниченное количество успейте записаться",
            "не упустите шанс уникальное предложение акция заканчивается",
            
            # Эмодзи-спам паттерны
            "огонь топ лучший номер один советую рекомендую проверено",
            "ссылка переход регистрация бонус подарок фриспины",
            
            # Вакансии с завышенной зп
            "вакансия работа зарплата тысяч миллион неделя месяц набор",
            "без опыта обучение бесплатно начать сразу сегодня",
            
            # Переходы и ссылки
            "перейти ссылка бот канал группа подписаться вступить",
            "жми кликай нажми скорее быстрее успей",
            
            # Разное подозрительное
            "помощь нужна срочно быстро заплачу оплата наличные руки",
            "муж час разнорабочий помощник уборка покраска вынести",
            
            # Финансовые пирамиды
            "пассив структура уровень доход процент команда лидер",
            "первая линия реферал партнер бонус вознаграждение"
        ]
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Получаем эмбеддинг для текста сообщения
                response = await client.post(
                    "https://api.mistral.ai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "input": [text] + spam_indicators
                    }
                )
                
                if response.status_code != 200:
                    LOGGER.error(f"Mistral API error: {response.status_code} {response.text}")
                    return 0.5, f"API error: {response.status_code}"
                
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                
                message_embedding = embeddings[0]
                spam_embeddings = embeddings[1:]
                
                # Вычисляем максимальную схожесть со спам-индикаторами
                max_similarity = 0.0
                most_similar_indicator = ""
                
                for idx, spam_emb in enumerate(spam_embeddings):
                    similarity = self._cosine_similarity(message_embedding, spam_emb)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_indicator = spam_indicators[idx][:40]
                
                # Нормализуем от [-1, 1] к [0, 1]
                # Косинусная схожесть обычно > 0.3 для связанных текстов
                # > 0.6 для очень похожих текстов
                normalized_score = max(0.0, min(1.0, (max_similarity - 0.3) / 0.4))
                
                reasoning = f"Max similarity: {max_similarity:.2f} ({most_similar_indicator}...)"
                
                return normalized_score, reasoning
        
        except Exception as e:
            LOGGER.error(f"Mistral API request failed: {e}")
            return 0.5, f"Error: {str(e)}"
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        if len(vec1) != len(vec2):
            return 0.5
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.5
        
        return dot_product / (magnitude1 * magnitude2)


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
        model_path: Path | str | None = None
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
        if not self.provider:
            return FilterResult(
                filter_name=self.name,
                score=0.5,
                confidence=0.0,
                details={"status": "disabled"}
            )
        
        try:
            score, reasoning = await self.provider.get_spam_score(text)
            
            return FilterResult(
                filter_name=self.name,
                score=score,
                confidence=0.8,
                details={"reasoning": reasoning}
            )
        except Exception as e:
            LOGGER.error(f"Embedding analysis failed: {e}")
            return FilterResult(
                filter_name=self.name,
                score=0.5,
                confidence=0.0,
                details={"error": str(e)}
            )
    
    def is_ready(self) -> bool:
        return self.provider is not None
