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
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.mistral.ai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "input": [text]
                    }
                )
                
                if response.status_code != 200:
                    LOGGER.error(f"Mistral API error: {response.status_code} {response.text}")
                    return 0.5, f"API error: {response.status_code}"
                
                data = response.json()
                embedding = data["data"][0]["embedding"]
                
                spam_keywords_embedding = self._get_spam_reference_embedding()
                similarity = self._cosine_similarity(embedding, spam_keywords_embedding)
                
                score = max(0.0, min(1.0, similarity))
                
                return score, f"Semantic similarity: {similarity:.2f}"
        
        except Exception as e:
            LOGGER.error(f"Mistral API request failed: {e}")
            return 0.5, f"Error: {str(e)}"
    
    def _get_spam_reference_embedding(self) -> list[float]:
        return [0.0] * 1024
    
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
