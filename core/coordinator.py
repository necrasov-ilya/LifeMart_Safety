from __future__ import annotations

from core.types import AnalysisResult
from filters.base import BaseFilter
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class FilterCoordinator:
    def __init__(
        self,
        keyword_filter: BaseFilter,
        tfidf_filter: BaseFilter,
        embedding_filter: BaseFilter | None = None
    ):
        self.keyword_filter = keyword_filter
        self.tfidf_filter = tfidf_filter
        self.embedding_filter = embedding_filter
    
    async def analyze(self, text: str) -> AnalysisResult:
        keyword_result = await self.keyword_filter.analyze(text)
        LOGGER.debug(f"Keyword: {keyword_result.score:.2f}")
        
        tfidf_result = await self.tfidf_filter.analyze(text)
        LOGGER.debug(f"TF-IDF: {tfidf_result.score:.2f}")
        
        embedding_result = None
        if self.embedding_filter and self.embedding_filter.is_ready():
            embedding_result = await self.embedding_filter.analyze(text)
            LOGGER.debug(f"Embedding: {embedding_result.score:.2f}")
        
        result = AnalysisResult(
            keyword_result=keyword_result,
            tfidf_result=tfidf_result,
            embedding_result=embedding_result
        )
        
        LOGGER.info(
            f"Analysis complete: avg={result.average_score:.2f}, "
            f"max={result.max_score:.2f}, all_high={result.all_high}"
        )
        
        return result
    
    def is_ready(self) -> bool:
        return (
            self.keyword_filter.is_ready() and
            self.tfidf_filter.is_ready()
        )
