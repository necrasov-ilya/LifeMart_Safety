from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import FilterResult


class BaseFilter(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def analyze(self, text: str) -> FilterResult:
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        pass
