"""
Динамическая конфигурация для управления настройками в runtime.
.env файл содержит дефолтные значения, которые могут быть переопределены.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from config.config import settings
from utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class RuntimeConfig:
    """Конфигурация которая может изменяться во время работы бота"""
    
    # Policy settings
    policy_mode: str = field(default_factory=lambda: settings.POLICY_MODE)
    auto_delete_threshold: float = field(default_factory=lambda: settings.AUTO_DELETE_THRESHOLD)
    auto_kick_threshold: float = field(default_factory=lambda: settings.AUTO_KICK_THRESHOLD)
    notify_threshold: float = field(default_factory=lambda: settings.NOTIFY_THRESHOLD)
    
    # Filter thresholds
    keyword_threshold: float = field(default_factory=lambda: settings.KEYWORD_THRESHOLD)
    tfidf_threshold: float = field(default_factory=lambda: settings.TFIDF_THRESHOLD)
    embedding_threshold: float = field(default_factory=lambda: settings.EMBEDDING_THRESHOLD)
    
    # NEW: Meta classifier settings
    use_meta_classifier: bool = field(default_factory=lambda: settings.USE_META_CLASSIFIER)
    meta_threshold_high: float = field(default_factory=lambda: settings.META_THRESHOLD_HIGH)
    meta_threshold_medium: float = field(default_factory=lambda: settings.META_THRESHOLD_MEDIUM)
    
    # Tracking overrides
    _overrides: Dict[str, any] = field(default_factory=dict, repr=False)
    
    def set_policy_mode(self, mode: str) -> bool:
        """Изменить режим политики"""
        mode = mode.lower()
        if mode not in {"manual", "semi-auto", "auto"}:
            return False
        
        old_value = self.policy_mode
        self.policy_mode = mode
        self._overrides["policy_mode"] = mode
        
        LOGGER.info(f"Policy mode changed: {old_value} → {mode}")
        return True
    
    def set_threshold(self, name: str, value: float) -> bool:
        """Изменить порог фильтра или политики"""
        if value < 0.0 or value > 1.0:
            return False
        
        valid_thresholds = {
            "auto_delete", "auto_kick", "notify",
            "keyword", "tfidf", "embedding",
            "meta_high", "meta_medium"  # NEW
        }
        
        threshold_name = name.lower().replace("-", "_").replace(".", "_")
        
        # Обработка meta.high -> meta_threshold_high
        if threshold_name == "meta_high":
            threshold_name = "meta_threshold_high"
        elif threshold_name == "meta_medium":
            threshold_name = "meta_threshold_medium"
        elif not threshold_name.endswith("_threshold"):
            threshold_name = f"{threshold_name}_threshold"
        
        # Проверяем что это валидный threshold
        if not hasattr(self, threshold_name):
            return False
        
        old_value = getattr(self, threshold_name)
        setattr(self, threshold_name, value)
        self._overrides[threshold_name] = value
        
        LOGGER.info(f"Threshold changed: {threshold_name} = {old_value:.2f} → {value:.2f}")
        return True
    
    def get_overrides(self) -> Dict[str, any]:
        """Получить словарь переопределенных значений"""
        return self._overrides.copy()
    
    def reset_overrides(self) -> None:
        """Сбросить все переопределения к дефолтным значениям из .env"""
        LOGGER.info("Resetting all overrides to .env defaults")
        
        self.policy_mode = settings.POLICY_MODE
        self.auto_delete_threshold = settings.AUTO_DELETE_THRESHOLD
        self.auto_kick_threshold = settings.AUTO_KICK_THRESHOLD
        self.notify_threshold = settings.NOTIFY_THRESHOLD
        
        self.keyword_threshold = settings.KEYWORD_THRESHOLD
        self.tfidf_threshold = settings.TFIDF_THRESHOLD
        self.embedding_threshold = settings.EMBEDDING_THRESHOLD
        
        # NEW: Meta classifier
        self.use_meta_classifier = settings.USE_META_CLASSIFIER
        self.meta_threshold_high = settings.META_THRESHOLD_HIGH
        self.meta_threshold_medium = settings.META_THRESHOLD_MEDIUM
        
        self._overrides.clear()
    
    def is_default(self, name: str) -> bool:
        """Проверить использует ли параметр дефолтное значение"""
        return name not in self._overrides


# Глобальный singleton для runtime конфигурации
runtime_config = RuntimeConfig()

__all__ = ["runtime_config", "RuntimeConfig"]
