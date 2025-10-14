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
    
    # Meta classifier thresholds
    meta_notify: float = field(default_factory=lambda: settings.META_NOTIFY)
    meta_delete: float = field(default_factory=lambda: settings.META_DELETE)
    meta_kick: float = field(default_factory=lambda: settings.META_KICK)
    
    # Downweight multipliers
    meta_downweight_announcement: float = field(default_factory=lambda: settings.META_DOWNWEIGHT_ANNOUNCEMENT)
    meta_downweight_reply_to_staff: float = field(default_factory=lambda: settings.META_DOWNWEIGHT_REPLY_TO_STAFF)
    meta_downweight_whitelist: float = field(default_factory=lambda: settings.META_DOWNWEIGHT_WHITELIST)
    
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
        """Изменить порог мета-классификатора"""
        if value < 0.0 or value > 1.0:
            return False
        
        valid_thresholds = {"meta_notify", "meta_delete", "meta_kick"}
        threshold_name = name.lower().replace("-", "_")
        
        if threshold_name not in valid_thresholds:
            return False
        
        old_value = getattr(self, threshold_name)
        setattr(self, threshold_name, value)
        self._overrides[threshold_name] = value
        
        LOGGER.info(f"Threshold changed: {threshold_name} = {old_value:.2f} → {value:.2f}")
        return True
    
    def set_downweight(self, name: str, value: float) -> bool:
        """
        Изменить понижающий множитель.
        
        Args:
            name: announcement|reply_to_staff|whitelist
            value: множитель (обычно 0.7-0.95)
        
        Returns:
            True если успешно
        """
        if value < 0.0 or value > 1.0:
            LOGGER.warning(f"Invalid downweight value: {value} (must be 0.0-1.0)")
            return False
        
        downweight_map = {
            "announcement": "meta_downweight_announcement",
            "reply_to_staff": "meta_downweight_reply_to_staff",
            "whitelist": "meta_downweight_whitelist"
        }
        
        name_normalized = name.lower().replace("-", "_")
        attr_name = downweight_map.get(name_normalized)
        
        if not attr_name or not hasattr(self, attr_name):
            LOGGER.warning(f"Unknown downweight: {name}")
            return False
        
        old_value = getattr(self, attr_name)
        setattr(self, attr_name, value)
        self._overrides[attr_name] = value
        
        LOGGER.info(f"Downweight changed: {attr_name} = {old_value:.2f} → {value:.2f}")
        return True
    
    def get_overrides(self) -> Dict[str, any]:
        """Получить словарь переопределенных значений"""
        return self._overrides.copy()
    
    def reset_overrides(self) -> None:
        """Сбросить все переопределения к дефолтным значениям из .env"""
        LOGGER.info("Resetting all overrides to .env defaults")
        
        self.policy_mode = settings.POLICY_MODE
        self.meta_notify = settings.META_NOTIFY
        self.meta_delete = settings.META_DELETE
        self.meta_kick = settings.META_KICK
        self.meta_downweight_announcement = settings.META_DOWNWEIGHT_ANNOUNCEMENT
        self.meta_downweight_reply_to_staff = settings.META_DOWNWEIGHT_REPLY_TO_STAFF
        self.meta_downweight_whitelist = settings.META_DOWNWEIGHT_WHITELIST
        
        self._overrides.clear()
    
    def is_default(self, name: str) -> bool:
        """Проверить использует ли параметр дефолтное значение"""
        return name not in self._overrides


# Глобальный singleton для runtime конфигурации
runtime_config = RuntimeConfig()

__all__ = ["runtime_config", "RuntimeConfig"]
