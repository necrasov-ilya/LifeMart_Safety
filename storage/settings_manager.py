"""
storage/settings_manager.py
────────────────────────────────────────────────────────
Менеджер настроек с персистентным хранением.

• Заменяет глобальные переменные SPAM_POLICY и ANNOUNCE_BLOCKS
• Автоматически сохраняет изменения в базу данных
• Предоставляет thread-safe доступ к настройкам
"""

from __future__ import annotations

import threading
from typing import Any

from storage.database import get_db, BotSettings
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class SettingsManager:
    """Thread-safe менеджер настроек с автосохранением."""

    def __init__(self):
        self._lock = threading.RLock()
        self._db = get_db()
        self._settings = self._db.load_settings()

        LOGGER.info(f"Загружены настройки: spam_policy={self._settings.spam_policy}, "
                   f"announce_blocks={self._settings.announce_blocks}")

    @property
    def spam_policy(self) -> str:
        """Политика обработки спама: notify | delete | kick"""
        with self._lock:
            return self._settings.spam_policy

    @spam_policy.setter
    def spam_policy(self, value: str) -> None:
        if value not in ("notify", "delete", "kick"):
            raise ValueError(f"Недопустимая политика спама: {value}")

        with self._lock:
            old_value = self._settings.spam_policy
            self._settings.spam_policy = value
            self._save_settings()

        LOGGER.info(f"Изменена политика спама: {old_value} → {value}")

    @property
    def announce_blocks(self) -> bool:
        """Публиковать ли уведомления о блокировках в чате"""
        with self._lock:
            return self._settings.announce_blocks

    @announce_blocks.setter
    def announce_blocks(self, value: bool) -> None:
        with self._lock:
            old_value = self._settings.announce_blocks
            self._settings.announce_blocks = value
            self._save_settings()

        LOGGER.info(f"Изменены уведомления о блокировках: {old_value} → {value}")

    @property
    def notification_enabled(self) -> bool:
        """Включены ли уведомления модераторам"""
        with self._lock:
            return self._settings.notification_enabled

    @notification_enabled.setter
    def notification_enabled(self, value: bool) -> None:
        with self._lock:
            old_value = self._settings.notification_enabled
            self._settings.notification_enabled = value
            self._save_settings()

        LOGGER.info(f"Изменены уведомления модераторам: {old_value} → {value}")

    @property
    def auto_delete_threshold(self) -> float:
        """Порог вероятности для автоматического удаления"""
        with self._lock:
            return self._settings.auto_delete_threshold

    @auto_delete_threshold.setter
    def auto_delete_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Порог должен быть от 0.0 до 1.0, получен: {value}")

        with self._lock:
            old_value = self._settings.auto_delete_threshold
            self._settings.auto_delete_threshold = value
            self._save_settings()

        LOGGER.info(f"Изменен порог автоудаления: {old_value} → {value}")

    def get_all_settings(self) -> dict[str, Any]:
        """Получить все настройки в виде словаря"""
        with self._lock:
            return {
                'spam_policy': self._settings.spam_policy,
                'announce_blocks': self._settings.announce_blocks,
                'notification_enabled': self._settings.notification_enabled,
                'auto_delete_threshold': self._settings.auto_delete_threshold,
            }

    def update_settings(self, **kwargs) -> None:
        """Массовое обновление настроек"""
        with self._lock:
            changes = []

            for key, value in kwargs.items():
                if hasattr(self._settings, key):
                    old_value = getattr(self._settings, key)
                    if old_value != value:
                        setattr(self._settings, key, value)
                        changes.append(f"{key}: {old_value} → {value}")
                else:
                    raise ValueError(f"Неизвестная настройка: {key}")

            if changes:
                self._save_settings()
                LOGGER.info(f"Обновлены настройки: {', '.join(changes)}")

    def _save_settings(self) -> None:
        """Сохранить настройки в базу данных (вызывается с блокировкой)"""
        try:
            self._db.save_settings(self._settings)
        except Exception as e:
            LOGGER.error(f"Ошибка сохранения настроек: {e}")
            raise

    def reset_to_defaults(self) -> None:
        """Сбросить настройки к значениям по умолчанию"""
        with self._lock:
            default_settings = BotSettings()
            old_settings = self.get_all_settings()
            self._settings = default_settings
            self._save_settings()

        LOGGER.info(f"Настройки сброшены к значениям по умолчанию. Было: {old_settings}")


# Глобальный экземпляр менеджера настроек
_settings_manager: SettingsManager | None = None

def get_settings() -> SettingsManager:
    """Получить глобальный экземпляр менеджера настроек."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
