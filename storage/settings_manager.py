"""
storage/settings_manager.py
────────────────────────────────────────────────────────
Менеджер настроек с персистентным хранением.

• Заменяет глобальные переменные SPAM_POLICY и ANNOUNCE_BLOCKS
• Автоматически сохраняет изменения в базу данных
• Предоставляет thread-safe доступ к настройкам
• Управляет списком модераторов
"""

from __future__ import annotations

import threading
import json
from typing import Any, List, Set
from pathlib import Path

from storage.database import get_db, BotSettings
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class SettingsManager:
    """Thread-safe менеджер настроек с автосохранением."""

    def __init__(self):
        self._lock = threading.RLock()
        self._db = get_db()
        self._settings = self._db.load_settings()

        # Файл для хранения дополнительных модераторов
        self._moderators_file = Path("data/moderators.json")
        self._additional_moderators: Set[int] = self._load_moderators()

        LOGGER.info(f"Загружены настройки: spam_policy={self._settings.spam_policy}, "
                   f"announce_blocks={self._settings.announce_blocks}, "
                   f"moderators={len(self._additional_moderators)}")

    def _load_moderators(self) -> Set[int]:
        """Загружает список дополнительных модераторов из файла."""
        try:
            if self._moderators_file.exists():
                with open(self._moderators_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return set(data.get('moderators', []))
        except Exception as e:
            LOGGER.error(f"Ошибка загрузки модераторов: {e}")
        return set()

    def _save_moderators(self) -> None:
        """Сохраняет список дополнительных модераторов в файл."""
        try:
            self._moderators_file.parent.mkdir(exist_ok=True)
            with open(self._moderators_file, 'w', encoding='utf-8') as f:
                json.dump({'moderators': list(self._additional_moderators)}, f, indent=2)
        except Exception as e:
            LOGGER.error(f"Ошибка сохранения модераторов: {e}")

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
        """Объявлять ли о блокировках в чате"""
        with self._lock:
            return self._settings.announce_blocks

    @announce_blocks.setter
    def announce_blocks(self, value: bool) -> None:
        with self._lock:
            old_value = self._settings.announce_blocks
            self._settings.announce_blocks = value
            self._save_settings()

        LOGGER.info(f"Изменены уведомления о блоках: {old_value} → {value}")

    @property
    def notification_enabled(self) -> bool:
        """Включены ли уведомления модераторам"""
        with self._lock:
            return getattr(self._settings, 'notification_enabled', True)

    @notification_enabled.setter
    def notification_enabled(self, value: bool) -> None:
        with self._lock:
            old_value = getattr(self._settings, 'notification_enabled', True)
            self._settings.notification_enabled = value
            self._save_settings()

        LOGGER.info(f"Изменены уведомления модераторам: {old_value} → {value}")

    @property
    def auto_delete_threshold(self) -> float:
        """Порог автоматического удаления сообщений"""
        with self._lock:
            return getattr(self._settings, 'auto_delete_threshold', 0.8)

    @auto_delete_threshold.setter
    def auto_delete_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Порог должен быть между 0.0 и 1.0")

        with self._lock:
            old_value = getattr(self._settings, 'auto_delete_threshold', 0.8)
            self._settings.auto_delete_threshold = value
            self._save_settings()

        LOGGER.info(f"Изменен порог автоудаления: {old_value} → {value}")

    def add_moderator(self, user_id: int) -> bool:
        """Добавляет пользователя в список модераторов."""
        with self._lock:
            if user_id not in self._additional_moderators:
                self._additional_moderators.add(user_id)
                self._save_moderators()
                LOGGER.info(f"Добавлен новый модератор: {user_id}")
                return True
            return False

    def remove_moderator(self, user_id: int) -> bool:
        """Удаляет пользователя из списка модераторов."""
        with self._lock:
            if user_id in self._additional_moderators:
                self._additional_moderators.remove(user_id)
                self._save_moderators()
                LOGGER.info(f"Удален модератор: {user_id}")
                return True
            return False

    def is_moderator(self, user_id: int) -> bool:
        """Проверяет, является ли пользователь модератором."""
        from config.config import settings
        with self._lock:
            return (user_id in settings.WHITELIST_USER_IDS or
                   user_id in self._additional_moderators)

    def get_all_moderators(self) -> List[int]:
        """Возвращает список всех модераторов."""
        from config.config import settings
        with self._lock:
            all_mods = set(settings.WHITELIST_USER_IDS)
            all_mods.update(self._additional_moderators)
            return list(all_mods)

    def get_additional_moderators(self) -> List[int]:
        """Возвращает список дополнительных модераторов (не из whitelist)."""
        with self._lock:
            return list(self._additional_moderators)

    def _save_settings(self) -> None:
        """Сохраняет настройки в базу данных."""
        try:
            self._db.save_settings(self._settings)
        except Exception as e:
            LOGGER.error(f"Ошибка сохранения настроек: {e}")


# Глобальный экземпляр менеджера настроек
_settings_manager = None
_manager_lock = threading.Lock()


def get_settings() -> SettingsManager:
    """Возвращает глобальный экземпляр менеджера настроек (singleton)."""
    global _settings_manager
    if _settings_manager is None:
        with _manager_lock:
            if _settings_manager is None:
                _settings_manager = SettingsManager()
    return _settings_manager
