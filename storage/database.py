"""
storage/database.py
────────────────────────────────────────────────────────
Система хранения данных для LifeMart Safety Bot.

• Логирование всех сообщений и решений по ним
• Хранение настроек бота (персистентность между перезапусками)
• История модерации и статистика
"""

from __future__ import annotations

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class MessageLog:
    """Лог обработанного сообщения."""
    message_id: int
    chat_id: int
    user_id: int
    username: Optional[str]
    full_name: str
    text: str
    timestamp: datetime
    spam_probability: float
    predicted_label: str  # "spam" | "ham"
    action_taken: str     # "notify" | "delete" | "kick" | "none"
    moderator_decision: Optional[str] = None  # "spam" | "ham" если модератор решил
    moderator_user_id: Optional[int] = None


@dataclass
class BotSettings:
    """Настройки бота для сохранения между перезапусками."""
    spam_policy: str = "notify"      # notify | delete | kick
    announce_blocks: bool = True     # публиковать ли уведомления в чате
    notification_enabled: bool = True
    auto_delete_threshold: float = 0.9  # порог для автоудаления


class DatabaseManager:
    """Менеджер базы данных для хранения логов и настроек."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "bot_data.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)

        self._init_database()
        LOGGER.info(f"Инициализирована база данных: {self.db_path}")

    def _init_database(self) -> None:
        """Создание таблиц при первом запуске."""
        with self._get_connection() as conn:
            # Таблица логов сообщений
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    username TEXT,
                    full_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    spam_probability REAL NOT NULL,
                    predicted_label TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    moderator_decision TEXT,
                    moderator_user_id INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Таблица настроек бота
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bot_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    settings_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Таблица статистики
            conn.execute("""
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_messages INTEGER DEFAULT 0,
                    spam_detected INTEGER DEFAULT 0,
                    ham_detected INTEGER DEFAULT 0,
                    auto_deleted INTEGER DEFAULT 0,
                    moderator_corrections INTEGER DEFAULT 0,
                    UNIQUE(date)
                )
            """)

            # Индексы для оптимизации
            conn.execute("CREATE INDEX IF NOT EXISTS idx_message_logs_chat_id ON message_logs(chat_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_message_logs_timestamp ON message_logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_message_logs_user_id ON message_logs(user_id)")

    @contextmanager
    def _get_connection(self):
        """Контекстный менеджер для работы с соединением."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def log_message(self, log_entry: MessageLog) -> None:
        """Сохранить лог обработанного сообщения."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO message_logs (
                    message_id, chat_id, user_id, username, full_name,
                    text, timestamp, spam_probability, predicted_label,
                    action_taken, moderator_decision, moderator_user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry.message_id,
                log_entry.chat_id,
                log_entry.user_id,
                log_entry.username,
                log_entry.full_name,
                log_entry.text,
                log_entry.timestamp.isoformat(),
                log_entry.spam_probability,
                log_entry.predicted_label,
                log_entry.action_taken,
                log_entry.moderator_decision,
                log_entry.moderator_user_id
            ))

        LOGGER.info(f"Залогировано сообщение: chat_id={log_entry.chat_id}, "
                   f"user_id={log_entry.user_id}, prediction={log_entry.predicted_label}")

    def update_moderator_decision(
        self,
        chat_id: int,
        message_id: int,
        decision: str,
        moderator_user_id: int
    ) -> None:
        """Обновить решение модератора для сообщения."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE message_logs 
                SET moderator_decision = ?, moderator_user_id = ?
                WHERE chat_id = ? AND message_id = ?
            """, (decision, moderator_user_id, chat_id, message_id))

            if cursor.rowcount > 0:
                LOGGER.info(f"Обновлено решение модератора: {decision} для сообщения {message_id}")
                # Обновляем статистику
                self._update_daily_stats(moderator_corrections=1)

    def save_settings(self, settings: BotSettings) -> None:
        """Сохранить настройки бота."""
        settings_json = json.dumps(asdict(settings))

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO bot_settings (id, settings_json, updated_at)
                VALUES (1, ?, CURRENT_TIMESTAMP)
            """, (settings_json,))

        LOGGER.info("Настройки бота сохранены в базу данных")

    def load_settings(self) -> BotSettings:
        """Загрузить настройки бота."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT settings_json FROM bot_settings WHERE id = 1").fetchone()

        if row:
            settings_dict = json.loads(row['settings_json'])
            return BotSettings(**settings_dict)
        else:
            # Возвращаем настройки по умолчанию
            default_settings = BotSettings()
            self.save_settings(default_settings)
            return default_settings

    def get_user_stats(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Получить статистику пользователя за последние N дней."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_messages,
                    SUM(CASE WHEN predicted_label = 'spam' THEN 1 ELSE 0 END) as spam_messages,
                    AVG(spam_probability) as avg_spam_probability,
                    MAX(timestamp) as last_message
                FROM message_logs 
                WHERE user_id = ? 
                AND datetime(timestamp) > datetime('now', '-{} days')
            """.format(days), (user_id,))

            row = cursor.fetchone()

        return {
            'user_id': user_id,
            'total_messages': row['total_messages'] or 0,
            'spam_messages': row['spam_messages'] or 0,
            'avg_spam_probability': row['avg_spam_probability'] or 0.0,
            'last_message': row['last_message'],
            'period_days': days
        }

    def get_chat_stats(self, chat_id: int, days: int = 7) -> Dict[str, Any]:
        """Получить статистику чата за последние N дней."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(DISTINCT user_id) as unique_users,
                    SUM(CASE WHEN predicted_label = 'spam' THEN 1 ELSE 0 END) as spam_detected,
                    SUM(CASE WHEN action_taken = 'delete' THEN 1 ELSE 0 END) as auto_deleted,
                    SUM(CASE WHEN moderator_decision IS NOT NULL THEN 1 ELSE 0 END) as moderator_reviewed
                FROM message_logs 
                WHERE chat_id = ? 
                AND datetime(timestamp) > datetime('now', '-{} days')
            """.format(days), (chat_id,))

            row = cursor.fetchone()

        return {
            'chat_id': chat_id,
            'total_messages': row['total_messages'] or 0,
            'unique_users': row['unique_users'] or 0,
            'spam_detected': row['spam_detected'] or 0,
            'auto_deleted': row['auto_deleted'] or 0,
            'moderator_reviewed': row['moderator_reviewed'] or 0,
            'period_days': days
        }

    def _update_daily_stats(
        self,
        total_messages: int = 0,
        spam_detected: int = 0,
        ham_detected: int = 0,
        auto_deleted: int = 0,
        moderator_corrections: int = 0
    ) -> None:
        """Обновить ежедневную статистику."""
        today = datetime.now().date().isoformat()

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO statistics (
                    date, total_messages, spam_detected, ham_detected, 
                    auto_deleted, moderator_corrections
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_messages = total_messages + excluded.total_messages,
                    spam_detected = spam_detected + excluded.spam_detected,
                    ham_detected = ham_detected + excluded.ham_detected,
                    auto_deleted = auto_deleted + excluded.auto_deleted,
                    moderator_corrections = moderator_corrections + excluded.moderator_corrections
            """, (today, total_messages, spam_detected, ham_detected, auto_deleted, moderator_corrections))

    def update_message_stats(self, predicted_label: str, action_taken: str) -> None:
        """Обновить статистику обработанных сообщений."""
        spam_detected = 1 if predicted_label == 'spam' else 0
        ham_detected = 1 if predicted_label == 'ham' else 0
        auto_deleted = 1 if action_taken == 'delete' else 0

        self._update_daily_stats(
            total_messages=1,
            spam_detected=spam_detected,
            ham_detected=ham_detected,
            auto_deleted=auto_deleted
        )

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Получить последние логи сообщений."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM message_logs 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]


# Глобальный экземпляр менеджера базы данных
_db_manager: Optional[DatabaseManager] = None

def get_db() -> DatabaseManager:
    """Получить глобальный экземпляр менеджера базы данных."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
