"""
storage/message_logger.py
────────────────────────────────────────────────────────
Логгер сообщений для интеграции с ML-моделью и системой модерации.

• Автоматическое логирование всех обработанных сообщений
• Интеграция с решениями ML-модели
• Трекинг действий модераторов
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from telegram import Message, User

from storage.database import get_db, MessageLog
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class MessageLogger:
    """Логгер для сообщений и решений по ним."""

    def __init__(self):
        self._db = get_db()

    def log_message_processed(
        self,
        message: Message,
        spam_probability: float,
        predicted_label: str,
        action_taken: str
    ) -> None:
        """
        Залогировать обработанное сообщение.

        Args:
            message: Telegram сообщение
            spam_probability: Вероятность спама (0.0 - 1.0)
            predicted_label: Предсказанная метка ("spam" | "ham")
            action_taken: Предпринятое действие ("notify" | "delete" | "kick" | "none")
        """
        user = message.from_user
        if not user:
            LOGGER.warning(f"Сообщение {message.message_id} без пользователя")
            return

        full_name = self._get_full_name(user)

        log_entry = MessageLog(
            message_id=message.message_id,
            chat_id=message.chat_id,
            user_id=user.id,
            username=user.username,
            full_name=full_name,
            text=message.text or "",
            timestamp=datetime.now(),
            spam_probability=spam_probability,
            predicted_label=predicted_label,
            action_taken=action_taken
        )

        try:
            self._db.log_message(log_entry)
            self._db.update_message_stats(predicted_label, action_taken)
        except Exception as e:
            LOGGER.error(f"Ошибка логирования сообщения {message.message_id}: {e}")

    def log_moderator_decision(
        self,
        chat_id: int,
        message_id: int,
        decision: str,
        moderator_user_id: int
    ) -> None:
        """
        Залогировать решение модератора.

        Args:
            chat_id: ID чата
            message_id: ID сообщения
            decision: Решение модератора ("spam" | "ham")
            moderator_user_id: ID модератора
        """
        try:
            self._db.update_moderator_decision(
                chat_id=chat_id,
                message_id=message_id,
                decision=decision,
                moderator_user_id=moderator_user_id
            )
            LOGGER.info(f"Модератор {moderator_user_id} отметил сообщение {message_id} как {decision}")
        except Exception as e:
            LOGGER.error(f"Ошибка логирования решения модератора: {e}")

    def get_user_history(self, user_id: int, limit: int = 20) -> list[dict]:
        """Получить историю сообщений пользователя."""
        try:
            with self._db._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT message_id, chat_id, text, timestamp, spam_probability, 
                           predicted_label, action_taken, moderator_decision
                    FROM message_logs 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, limit))

                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            LOGGER.error(f"Ошибка получения истории пользователя {user_id}: {e}")
            return []

    def get_pending_moderation(self, chat_id: Optional[int] = None) -> list[dict]:
        """Получить сообщения, ожидающие модерации."""
        try:
            with self._db._get_connection() as conn:
                if chat_id:
                    cursor = conn.execute("""
                        SELECT message_id, chat_id, user_id, username, full_name, 
                               text, timestamp, spam_probability, predicted_label
                        FROM message_logs 
                        WHERE chat_id = ? AND moderator_decision IS NULL 
                              AND predicted_label = 'spam'
                        ORDER BY timestamp DESC
                        LIMIT 50
                    """, (chat_id,))
                else:
                    cursor = conn.execute("""
                        SELECT message_id, chat_id, user_id, username, full_name, 
                               text, timestamp, spam_probability, predicted_label
                        FROM message_logs 
                        WHERE moderator_decision IS NULL AND predicted_label = 'spam'
                        ORDER BY timestamp DESC
                        LIMIT 100
                    """)

                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            LOGGER.error(f"Ошибка получения сообщений на модерацию: {e}")
            return []

    def get_model_accuracy_stats(self, days: int = 7) -> dict:
        """Получить статистику точности модели за последние дни."""
        try:
            with self._db._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_reviewed,
                        SUM(CASE WHEN predicted_label = moderator_decision THEN 1 ELSE 0 END) as correct_predictions,
                        SUM(CASE WHEN predicted_label = 'spam' AND moderator_decision = 'ham' THEN 1 ELSE 0 END) as false_positives,
                        SUM(CASE WHEN predicted_label = 'ham' AND moderator_decision = 'spam' THEN 1 ELSE 0 END) as false_negatives
                    FROM message_logs 
                    WHERE moderator_decision IS NOT NULL 
                    AND datetime(timestamp) > datetime('now', '-{} days')
                """.format(days), ())

                row = cursor.fetchone()

                total = row['total_reviewed'] or 0
                correct = row['correct_predictions'] or 0

                return {
                    'total_reviewed': total,
                    'correct_predictions': correct,
                    'accuracy': (correct / total * 100) if total > 0 else 0.0,
                    'false_positives': row['false_positives'] or 0,
                    'false_negatives': row['false_negatives'] or 0,
                    'period_days': days
                }
        except Exception as e:
            LOGGER.error(f"Ошибка получения статистики точности модели: {e}")
            return {
                'total_reviewed': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'false_positives': 0,
                'false_negatives': 0,
                'period_days': days
            }

    @staticmethod
    def _get_full_name(user: User) -> str:
        """Получить полное имя пользователя."""
        parts = []
        if user.first_name:
            parts.append(user.first_name)
        if user.last_name:
            parts.append(user.last_name)

        full_name = " ".join(parts)
        return full_name if full_name else f"User_{user.id}"


# Глобальный экземпляр логгера сообщений
_message_logger: MessageLogger | None = None

def get_message_logger() -> MessageLogger:
    """Получить глобальный экземпляр логгера сообщений."""
    global _message_logger
    if _message_logger is None:
        _message_logger = MessageLogger()
    return _message_logger
