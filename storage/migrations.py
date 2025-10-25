from __future__ import annotations

MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        -- moderation events and related tables
        CREATE TABLE IF NOT EXISTS moderation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
            chat_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            username TEXT,
            text_hash TEXT,
            text_length INTEGER,
            action TEXT NOT NULL,
            action_confidence REAL,
            filter_keyword_score REAL,
            filter_tfidf_score REAL,
            filter_embedding_score REAL,
            meta_debug TEXT,
            source TEXT NOT NULL DEFAULT 'bot'
        );
        CREATE INDEX IF NOT EXISTS idx_moderation_events_chat_time
            ON moderation_events(chat_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_moderation_events_user_time
            ON moderation_events(user_id, created_at);

        CREATE TABLE IF NOT EXISTS moderation_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            performed_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
            moderator_id INTEGER NOT NULL,
            decision TEXT NOT NULL,
            reason TEXT,
            took_ms INTEGER,
            FOREIGN KEY(event_id) REFERENCES moderation_events(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_moderation_actions_event
            ON moderation_actions(event_id);
        CREATE INDEX IF NOT EXISTS idx_moderation_actions_moderator
            ON moderation_actions(moderator_id, performed_at);

        CREATE TABLE IF NOT EXISTS users (
            telegram_id INTEGER PRIMARY KEY,
            username TEXT,
            first_seen_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
            last_seen_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
            is_whitelisted BOOLEAN NOT NULL DEFAULT 0,
            is_banned BOOLEAN NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS log_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
            level TEXT NOT NULL,
            logger TEXT NOT NULL,
            message TEXT NOT NULL,
            context TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_log_events_level_time
            ON log_events(level, created_at);

        CREATE TABLE IF NOT EXISTS metrics_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_start DATETIME NOT NULL,
            period TEXT NOT NULL,
            spam_detected INTEGER NOT NULL DEFAULT 0,
            manual_overrides INTEGER NOT NULL DEFAULT 0,
            avg_spam_score REAL,
            embed_failures INTEGER NOT NULL DEFAULT 0,
            UNIQUE(period_start, period)
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)
        );
        """,
    ),
)

__all__ = ["MIGRATIONS"]

