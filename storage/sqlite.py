from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from threading import RLock
from typing import Iterable, Mapping, Sequence

from .interfaces import (
    MetricsSnapshot,
    MetricsStore,
    ModerationAction,
    ModerationActionInput,
    ModerationEvent,
    ModerationEventInput,
    ModerationStore,
    SettingsStore,
    UserProfile,
    UserStore,
)


class Storage:
    """
    Entry point for interacting with SQLite-backed repositories. Keeps a single
    connection guarded by an RLock; operations are small and executed serially.
    """

    def __init__(self, *, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = RLock()
        self.events: ModerationStore = _ModerationStore(conn, self._lock)
        self.users: UserStore = _UserStore(conn, self._lock)
        self.metrics: MetricsStore = _MetricsStore(conn, self._lock)
        self.settings: SettingsStore = _SettingsStore(conn, self._lock)
        self.logs = _LogStore(conn, self._lock)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class _SQLiteRepoBase:
    def __init__(self, conn: sqlite3.Connection, lock: RLock):
        self._conn = conn
        self._lock = lock

    @contextlib.contextmanager
    def _cursor(self) -> Iterable[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()


class _ModerationStore(_SQLiteRepoBase, ModerationStore):
    def record_event(self, data: ModerationEventInput) -> int:
        payload = (
            data.chat_id,
            data.message_id,
            data.user_id,
            data.username,
            data.text_hash,
            data.text_length,
            data.action,
            data.action_confidence,
            data.filter_keyword_score,
            data.filter_tfidf_score,
            data.filter_embedding_score,
            data.meta_debug,
            data.source,
        )

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO moderation_events (
                    chat_id, message_id, user_id, username, text_hash, text_length,
                    action, action_confidence, filter_keyword_score,
                    filter_tfidf_score, filter_embedding_score, meta_debug, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            return int(cur.lastrowid)

    def fetch_recent(self, limit: int = 100) -> Sequence[ModerationEvent]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, created_at, chat_id, message_id, user_id, username, text_hash,
                    text_length, action, action_confidence, filter_keyword_score,
                    filter_tfidf_score, filter_embedding_score, meta_debug, source
                FROM moderation_events
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()

        return [
            ModerationEvent(
                id=row["id"],
                created_at=_parse_dt(row["created_at"]),
                chat_id=row["chat_id"],
                message_id=row["message_id"],
                user_id=row["user_id"],
                username=row["username"],
                text_hash=row["text_hash"],
                text_length=row["text_length"],
                action=row["action"],
                action_confidence=row["action_confidence"],
                filter_keyword_score=row["filter_keyword_score"],
                filter_tfidf_score=row["filter_tfidf_score"],
                filter_embedding_score=row["filter_embedding_score"],
                meta_debug=row["meta_debug"],
                source=row["source"],
            )
            for row in rows
        ]

    def record_action(self, data: ModerationActionInput) -> int:
        payload = (
            data.event_id,
            data.moderator_id,
            data.decision,
            data.reason,
            data.took_ms,
        )

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO moderation_actions (
                    event_id, moderator_id, decision, reason, took_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                payload,
            )
            return int(cur.lastrowid)

    def fetch_actions(self, event_id: int) -> Sequence[ModerationAction]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, event_id, performed_at, moderator_id, decision, reason, took_ms
                FROM moderation_actions
                WHERE event_id = ?
                ORDER BY performed_at ASC
                """,
                (event_id,),
            )
            rows = cur.fetchall()

        return [
            ModerationAction(
                id=row["id"],
                event_id=row["event_id"],
                performed_at=_parse_dt(row["performed_at"]),
                moderator_id=row["moderator_id"],
                decision=row["decision"],
                reason=row["reason"],
                took_ms=row["took_ms"],
            )
            for row in rows
        ]


class _UserStore(_SQLiteRepoBase, UserStore):
    def upsert(
        self,
        telegram_id: int,
        *,
        username: str | None,
        is_whitelisted: bool | None = None,
        is_banned: bool | None = None,
    ) -> None:
        flags = {}
        if is_whitelisted is not None:
            flags["is_whitelisted"] = int(is_whitelisted)
        if is_banned is not None:
            flags["is_banned"] = int(is_banned)

        set_clause = ", ".join(f"{k} = :{k}" for k in flags)
        sql = """
            INSERT INTO users(telegram_id, username, is_whitelisted, is_banned)
            VALUES (:telegram_id, :username,
                    COALESCE(:is_whitelisted, 0),
                    COALESCE(:is_banned, 0))
            ON CONFLICT(telegram_id) DO UPDATE SET
                username = excluded.username,
                last_seen_at = CURRENT_TIMESTAMP
        """
        if set_clause:
            sql += ", " + set_clause

        payload = {
            "telegram_id": telegram_id,
            "username": username,
            "is_whitelisted": flags.get("is_whitelisted"),
            "is_banned": flags.get("is_banned"),
        }

        with self._cursor() as cur:
            cur.execute(sql, payload)

    def touch(self, telegram_id: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET last_seen_at = CURRENT_TIMESTAMP
                WHERE telegram_id = ?
                """,
                (telegram_id,),
            )

    def fetch(self, telegram_id: int) -> UserProfile | None:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT telegram_id, username, first_seen_at, last_seen_at,
                       is_whitelisted, is_banned
                FROM users
                WHERE telegram_id = ?
                """,
                (telegram_id,),
            )
            row = cur.fetchone()

        if not row:
            return None

        return UserProfile(
            telegram_id=row["telegram_id"],
            username=row["username"],
            first_seen_at=_parse_dt(row["first_seen_at"]),
            last_seen_at=_parse_dt(row["last_seen_at"]),
            is_whitelisted=bool(row["is_whitelisted"]),
            is_banned=bool(row["is_banned"]),
        )


class _MetricsStore(_SQLiteRepoBase, MetricsStore):
    def record_snapshot(self, snapshot: MetricsSnapshot) -> None:
        payload = asdict(snapshot)
        payload["period_start"] = _format_dt(snapshot.period_start)

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO metrics_snapshots (
                    period_start, period, spam_detected, manual_overrides,
                    avg_spam_score, embed_failures
                )
                VALUES (:period_start, :period, :spam_detected, :manual_overrides,
                        :avg_spam_score, :embed_failures)
                ON CONFLICT(period_start, period) DO UPDATE SET
                    spam_detected = excluded.spam_detected,
                    manual_overrides = excluded.manual_overrides,
                    avg_spam_score = excluded.avg_spam_score,
                    embed_failures = excluded.embed_failures
                """,
                payload,
            )

    def fetch_recent(self, limit: int = 30) -> Sequence[MetricsSnapshot]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT period_start, period, spam_detected, manual_overrides,
                       avg_spam_score, embed_failures
                FROM metrics_snapshots
                ORDER BY period_start DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()

        return [
            MetricsSnapshot(
                period_start=_parse_dt(row["period_start"]),
                period=row["period"],
                spam_detected=row["spam_detected"],
                manual_overrides=row["manual_overrides"],
                avg_spam_score=row["avg_spam_score"],
                embed_failures=row["embed_failures"],
            )
            for row in rows
        ]


class _SettingsStore(_SQLiteRepoBase, SettingsStore):
    def load_overrides(self) -> Mapping[str, str]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT key, value FROM settings ORDER BY key ASC",
            )
            rows = cur.fetchall()

        return {row["key"]: row["value"] for row in rows}

    def upsert(self, key: str, value: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO settings(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, value),
            )

    def remove(self, key: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM settings WHERE key = ?", (key,))


class _LogStore(_SQLiteRepoBase):
    def write(self, level: str, logger: str, message: str, context: dict | None = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO log_events(level, logger, message, context)
                VALUES (?, ?, ?, ?)
                """,
                (
                    level,
                    logger,
                    message,
                    json.dumps(context) if context else None,
                ),
            )


def _parse_dt(raw: str | datetime | None) -> datetime:
    if raw is None:
        return datetime.fromtimestamp(0)
    if isinstance(raw, datetime):
        return raw
    # SQLite returns ISO8601 strings
    return datetime.fromisoformat(raw)


def _format_dt(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")

