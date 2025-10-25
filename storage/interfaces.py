from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol, Sequence


@dataclass(slots=True)
class ModerationEventInput:
    chat_id: int
    message_id: int
    user_id: int
    username: str | None
    text_hash: str | None
    text_length: int | None
    action: str
    action_confidence: float | None
    filter_keyword_score: float | None
    filter_tfidf_score: float | None
    filter_embedding_score: float | None
    meta_debug: str | None
    source: str = "bot"


@dataclass(slots=True)
class ModerationEvent:
    id: int
    created_at: datetime
    chat_id: int
    message_id: int
    user_id: int
    username: str | None
    text_hash: str | None
    text_length: int | None
    action: str
    action_confidence: float | None
    filter_keyword_score: float | None
    filter_tfidf_score: float | None
    filter_embedding_score: float | None
    meta_debug: str | None
    source: str


@dataclass(slots=True)
class ModerationActionInput:
    event_id: int
    moderator_id: int
    decision: str
    reason: str | None
    took_ms: int | None


@dataclass(slots=True)
class ModerationAction:
    id: int
    event_id: int
    performed_at: datetime
    moderator_id: int
    decision: str
    reason: str | None
    took_ms: int | None


@dataclass(slots=True)
class UserProfile:
    telegram_id: int
    username: str | None
    first_seen_at: datetime
    last_seen_at: datetime
    is_whitelisted: bool
    is_banned: bool


@dataclass(slots=True)
class MetricsSnapshot:
    period_start: datetime
    period: str
    spam_detected: int
    manual_overrides: int
    avg_spam_score: float | None
    embed_failures: int


class ModerationStore(Protocol):
    def record_event(self, data: ModerationEventInput) -> int: ...

    def fetch_recent(self, limit: int = 100) -> Sequence[ModerationEvent]: ...

    def record_action(self, data: ModerationActionInput) -> int: ...

    def fetch_actions(self, event_id: int) -> Sequence[ModerationAction]: ...


class UserStore(Protocol):
    def upsert(
        self,
        telegram_id: int,
        *,
        username: str | None,
        is_whitelisted: bool | None = None,
        is_banned: bool | None = None,
    ) -> None: ...

    def touch(self, telegram_id: int) -> None: ...

    def fetch(self, telegram_id: int) -> UserProfile | None: ...


class MetricsStore(Protocol):
    def record_snapshot(self, snapshot: MetricsSnapshot) -> None: ...

    def fetch_recent(self, limit: int = 30) -> Sequence[MetricsSnapshot]: ...


class SettingsStore(Protocol):
    def load_overrides(self) -> Mapping[str, str]: ...

    def upsert(self, key: str, value: str) -> None: ...

    def remove(self, key: str) -> None: ...

