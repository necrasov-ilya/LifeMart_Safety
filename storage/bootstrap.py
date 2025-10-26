from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock
from typing import Optional

from .migrations import MIGRATIONS
from .sqlite import Storage

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "storage.sqlite"

_storage_instance: Optional[Storage] = None
_storage_lock = RLock()


def init_storage(*, db_path: Optional[Path] = None) -> Storage:
    """
    Initialise storage singleton. Ensures migrations are applied and the connection
    is configured with sane defaults for concurrent access from async handlers.
    """
    global _storage_instance

    if _storage_instance is not None:
        return _storage_instance

    with _storage_lock:
        if _storage_instance is not None:
            return _storage_instance

        path = Path(db_path) if db_path else DEFAULT_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row

        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")

        _apply_migrations(conn)

        _storage_instance = Storage(conn=conn)
        return _storage_instance


def get_storage() -> Storage:
    if _storage_instance is None:
        raise RuntimeError("Storage was not initialised. Call init_storage() first.")
    return _storage_instance


def _apply_migrations(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)")

    applied = {
        row["version"]
        for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")
    }

    for version, sql in MIGRATIONS:
        if version in applied:
            continue

        with conn:
            conn.executescript(sql)
            conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))

