from __future__ import annotations

from pathlib import Path
from typing import Optional

from .bootstrap import init_storage, get_storage

__all__ = ["init_storage", "get_storage"]


def ensure_storage_initialized(db_path: Optional[Path] = None) -> None:
    """
    Convenience helper for modules that only need side-effects from storage startup
    (directory creation, migrations). Calls init_storage() without returning the instance.
    """
    init_storage(db_path=db_path)

