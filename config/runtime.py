"""
Runtime configuration overrides that can be changed without restarting the bot.
.env provides defaults, while overrides are persisted in storage/settings.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Mapping

from config.config import settings
from storage import init_storage
from utils.logger import get_logger

LOGGER = get_logger(__name__)
STORAGE = init_storage()


def _serialize_override(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _deserialize_override(raw: str) -> object:
    return json.loads(raw)


def _values_equal(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    return a == b


@dataclass
class RuntimeConfig:
    """Mutable configuration that can be overridden at runtime."""

    policy_mode: str = field(default_factory=lambda: settings.POLICY_MODE)

    meta_notify: float = field(default_factory=lambda: settings.META_NOTIFY)
    meta_delete: float = field(default_factory=lambda: settings.META_DELETE)
    meta_kick: float = field(default_factory=lambda: settings.META_KICK)

    meta_downweight_announcement: float = field(
        default_factory=lambda: settings.META_DOWNWEIGHT_ANNOUNCEMENT
    )
    meta_downweight_reply_to_staff: float = field(
        default_factory=lambda: settings.META_DOWNWEIGHT_REPLY_TO_STAFF
    )
    meta_downweight_whitelist: float = field(
        default_factory=lambda: settings.META_DOWNWEIGHT_WHITELIST
    )

    _overrides: Dict[str, object] = field(default_factory=dict, repr=False)

    def apply_overrides(self, overrides: Mapping[str, str], *, persist: bool = False) -> None:
        """Apply overrides loaded from storage. persist=False prevents double writes."""
        for name, raw_value in overrides.items():
            try:
                value = _deserialize_override(raw_value)
            except (TypeError, json.JSONDecodeError) as exc:
                LOGGER.warning("Failed to decode override %s: %s (%s)", name, raw_value, exc)
                continue

            if name == "policy_mode":
                self.set_policy_mode(str(value), persist=persist)
            elif name in {"meta_notify", "meta_delete", "meta_kick"}:
                self.set_threshold(name, float(value), persist=persist)
            elif name in {
                "meta_downweight_announcement",
                "meta_downweight_reply_to_staff",
                "meta_downweight_whitelist",
            }:
                suffix = name[len("meta_downweight_") :]
                self.set_downweight(suffix, float(value), persist=persist)
            else:
                LOGGER.debug("Ignoring unknown override key: %s", name)

    def set_policy_mode(self, mode: str, *, persist: bool = True) -> bool:
        if not mode:
            return False
        normalized = mode.strip().lower().replace("_", "-")
        if normalized not in {"manual", "semi-auto", "auto", "legacy-manual"}:
            return False

        current = self.policy_mode
        if normalized == current:
            return False

        self.policy_mode = normalized
        self._record_override("policy_mode", normalized, persist=persist)

        if persist:
            LOGGER.info("Policy mode changed: %s -> %s", current, normalized)
        return True

    def set_threshold(self, name: str, value: float, *, persist: bool = True) -> bool:
        if value < 0.0 or value > 1.0:
            return False

        key = name.lower().replace("-", "_")
        if key not in {"meta_notify", "meta_delete", "meta_kick"}:
            return False

        current = getattr(self, key)
        if _values_equal(current, value):
            return False

        setattr(self, key, value)
        self._record_override(key, value, persist=persist)

        if persist:
            LOGGER.info("Threshold %s changed: %.2f -> %.2f", key, current, value)
        return True

    def set_downweight(self, name: str, value: float, *, persist: bool = True) -> bool:
        if value < 0.0 or value > 1.0:
            LOGGER.warning("Invalid downweight value: %s (must be 0.0-1.0)", value)
            return False

        mapping = {
            "announcement": "meta_downweight_announcement",
            "reply_to_staff": "meta_downweight_reply_to_staff",
            "whitelist": "meta_downweight_whitelist",
        }

        key = name.lower().replace("-", "_")
        attr = mapping.get(key)
        if not attr:
            LOGGER.warning("Unknown downweight: %s", name)
            return False

        current = getattr(self, attr)
        if _values_equal(current, value):
            return False

        setattr(self, attr, value)
        self._record_override(attr, value, persist=persist)

        if persist:
            LOGGER.info("Downweight %s changed: %.2f -> %.2f", attr, current, value)
        return True

    def get_overrides(self) -> Dict[str, object]:
        return self._overrides.copy()

    def reset_overrides(self) -> None:
        LOGGER.info("Resetting all overrides to .env defaults")

        self.policy_mode = settings.POLICY_MODE
        self.meta_notify = settings.META_NOTIFY
        self.meta_delete = settings.META_DELETE
        self.meta_kick = settings.META_KICK
        self.meta_downweight_announcement = settings.META_DOWNWEIGHT_ANNOUNCEMENT
        self.meta_downweight_reply_to_staff = settings.META_DOWNWEIGHT_REPLY_TO_STAFF
        self.meta_downweight_whitelist = settings.META_DOWNWEIGHT_WHITELIST

        for key in list(self._overrides.keys()):
            STORAGE.settings.remove(key)

        self._overrides.clear()

    def is_default(self, name: str) -> bool:
        return name not in self._overrides

    def _record_override(self, name: str, value: object, *, persist: bool) -> None:
        default_attr = name.upper()
        default_value = getattr(settings, default_attr, None)

        if default_value is not None and _values_equal(default_value, value):
            if name in self._overrides:
                self._overrides.pop(name, None)
                if persist:
                    STORAGE.settings.remove(name)
            return

        self._overrides[name] = value
        if persist:
            STORAGE.settings.upsert(name, _serialize_override(value))


runtime_config = RuntimeConfig()
try:
    runtime_config.apply_overrides(STORAGE.settings.load_overrides(), persist=False)
except Exception as exc:
    LOGGER.warning("Failed to apply runtime overrides: %s", exc)

__all__ = ["runtime_config", "RuntimeConfig"]

