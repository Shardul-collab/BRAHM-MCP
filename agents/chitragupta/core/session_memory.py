# core/session_memory.py

"""
Session Memory — lightweight persistence for two Jarvis-style features.

Stored at: data/session_memory.json
No external dependencies — pure stdlib.

Features
--------
1. Skip learning
   record_skip(db, field)   → increment skip counter
   should_skip(db, field)   → True after _SKIP_THRESHOLD consecutive skips
   reset_skip(db, field)    → clear counter when user actually answers

2. Last-session values
   save_session(db, values) → persist field values after a confirmed save
   get_last_value(db, field) → retrieve last value for "same as last time?" offer
"""

import json
import logging
from pathlib import Path
from typing import Any

from config.settings import DATA_DIR

logger = logging.getLogger("chitragupta.session_memory")

_MEMORY_PATH: Path   = DATA_DIR / "session_memory.json"
_SKIP_THRESHOLD: int = 3


def _load() -> dict:
    if not _MEMORY_PATH.exists():
        return {"skip_counts": {}, "last_values": {}}
    try:
        data = json.loads(_MEMORY_PATH.read_text(encoding="utf-8"))
        data.setdefault("skip_counts", {})
        data.setdefault("last_values", {})
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("session_memory: read failed — %s", exc)
        return {"skip_counts": {}, "last_values": {}}


def _save(data: dict) -> None:
    try:
        _MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MEMORY_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("session_memory: write failed — %s", exc)


# ── Skip learning ─────────────────────────────────────────────────────────────

def record_skip(database_name: str, field_name: str) -> int:
    """Increment skip counter. Returns new count."""
    data = _load()
    db = data["skip_counts"].setdefault(database_name, {})
    db[field_name] = db.get(field_name, 0) + 1
    _save(data)
    return db[field_name]


def should_skip(database_name: str, field_name: str) -> bool:
    """True if field has been skipped >= _SKIP_THRESHOLD times."""
    data = _load()
    return data["skip_counts"].get(database_name, {}).get(field_name, 0) >= _SKIP_THRESHOLD


def reset_skip(database_name: str, field_name: str) -> None:
    """Clear skip counter when user actually answers the field."""
    data = _load()
    if database_name in data["skip_counts"]:
        data["skip_counts"][database_name].pop(field_name, None)
    _save(data)


# ── Last-session values ───────────────────────────────────────────────────────

def save_session(database_name: str, field_values: dict[str, Any]) -> None:
    """Persist confirmed field values from a completed session."""
    data = _load()
    data["last_values"][database_name] = {
        k: v for k, v in field_values.items()
        if v is not None and v != "" and v != [] and v is not False
    }
    _save(data)


def get_last_value(database_name: str, field_name: str) -> Any:
    """Return last confirmed value for a field, or None."""
    data = _load()
    return data["last_values"].get(database_name, {}).get(field_name)
