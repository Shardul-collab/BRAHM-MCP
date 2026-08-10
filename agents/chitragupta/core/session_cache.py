# core/session_cache.py

"""
Session Cache — stores the last successfully saved entry per database.

Used by the FSM to offer "same as last time?" for multi-select and select
fields so the user doesn't have to repeat common values every session.

Storage: data/session_cache.json
Format:
    {
        "Daily Log": {
            "values":     {"Activities": ["Yoga", "Reading"], "Mood": 7},
            "saved_at":   "2026-04-21T09:55:00+00:00"
        }
    }
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import DATA_DIR

logger = logging.getLogger("chitragupta.session_cache")

_CACHE_PATH = DATA_DIR / "session_cache.json"


# ── Internal I/O ──────────────────────────────────────────────────────────────

def _load() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("session_cache: could not read — %s", exc)
        return {}


def _save(data: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.error("session_cache: could not write — %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def save_session(database_name: str, values: dict) -> None:
    """
    Persist the field values from a completed, saved entry.
    Called by main.py after a successful Notion page creation.

    Only stores fields that are useful for "same as last time?" suggestions:
    multi_select, select, number, and checkbox.  Omits title, date, rich_text
    since those are almost always different each session.
    """
    # Filter to repeatable fields only (avoid "today's title" or "yesterday's notes")
    _REPEATABLE_TYPES = {"multi_select", "select", "number", "checkbox"}
    filtered = {
        k: v for k, v in values.items()
        if not isinstance(v, str) or k.lower() not in ("title", "notes", "date")
    }

    cache = _load()
    cache[database_name] = {
        "values":   filtered,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(cache)
    logger.info(
        "session_cache: saved | db='%s' fields=%d",
        database_name, len(filtered),
    )


def get_last_values(database_name: str) -> dict:
    """
    Return the cached field values from the last saved session.
    Returns {} if no cache exists for this database.
    """
    cache = _load()
    entry = cache.get(database_name, {})
    values = entry.get("values", {})
    if values:
        saved_at = entry.get("saved_at", "unknown")
        logger.info(
            "session_cache: loaded | db='%s' fields=%d saved_at=%s",
            database_name, len(values), saved_at,
        )
    return values


def clear_cache(database_name: str) -> None:
    """Remove cached session for a specific database."""
    cache = _load()
    if database_name in cache:
        del cache[database_name]
        _save(cache)
        logger.info("session_cache: cleared | db='%s'", database_name)
