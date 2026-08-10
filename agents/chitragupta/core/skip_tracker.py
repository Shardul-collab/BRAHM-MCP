# core/skip_tracker.py

"""
Skip Tracker — learns which fields a user habitually skips.

Persists a per-database skip count for each field.  When a field has been
skipped _AUTO_SKIP_THRESHOLD times in a row without being answered, the FSM
auto-skips it in future sessions with a short spoken notice.  The count resets
whenever the user actually answers the field, so deliberately skipping a field
for a stretch doesn't permanently disable it.

Storage: data/skip_tracker.json
Format:
    {
        "Daily Log": {
            "Goals": {"skips": 5, "answers": 1},
            "URL":   {"skips": 3, "answers": 0}
        }
    }
"""

import json
import logging
from pathlib import Path

from config.settings import DATA_DIR

logger = logging.getLogger("chitragupta.skip_tracker")

_TRACKER_PATH    = DATA_DIR / "skip_tracker.json"
_AUTO_SKIP_THRESHOLD = 3   # skip N consecutive times → auto-skip next session


# ── Internal I/O ──────────────────────────────────────────────────────────────

def _load() -> dict:
    if not _TRACKER_PATH.exists():
        return {}
    try:
        return json.loads(_TRACKER_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("skip_tracker: could not read — %s", exc)
        return {}


def _save(data: dict) -> None:
    try:
        _TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TRACKER_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error("skip_tracker: could not write — %s", exc)


def _field_record(data: dict, db: str, field: str) -> dict:
    """Return (and create if missing) the record for a db/field pair."""
    db_data    = data.setdefault(db, {})
    field_data = db_data.setdefault(field, {"skips": 0, "answers": 0})
    return field_data


# ── Public API ────────────────────────────────────────────────────────────────

def record_skip(database_name: str, field_name: str) -> None:
    """Increment the skip counter for this field."""
    data   = _load()
    record = _field_record(data, database_name, field_name)
    record["skips"] += 1
    _save(data)
    logger.debug(
        "skip_tracker: skip | db='%s' field='%s' total_skips=%d",
        database_name, field_name, record["skips"],
    )


def record_answered(database_name: str, field_name: str) -> None:
    """
    Increment the answer counter and reset the skip counter.
    Called whenever the user provides a real answer for a field.
    """
    data   = _load()
    record = _field_record(data, database_name, field_name)
    record["answers"] += 1
    record["skips"]    = 0   # reset — user is engaging with this field again
    _save(data)
    logger.debug(
        "skip_tracker: answered | db='%s' field='%s' total_answers=%d",
        database_name, field_name, record["answers"],
    )


def should_auto_skip(database_name: str, field_name: str) -> bool:
    """
    Return True if this field should be auto-skipped in the current session.
    Fires when skip count >= _AUTO_SKIP_THRESHOLD consecutive skips.
    """
    data   = _load()
    record = data.get(database_name, {}).get(field_name)
    if not record:
        return False
    return record.get("skips", 0) >= _AUTO_SKIP_THRESHOLD


def get_skip_count(database_name: str, field_name: str) -> int:
    """Return current consecutive skip count for a field."""
    data = _load()
    return data.get(database_name, {}).get(field_name, {}).get("skips", 0)


def reset_field(database_name: str, field_name: str) -> None:
    """Manually reset skip and answer counts for a field."""
    data   = _load()
    db_data = data.get(database_name, {})
    if field_name in db_data:
        db_data[field_name] = {"skips": 0, "answers": 0}
        _save(data)
