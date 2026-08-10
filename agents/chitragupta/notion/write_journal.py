# notion/write_journal.py

"""
Write Journal — two-phase write model for Notion page creation.

Responsibilities
----------------
- Record write intent BEFORE the API call (WRITE_PENDING)
- Mark completion AFTER successful API response (WRITE_WRITTEN)
- Expose pending submissions so callers can warn before retry

On any retry after a network fault, the caller can detect that a
submission was already attempted and warn the user instead of silently
creating a duplicate page in Notion.

Journal file: data/write_journal.json
Format: {submission_id: {database_id, status, created_at, page_id}}
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("chitragupta.write_journal")

_JOURNAL_PATH = Path(__file__).resolve().parent.parent / "data" / "write_journal.json"
_STATUS_PENDING = "pending"
_STATUS_WRITTEN = "written"


# ── Internal I/O (never raises — journal failure must not block writes) ────────

def _load() -> dict:
    if not _JOURNAL_PATH.exists():
        return {}
    try:
        return json.loads(_JOURNAL_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("write_journal: could not read journal — %s", exc)
        return {}


def _save(journal: dict) -> None:
    try:
        _JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        _JOURNAL_PATH.write_text(json.dumps(journal, indent=2), encoding="utf-8")
    except OSError as exc:
        # Journal write failure MUST NOT block the caller
        logger.error("write_journal: could not write journal — %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def new_submission_id() -> str:
    """Generate a unique submission ID for one write attempt."""
    return str(uuid.uuid4())


def mark_pending(submission_id: str, database_id: str) -> None:
    """
    Record write intent before hitting the API.
    Call this immediately before create_page().
    """
    journal = _load()
    journal[submission_id] = {
        "database_id": database_id,
        "status":      _STATUS_PENDING,
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "page_id":     None,
    }
    _save(journal)
    logger.debug("write_journal: pending | submission=%s db=%s", submission_id, database_id)


def mark_written(submission_id: str, page_id: str) -> None:
    """
    Mark a submission as successfully written.
    Call this immediately after create_page() returns.
    """
    journal = _load()
    if submission_id in journal:
        journal[submission_id]["status"]  = _STATUS_WRITTEN
        journal[submission_id]["page_id"] = page_id
        _save(journal)
    logger.debug("write_journal: written | submission=%s page=%s", submission_id, page_id)


def pending_for_database(database_id: str) -> list[dict]:
    """
    Return all PENDING (unconfirmed) submissions for a given database.
    A non-empty list means a previous write attempt may have succeeded
    in Notion but the response was never received — duplicate risk.
    """
    journal = _load()
    return [
        {"submission_id": sid, **entry}
        for sid, entry in journal.items()
        if entry.get("status") == _STATUS_PENDING
        and entry.get("database_id") == database_id
    ]


def clear_stale(older_than_hours: int = 24) -> int:
    """
    Remove WRITTEN entries older than `older_than_hours`.
    Returns the number of entries removed.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - (older_than_hours * 3600)
    journal = _load()
    before  = len(journal)
    journal = {
        sid: entry for sid, entry in journal.items()
        if not (
            entry.get("status") == _STATUS_WRITTEN
            and _parse_ts(entry.get("created_at", "")) < cutoff
        )
    }
    _save(journal)
    removed = before - len(journal)
    if removed:
        logger.debug("write_journal: cleared %d stale entries.", removed)
    return removed


def _parse_ts(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError):
        return 0.0
