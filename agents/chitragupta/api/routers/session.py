# api/routers/session.py

"""
Session router — expose session memory for skip learning and last values.

Endpoints
---------
GET    /session/{database_name}/last-values          — get last confirmed values
GET    /session/{database_name}/skip-counts          — get per-field skip counts
DELETE /session/{database_name}/skip/{field_name}    — reset skip counter for a field
DELETE /session/{database_name}                      — clear all session data for a DB
"""

from __future__ import annotations

import json
import logging
from fastapi import APIRouter, Depends, status

from api.models import LastValuesResponse, SkipCountsResponse, StatusResponse
from api.dependencies import api_key_auth, http_error

from config.settings import DATA_DIR

logger = logging.getLogger("chitragupta.api.session")

_MEMORY_PATH = DATA_DIR / "session_memory.json"

router = APIRouter(
    prefix="/session",
    tags=["Session"],
    dependencies=[Depends(api_key_auth)],
)


def _load_memory() -> dict:
    if not _MEMORY_PATH.exists():
        return {"skip_counts": {}, "last_values": {}}
    try:
        data = json.loads(_MEMORY_PATH.read_text(encoding="utf-8"))
        data.setdefault("skip_counts", {})
        data.setdefault("last_values",  {})
        return data
    except Exception:
        return {"skip_counts": {}, "last_values": {}}


def _save_memory(data: dict) -> None:
    _MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MEMORY_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@router.get("/{database_name}/last-values", response_model=LastValuesResponse)
async def get_last_values(database_name: str) -> LastValuesResponse:
    """Return the last confirmed field values for a database."""
    data   = _load_memory()
    values = data["last_values"].get(database_name, {})
    return LastValuesResponse(database_name=database_name, last_values=values)


@router.get("/{database_name}/skip-counts", response_model=SkipCountsResponse)
async def get_skip_counts(database_name: str) -> SkipCountsResponse:
    """Return the per-field skip counters for a database."""
    data   = _load_memory()
    counts = data["skip_counts"].get(database_name, {})
    return SkipCountsResponse(database_name=database_name, skip_counts=counts)


@router.delete("/{database_name}/skip/{field_name}", response_model=StatusResponse)
async def reset_field_skip(database_name: str, field_name: str) -> StatusResponse:
    """Reset the skip counter for a specific field."""
    try:
        from core.session_memory import reset_skip
        reset_skip(database_name, field_name)
    except Exception as exc:
        raise http_error(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))
    return StatusResponse(
        ok=True,
        message=f"Skip counter for '{field_name}' in '{database_name}' reset.",
    )


@router.delete("/{database_name}", response_model=StatusResponse)
async def clear_session(database_name: str) -> StatusResponse:
    """Clear all session memory (last values + skip counts) for a database."""
    data = _load_memory()
    data["skip_counts"].pop(database_name, None)
    data["last_values"].pop(database_name, None)
    _save_memory(data)
    logger.info("API: session cleared | db='%s'", database_name)
    return StatusResponse(
        ok=True,
        message=f"Session data cleared for '{database_name}'.",
    )
