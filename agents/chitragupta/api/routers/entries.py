# api/routers/entries.py

"""
Entries router.

FIX 4 — Background processing
    POST /entries/{name} validates the payload synchronously then queues
    the Notion write as a BackgroundTask.  The endpoint returns 202
    immediately with a submission_id.  The background task calls
    create_page() which uses the two-phase write journal (mark_pending →
    mark_written) so /pending always reflects real queue state.

FIX 6 — Session auto-update
    After a successful Notion write the background task updates
    session_memory last-values and resets skip counters for all fields
    that received a non-trivial value.  No manual sync needed.

FIX 7 — Pagination (originally offset-based, now cursor-based)
    GET /entries/{name} accepts limit (1–100, default 10) and an optional
    cursor (opaque string from previous response's next_cursor field).
    Returns PaginatedEntriesResponse with items / limit / next_cursor.

    AUDIT FIX MAJ-1 — True cursor-based pagination replacing fake offset.
    The previous implementation called query_database() which fetched ALL
    Notion pages, then sliced in memory.  A limit=10 request on a 1,000-row
    database issued 10 Notion API calls and loaded the full dataset.  The
    endpoint now delegates pagination to query_database_page() which makes
    exactly ONE Notion API call and returns a cursor for the next page.

    NOTE: The `offset` query parameter has been removed.  If you need
    offset-style access, use cursor pagination and walk forward, or use the
    Notion filter API to narrow your result set before paginating.

    MODEL CHANGE REQUIRED: PaginatedEntriesResponse in api/models.py must
    have `next_cursor: str | None = None` added and `offset: int` can be
    removed (or kept as deprecated for backward compat).

FIX 8 — Structured Notion errors
    notion_error() converts NotionAPIError into retryable/non-retryable
    structured JSON.  Timeout and rate-limit cases are identified and
    marked retryable=True.

AUDIT FIX MIN-1 — /pending now has a response_model
    The endpoint previously returned a bare dict with no Swagger schema.
    A typed PendingWritesResponse model is used (see models.py).

AUDIT FIX MIN-6 — limit validated by FastAPI Query() constraint
    Previously limit/offset were validated by hand with if/raise logic.
    FastAPI Query(ge=1, le=100) enforces bounds declaratively and produces
    proper 422 Unprocessable Entity responses with field-level detail.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Query, status

from api.models import (
    LogEntryRequest, LogEntryResponse,
    PaginatedEntriesResponse,
    PendingWritesResponse,          # MIN-1: new model — add to api/models.py
)
from api.dependencies import api_key_auth, http_error, notion_error

from notion.schema_manager import load_schema, SchemaMissingError
from notion.notion_client import (
    create_page, query_database_page, NotionAPIError,
)
from notion.write_journal import (
    new_submission_id, mark_pending, pending_for_database,
)
from core.json_builder import build_json, json_to_notion_properties, ValidationError

logger = logging.getLogger("chitragupta.api.entries")

router = APIRouter(
    prefix="/entries",
    tags=["Entries"],
    dependencies=[Depends(api_key_auth)],
)


# ── Background task ───────────────────────────────────────────────────────────

def _write_entry_to_notion(
    database_name:  str,
    notion_db_id:   str,
    notion_payload: dict[str, Any],
    validated:      dict[str, Any],
    submission_id:  str,
) -> None:
    """
    Runs in background after the endpoint returns 202.

    Steps:
    1. Call create_page() — the caller (this router) has already called
       mark_pending(), so create_page skips its own mark_pending call
       (CRIT-2 fix in notion_client.py).  create_page calls mark_written()
       on success.
    2. On success update session_memory last-values and reset skip counters
       for all answered fields.
    """
    try:
        create_page(notion_db_id, notion_payload, submission_id=submission_id)
        logger.info(
            "Background write complete | db='%s' submission=%s",
            database_name, submission_id,
        )

        # Auto-update session memory after successful write (non-fatal)
        try:
            from core.session_memory import save_session, reset_skip
            save_session(database_name, validated)
            for field_name, value in validated.items():
                if value is not None and value != "" and value != []:
                    reset_skip(database_name, field_name)
        except Exception as mem_exc:
            logger.debug("Session memory update failed (non-fatal): %s", mem_exc)

    except NotionAPIError as exc:
        logger.error(
            "Background write failed | db='%s' submission=%s error=%s",
            database_name, submission_id, exc,
        )
    except Exception as exc:
        logger.exception(
            "Background write unexpected error | db='%s' submission=%s",
            database_name, submission_id,
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/{database_name}",
    response_model=LogEntryResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def log_entry(
    database_name:    str,
    body:             LogEntryRequest,
    background_tasks: BackgroundTasks,
) -> LogEntryResponse:
    """
    Validate fields synchronously, queue Notion write in background.

    Returns 202 immediately with a submission_id.
    Use GET /entries/{name}/pending to check if the write has completed.

    Example body:
    {
        "fields": {
            "Title":      "Evening log",
            "Mood":       8,
            "Activities": ["Yoga", "Reading"],
            "Notes":      "Felt focused today."
        }
    }
    """
    # Load schema
    try:
        schema = load_schema(database_name)
    except SchemaMissingError as exc:
        raise http_error(status.HTTP_404_NOT_FOUND, str(exc))

    notion_db_id = schema.get("notion_database_id", "").strip()
    if not notion_db_id:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"'{database_name}' has no Notion ID. Run 'Create Database' first.",
        )

    # Validate payload synchronously — fail fast before queuing
    try:
        validated      = build_json(database_name, body.fields)
        notion_payload = json_to_notion_properties(database_name, validated)
    except ValidationError as exc:
        raise http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))

    # Allocate submission ID and mark pending BEFORE handing off to background.
    # create_page() will NOT call mark_pending again because we pass sid here
    # (see CRIT-2 fix in notion_client.py).
    sid = new_submission_id()
    mark_pending(sid, notion_db_id)

    background_tasks.add_task(
        _write_entry_to_notion,
        database_name,
        notion_db_id,
        notion_payload,
        validated,
        sid,
    )

    logger.info("Entry queued | db='%s' submission=%s", database_name, sid)
    return LogEntryResponse(
        status="queued",
        submission_id=sid,
        database_name=database_name,
        message=(
            f"Entry queued for '{database_name}'. "
            f"Check /entries/{database_name}/pending with submission_id={sid} to confirm."
        ),
    )


@router.get("/{database_name}", response_model=PaginatedEntriesResponse)
async def get_entries(
    database_name: str,
    # MIN-6: bounds enforced declaratively via Query so FastAPI returns a
    # proper 422 with field-level detail instead of a hand-rolled http_error.
    limit:  int          = Query(default=10, ge=1, le=100, description="Max results to return"),
    cursor: str | None   = Query(default=None,             description="Pagination cursor from previous response's next_cursor"),
) -> PaginatedEntriesResponse:
    """
    MAJ-1 FIX: Fetch a single page of entries with true cursor-based pagination.

    Each response includes a next_cursor field.  Pass it as the cursor
    query parameter on the next request to get the following page.  A null
    next_cursor means you have reached the last page.

    - limit:  max items to return (1–100, default 10)
    - cursor: opaque pagination token from the previous response
    """
    try:
        schema = load_schema(database_name)
    except SchemaMissingError as exc:
        raise http_error(status.HTTP_404_NOT_FOUND, str(exc))

    notion_db_id = schema.get("notion_database_id", "").strip()
    if not notion_db_id:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"'{database_name}' has no Notion ID.",
        )

    try:
        # MAJ-1: single Notion API call — returns exactly `limit` items.
        items, next_cursor = query_database_page(
            notion_db_id,
            page_size=limit,
            start_cursor=cursor,
        )
    except NotionAPIError as exc:
        raise notion_error(exc)

    logger.info(
        "Entries queried | db='%s' limit=%d cursor=%s returned=%d has_more=%s",
        database_name, limit, cursor, len(items), next_cursor is not None,
    )
    return PaginatedEntriesResponse(
        database_name=database_name,
        items=items,
        total=len(items),
        limit=limit,
        next_cursor=next_cursor,   # MODEL CHANGE: add next_cursor to PaginatedEntriesResponse
    )


@router.get(
    "/{database_name}/pending",
    response_model=PendingWritesResponse,   # MIN-1: was untyped bare dict
)
async def get_pending_writes(database_name: str) -> PendingWritesResponse:
    """
    List unconfirmed write journal entries for a database.

    A pending entry means the background task queued the write but the
    Notion API has not yet confirmed it.  Non-empty results indicate a
    write is in flight or failed silently.

    MODEL CHANGE: add PendingWritesResponse to api/models.py:
        class PendingWritesResponse(BaseModel):
            database_name: str
            pending_count: int
            entries: list[dict]
    """
    try:
        schema = load_schema(database_name)
    except SchemaMissingError as exc:
        raise http_error(status.HTTP_404_NOT_FOUND, str(exc))

    notion_db_id = schema.get("notion_database_id", "").strip()
    pending      = pending_for_database(notion_db_id) if notion_db_id else []

    return PendingWritesResponse(
        database_name=database_name,
        pending_count=len(pending),
        entries=pending,
    )
