# notion/notion_client.py

"""
Thin, reusable wrapper around the Notion REST API.  (v0.5 — audit fixes)

Changes from v0.4
-----------------
AUDIT FIX C-001 / SECURITY — Sanitised NotionAPIError + notion_status field

    The previous _request() raised:
        NotionAPIError(f"[{status_code}] {code}: {message}")
    where `message = data.get("message", response.text)` is the verbatim
    Notion response body (e.g. "API token is invalid").  Callers that
    included str(exc) in their own error responses were therefore leaking
    raw Notion internals to end-users — the exact information leak the
    audit flagged under the Security Risks section.

    Fix: the full Notion error (status, code, message) is now written to
    the internal logger at ERROR level.  The NotionAPIError that propagates
    to callers carries only "[{status_code}] {code}" — enough for
    notion_error() in dependencies.py to classify the error, but containing
    no raw Notion message content that could expose integration details.

    NotionAPIError gains a `notion_status` integer field so that
    dependencies.notion_error() can classify errors by status code
    directly instead of parsing the string for "[429]" etc.  This makes
    the classification logic in dependencies.py more robust and removes
    the need for string-based status-code detection entirely.

    Network-level exceptions (ConnectionError, Timeout) already produced
    safe messages and are unchanged.

Previous changes (v0.4)
-----------------------
AUDIT FIX CRIT-2 — Double mark_pending() eliminated
    create_page() previously called mark_pending(sid, database_id)
    unconditionally, even when the caller (entries router, voice router)
    had already pre-allocated a submission_id and called mark_pending()
    itself before dispatching the background task.  The same sid was thus
    recorded as pending twice, meaning mark_written() could only ever
    satisfy one of the two records — the other remained pending forever
    and GET /entries/{name}/pending always returned stale data.

    Fix: mark_pending() is now called inside create_page() ONLY when no
    submission_id is supplied by the caller (i.e. when create_page is
    invoked directly without prior journal registration).  When the caller
    pre-registers the sid, create_page skips the redundant mark_pending.

AUDIT FIX MAJ-1 — True cursor-based single-page query
    The previous query_database() fetched ALL pages before returning,
    meaning GET /entries?limit=10 on a 1,000-row database still issued
    10 Notion API calls and loaded the full dataset into memory.  A new
    query_database_page() function fetches exactly one page of results
    from Notion and returns the Notion next_cursor for the caller to use
    on the next request.  The original query_database() is preserved for
    internal callers (analysis, drift check) that genuinely need all rows.
"""

import logging
import requests
from typing import Any

from config.settings import NOTION_TOKEN, NOTION_VERSION, NOTION_BASE_URL
from core.validator import ValidationError, guard_notion_payload
from notion.write_journal import new_submission_id, mark_pending, mark_written

logger = logging.getLogger("chitragupta.notion_client")


# ── Custom exception ──────────────────────────────────────────────────────────

class NotionAPIError(Exception):
    """
    Raised when the Notion API returns an error or is unreachable.

    AUDIT FIX C-001: Added `notion_status` field so that callers and
    dependencies.notion_error() can classify errors by HTTP status code
    directly, without parsing the exception message string.

    Attributes:
        notion_status: The HTTP status code returned by Notion, or 0 for
                       network-level failures (timeout, connection error).
    """
    def __init__(self, message: str, notion_status: int = 0) -> None:
        super().__init__(message)
        self.notion_status = notion_status


# ── Internal helpers ──────────────────────────────────────────────────────────

def _base_url() -> str:
    """
    Return the canonical Notion base URL with no trailing slash.

    NOTION_BASE_URL in .env may or may not carry a trailing "/".  Stripping
    it here means every call site can safely use "/endpoint" without worrying
    about configuration inconsistencies producing double-slash URLs.
    """
    return NOTION_BASE_URL.rstrip("/")


def _headers() -> dict[str, str]:
    """Build Notion API auth headers. Called fresh on every request."""
    return {
        "Authorization":  f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type":   "application/json",
    }


def _request(
    method: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Central HTTP dispatcher for all Notion API calls.

    Args:
        method:   HTTP verb — "GET", "POST", "PATCH"
        endpoint: Path after base URL, e.g. "/databases/{id}"
        payload:  JSON body (optional)

    Returns:
        Parsed JSON response as a dict.

    Raises:
        NotionAPIError: on any non-2xx response or network failure.

    AUDIT FIX C-001 / SECURITY:
        On a non-2xx response the full Notion error (status + code + raw
        message) is written to the internal logger at ERROR level so that
        operators can diagnose failures from the server logs.

        The NotionAPIError that propagates to callers contains only the
        HTTP status code and the Notion error code (e.g. "[401] unauthorized")
        — it deliberately omits the raw Notion message body.  This prevents
        router-level code that does `str(exc)` from accidentally forwarding
        Notion internals (e.g. "API token is invalid") to end-users.
    """
    url = f"{_base_url()}{endpoint}"

    logger.debug(
        "%s %s | payload_keys=%s", method, url,
        list(payload.keys()) if payload else [],
    )

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=_headers(),
            json=payload,
            timeout=30,
        )
    except requests.exceptions.ConnectionError as exc:
        raise NotionAPIError(
            f"Network error reaching Notion API: {exc}",
            notion_status=0,
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise NotionAPIError(
            "Notion API request timed out after 30 s.",
            notion_status=0,
        ) from exc

    try:
        data: dict[str, Any] = response.json()
    except ValueError:
        data = {"raw": response.text}

    if not response.ok:
        code    = data.get("code", "unknown")
        message = data.get("message", response.text)

        # AUDIT FIX C-001 / SECURITY: Log the full Notion error internally.
        # This is the only place the raw `message` value is used — it goes
        # to the server log, never into the exception that propagates upward.
        logger.error(
            "Notion API error | status=%s code=%s message=%s url=%s",
            response.status_code, code, message, url,
        )

        # Raise with a sanitised message: status code + Notion error code only.
        # The raw `message` string is intentionally excluded here.
        raise NotionAPIError(
            f"[{response.status_code}] {code}",
            notion_status=response.status_code,
        )

    logger.debug("Notion response status=%s", response.status_code)
    return data


# ── Public API ────────────────────────────────────────────────────────────────

def create_database(
    parent_page_id: str,
    title: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    """
    Create a new Notion database as a child of a given page.

    Args:
        parent_page_id: The Notion page ID that will contain this database.
        title:          Human-readable database name.
        properties:     Notion property schema dict.

    Returns:
        Full Notion API response (includes the new database's ID).
    """
    if not parent_page_id or not parent_page_id.strip():
        raise ValidationError(
            "parent_page_id is required to create a Notion database. "
            "Set NOTION_PAGE_ID in your .env or enter it when prompted."
        )

    payload: dict[str, Any] = {
        "parent": {
            "type":    "page_id",
            "page_id": parent_page_id,
        },
        "title": [
            {
                "type": "text",
                "text": {"content": title},
            }
        ],
        "properties": properties,
    }

    logger.info(
        "Creating Notion database | title='%s' parent=%s", title, parent_page_id
    )
    result = _request("POST", "/databases", payload)
    logger.info("Database created | id=%s", result.get("id"))
    return result


def get_database(database_id: str) -> dict[str, Any]:
    """
    Fetch metadata for an existing Notion database.

    Args:
        database_id: Notion database ID (32-char hex, with or without hyphens).

    Returns:
        Full Notion API response for the database object.
    """
    logger.info("Fetching database | id=%s", database_id)
    return _request("GET", f"/databases/{database_id}")


def create_page(
    database_id: str,
    properties: dict[str, Any],
    submission_id: str | None = None,
) -> dict[str, Any]:
    """
    Insert a new row (page) into a Notion database.

    Uses a two-phase write model:
      Phase 1 — mark_pending() before the HTTP call (intent recorded)
      Phase 2 — mark_written() after a successful response (confirmed)

    Args:
        database_id:   Target database ID.
        properties:    Notion-formatted property payload from json_builder.
        submission_id: Optional pre-allocated ID for the write journal entry.
                       When supplied the caller is assumed to have already
                       called mark_pending() — this function will NOT call it
                       again to avoid double-recording.

    Returns:
        Full Notion API response (includes the new page's ID).

    Raises:
        ValidationError: if payload fails pre-flight checks.
        NotionAPIError:  if the API returns an error.

    CRIT-2 FIX: Previously mark_pending(sid, database_id) was called
    unconditionally.  The entries and voice routers pre-allocate a sid and
    call mark_pending() before dispatching the background task, so create_page
    was marking the same sid pending a second time.  Now mark_pending() is
    only called when this function allocates a fresh sid itself.
    """
    guard_notion_payload(properties)

    caller_provided_sid = submission_id is not None
    sid = submission_id if caller_provided_sid else new_submission_id()

    # Only mark pending when WE allocated the sid.  If the caller pre-marked
    # it (e.g. entries router), skip to avoid double-recording in the journal.
    if not caller_provided_sid:
        mark_pending(sid, database_id)

    payload: dict[str, Any] = {
        "parent": {
            "type":        "database_id",
            "database_id": database_id,
        },
        "properties": properties,
    }

    logger.info(
        "Creating page in database | db=%s submission=%s", database_id, sid
    )
    result  = _request("POST", "/pages", payload)
    page_id = result.get("id", "")

    mark_written(sid, page_id)
    logger.info("Page created | id=%s", page_id)
    return result


def update_page(
    page_id: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    """
    Update properties on an existing Notion page (row).

    Args:
        page_id:    The Notion page ID to update.
        properties: Partial or full property payload in Notion format.

    Returns:
        Full Notion API response for the updated page.

    Raises:
        ValidationError: if payload fails pre-flight checks.
        NotionAPIError:  if the API returns an error.
    """
    guard_notion_payload(properties)
    payload: dict[str, Any] = {"properties": properties}
    logger.info("Updating page | id=%s", page_id)
    result = _request("PATCH", f"/pages/{page_id}", payload)
    logger.info("Page updated | id=%s", result.get("id"))
    return result


def check_schema_drift(
    database_id: str,
    local_fields: list[dict[str, Any]],
) -> list[str]:
    """
    Compare local schema fields against live Notion database properties.

    Returns a list of human-readable drift messages, empty if none.
    Never raises — a drift-check failure must not block a log entry.
    """
    try:
        remote = get_database(database_id)
    except NotionAPIError as exc:
        logger.warning("check_schema_drift: API call failed — %s", exc)
        return []

    remote_keys_raw: dict[str, str] = {
        k.lower(): k for k in remote.get("properties", {}).keys()
    }
    local_keys_raw: dict[str, str] = {
        f["name"].lower(): f["name"] for f in local_fields
    }

    remote_lower = set(remote_keys_raw.keys())
    local_lower  = set(local_keys_raw.keys())

    notion_only_lower = (remote_lower - local_lower) - {"title"}
    local_only_lower  = local_lower - remote_lower

    issues: list[str] = []

    for key in sorted(local_only_lower):
        display = local_keys_raw[key]
        issues.append(
            f"LOCAL ONLY  '{display}' — in schema file but missing from Notion. "
            "Values for this field will be silently dropped."
        )
    for key in sorted(notion_only_lower):
        display = remote_keys_raw[key]
        issues.append(
            f"NOTION ONLY '{display}' — exists in Notion but not in local schema. "
            "This field will never be populated by Chitragupta."
        )

    if issues:
        logger.warning(
            "Schema drift detected | db=%s issues=%d", database_id, len(issues)
        )
    return issues


def query_database_page(
    database_id: str,
    filter_payload: dict[str, Any] | None = None,
    sorts: list[dict[str, Any]] | None = None,
    page_size: int = 10,
    start_cursor: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    MAJ-1 FIX: Fetch a single page of results from Notion and return the
    cursor for the next page.

    Unlike query_database() which fetches ALL pages, this function makes
    exactly ONE API call.  The caller uses the returned next_cursor on a
    subsequent call to walk forward through the result set — without ever
    loading the full dataset into memory.

    Args:
        database_id:    Target database ID.
        filter_payload: Notion filter object (optional).
        sorts:          List of sort objects (optional).
        page_size:      Number of results to return (1–100, default 10).
        start_cursor:   Opaque cursor from the previous call's next_cursor.
                        Pass None to start from the beginning.

    Returns:
        (items, next_cursor) where next_cursor is None when there are no
        further pages.
    """
    page_size = max(1, min(100, page_size))

    payload: dict[str, Any] = {"page_size": page_size}
    if filter_payload:
        payload["filter"] = filter_payload
    if sorts:
        payload["sorts"] = sorts
    if start_cursor:
        payload["start_cursor"] = start_cursor

    logger.debug(
        "query_database_page | id=%s page_size=%d cursor=%s",
        database_id, page_size, start_cursor,
    )
    data = _request("POST", f"/databases/{database_id}/query", payload)

    items       = data.get("results", [])
    next_cursor = data.get("next_cursor") if data.get("has_more") else None

    logger.info(
        "Page query complete | db=%s returned=%d has_more=%s",
        database_id, len(items), data.get("has_more", False),
    )
    return items, next_cursor


def query_database(
    database_id: str,
    filter_payload: dict[str, Any] | None = None,
    sorts: list[dict[str, Any]] | None = None,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """
    Query ALL pages inside a Notion database with optional filters + sorts.

    Handles Notion's cursor-based pagination automatically, fetching every
    page until has_more is False.  Use this for internal operations that
    genuinely need the full dataset (analysis, drift check).  For the
    paginated HTTP endpoint use query_database_page() instead.

    Args:
        database_id:    Target database ID.
        filter_payload: Notion filter object (optional).
        sorts:          List of sort objects (optional).
        page_size:      Results per API call (max 100).

    Returns:
        List of all Notion page objects.
    """
    results: list[dict[str, Any]] = []
    cursor: str | None = None

    while True:
        page, cursor = query_database_page(
            database_id,
            filter_payload=filter_payload,
            sorts=sorts,
            page_size=page_size,
            start_cursor=cursor,
        )
        results.extend(page)
        if cursor is None:
            break

    logger.info(
        "query_database complete | db=%s total_pages=%d", database_id, len(results)
    )
    return results
