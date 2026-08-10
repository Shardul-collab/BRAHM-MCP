# api/dependencies.py

"""
Shared FastAPI dependencies.

FIX 2 — Strict API key enforcement
    - If API_KEY is missing or empty the server REFUSES to start.
    - No silent dev-mode fallback.
    - /health is the only route that does NOT use this dependency.

FIX 8 — Structured Notion error helper
    notion_error(): converts NotionAPIError messages into structured
    JSON responses with a `retryable` flag so clients can decide
    whether to retry automatically.

AUDIT FIX CRIT-4 — Timing-safe API key comparison
    The previous `x_api_key != _API_KEY` short-circuits on the first
    mismatched character, leaking key length and prefix via response-time
    variance.  Replaced with hmac.compare_digest() which runs in constant
    time regardless of where the strings diverge.

AUDIT FIX MIN-4 — Retry-After header on 429 responses
    notion_error() now attaches a Retry-After: 5 header on rate-limit
    responses so clients can implement correct backoff without guessing.

AUDIT FIX C-002 — Correct 401 vs 403 semantics in api_key_auth
    Previously both a missing key and a wrong key returned HTTP 401.
    RFC 7235 semantics:
      - 401 Unauthorized → no credentials were provided at all
      - 403 Forbidden    → credentials were provided but are invalid/wrong
    api_key_auth() now checks for an empty/absent header first (→ 401),
    then runs the constant-time digest comparison; a mismatch now returns
    403 instead of 401.

AUDIT FIX SECURITY — notion_error() generic message no longer leaks internals
    The previous generic branch returned `"message": msg` where `msg =
    str(exc)`.  After the notion_client.py C-001 fix, the NotionAPIError
    string is "[{status}] {code}" — safe, but still an implementation
    detail.  The generic branch now returns a fixed user-facing message
    ("Notion integration error. Check server logs for details.") so that
    no Notion internal error codes or status codes are ever forwarded to
    the end-user.  The notion_status integer field on NotionAPIError is
    used for classification instead of string parsing, making the rate-limit
    and timeout branches more robust.
"""

from __future__ import annotations

import hmac
import os
import logging

from fastapi import Header, HTTPException, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("chitragupta.api")

# Read once at import time — change requires restart.
_API_KEY: str = os.getenv("API_KEY", "").strip()


def verify_api_key_at_startup() -> None:
    """
    Called from app lifespan.
    Raises RuntimeError (crashes the server) if API_KEY is not configured.
    This makes accidental insecure deployment impossible — the process
    simply refuses to start rather than silently accepting all requests.
    """
    if not _API_KEY:
        raise RuntimeError(
            "STARTUP FAILURE: API_KEY is not set in your .env file.\n"
            "Add API_KEY=<your-secret> to .env and restart the server.\n"
            "Chitragupta API will not start without a configured API key."
        )
    logger.info("API key verified at startup.")


async def api_key_auth(x_api_key: str = Header(default="")) -> None:
    """
    Dependency injected into every protected router.

    AUDIT FIX C-002: Two distinct failure modes with correct HTTP codes:

      1. Missing key (empty/absent X-API-Key header) → 401 Unauthorized
         The request carries no recognisable credentials at all.
         RFC 7235: 401 means "unauthenticated" — the client must provide
         credentials before the server will respond to this request.

      2. Wrong key (header present but value is incorrect) → 403 Forbidden
         The request carried credentials, but they are invalid.
         RFC 7235: 403 means "authenticated but not authorised" — the
         server understood who the client is claiming to be, but refuses
         to fulfil the request.

    CRIT-4: The digest comparison still uses hmac.compare_digest() for
    constant-time equality so that response-time variance leaks no
    information about how much of the submitted key matched the expected key.
    """
    # AUDIT FIX C-002 Step 1: missing key → 401
    if not x_api_key.strip():
        logger.warning("Auth failure — X-API-Key header is missing.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error":   "unauthorized",
                "message": "X-API-Key header is missing.",
            },
        )

    # AUDIT FIX C-002 Step 2: wrong key → 403
    # hmac.compare_digest runs in constant time — no timing oracle.
    provided = x_api_key.encode("utf-8")
    expected = _API_KEY.encode("utf-8")

    if not hmac.compare_digest(provided, expected):
        logger.warning("Auth failure — invalid X-API-Key provided.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error":   "forbidden",
                "message": "X-API-Key is invalid.",
            },
        )


def http_error(code: int, message: str) -> HTTPException:
    """Return an HTTPException with a standard body shape."""
    return HTTPException(
        status_code=code,
        detail={"ok": False, "message": message},
    )


def notion_error(exc: Exception) -> HTTPException:
    """
    FIX 8 + MIN-4 + SECURITY: Convert a NotionAPIError into a structured
    HTTP response.

    Classification order (checked in priority order):
      1. Rate limit  (Notion status 429) → HTTP 429, retryable=True, Retry-After header
      2. Timeout     (notion_status 0 + "timed out" in message) → HTTP 504, retryable=True
      3. Everything else → HTTP 502, retryable=False

    AUDIT FIX SECURITY: The generic branch previously returned `"message": msg`
    where `msg = str(exc)`.  Even after the notion_client sanitisation fix,
    this forwarded the Notion error code string to end-users.  The generic
    branch now returns a fixed safe message.  Full error details are available
    in server logs via notion_client._request()'s ERROR-level log entry.

    AUDIT FIX C-002 (robustness): Classification now uses `exc.notion_status`
    (the integer HTTP status returned by Notion) rather than searching for
    "[429]" in the string.  This is more reliable and does not depend on the
    specific format of the NotionAPIError message.

    MIN-4: 429 responses still include a Retry-After: 5 header.
    """
    msg           = str(exc)
    notion_status = getattr(exc, "notion_status", 0)

    # Rate limited — Notion returned 429
    if notion_status == 429 or "rate_limited" in msg.lower():
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": "5"},
            detail={
                "error":       "notion_rate_limit",
                "message":     "Notion API rate limit reached. Retry after 5 seconds.",
                "retryable":   True,
                "retry_after": 5,
            },
        )

    # Timeout — network-level (notion_status == 0) and message says timed out
    if notion_status == 0 and (
        "timeout" in msg.lower() or "timed out" in msg.lower()
    ):
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error":     "notion_timeout",
                "message":   "Notion API request timed out. Retry shortly.",
                "retryable": True,
            },
        )

    # All other Notion errors → 502 Bad Gateway.
    # AUDIT FIX SECURITY: Fixed safe message — no Notion internals forwarded.
    # Full error details (status, code, raw message) are in the server log.
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail={
            "error":     "notion_error",
            "message":   "Notion integration error. Check server logs for details.",
            "retryable": False,
        },
    )
