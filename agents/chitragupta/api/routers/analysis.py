# api/routers/analysis.py

"""
Analysis router — run the pattern analyzer over a Notion database.

Endpoints
---------
GET /analysis/{database_name}  — fetch + analyze all entries, return insights

AUDIT FIX MIN-5 — database_name path parameter sanitisation
    The raw path value is passed through _safe_db_name() before reaching
    any schema or file-system operation, preventing path traversal attacks
    (e.g. "../../etc/passwd").

AUDIT FIX CRIT-6 (partial) — 500 handler no longer leaks exc detail
    The previous catch-all raised http_error(500, f"Analysis error: {exc}")
    which could expose internal stack frames, schema paths, or Notion IDs.
    The exception is now logged server-side and a generic message is returned
    to the caller.  The SchemaMissingError (404) path is unchanged as that
    message is intentionally user-facing.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, status

from api.models import AnalysisResponse
from api.dependencies import api_key_auth, http_error

from notion.schema_manager import load_schema, SchemaMissingError
from analysis.pattern_analyzer import analyze_database

logger = logging.getLogger("chitragupta.api.analysis")

router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"],
    dependencies=[Depends(api_key_auth)],
)

# MIN-5: reuse the same pattern as databases.py
_DB_NAME_RE = re.compile(r"^[\w\- ]{1,128}$")


def _safe_db_name(name: str) -> str:
    """Validate database_name before any I/O.  Raises 422 on bad input."""
    name = name.strip()
    if not name:
        raise http_error(status.HTTP_422_UNPROCESSABLE_ENTITY,
                         "database_name must not be empty.")
    if "\x00" in name or "/" in name or "\\" in name:
        raise http_error(status.HTTP_422_UNPROCESSABLE_ENTITY,
                         "database_name contains invalid characters.")
    if not _DB_NAME_RE.match(name):
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "database_name may only contain letters, digits, hyphens, "
            "underscores, and spaces (max 128 characters).",
        )
    return name


@router.get("/{database_name}", response_model=AnalysisResponse)
async def get_analysis(database_name: str) -> AnalysisResponse:
    """
    Fetch all entries from a Notion database and return pattern analysis insights.

    Mirrors the CLI 'Analyze Data' feature.  The response contains whatever
    the pattern_analyzer returns for the given database.
    """
    db_name = _safe_db_name(database_name)   # MIN-5

    try:
        load_schema(db_name)   # validate schema exists before hitting Notion
    except SchemaMissingError as exc:
        raise http_error(status.HTTP_404_NOT_FOUND, str(exc))

    try:
        insights = analyze_database(db_name)
    except SchemaMissingError as exc:
        raise http_error(status.HTTP_404_NOT_FOUND, str(exc))
    except Exception:
        # CRIT-6: log full traceback server-side, return generic message to caller
        logger.exception("API: analysis failed for '%s'", db_name)
        raise http_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Analysis failed. See server logs for details.",
        )

    logger.info("API: analysis complete | db='%s'", db_name)
    return AnalysisResponse(database_name=db_name, insights=insights)
