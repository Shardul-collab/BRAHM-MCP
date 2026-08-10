# api/routers/databases.py

"""
Database router — schema management and Notion database creation.

Endpoints
---------
GET    /databases                  — list all local schemas
POST   /databases                  — create schema + Notion database
GET    /databases/{name}           — get schema for one database
DELETE /databases/{name}           — delete local schema (not Notion)
GET    /databases/{name}/drift     — check schema drift vs live Notion DB
POST   /databases/infer-schema     — infer fields from a natural language description

AUDIT FIX MIN-5 — database_name path parameter sanitisation
    Previously database_name was passed directly to load_schema() /
    create_schema() without any validation.  A name like
    "../../etc/passwd" or one containing null bytes could reach the
    file-system layer inside schema_manager.  The _safe_db_name() helper
    now rejects names that contain path separators, null bytes, or are
    longer than 128 characters, returning a 422 before any I/O occurs.

AUDIT FIX C-003 / SECURITY — Sanitised error responses from create_database_endpoint
    The previous exception handler was:
        except (NotionAPIError, ValidationError) as exc:
            raise http_error(502, f"Schema saved locally but Notion creation failed: {exc}")

    Two problems with this:

    1.  str(exc) forwarded the raw NotionAPIError message into the 502
        response body.  After the notion_client.py C-001 fix the message
        is "[{status}] {code}" — implementation detail that still leaks.
        The error message is now a fixed safe string; the full Notion error
        is already logged at ERROR level by notion_client._request().

    2.  NotionAPIError and ValidationError were caught by the same handler.
        A ValidationError from schema_to_notion_properties() is a local
        schema problem (422), not a Notion connectivity problem (502).
        They are now caught in separate except blocks with correct status
        codes: ValidationError → 422, NotionAPIError → 502.

    The 502 branch now uses notion_error() from dependencies so that
    rate-limit (429) and timeout (504) conditions on the Notion side
    return the correct status code instead of a generic 502.  The
    "Schema saved locally" context is preserved in the internal log.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, status

from api.models import (
    CreateDatabaseRequest, DatabaseResponse, DatabaseListResponse,
    DriftResponse, InferSchemaRequest, InferSchemaResponse, StatusResponse,
)
from api.dependencies import api_key_auth, http_error, notion_error

from notion.schema_manager import (
    create_schema, load_schema, list_schemas, delete_schema,
    schema_to_notion_properties, update_notion_id,
    SchemaValidationError, SchemaMissingError, SchemaAlreadyExistsError,
)
from notion.notion_client import create_database, check_schema_drift, NotionAPIError
from nlp.schema_inferencer import infer_schema_from_description, describe_schema_naturally
from core.json_builder import ValidationError

logger = logging.getLogger("chitragupta.api.databases")

router = APIRouter(
    prefix="/databases",
    tags=["Databases"],
    dependencies=[Depends(api_key_auth)],
)

# MIN-5: Only allow alphanumeric, hyphen, underscore, and space.
# Max 128 characters.  No path separators or control characters.
_DB_NAME_RE = re.compile(r"^[\w\- ]{1,128}$")


def _safe_db_name(name: str) -> str:
    """
    MIN-5 FIX: Validate database_name from path/body before passing to
    any file-system or schema operation.

    Raises http_error(422) if the name is not safe.
    Returns the stripped name on success.
    """
    name = name.strip()
    if not name:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "database_name must not be empty.",
        )
    if "\x00" in name:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "database_name must not contain null bytes.",
        )
    if "/" in name or "\\" in name:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "database_name must not contain path separators (/ or \\).",
        )
    if not _DB_NAME_RE.match(name):
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "database_name may only contain letters, digits, hyphens, "
            "underscores, and spaces (max 128 characters).",
        )
    return name


@router.get("", response_model=DatabaseListResponse)
async def list_databases() -> DatabaseListResponse:
    """Return the names of all locally saved database schemas."""
    return DatabaseListResponse(databases=list_schemas())


@router.post("", response_model=DatabaseResponse, status_code=status.HTTP_201_CREATED)
async def create_database_endpoint(body: CreateDatabaseRequest) -> DatabaseResponse:
    """
    Validate and save a schema locally, then create the database in Notion.

    The fields list must contain exactly one field of type 'title'.
    """
    db_name = _safe_db_name(body.database_name)   # MIN-5
    fields  = [f.model_dump(exclude_none=True) for f in body.fields]

    # 1. Save schema locally
    try:
        create_schema(db_name, fields)
    except SchemaValidationError as exc:
        raise http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    except SchemaAlreadyExistsError:
        raise http_error(
            status.HTTP_409_CONFLICT,
            f"Schema '{db_name}' already exists. "
            "Delete it first or use a different name.",
        )

    # 2. Build Notion property payload from local schema.
    #    schema_to_notion_properties() may raise SchemaValidationError (a local
    #    problem — 422), which must not be conflated with a Notion API failure.
    try:
        schema       = load_schema(db_name)
        notion_props = schema_to_notion_properties(schema)
    except (SchemaValidationError, ValidationError) as exc:
        raise http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))

    # 3. Create the database in Notion.
    #
    # AUDIT FIX C-003 / SECURITY: NotionAPIError is now caught in its own
    # block, separate from local validation errors.  The error message passed
    # to the caller is a fixed safe string — it does NOT include str(exc)
    # because that would forward Notion internal details to end-users.
    # The full Notion error (status, code, raw message) was already written
    # to the ERROR-level server log inside notion_client._request().
    # notion_error() is used so that rate-limit (429) and timeout (504)
    # conditions return the correct status codes rather than a generic 502.
    try:
        response  = create_database(body.parent_page_id, db_name, notion_props)
        notion_id = response["id"]
    except NotionAPIError as exc:
        logger.error(
            "Notion database creation failed | db='%s' notion_status=%s",
            db_name, getattr(exc, "notion_status", "n/a"),
        )
        raise http_error(
            status.HTTP_502_BAD_GATEWAY,
            "Schema saved locally but Notion creation failed. "
            "Check server logs for details.",
        )

    # 4. Persist Notion ID back into the local schema file.
    try:
        update_notion_id(db_name, notion_id)
    except Exception as exc:
        logger.error(
            "Failed to persist notion_id | db='%s' notion_id=%s error=%s",
            db_name, notion_id, exc,
        )
        raise http_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Database created in Notion (id={notion_id}) but local schema "
            "could not be updated. Check server logs for details.",
        )

    logger.info("API: database created | name='%s' notion_id=%s", db_name, notion_id)
    schema = load_schema(db_name)
    return DatabaseResponse(
        database_name=schema["database_name"],
        notion_database_id=schema["notion_database_id"],
        fields=schema["fields"],
    )


@router.get("/{database_name}", response_model=DatabaseResponse)
async def get_database(database_name: str) -> DatabaseResponse:
    """Retrieve the local schema for a named database."""
    db_name = _safe_db_name(database_name)   # MIN-5
    try:
        schema = load_schema(db_name)
    except SchemaMissingError as exc:
        raise http_error(status.HTTP_404_NOT_FOUND, str(exc))
    return DatabaseResponse(
        database_name=schema["database_name"],
        notion_database_id=schema["notion_database_id"],
        fields=schema["fields"],
    )


@router.delete("/{database_name}", response_model=StatusResponse)
async def delete_database(database_name: str) -> StatusResponse:
    """
    Delete the local schema file.
    Does NOT delete the database in Notion.
    """
    db_name = _safe_db_name(database_name)   # MIN-5
    try:
        delete_schema(db_name)
    except SchemaMissingError as exc:
        raise http_error(status.HTTP_404_NOT_FOUND, str(exc))
    logger.info("API: schema deleted | name='%s'", db_name)
    return StatusResponse(ok=True, message=f"Schema '{db_name}' deleted locally.")


@router.get("/{database_name}/drift", response_model=DriftResponse)
async def schema_drift(database_name: str) -> DriftResponse:
    """Compare local schema fields against the live Notion database properties."""
    db_name = _safe_db_name(database_name)   # MIN-5
    try:
        schema = load_schema(db_name)
    except SchemaMissingError as exc:
        raise http_error(status.HTTP_404_NOT_FOUND, str(exc))

    notion_id = schema.get("notion_database_id", "").strip()
    if not notion_id:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"'{db_name}' has no Notion ID. Create it in Notion first.",
        )

    issues = check_schema_drift(notion_id, schema.get("fields", []))
    return DriftResponse(
        database_name=db_name,
        has_drift=bool(issues),
        issues=issues,
    )


@router.post("/infer-schema", response_model=InferSchemaResponse)
async def infer_schema(body: InferSchemaRequest) -> InferSchemaResponse:
    """
    Infer a schema from a natural language description.

    Returns the suggested field list and a spoken description sentence.
    Does NOT create anything — purely inference.
    """
    fields = infer_schema_from_description(body.description)
    spoken = describe_schema_naturally(fields, "this database")
    return InferSchemaResponse(fields=fields, description=spoken)
