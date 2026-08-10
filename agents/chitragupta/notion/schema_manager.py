# notion/schema_manager.py

"""
Schema Manager — the source of truth for all database structures.

Responsibilities:
- Accept user-defined field definitions
- Validate them strictly
- Persist them locally as JSON (one file per database)
- Load them back on demand
- Convert them to Notion API property format

Nothing here touches the Notion API directly.
Nothing here knows about voice, NLP, or business logic.

(v0.2 — directory auto-create + title field enforcement)

Changes from v0.1
-----------------
- create_schema(): calls _ensure_schema_dir() before write_text().  On a fresh
  install SCHEMA_DIR does not exist; previously every first-run schema save
  raised FileNotFoundError.
- validate_schema(): enforces exactly-one-title rule.  Notion requires every
  database to have precisely one property of type "title".  Zero-title and
  multi-title schemas previously passed validation and caused cryptic HTTP 400
  responses from the Notion API.
- schema_to_notion_properties(): re-validates that at least one title field is
  present before building the Notion payload — second guard in case
  validate_schema() is bypassed by direct calls.
- _ensure_schema_dir(): new private helper that creates SCHEMA_DIR (and all
  parent directories) if they do not already exist.
"""

import json
import logging
from pathlib import Path
from typing import Any

from config.settings import SCHEMA_DIR, SUPPORTED_PROPERTY_TYPES

logger = logging.getLogger("chitragupta.schema_manager")


# ── Directory bootstrap ───────────────────────────────────────────────────────

def _ensure_schema_dir() -> None:
    """
    Create SCHEMA_DIR (and parents) if it does not already exist.

    Called before every write operation so the first-run experience never
    fails with FileNotFoundError. Safe to call multiple times — mkdir with
    exist_ok=True is a no-op when the directory already exists.
    """
    try:
        SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # Surface a clear error rather than a confusing FileNotFoundError
        raise SchemaValidationError(
            f"Cannot create schema directory '{SCHEMA_DIR}': {exc}"
        ) from exc


# ── Schema file path convention ───────────────────────────────────────────────

def _schema_path(database_name: str) -> Path:
    """
    Return the local file path for a given database's schema.
    Normalises the name: lowercase, spaces → underscores.

    Example: "Daily Log" → data/schemas/daily_log.json
    """
    safe_name = database_name.strip().lower().replace(" ", "_")
    return SCHEMA_DIR / f"{safe_name}.json"


# ── Validation ────────────────────────────────────────────────────────────────

def validate_schema(fields: list[dict[str, Any]]) -> None:
    """
    Validate a list of field definitions before saving.

    Each field must be a dict with at minimum:
        {
            "name": str,          # non-empty
            "type": str,          # must be in SUPPORTED_PROPERTY_TYPES
        }

    Optional keys per type:
        "options"   → list[str]   for "select" and "multi_select"
        "relation_database_id"    required for "relation"
        "format"    → str         optional for "number" (e.g. "percent")
        "formula"   → str         required for "formula"

    FIX: enforces exactly-one-title rule.  Notion requires every database to
    have precisely one property of type "title".  Previously zero-title schemas
    (or schemas with multiple title fields) passed validation and caused cryptic
    HTTP 400 responses when Chitragupta tried to create the database.

    Raises:
        SchemaValidationError with a descriptive message on any violation.
    """
    if not fields:
        raise SchemaValidationError("Schema must contain at least one field.")

    seen_names: set[str] = set()
    title_count: int = 0   # FIX: track title fields for enforcement

    for idx, field in enumerate(fields):
        position = f"Field #{idx + 1}"

        if not isinstance(field, dict):
            raise SchemaValidationError(
                f"{position}: each field must be a dict, got {type(field).__name__}."
            )

        name       = field.get("name", "").strip()
        field_type = field.get("type", "").strip()

        if not name:
            raise SchemaValidationError(
                f"{position}: 'name' is required and cannot be empty."
            )
        if not field_type:
            raise SchemaValidationError(
                f"Field '{name}': 'type' is required and cannot be empty."
            )

        if field_type not in SUPPORTED_PROPERTY_TYPES:
            raise SchemaValidationError(
                f"Field '{name}': unsupported type '{field_type}'. "
                f"Supported types: {', '.join(SUPPORTED_PROPERTY_TYPES)}."
            )

        normalised = name.lower()
        if normalised in seen_names:
            raise SchemaValidationError(
                f"Duplicate field name '{name}' (names are case-insensitive)."
            )
        seen_names.add(normalised)

        # FIX: count title fields as we scan
        if field_type == "title":
            title_count += 1

        if field_type in ("select", "multi_select"):
            options = field.get("options")
            if options is not None:
                if not isinstance(options, list) or not all(
                    isinstance(o, str) for o in options
                ):
                    raise SchemaValidationError(
                        f"Field '{name}': 'options' must be a list of strings."
                    )

        if field_type == "relation":
            if not field.get("relation_database_id", "").strip():
                raise SchemaValidationError(
                    f"Field '{name}': 'relation_database_id' is required "
                    "for type 'relation'."
                )

        if field_type == "formula":
            if not field.get("formula", "").strip():
                raise SchemaValidationError(
                    f"Field '{name}': 'formula' expression is required "
                    "for type 'formula'."
                )

    # FIX: enforce exactly-one-title rule
    if title_count == 0:
        raise SchemaValidationError(
            "Schema must contain exactly one field of type 'title'. "
            "Notion requires a title property on every database. "
            "Add a field like {'name': 'Title', 'type': 'title'} to your schema."
        )
    if title_count > 1:
        raise SchemaValidationError(
            f"Schema contains {title_count} fields of type 'title'. "
            "Notion only allows one title property per database. "
            "Remove the extra title field(s)."
        )

    logger.debug("Schema validation passed | fields=%d", len(fields))


# ── Schema CRUD ───────────────────────────────────────────────────────────────

def create_schema(
    database_name: str,
    fields: list[dict[str, Any]],
    notion_database_id: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Validate, build, and persist a schema for a named database.

    FIX: calls _ensure_schema_dir() before writing so first-run installs
    don't fail with FileNotFoundError when SCHEMA_DIR doesn't exist yet.

    Args:
        database_name:      Human-readable name, e.g. "Daily Log".
        fields:             List of field definition dicts.
        notion_database_id: Notion DB ID — stored after remote creation.
        overwrite:          If False (default), raises if schema already exists.

    Returns:
        The saved schema dict.

    Raises:
        SchemaValidationError:    on invalid fields or missing title.
        SchemaAlreadyExistsError: if schema exists and overwrite=False.
    """
    path = _schema_path(database_name)

    if path.exists() and not overwrite:
        raise SchemaAlreadyExistsError(
            f"Schema for '{database_name}' already exists at {path}. "
            "Pass overwrite=True to replace it."
        )

    validate_schema(fields)

    schema: dict[str, Any] = {
        "database_name":      database_name.strip(),
        "notion_database_id": notion_database_id.strip(),
        "fields":             fields,
    }

    # FIX: ensure directory exists before writing
    _ensure_schema_dir()
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Schema saved | db='%s' path=%s fields=%d",
        database_name, path, len(fields),
    )
    return schema


def load_schema(database_name: str) -> dict[str, Any]:
    """
    Load the persisted schema for a named database.

    Raises:
        SchemaMissingError: if no schema file exists for this database.
    """
    path = _schema_path(database_name)

    if not path.exists():
        available = list_schemas()
        hint = (
            f" Available schemas: {', '.join(available)}"
            if available
            else " No schemas have been created yet."
        )
        raise SchemaMissingError(
            f"No schema found for '{database_name}' at {path}.{hint}"
        )

    try:
        schema: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaMissingError(
            f"Schema file for '{database_name}' is corrupted: {exc}"
        ) from exc

    logger.info(
        "Schema loaded | db='%s' fields=%d",
        database_name, len(schema.get("fields", [])),
    )
    return schema


def update_notion_id(database_name: str, notion_database_id: str) -> None:
    """
    Patch the stored notion_database_id after a database is created remotely.
    """
    schema = load_schema(database_name)
    schema["notion_database_id"] = notion_database_id.strip()
    path = _schema_path(database_name)
    # FIX: guard against the directory disappearing between load and write
    _ensure_schema_dir()
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "notion_database_id updated | db='%s' id=%s",
        database_name, notion_database_id,
    )


def list_schemas() -> list[str]:
    """Return the names of all locally saved schemas."""
    names: list[str] = []
    # FIX: if SCHEMA_DIR doesn't exist yet, return [] instead of crashing
    if not SCHEMA_DIR.exists():
        return names
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            names.append(data.get("database_name", path.stem))
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping unreadable schema file: %s", path)
    return names


def delete_schema(database_name: str) -> None:
    """
    Remove the local schema file for a database.

    Does NOT delete anything in Notion — only removes local metadata.

    Raises:
        SchemaMissingError: if the schema file does not exist.
    """
    path = _schema_path(database_name)
    if not path.exists():
        raise SchemaMissingError(
            f"Cannot delete: no schema found for '{database_name}'."
        )
    path.unlink()
    logger.info("Schema deleted locally | db='%s'", database_name)


# ── Notion conversion ─────────────────────────────────────────────────────────

def schema_to_notion_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a local schema dict → Notion API property definitions.

    Used when calling notion_client.create_database().

    FIX: explicitly checks that exactly one 'title' type field is present
    before building the payload.  If validate_schema() was somehow bypassed
    (e.g. a hand-edited schema file), this guard catches the problem here
    with a clear message rather than letting it surface as a Notion 400.

    Raises:
        SchemaValidationError: if an unsupported type slips through or the
                               title field requirement is unmet.
    """
    fields: list[dict[str, Any]] = schema.get("fields", [])

    # FIX: second guard — enforce title presence even if validate_schema() was skipped
    title_fields = [f for f in fields if f.get("type") == "title"]
    if not title_fields:
        raise SchemaValidationError(
            f"Schema for '{schema.get('database_name', '?')}' has no 'title' field. "
            "Notion requires exactly one title property. "
            "Re-run 'Create Database' to rebuild the schema."
        )
    if len(title_fields) > 1:
        raise SchemaValidationError(
            f"Schema for '{schema.get('database_name', '?')}' has "
            f"{len(title_fields)} 'title' fields — Notion allows only one. "
            "Edit the schema file to remove the extra title field."
        )

    notion_properties: dict[str, Any] = {}
    for field in fields:
        name:  str = field["name"]
        ftype: str = field["type"]
        notion_properties[name] = _field_to_notion(name, ftype, field)

    logger.debug(
        "Converted schema → Notion properties | keys=%s",
        list(notion_properties.keys()),
    )
    return notion_properties


def _field_to_notion(
    name: str,
    ftype: str,
    field: dict[str, Any],
) -> dict[str, Any]:
    """Convert a single field definition to its Notion property config block."""
    simple_types = {
        "title", "rich_text", "date", "checkbox",
        "url", "email", "phone_number", "people", "files",
    }
    if ftype in simple_types:
        return {ftype: {}}

    if ftype == "number":
        fmt = field.get("format", "number")
        return {"number": {"format": fmt}}

    if ftype == "select":
        options = field.get("options", [])
        return {"select": {"options": [{"name": opt} for opt in options]}}

    if ftype == "multi_select":
        options = field.get("options", [])
        return {"multi_select": {"options": [{"name": opt} for opt in options]}}

    if ftype == "relation":
        db_id = field.get("relation_database_id", "")
        return {"relation": {"database_id": db_id, "single_property": {}}}

    if ftype == "formula":
        expression = field.get("formula", "")
        return {"formula": {"expression": expression}}

    raise SchemaValidationError(
        f"Field '{name}': no Notion mapping for type '{ftype}'."
    )


# ── Custom exceptions ─────────────────────────────────────────────────────────

class SchemaValidationError(Exception):
    """Raised when a field definition fails validation."""


class SchemaMissingError(Exception):
    """Raised when a requested schema file does not exist or is unreadable."""


class SchemaAlreadyExistsError(Exception):
    """Raised when trying to create a schema that already exists."""
