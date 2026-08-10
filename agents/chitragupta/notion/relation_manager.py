# notion/relation_manager.py

"""
Relation Manager — dynamic cross-database linking in Notion.

Responsibilities:
- Load both schemas by name
- Validate both have a Notion database ID (i.e. exist remotely)
- Create a relation property in Database A pointing to Database B
  via the Notion API
- Reflect the new relation field in both local schema files

Design contract:
- No hardcoded database names or field names anywhere
- Caller provides everything: db names, relation property name, direction
- Both local schemas are updated atomically after a successful API call
- Nothing here touches voice, NLP, or JSON building
"""

import logging
from typing import Any

from notion.schema_manager import (
    load_schema,
    update_notion_id,          # reused to persist schema changes
    SchemaMissingError,
)
from notion import notion_client
from notion.notion_client import NotionAPIError

logger = logging.getLogger("chitragupta.relation_manager")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _assert_has_notion_id(schema: dict[str, Any]) -> str:
    """
    Return the notion_database_id from a schema, or raise if missing.

    A schema can exist locally before the Notion database is created
    (e.g. during setup). We must reject that case here because the
    Notion API needs a real database ID to create a relation.
    """
    db_id: str = schema.get("notion_database_id", "").strip()
    name: str  = schema.get("database_name", "?")

    if not db_id:
        raise RelationError(
            f"Database '{name}' has no Notion ID in its local schema. "
            "Create the Notion database first (schema_manager → notion_client) "
            "so the ID is stored, then retry linking."
        )
    return db_id


def _relation_field_exists(schema: dict[str, Any], relation_name: str) -> bool:
    """
    Return True if a field with relation_name already exists in the schema.
    Comparison is case-insensitive.
    """
    target = relation_name.strip().lower()
    return any(
        f["name"].lower() == target
        for f in schema.get("fields", [])
    )


def _append_relation_field(
    schema: dict[str, Any],
    relation_name: str,
    target_database_id: str,
) -> None:
    """
    Add a relation field entry to a schema dict IN PLACE and persist it.

    Args:
        schema:               The schema dict (loaded from disk).
        relation_name:        Name of the new relation property.
        target_database_id:   Notion ID of the database being pointed to.
    """
    new_field: dict[str, Any] = {
        "name": relation_name,
        "type": "relation",
        "relation_database_id": target_database_id,
    }

    schema["fields"].append(new_field)

    # Persist: re-write the schema file using its stored database name
    import json
    from config.settings import SCHEMA_DIR

    safe_name = schema["database_name"].strip().lower().replace(" ", "_")
    path = SCHEMA_DIR / f"{safe_name}.json"
    path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Schema updated | db='%s' added relation field='%s' → target=%s",
        schema["database_name"], relation_name, target_database_id,
    )


def _create_notion_relation_property(
    database_id: str,
    relation_name: str,
    target_database_id: str,
) -> dict[str, Any]:
    """
    PATCH an existing Notion database to add one relation property.

    Uses the databases/{id} PATCH endpoint which merges new properties
    into the existing schema without touching other fields.

    Args:
        database_id:          Notion ID of the database receiving the property.
        relation_name:        Name of the new property.
        target_database_id:   Notion ID of the database being linked to.

    Returns:
        Notion API response dict.

    Raises:
        NotionAPIError on any API failure.
    """
    payload: dict[str, Any] = {
        "properties": {
            relation_name: {
                "relation": {
                    "database_id": target_database_id,
                    "single_property": {},   # single-direction relation
                }
            }
        }
    }

    logger.info(
        "PATCH Notion DB | id=%s  adding relation '%s' → %s",
        database_id, relation_name, target_database_id,
    )
    return notion_client._request(
        "PATCH",
        f"/databases/{database_id}",
        payload,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def add_relation_field_to_schema(
    database_name: str,
    relation_name: str,
    target_database_id: str,
) -> None:
    """
    Add a relation field to a local schema without touching the Notion API.

    Use this when you need to record a relation that already exists in
    Notion (e.g. syncing after manual setup) or when testing locally.

    Args:
        database_name:        Local schema to update (e.g. "Daily Log").
        relation_name:        Name of the relation property to add.
        target_database_id:   Notion ID of the linked database.

    Raises:
        SchemaMissingError:   if database_name has no local schema.
        RelationError:        if relation_name already exists in the schema.
    """
    schema = load_schema(database_name)

    if _relation_field_exists(schema, relation_name):
        raise RelationError(
            f"Field '{relation_name}' already exists in schema "
            f"'{database_name}'. No changes made."
        )

    _append_relation_field(schema, relation_name, target_database_id)
    logger.info(
        "add_relation_field_to_schema | db='%s' field='%s'",
        database_name, relation_name,
    )


def create_relation(
    database_a: str,
    database_b: str,
    relation_name: str,
    bidirectional: bool = False,
    reverse_relation_name: str = "",
) -> None:
    """
    Link two Notion databases by creating a relation property.

    Default (unidirectional):
        Database A gets a new relation property → Database B.

    Optional (bidirectional):
        Database A → Database B  AND  Database B → Database A.
        Requires reverse_relation_name when bidirectional=True.

    Steps:
        1. Load both local schemas
        2. Validate both have a Notion database ID
        3. Check the relation doesn't already exist
        4. PATCH Database A in Notion to add the relation property
        5. Update Database A's local schema
        6. If bidirectional: PATCH Database B + update its schema

    Args:
        database_a:            Source database (receives the relation field).
        database_b:            Target database (what A points to).
        relation_name:         Name for the relation property in Database A.
        bidirectional:         Also create the reverse relation in Database B.
        reverse_relation_name: Required if bidirectional=True; name for the
                               relation property added to Database B.

    Raises:
        SchemaMissingError:  if either schema file is not found locally.
        RelationError:       if a Notion ID is missing, the relation already
                             exists, or the API call fails.
    """
    # ── 1. Load schemas ───────────────────────────────────────────────────────
    logger.info(
        "create_relation | '%s' → '%s'  field='%s'  bidirectional=%s",
        database_a, database_b, relation_name, bidirectional,
    )

    schema_a = load_schema(database_a)
    schema_b = load_schema(database_b)

    # ── 2. Validate Notion IDs ────────────────────────────────────────────────
    id_a = _assert_has_notion_id(schema_a)
    id_b = _assert_has_notion_id(schema_b)

    # ── 3. Guard: relation must not already exist ─────────────────────────────
    if _relation_field_exists(schema_a, relation_name):
        raise RelationError(
            f"Relation field '{relation_name}' already exists in "
            f"schema '{database_a}'. No changes made."
        )

    if bidirectional:
        if not reverse_relation_name.strip():
            raise RelationError(
                "reverse_relation_name is required when bidirectional=True."
            )
        if _relation_field_exists(schema_b, reverse_relation_name):
            raise RelationError(
                f"Reverse relation field '{reverse_relation_name}' already "
                f"exists in schema '{database_b}'. No changes made."
            )

    # ── 4. Create relation in Notion (A → B) ──────────────────────────────────
    try:
        _create_notion_relation_property(
            database_id=id_a,
            relation_name=relation_name,
            target_database_id=id_b,
        )
    except NotionAPIError as exc:
        raise RelationError(
            f"Notion API failed while adding relation '{relation_name}' "
            f"to '{database_a}': {exc}"
        ) from exc

    # ── 5. Persist to local schema A ──────────────────────────────────────────
    _append_relation_field(schema_a, relation_name, id_b)

    # ── 6. Bidirectional: create reverse relation (B → A) ────────────────────
    if bidirectional:
        try:
            _create_notion_relation_property(
                database_id=id_b,
                relation_name=reverse_relation_name,
                target_database_id=id_a,
            )
        except NotionAPIError as exc:
            # Forward relation already created — warn but don't undo
            logger.error(
                "Reverse relation creation failed. Forward relation "
                "('%s' → '%s') was already applied. Manual cleanup may "
                "be needed in Notion. Error: %s",
                database_a, database_b, exc,
            )
            raise RelationError(
                f"Forward relation '{relation_name}' was created in Notion "
                f"but the reverse relation '{reverse_relation_name}' failed: "
                f"{exc}. Check Notion manually."
            ) from exc

        _append_relation_field(schema_b, reverse_relation_name, id_a)

    logger.info(
        "create_relation complete | '%s' → '%s'  bidirectional=%s",
        database_a, database_b, bidirectional,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  ✓  Relation created: '{database_a}' → '{database_b}'")
    print(f"     Property name in '{database_a}': '{relation_name}'")
    if bidirectional:
        print(f"     Property name in '{database_b}': '{reverse_relation_name}'")
    print()


def list_relations(database_name: str) -> list[dict[str, Any]]:
    """
    Return all relation fields defined in a database's local schema.

    Args:
        database_name: Name of the database to inspect.

    Returns:
        List of field dicts where type == "relation". May be empty.

    Raises:
        SchemaMissingError: if no local schema exists.
    """
    schema = load_schema(database_name)
    relations = [
        f for f in schema.get("fields", [])
        if f.get("type") == "relation"
    ]
    logger.info(
        "list_relations | db='%s' found=%d", database_name, len(relations)
    )
    return relations


# ── Custom exceptions ─────────────────────────────────────────────────────────

class RelationError(Exception):
    """
    Raised for any relation operation failure:
    missing Notion ID, duplicate field, API error, or bad arguments.
    """