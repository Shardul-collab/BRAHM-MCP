# core/json_builder.py

"""
JSON Builder — pure data transformation layer.  (v0.5 — input cleaning + list parsing)

Changes from v0.4
-----------------
- _to_list(): now splits on " and ", " or ", "&", and "+" in addition to
  commas.  Previously "running and swimming" was treated as a single string,
  so multi-select / activities fields received one item instead of many.
  Items are title-cased for consistent Notion option naming.
- _to_string(): filters _NOISE_WORDS before returning.  Words like "okay",
  "done", "next", "skip" that Whisper sometimes captures as filler no longer
  end up stored as field values in rich_text or select fields.
- _NOISE_WORDS constant added at module level.
- All other logic (coercion, Notion payload formatting, validation) unchanged.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any

from notion.schema_manager import load_schema
from core.validator import (
    ValidationError,   # re-exported so callers only need one import
    validate_record,
    json_serial,
    safe_dumps,        # re-exported
)

logger = logging.getLogger("chitragupta.json_builder")

# Public re-exports so callers don't need to import validator separately
__all__ = [
    "build_json",
    "json_to_notion_properties",
    "safe_dumps",
    "ValidationError",
]

# ── Noise word filter ─────────────────────────────────────────────────────────

# FIX: words that Whisper frequently captures as standalone utterances when
# the user is thinking or transitioning between answers.  If any of these are
# the *entire* cleaned value of a string field, the field is returned as "".
# They are not filtered when embedded in longer text (e.g. "I feel okay today").
_NOISE_WORDS: frozenset[str] = frozenset({
    "okay", "ok", "done", "next", "skip", "yeah", "yes", "no",
    "uh", "um", "uh huh", "hmm", "hm", "right", "sure",
    "next question", "move on", "pass", "nothing", "none",
    "go ahead", "continue", "go on",
})


# ── Slot unwrapping ───────────────────────────────────────────────────────────

def _unwrap(value: Any) -> Any:
    """
    If value is a slot dict {value, confidence, updated_at}, return .value.
    Otherwise return as-is (flat dict compatibility).
    """
    if (
        isinstance(value, dict)
        and "value" in value
        and "confidence" in value
    ):
        return value["value"]
    return value


# ── Type coercion helpers ─────────────────────────────────────────────────────

def _to_number(value: Any, field_name: str) -> float | int | None:
    value = _unwrap(value)
    if value is None or value == "":
        return None
    try:
        f = float(str(value).strip())
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        logger.warning(
            "Field '%s': cannot coerce '%s' to number — defaulting to None.",
            field_name, value,
        )
        return None


def _to_string(value: Any, field_name: str) -> str:
    """
    Coerce value to a clean string.

    FIX: if the entire cleaned value is a known noise word (captured by Whisper
    during pauses or transitions), return "" so the field is treated as empty
    rather than storing "okay" or "done" as an actual data value.
    Noise words embedded in longer sentences are preserved unchanged.
    """
    value = _unwrap(value)
    if value is None:
        return ""
    if isinstance(value, str):
        cleaned = value.strip()
    else:
        cleaned = str(value).strip()

    # Only filter if the *entire* value is a noise word, not a substring
    if cleaned.lower() in _NOISE_WORDS:
        logger.debug(
            "Field '%s': noise word '%s' filtered out — returning ''.",
            field_name, cleaned,
        )
        return ""

    return cleaned


def _to_bool(value: Any, field_name: str) -> bool:
    value = _unwrap(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in ("true", "yes", "1"):
            return True
        if normalised in ("false", "no", "0", ""):
            return False
    logger.warning(
        "Field '%s': cannot coerce '%s' to bool — defaulting to False.",
        field_name, value,
    )
    return False


def _to_list(value: Any, field_name: str) -> list[str]:
    """
    Coerce value to a list of title-cased strings.

    FIX: splits on "and", "or", "&", and "+" in addition to commas so that
    voice input like "running and swimming" or "yoga & meditation" is correctly
    parsed as multiple items rather than stored as a single string.

    Items are title-cased to match Notion multi-select option naming conventions
    (consistent with what confirmation.py._coerce() produces for the same field
    type when the user edits an entry).

    Noise words that are the *sole* item after splitting are dropped.
    """
    value = _unwrap(value)
    if value is None or value == "":
        return []
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
    elif isinstance(value, str):
        # FIX: normalise multi-value separators before splitting
        normalised = re.sub(
            r"\s+(and|or|&|\+)\s+",   # "and", "or", "&", "+"
            ",",
            value,
            flags=re.IGNORECASE,
        )
        # Also handle bare "&" or "+" without surrounding spaces
        normalised = re.sub(r"[&+]", ",", normalised)
        items = [v.strip() for v in normalised.split(",") if v.strip()]
    else:
        items = [str(value).strip()]

    # Title-case each item; drop pure noise words
    result = []
    for item in items:
        if item.lower() in _NOISE_WORDS:
            logger.debug(
                "Field '%s': dropping noise word '%s' from list.", field_name, item
            )
            continue
        result.append(item.title())

    return result


def _to_date(value: Any, field_name: str) -> str | None:
    """Coerce value to ISO date string.  datetime objects are always safe here."""
    value = _unwrap(value)
    if value is None or value == "":
        return None
    # datetime / date objects → always serialisable to ISO string
    if isinstance(value, datetime):
        return value.date().isoformat()
    try:
        from datetime import date as _date
        if isinstance(value, _date):
            return value.isoformat()
    except Exception:
        pass
    if isinstance(value, str):
        clean = value.strip()
        try:
            datetime.fromisoformat(clean)
            return clean
        except ValueError:
            pass
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y%m%d"):
            try:
                return datetime.strptime(clean, fmt).date().isoformat()
            except ValueError:
                continue
    logger.warning(
        "Field '%s': cannot coerce '%s' to ISO date — defaulting to None.",
        field_name, value,
    )
    return None


# ── Coercion dispatcher ───────────────────────────────────────────────────────

_COERCE = {
    "number":       _to_number,
    "title":        _to_string,
    "rich_text":    _to_string,
    "select":       _to_string,
    "multi_select": _to_list,
    "date":         _to_date,
    "checkbox":     _to_bool,
    "url":          _to_string,
    "email":        _to_string,
    "phone_number": _to_string,
    "people":       _to_list,
    "relation":     _to_list,
}

_SKIP_IN_PAYLOAD = {"formula", "files"}


# ── Public API ────────────────────────────────────────────────────────────────

def build_json(
    database_name: str,
    extracted_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Align raw extracted data to the stored schema for a named database,
    then validate the result.

    Accepts both flat dicts and slot-format dicts (from DialogueFSM).
    Slot values are unwrapped (.value extracted) before coercion.

    Args:
        database_name:  Matches a locally saved schema (e.g. "Daily Log").
        extracted_data: Raw dict from NLP extraction, manual input, or FSM.
                        Keys should match schema field names (case-insensitive).

    Returns:
        Clean, schema-aligned, validated dict with properly typed values.

    Raises:
        SchemaMissingError:  if no local schema exists for database_name.
        ValidationError:     if any value fails range / type validation.
                             Pipeline must stop — do not send to Notion.
    """
    if not extracted_data:
        raise ValidationError(
            f"build_json called with empty data for database '{database_name}'."
        )

    schema = load_schema(database_name)
    fields: list[dict[str, Any]] = schema["fields"]

    # Build a lowercase → original-case lookup (unwrap slot values)
    normalised_input: dict[str, Any] = {}
    for k, v in extracted_data.items():
        normalised_input[k.strip().lower()] = _unwrap(v)

    raw_result: dict[str, Any] = {}

    for field in fields:
        name: str  = field["name"]
        ftype: str = field["type"]
        key        = name.lower()

        raw_value  = normalised_input.get(key)   # None if not provided

        coerce_fn  = _COERCE.get(ftype)
        if coerce_fn is None:
            logger.debug(
                "Field '%s' type '%s' has no coercion rule — storing raw.",
                name, ftype,
            )
            raw_result[name] = raw_value
            continue

        raw_result[name] = coerce_fn(raw_value, name)

    # Log unrecognised keys
    schema_keys = {f["name"].lower() for f in fields}
    ignored = [k for k in normalised_input if k not in schema_keys]
    if ignored:
        logger.debug(
            "build_json: ignored %d unrecognised key(s): %s",
            len(ignored), ignored,
        )

    # ── Validation pass (fail-fast) ───────────────────────────────────────────
    # validate_record re-checks range bounds and list types.
    # Raises ValidationError immediately on any violation.
    validated = validate_record(schema, raw_result)

    logger.info(
        "build_json complete | db='%s' fields_built=%d ignored=%d",
        database_name, len(validated), len(ignored),
    )
    return validated


def json_to_notion_properties(
    database_name: str,
    json_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert a schema-aligned JSON dict → Notion page property payload.

    Accepts both flat dicts and slot-format dicts.

    Args:
        database_name: Used to reload the schema (field order + types).
        json_data:     Output of build_json() for the same database,
                       or a raw slot dict from DialogueFSM.export_slots().

    Returns:
        Dict ready to pass as `properties` to notion_client.create_page().

    Raises:
        ValidationError: if json_data is empty.
    """
    if not json_data:
        raise ValidationError(
            f"json_to_notion_properties called with empty data for '{database_name}'."
        )

    schema = load_schema(database_name)
    fields: list[dict[str, Any]] = schema["fields"]

    notion_props: dict[str, Any] = {}

    for field in fields:
        name: str  = field["name"]
        ftype: str = field["type"]

        if ftype in _SKIP_IN_PAYLOAD:
            logger.debug("Skipping read-only field '%s' (type=%s).", name, ftype)
            continue

        # Unwrap slot format if necessary
        value = _unwrap(json_data.get(name))
        notion_props[name] = _to_notion_value(name, ftype, value)

    logger.info(
        "json_to_notion_properties complete | db='%s' properties=%d",
        database_name, len(notion_props),
    )
    return notion_props


# ── Notion value formatter ────────────────────────────────────────────────────

def _to_notion_value(name: str, ftype: str, value: Any) -> dict[str, Any]:
    """Wrap a single typed Python value in the correct Notion property structure."""
    if ftype == "title":
        return {"title": [{"text": {"content": value or ""}}]}

    if ftype == "rich_text":
        return {"rich_text": [{"text": {"content": value or ""}}]}

    if ftype == "number":
        return {"number": value}

    if ftype == "select":
        return {"select": {"name": value} if value else None}

    if ftype == "multi_select":
        items = value if isinstance(value, list) else []
        return {"multi_select": [{"name": opt} for opt in items if opt]}

    if ftype == "date":
        # Serialise datetime objects that may have slipped through
        if isinstance(value, datetime):
            value = value.isoformat()
        return {"date": {"start": value} if value else None}

    if ftype == "checkbox":
        return {"checkbox": bool(value)}

    if ftype in ("url", "email", "phone_number"):
        return {ftype: value or None}

    if ftype == "relation":
        ids = value if isinstance(value, list) else []
        return {"relation": [{"id": pid} for pid in ids if pid]}

    if ftype == "people":
        ids = value if isinstance(value, list) else []
        return {"people": [{"object": "user", "id": uid} for uid in ids if uid]}

    logger.warning(
        "Field '%s': no Notion formatter for type '%s' — skipping.",
        name, ftype,
    )
    return {}
