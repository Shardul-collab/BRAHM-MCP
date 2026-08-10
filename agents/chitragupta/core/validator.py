# core/validator.py

"""
Central validation layer for Chitragupta.  (v1.0)

Responsibilities
----------------
- Enforce value ranges for bounded numeric fields (mood, energy, productivity → 1–10)
- Coerce and validate list fields (keywords / multi_select → list[str])
- Reject None / empty required inputs with clear errors
- Provide a safe JSON serializer that handles datetime objects
- Validate a full data record against its schema in one call

Design principles
-----------------
- Raise ValidationError immediately on the first violation (fail-fast)
- No UI, no logging to stdout — pure input → output transformation
- Imported by json_builder, confirmation, and dialogue_fsm
"""

import json
import logging
from datetime import date, datetime
from typing import Any

logger = logging.getLogger("chitragupta.validator")


# ── Custom exception ──────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when input data fails validation. Pipeline must stop."""


# ── Bounded numeric field registry ────────────────────────────────────────────
# Maps lowercase field name → (min, max).  Extend here as needed.

_BOUNDED_FIELDS: dict[str, tuple[int, int]] = {
    "mood":            (1, 10),
    "energy":          (1, 10),
    "productivity":    (1, 10),
    "stress level":    (1, 10),
    "stress":          (1, 10),
    "pain level":      (1, 10),
    "pain":            (1, 10),
    "score":           (1, 10),
    "focus":           (1, 10),
    "happiness":       (1, 10),
}


def _get_bounds(field_name: str) -> tuple[int, int] | None:
    return _BOUNDED_FIELDS.get(field_name.strip().lower())


# ── Per-type validators ───────────────────────────────────────────────────────

def validate_number(
    field_name: str,
    value: Any,
    field_def: dict | None = None,
) -> int | float | None:
    """
    Coerce value to a number and enforce range bounds.

    Bounds priority:
      1. _BOUNDED_FIELDS hardcoded registry (by name)
      2. field_def.get("min") / field_def.get("max") from schema file

    Returns:
        Coerced int or float, or None if value is empty.

    Raises:
        ValidationError: if value cannot be coerced or is out of range.
    """
    if value is None or value == "":
        return None

    try:
        f = float(str(value).strip())
    except (ValueError, TypeError):
        raise ValidationError(
            f"Field '{field_name}': '{value}' is not a valid number."
        )

    bounds = _get_bounds(field_name)
    if bounds is None and field_def is not None:
        lo = field_def.get("min")
        hi = field_def.get("max")
        if lo is not None and hi is not None:
            try:
                bounds = (float(lo), float(hi))
            except (TypeError, ValueError):
                logger.warning(
                    "Field '%s': schema min/max are not numeric (%r, %r) — ignoring bounds.",
                    field_name, lo, hi,
                )

    if bounds is not None:
        lo, hi = bounds
        if not (lo <= f <= hi):
            raise ValidationError(
                f"Field '{field_name}': value {f} is out of range "
                f"[{lo}–{hi}]. Received: {value!r}"
            )

    return int(f) if float(f).is_integer() else f


def validate_keywords(field_name: str, value: Any) -> list[str]:
    """
    Ensure value is a list of non-empty strings.
    Accepts: list, comma-separated string, or a single value.

    Returns:
        Cleaned list[str] (may be empty — callers decide if empty is valid).
    """
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value).strip()]


def sanitize_text(value: Any) -> str:
    """Strip and return string; return empty string for None/whitespace-only."""
    if value is None:
        return ""
    return str(value).strip()


def validate_required(field_name: str, value: Any) -> None:
    """
    Raise ValidationError if value is None, empty string, or empty list.

    Use only for fields explicitly marked required — most fields are optional.
    """
    if value is None:
        raise ValidationError(
            f"Field '{field_name}' is required but received None."
        )
    if isinstance(value, str) and not value.strip():
        raise ValidationError(
            f"Field '{field_name}' is required but received an empty string."
        )
    if isinstance(value, list) and not value:
        raise ValidationError(
            f"Field '{field_name}' is required but received an empty list."
        )


# ── Full-record validation ────────────────────────────────────────────────────

def validate_record(
    schema: dict,
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate and coerce a full data dict against a loaded schema.

    - Number fields: coerced to int/float and range-checked if bounded.
    - Multi-select / relation / people: coerced to list[str].
    - Text fields: stripped of leading/trailing whitespace.
    - Date / checkbox: passed through unchanged.

    Args:
        schema: loaded schema dict (has a 'fields' list).
        data:   raw data dict keyed by field name.

    Returns:
        New dict with validated/coerced values.

    Raises:
        ValidationError: on the first invalid value found (fail-fast).
    """
    fields: list[dict] = schema.get("fields", [])
    validated: dict[str, Any] = {}

    for field in fields:
        name  = field["name"]
        ftype = field["type"]
        value = data.get(name)

        try:
            if ftype == "number":
                validated[name] = validate_number(name, value, field_def=field)

            elif ftype in ("multi_select", "relation", "people"):
                validated[name] = validate_keywords(name, value)

            elif ftype in ("title", "rich_text", "select", "url", "email", "phone_number"):
                validated[name] = sanitize_text(value)

            else:
                # date, checkbox, formula, files — pass through
                validated[name] = value

        except ValidationError:
            raise   # propagate immediately (fail-fast)
        except Exception as exc:
            raise ValidationError(
                f"Field '{name}': unexpected validation error — {exc}"
            ) from exc

    logger.debug(
        "validate_record OK | schema='%s' fields=%d",
        schema.get("name", "?"), len(validated),
    )
    return validated


# ── Payload guard ─────────────────────────────────────────────────────────────

def guard_notion_payload(properties: dict[str, Any]) -> None:
    """
    Raise ValidationError if the Notion properties payload is obviously broken.

    Checks:
    - Not empty
    - No value is a raw Python datetime (should have been serialised already)

    Call this in notion_client just before hitting the API.
    """
    if not properties:
        raise ValidationError(
            "Notion payload is empty — refusing to send a blank page to the API."
        )

    for key, val in properties.items():
        if isinstance(val, (datetime, date)):
            raise ValidationError(
                f"Notion payload field '{key}' contains a raw datetime object. "
                "It must be serialised to an ISO string before sending."
            )


# ── JSON serialization safety ─────────────────────────────────────────────────

def json_serial(obj: Any) -> str:
    """
    Fallback serializer for json.dumps — handles datetime and date objects.

    Usage:
        import json
        from core.validator import json_serial
        safe_str = json.dumps(data, default=json_serial)
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serialisable."
    )


def safe_dumps(data: Any, **kwargs) -> str:
    """
    json.dumps wrapper that never crashes on datetime values.

    Usage:
        from core.validator import safe_dumps
        print(safe_dumps(my_dict))
    """
    return json.dumps(data, default=json_serial, **kwargs)
