# core/confirmation.py

"""
Voice-first confirmation and edit layer.  (v0.6 — deduplication-aware)

Changes from v0.5
-----------------
- confirm_data() accepts an optional `already_confirmed` flag.
  When True (set by conversation_engine after the FSM has already reached
  STORE), the function skips the show_entry + voice-capture loop entirely and
  returns the data immediately after validation.  This eliminates the third
  "yes" the user had to speak: FSM-SUMMARY yes → FSM-STORE → confirm_data yes.
  The full loop still runs when called in standalone mode (already_confirmed=False,
  the default), so nothing breaks if confirm_data is used outside the FSM flow.
- edit_data() return-None path is unchanged; caller must check.
- All normalisation, coercion, field resolution logic unchanged.
"""

import logging
from typing import Any

import core.conversation_ui as ui
from voice.whisper_handler import capture_command, manual_input
from nlp.intent_parser import parse_command
from core.validator import ValidationError, validate_record

logger = logging.getLogger("chitragupta.confirmation")


# ── Slot normalisation ────────────────────────────────────────────────────────

def _normalise(data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict) and "value" in v and "confidence" in v:
            flat[k] = v["value"]
        else:
            flat[k] = v
    return flat


# ── Voice capture (thin wrapper) ──────────────────────────────────────────────

def _listen(prompt: str, timeout: int = 5) -> str:
    return ui.ask(prompt, timeout=timeout)


def _listen_command(prompt: str, timeout: int = 5) -> str:
    raw = _listen(prompt, timeout)
    return parse_command(raw)


# ── Coercion ──────────────────────────────────────────────────────────────────

def _coerce(raw: str, current: Any, field_name: str) -> Any:
    if isinstance(current, bool):
        return raw.lower() in ("yes", "true", "1", "on", "yeah", "yep")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(float(raw))
        except ValueError:
            logger.warning("Could not coerce '%s' to int for '%s'.", raw, field_name)
            return current
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError:
            logger.warning("Could not coerce '%s' to float for '%s'.", raw, field_name)
            return current
    if isinstance(current, list):
        return [v.strip().title() for v in raw.replace(" and ", ",").split(",") if v.strip()]
    return raw.strip()


# ── Field target resolver ─────────────────────────────────────────────────────

def _resolve_field_target(raw: str, fields: list[str]) -> str | None:
    import re
    raw_lower = raw.strip().lower()
    if not raw_lower:
        return None

    cmd = parse_command(raw_lower)
    if cmd.isdigit():
        idx = int(cmd) - 1
        if 0 <= idx < len(fields):
            return fields[idx]

    m = re.search(r"\b(\d{1,2})\b", raw_lower)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(fields):
            return fields[idx]

    for f in fields:
        if f.lower() in raw_lower or raw_lower in f.lower():
            return f

    return None


# ── Single field edit ─────────────────────────────────────────────────────────

def _edit_field_voice(field: str, current: Any) -> Any:
    type_hint = "comma-separated list" if isinstance(current, list) else type(current).__name__
    ui.speak(
        f"Editing '{field}'. Current value: {current}. "
        f"Type: {type_hint}. Say the new value, 'skip' to keep, or 'cancel' to abort."
    )

    raw = ui.ask("New value", timeout=8)

    if not raw:
        ui.acknowledge(f"Keeping: {current}")
        return current

    if parse_command(raw) == "cancel":
        return "__CANCEL__"

    if any(w in raw.lower() for w in ("skip", "keep", "same", "unchanged")):
        ui.acknowledge(f"Keeping: {current}")
        return current

    new_value = _coerce(raw, current, field)
    ui.acknowledge(f"Updated → {new_value}")
    return new_value


# ── Edit loop ─────────────────────────────────────────────────────────────────

def edit_data(json_data: dict[str, Any]) -> dict[str, Any] | None:
    flat_data = _normalise(json_data)
    updated   = dict(flat_data)
    fields    = list(flat_data.keys())

    ui.speak("Which field do you want to change? Say its name or number. Say 'done' when finished.")

    while True:
        ui.show_entry(updated, title="Current values")

        raw = _listen("Field name or number — or 'done' to save", timeout=7)
        raw_lower = raw.strip().lower()

        if raw_lower in ("done", "finish", "finished", "save", "confirm") or parse_command(raw) == "yes":
            ui.acknowledge("Done editing.")
            return updated

        if parse_command(raw) == "cancel":
            ui.speak("Edit cancelled — changes discarded.")
            return None

        target = _resolve_field_target(raw, fields)
        if not target:
            ui.speak("Couldn't find that field. Say its name or number from the list above.")
            continue

        result = _edit_field_voice(target, updated[target])
        if result == "__CANCEL__":
            ui.speak("Edit cancelled — changes discarded.")
            return None

        updated[target] = result
        ui.speak("Edit another field, or say 'done' to save.")

_MAX_VALIDATION_RETRIES: int = 3


# ── Confirmation loop ─────────────────────────────────────────────────────────

def confirm_data(
    json_data: dict[str, Any],
    schema: dict,                       # required — no None fallback
    already_confirmed: bool = False,    # FIX: set True when FSM already got "yes"
) -> dict[str, Any] | None:
    """
    Show the collected data to the user and request final confirmation.

    Args:
        json_data:          Flat data dict (slot values already unwrapped).
        schema:             Loaded schema dict — required, never None.
        already_confirmed:  When True the FSM's SUMMARY state already collected
                            the user's 'yes'.  Skip the voice loop, validate,
                            and return immediately.  This removes the third
                            confirmation that the old architecture required:
                            FSM-SUMMARY "yes" → FSM-STORE → confirm_data "yes".

    Returns:
        Confirmed + validated dict, or None if cancelled / retries exhausted.
    """
    if not json_data:
        logger.warning("confirm_data called with empty dict.")
        return None

    current_data = _normalise(json_data)

    # FIX: fast-path — FSM already collected the user's approval.
    # Just validate and return; no need to ask the user again.
    if already_confirmed:
        try:
            current_data = validate_record(schema, current_data)
            logger.info("confirm_data: fast-path validation passed (already_confirmed=True).")
            return current_data
        except ValidationError as exc:
            logger.error("confirm_data fast-path validation failed: %s", exc)
            ui.error(
                f"Entry validation failed even though you confirmed it.\n{exc}\n"
                "Starting the edit flow so you can correct the field."
            )
            # Fall through to the full interactive loop below
            already_confirmed = False

    validation_failures = 0

    while True:
        ui.show_entry(current_data, title="Review before saving")
        ui.speak("Say 'yes' to save, 'edit' to change a field, or 'cancel' to discard.")

        command = _listen_command("yes / edit / cancel", timeout=6)

        if command == "yes":
            try:
                current_data = validate_record(schema, current_data)
            except ValidationError as exc:
                validation_failures += 1
                logger.warning(
                    "confirm_data: validation failed (%d/%d) — %s",
                    validation_failures, _MAX_VALIDATION_RETRIES, exc,
                )
                if validation_failures >= _MAX_VALIDATION_RETRIES:
                    ui.error(
                        f"Entry discarded after {_MAX_VALIDATION_RETRIES} failed "
                        f"validation attempts.\nLast error: {exc}"
                    )
                    return None

                remaining = _MAX_VALIDATION_RETRIES - validation_failures
                ui.error(
                    f"Cannot save — {exc}\n"
                    f"Edit the field and try again. ({remaining} attempt(s) left.)"
                )
                edited = edit_data(current_data)
                if edited is None:
                    ui.speak("Edit cancelled — entry discarded.")
                    return None
                current_data = edited
                continue

            logger.info("confirm_data: confirmed.")
            ui.acknowledge("Saving …")
            return current_data

        if command == "edit":
            logger.info("confirm_data: edit requested.")
            edited = edit_data(current_data)
            if edited is None:
                ui.speak("Edit cancelled — back to review.")
                continue
            current_data = edited
            continue

        if command == "cancel":
            logger.info("confirm_data: cancelled.")
            ui.speak("Entry discarded.")
            return None

        ui.speak("Say 'yes', 'edit', or 'cancel'.")
