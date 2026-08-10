# core/dialogue_fsm.py

"""
Dialogue FSM  (v0.7 — Jarvis intelligence layer)

Changes from v0.6
-----------------
Auto-date
    Date fields are filled silently with today's date in _next_field_prompt()
    unless the anchor response already mentioned a date.  The user is never
    asked an obvious question.

Pre-fill report
    After anchor extraction, _build_prefill_report() announces which fields
    were understood from the opening summary so the user knows what was heard.

"Same as last time?" for multi_select
    _build_field_question() checks session_memory for the last confirmed value
    of multi_select fields and appends a suggestion.  If the user says "same"
    the stored value is reused without re-asking.

Skip learning
    record_skip() / should_skip() from session_memory are wired in.  After
    three consecutive session skips the field is auto-bypassed with a spoken
    note.  reset_skip() fires when the user actually answers.

"Actually / wait / make that" inline correction
    _ACTUALLY_RE detects mid-answer corrections like "actually make that 8"
    and patches the PREVIOUS slot without leaving the current field.

"Change X to Y" field correction
    _CORRECTION_RE detects "change mood to 9" at any point in PROMPT state
    and applies it immediately, then continues with the current field.

"What have you got?" status query
    Any of the _STATUS_KEYWORDS returns the current summary text + remaining
    field name and stays on the current prompt.

"Start over" command
    _RESTART_KEYWORDS clears all slots and restarts the anchor question.

Field name aliases
    _FIELD_ALIASES maps natural synonyms ("vibe" → mood, "exercise" → activities)
    in _resolve_field() so the user doesn't need exact schema names.

Graceful empty-session exit
    Three consecutive silences trigger a clean exit with a spoken message.

Date re-prompt on low confidence
    After max retries, if the date field value is still empty, a spoken
    note is given before moving on (instead of silent advancement).

Activities repair → list
    _handle_repair() now calls extract_field_answer() for multi_select fields
    so repair responses are properly split into list items.

Earlier fixes (v0.6) preserved:
    - SUMMARY 'yes' → validate → STORE directly (no double confirmation).
    - 'edit' in SUMMARY stays in SUMMARY state.
    - Skip phrases cleanly advance without burning retries.
"""

from __future__ import annotations

import logging
import re
from datetime import date as _date_type
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any

from nlp.intent_parser import (
    parse_intent, validate_answer, extract_field_answer, parse_command,
)
from nlp.schema_inferencer import (
    generate_anchor_question, generate_field_question, generate_clarification_question,
)
from core.validator import ValidationError, validate_record, validate_number

logger = logging.getLogger("chitragupta.dialogue_fsm")

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_RETRIES:             int   = 2
_CONFLICT_THRESHOLD:      float = 0.6
_INTERRUPT_DEBOUNCE_MS:   int   = 500
_MAX_CONSECUTIVE_SILENCES: int  = 3

_SKIP_TYPES: frozenset[str] = frozenset({"formula", "files", "relation", "people"})

_INTERRUPT_KEYWORDS: frozenset[str] = frozenset({
    "stop", "wait", "hold on", "pause", "hang on",
})
_BACKTRACK_KEYWORDS: frozenset[str] = frozenset({
    "go back", "back", "change that", "redo", "undo", "previous",
    "last field", "wrong field",
})
_SKIP_PHRASES: frozenset[str] = frozenset({
    "skip", "next", "next question", "pass", "move on",
    "skip this", "skip that", "don't know", "no answer", "none",
    "not sure", "skip field", "no idea",
})
_STATUS_KEYWORDS: frozenset[str] = frozenset({
    "status", "what have you got", "what do you have", "summary so far",
    "what's collected", "show me", "what have you collected",
    "what did you get", "read back",
})
_RESTART_KEYWORDS: frozenset[str] = frozenset({
    "start over", "restart", "reset", "begin again", "start again",
    "clear everything", "wipe it",
})
# "same as last time" acceptance phrases
_SAME_PHRASES: frozenset[str] = frozenset({
    "same", "same as last time", "same as before", "keep it",
    "that again", "yes same", "yeah same",
})
# "actually" / inline correction for the PREVIOUS field
_ACTUALLY_RE = re.compile(
    r"\b(?:actually|wait|no|make that|i meant|correction|change that to|that should be)\s+(.+)",
    re.IGNORECASE,
)
# "change X to Y" correction for any named field
_CORRECTION_RE = re.compile(
    r"\b(?:change|set|update|make|correct)\s+(.+?)\s+(?:to|as)\s+(.+)",
    re.IGNORECASE,
)

# Natural-language aliases → canonical field name fragments
_FIELD_ALIASES: dict[str, str] = {
    "vibe":               "mood",
    "energy level":       "energy",
    "how i feel":         "mood",
    "feeling":            "mood",
    "exercise":           "activities",
    "workout":            "activities",
    "things done":        "activities",
    "today's activities": "activities",
    "reflection":         "notes",
    "thoughts":           "notes",
    "diary":              "notes",
    "stress":             "stress level",
    "pain":               "pain level",
    "meditate":           "meditation minutes",
    "meditation":         "meditation minutes",
    "sleep":              "sleep hours",
    "hours slept":        "sleep hours",
    "water":              "water intake",
    "heart":              "heart rate",
    "steps taken":        "steps",
    "screen":             "screen time",
}


# ── State enum ────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE      = auto()
    PROMPT    = auto()
    FEEDBACK  = auto()
    SUMMARY   = auto()
    CONFIRM   = auto()
    STORE     = auto()
    HOLD      = auto()
    ERROR     = auto()
    BACKTRACK = auto()
    REPAIR    = auto()


# ── Slot helpers ──────────────────────────────────────────────────────────────

def _make_slot(value: Any, confidence: float) -> dict[str, Any]:
    return {
        "value":      value,
        "confidence": round(float(confidence), 4),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _slot_value(slots: dict, field_name: str) -> Any:
    s = slots.get(field_name)
    return s["value"] if s else None


def _is_trivial(value: Any, ftype: str) -> bool:
    if value is None:                               return True
    if isinstance(value, str) and not value.strip(): return True
    if isinstance(value, list) and not value:        return True
    if ftype == "number"   and value == 0:           return True
    if ftype == "checkbox" and value is False:        return True
    return False


def _empty_value(ftype: str) -> Any:
    if ftype == "number":                                return None
    if ftype == "checkbox":                              return False
    if ftype in ("multi_select", "relation", "people"): return []
    return ""


# ── Conflict detection ────────────────────────────────────────────────────────

def _token_overlap(a: Any, b: Any) -> float:
    if a == b: return 1.0
    if type(a) != type(b): return 0.0
    if isinstance(a, (int, float)):
        denom = max(abs(a), abs(b), 1)
        return max(0.0, 1.0 - abs(a - b) / denom)
    if isinstance(a, str):
        sa = set(re.findall(r"\w+", a.lower()))
        sb = set(re.findall(r"\w+", b.lower()))
        if not sa and not sb: return 1.0
        if not sa or not sb:  return 0.0
        return len(sa & sb) / len(sa | sb)
    if isinstance(a, list):
        sa = {str(x).lower() for x in a}
        sb = {str(x).lower() for x in b}
        if not sa and not sb: return 1.0
        if not sa or not sb:  return 0.0
        return len(sa & sb) / len(sa | sb)
    return 0.0


# ── YAML overrides loader ─────────────────────────────────────────────────────

def _load_yaml_overrides(database_name: str) -> dict:
    from pathlib import Path
    try:
        import yaml
    except ImportError:
        return {}
    base = Path(__file__).resolve().parent.parent / "data" / "schemas"
    safe = re.sub(r"[^\w\-]", "_", database_name)
    path = base / f"{safe}.yaml"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("YAML overrides for '%s': %s", database_name, exc)
        return {}


# ── FSM class ─────────────────────────────────────────────────────────────────

class DialogueFSM:
    """
    Central controller for one conversational logging session.

    Usage
    -----
        fsm = DialogueFSM(database_name, schema)
        prompt = fsm.start()
        while fsm.state not in (State.STORE, State.ERROR):
            user_text = capture()
            prompt = fsm.step(user_text)
            display(prompt)

        if fsm.state == State.STORE:
            data = fsm.export_json()
    """

    def __init__(self, database_name: str, schema: dict) -> None:
        self.db_name: str  = database_name
        self.schema:  dict = schema
        self.state: State  = State.IDLE

        self.slots: dict[str, dict[str, Any]] = {}

        self._fields: list[dict[str, Any]] = [
            f for f in schema.get("fields", [])
            if f["type"] not in _SKIP_TYPES
        ]

        _yaml = _load_yaml_overrides(database_name)
        self._yaml_order:       list[str]      = _yaml.get("slot_order", [])
        self._yaml_prompts:     dict[str, str] = _yaml.get("prompts", {})
        self._yaml_transitions: dict[str, Any] = _yaml.get("transitions", {})

        if self._yaml_order:
            order_map = {n: i for i, n in enumerate(self._yaml_order)}
            self._fields.sort(key=lambda f: order_map.get(f["name"], 9999))

        self._field_index:          int   = 0
        self._retries:              int   = 0
        self._prev_index:           int   = 0
        self._hold_return:          State = State.PROMPT
        self._consecutive_silences: int   = 0

        self._repair_field:    str | None = None
        self._repair_old:      Any        = None
        self._repair_new:      Any        = None
        self._repair_new_conf: float      = 0.0

        self._last_interrupt_ts: float        = 0.0
        self._anchor_done:       bool         = False
        # Tracks fields where we've offered a "same as last time" suggestion
        self._pending_last_suggestion: dict[str, Any] = {}

        logger.info("DialogueFSM init | db='%s' fields=%d", database_name, len(self._fields))

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> str:
        self.state = State.PROMPT
        return generate_anchor_question(self.db_name, self._fields)

    def step(self, user_input: str) -> str:
        text = (user_input or "").strip()

        # Restart command — clears all slots and restarts
        if any(kw in text.lower() for kw in _RESTART_KEYWORDS):
            return self._restart()

        interrupt_prompt = self._check_interrupt(text)
        if interrupt_prompt is not None:
            return interrupt_prompt

        if self.state == State.IDLE:      return self.start()
        if self.state == State.PROMPT:    return self._handle_prompt(text)
        if self.state == State.FEEDBACK:  return self._handle_feedback(text)
        if self.state == State.SUMMARY:   return self._handle_summary(text)
        if self.state == State.CONFIRM:   return self._handle_confirm(text)
        if self.state == State.HOLD:      return self._handle_hold(text)
        if self.state == State.BACKTRACK: return self._handle_backtrack(text)
        if self.state == State.REPAIR:    return self._handle_repair(text)
        if self.state in (State.STORE, State.ERROR): return ""
        return self._error("FSM in unknown state.")

    def export_json(self) -> dict[str, Any]:
        return {f["name"]: _slot_value(self.slots, f["name"]) for f in self._fields}

    def export_slots(self) -> dict[str, dict[str, Any]]:
        return dict(self.slots)

    # ── Interrupt / backtrack ─────────────────────────────────────────────────

    def _check_interrupt(self, text: str) -> str | None:
        import time
        low = text.lower()
        if any(kw in low for kw in _BACKTRACK_KEYWORDS):
            return self._enter_backtrack()
        if any(kw in low for kw in _INTERRUPT_KEYWORDS):
            now = time.monotonic() * 1000
            if (now - self._last_interrupt_ts) > _INTERRUPT_DEBOUNCE_MS:
                self._last_interrupt_ts = now
                self._hold_return = self.state
                self.state = State.HOLD
                return (
                    "Paused. Say 'continue' to resume, "
                    "'cancel' to discard, or 'back' to change the last field."
                )
        return None

    # ── Restart ───────────────────────────────────────────────────────────────

    def _restart(self) -> str:
        self.slots.clear()
        self._field_index          = 0
        self._retries              = 0
        self._anchor_done          = False
        self._consecutive_silences = 0
        self._pending_last_suggestion.clear()
        self.state = State.PROMPT
        logger.info("DialogueFSM: restarted for '%s'.", self.db_name)
        return "Starting over. " + self.start()

    # ── State handlers ────────────────────────────────────────────────────────

    def _handle_prompt(self, text: str) -> str:
        # ── Anchor pass ───────────────────────────────────────────────────────
        if not self._anchor_done:
            self._anchor_done = True
            if text:
                self._extract_multi_slot(text, confidence_base=0.7)
            report = self._build_prefill_report()
            return (report + " " if report else "") + self._next_field_prompt()

        field = self._current_field()
        if field is None:
            return self._enter_summary()

        ftype = field["type"]
        name  = field["name"]

        # ── Status query ──────────────────────────────────────────────────────
        if text and any(kw in text.lower() for kw in _STATUS_KEYWORDS):
            remaining = self._current_field()
            suffix = f"\n\nStill need: {remaining['name']}." if remaining else ""
            return self._build_summary_text() + suffix

        # ── "Same as last time?" acceptance ───────────────────────────────────
        if name in self._pending_last_suggestion:
            low = text.lower().strip().rstrip(".,!?")
            if low in _SAME_PHRASES or any(p in low for p in ("same", "keep it")):
                value = self._pending_last_suggestion.pop(name)
                self._set_slot(name, value, 0.95)
                try:
                    from core.session_memory import reset_skip
                    reset_skip(self.db_name, name)
                except Exception:
                    pass
                self._retries     = 0
                self._field_index += 1
                self._consecutive_silences = 0
                self.state = State.FEEDBACK
                display = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
                return f"Got it — {name}: {display}."
            else:
                self._pending_last_suggestion.pop(name, None)

        # ── "Actually / wait / make that" — corrects PREVIOUS field ──────────
        if text and self._field_index > 0:
            m = _ACTUALLY_RE.search(text)
            if m:
                prev_field = self._fields[self._field_index - 1]
                new_text   = m.group(1).strip()
                is_valid, _ = validate_answer(prev_field, new_text)
                if is_valid:
                    value = extract_field_answer(prev_field, new_text)
                    self._set_slot(prev_field["name"], value, 0.9)
                    logger.info("Inline correction: %s → %r", prev_field["name"], value)
                    current = self._current_field()
                    return (
                        f"Updated {prev_field['name']} to '{value}'. "
                        + (self._build_field_question(current) if current else self._enter_summary())
                    )

        # ── "Change X to Y" — corrects any named field ────────────────────────
        if text:
            m = _CORRECTION_RE.search(text)
            if m:
                target_text = m.group(1).strip()
                new_text    = m.group(2).strip()
                target_field = self._resolve_field(target_text)
                if target_field:
                    is_valid, _ = validate_answer(target_field, new_text)
                    value = extract_field_answer(target_field, new_text) if is_valid else None
                    if value is not None:
                        self._set_slot(target_field["name"], value, 0.9)
                        logger.info("Field correction: %s → %r", target_field["name"], value)
                        current = self._current_field()
                        return (
                            f"Done — {target_field['name']} set to '{value}'. "
                            + (self._build_field_question(current) if current else self._enter_summary())
                        )

        # ── Skip phrase ───────────────────────────────────────────────────────
        if text and any(phrase in text.lower() for phrase in _SKIP_PHRASES):
            try:
                from core.session_memory import record_skip
                count = record_skip(self.db_name, name)
                if count >= 3:
                    msg = f"Noted — I'll skip '{name}' automatically from now on."
                else:
                    msg = f"Skipping '{name}'."
            except Exception:
                msg = f"Skipping '{name}'."
            self._set_slot(name, _empty_value(ftype), 0.0)
            self._retries              = 0
            self._field_index         += 1
            self._consecutive_silences = 0
            return msg + " " + self._next_field_prompt()

        # ── Empty input / silence ─────────────────────────────────────────────
        if not text:
            self._consecutive_silences += 1
            if self._consecutive_silences >= _MAX_CONSECUTIVE_SILENCES:
                self.state = State.ERROR
                return "Looks like you're done for now. Entry discarded."
            if self._retries < _MAX_RETRIES - 1:
                self._retries += 1
                self.state = State.FEEDBACK
                return generate_clarification_question(field, "", "empty")
            self._set_slot(name, _empty_value(ftype), 0.0)
            self._retries     = 0
            self._field_index += 1
            if ftype == "date":
                return f"No date captured — leaving blank. " + self._next_field_prompt()
            return self._next_field_prompt()

        # Real input — reset silence counter
        self._consecutive_silences = 0

        # ── Normal extraction ─────────────────────────────────────────────────
        is_valid, reason = validate_answer(field, text)

        if is_valid:
            value      = extract_field_answer(field, text)
            confidence = 0.9

            existing = self.slots.get(name)
            if existing and not _is_trivial(existing["value"], ftype):
                if _token_overlap(existing["value"], value) < _CONFLICT_THRESHOLD:
                    self._repair_field    = name
                    self._repair_old      = existing["value"]
                    self._repair_new      = value
                    self._repair_new_conf = confidence
                    self.state = State.REPAIR
                    return (
                        f"For '{name}' I have '{self._repair_old}' but you said '{value}'. "
                        "Which should I keep? Say 'old', 'new', or repeat the correct value."
                    )

            self._set_slot(name, value, confidence)
            stored = self.slots[name]["value"]

            if stored is None and value is not None:
                self._retries += 1
                self.state = State.FEEDBACK
                if self._retries <= _MAX_RETRIES:
                    return (
                        f"Sorry, {value} isn't valid for '{name}'. "
                        + generate_clarification_question(field, str(value), "no_number")
                    )
                logger.warning("Max retries for range field '%s'. Skipping.", name)
                self._retries     = 0
                self._field_index += 1
                return self._next_field_prompt()

            # Reset skip counter — user actually answered
            try:
                from core.session_memory import reset_skip
                reset_skip(self.db_name, name)
            except Exception:
                pass

            self._retries     = 0
            self._field_index += 1
            self.state = State.FEEDBACK
            return f"Got it — {name}: {stored}."

        if self._retries < _MAX_RETRIES - 1:
            self._retries += 1
            self.state = State.FEEDBACK
            return generate_clarification_question(field, text, reason)

        value = extract_field_answer(field, text)
        self._set_slot(name, value, 0.4)
        self._retries     = 0
        self._field_index += 1
        if ftype == "date" and not value:
            return "No date captured — leaving blank. " + self._next_field_prompt()
        return self._next_field_prompt()

    def _handle_feedback(self, text: str) -> str:
        self.state = State.PROMPT
        return self._handle_prompt(text)

    def _handle_summary(self, text: str) -> str:
        """
        SUMMARY: validate → STORE on 'yes' (single confirmation).
        'edit' stays in SUMMARY and asks which field.
        """
        cmd = parse_command(text)

        if cmd == "yes":
            draft = self.export_json()
            try:
                validate_record(self.schema, draft)
            except ValidationError as exc:
                return self._enter_error(f"Cannot save — {exc}. Please edit the field.")
            self.state = State.STORE
            logger.info("DialogueFSM: entry confirmed for '%s'.", self.db_name)
            return "✓ Entry confirmed."

        if cmd == "edit":
            return "Which field would you like to edit? Say its name or number."

        if cmd == "cancel":
            self.state = State.ERROR
            return "Entry discarded."

        target = self._resolve_field(text)
        if target:
            self._field_index = self._fields.index(target)
            self._retries     = 0
            self.state        = State.PROMPT
            return self._build_field_question(target)

        return "Say 'yes' to save, 'edit' to change a field, or 'cancel' to discard."

    def _handle_confirm(self, text: str) -> str:
        """Dead state — delegates to SUMMARY handler."""
        logger.warning("DialogueFSM: reached CONFIRM (unexpected). Delegating to SUMMARY.")
        self.state = State.SUMMARY
        return self._handle_summary(text)

    def _handle_hold(self, text: str) -> str:
        low = text.lower()
        if any(w in low for w in ("continue", "resume", "go on", "proceed")):
            self.state = self._hold_return
            field = self._current_field()
            if field and self.state == State.PROMPT:
                return self._build_field_question(field)
            return "Resuming where we left off."
        if parse_command(text) == "cancel":
            self.state = State.ERROR
            return "Entry discarded."
        if any(kw in low for kw in _BACKTRACK_KEYWORDS):
            return self._enter_backtrack()
        return "Still paused. Say 'continue' to go on or 'cancel' to discard."

    def _handle_backtrack(self, text: str) -> str:
        field = self._current_field()
        if field is None:
            self.state = State.SUMMARY
            return self._enter_summary()
        name  = field["name"]
        ftype = field["type"]
        if not text:
            self.state = State.PROMPT
            return self._build_field_question(field)
        is_valid, _ = validate_answer(field, text)
        value = extract_field_answer(field, text) if is_valid else _empty_value(ftype)
        conf  = 0.9 if is_valid else 0.4
        self._set_slot(name, value, conf)
        self._field_index = self._prev_index
        self._retries     = 0
        self.state        = State.PROMPT
        return f"Updated {name} to '{value}'. " + self._next_field_prompt()

    def _handle_repair(self, text: str) -> str:
        low        = text.lower()
        field_name = self._repair_field
        if not field_name:
            self.state = State.PROMPT
            return self._next_field_prompt()

        if "old" in low:
            self.state        = State.PROMPT
            self._retries     = 0
            self._field_index += 1
            return self._next_field_prompt()

        if "new" in low or parse_command(text) == "yes":
            self._set_slot(field_name, self._repair_new, self._repair_new_conf)
            self.state        = State.PROMPT
            self._retries     = 0
            self._field_index += 1
            return self._next_field_prompt()

        field = next((f for f in self._fields if f["name"] == field_name), None)
        if field:
            is_valid, _ = validate_answer(field, text)
            # FIX: multi_select repair response must go through extraction,
            # not be stored as a raw sentence.
            if field["type"] == "multi_select":
                value = extract_field_answer(field, text)
            elif is_valid:
                value = extract_field_answer(field, text)
            else:
                value = self._repair_old
            self._set_slot(field_name, value, 0.85)
            self.state        = State.PROMPT
            self._retries     = 0
            self._field_index += 1
            return f"Got it — {field_name}: '{value}'. " + self._next_field_prompt()

        self.state = State.PROMPT
        return self._next_field_prompt()

    # ── Navigation helpers ────────────────────────────────────────────────────

    def _current_field(self) -> dict[str, Any] | None:
        if 0 <= self._field_index < len(self._fields):
            return self._fields[self._field_index]
        return None

    def _next_field_prompt(self) -> str:
        while self._field_index < len(self._fields):
            field = self._fields[self._field_index]
            name  = field["name"]
            ftype = field["type"]

            # Auto-date: fill today silently if not already captured
            if ftype == "date":
                existing = self.slots.get(name)
                if not existing or _is_trivial(existing["value"], ftype):
                    today = _date_type.today().isoformat()
                    self._set_slot(name, today, 0.99)
                    logger.info("Auto-filled '%s' with today: %s", name, today)
                    self._field_index += 1
                    continue

            # Skip learning: auto-bypass repeatedly-skipped fields
            try:
                from core.session_memory import should_skip
                if should_skip(self.db_name, name):
                    logger.info("Auto-skipping learned field '%s'.", name)
                    self._set_slot(name, _empty_value(ftype), 0.0)
                    self._field_index += 1
                    continue
            except Exception:
                pass

            existing = self.slots.get(name)
            if existing and not _is_trivial(existing["value"], ftype) and existing["confidence"] >= 0.7:
                logger.debug("Skipping pre-filled '%s' (conf=%.2f).", name, existing["confidence"])
                self._field_index += 1
                continue
            break

        field = self._current_field()
        if field is None:
            return self._enter_summary()

        self.state = State.PROMPT
        return self._build_field_question(field)

    def _build_field_question(self, field: dict[str, Any]) -> str:
        name  = field["name"]
        ftype = field["type"]

        if name in self._yaml_prompts:
            return self._yaml_prompts[name]

        base_q = generate_field_question(field)

        # "Same as last time?" hint for multi_select fields
        if ftype == "multi_select":
            try:
                from core.session_memory import get_last_value
                last = get_last_value(self.db_name, name)
                if last and isinstance(last, list) and last:
                    last_str = ", ".join(str(v) for v in last[:3])
                    self._pending_last_suggestion[name] = last
                    return f"{base_q} Last time: {last_str}. Say 'same' to reuse, or give new values."
            except Exception:
                pass

        return base_q

    def _enter_summary(self) -> str:
        self.state = State.SUMMARY
        return (
            self._build_summary_text()
            + "\n\nSay 'yes' to save, 'edit' to change a field, or 'cancel' to discard."
        )

    def _enter_backtrack(self) -> str:
        self._prev_index  = self._field_index
        self._field_index = max(0, self._field_index - 1)
        self._retries     = 0
        self.state        = State.BACKTRACK
        field = self._current_field()
        if field:
            return f"Going back to '{field['name']}'. " + self._build_field_question(field)
        return "Nothing to go back to."

    def _enter_error(self, message: str) -> str:
        self.state = State.ERROR
        logger.error("DialogueFSM ERROR: %s", message)
        return f"⚠ {message}"

    def _error(self, message: str) -> str:
        return self._enter_error(message)

    # ── Multi-slot extraction ─────────────────────────────────────────────────

    def _extract_multi_slot(self, text: str, confidence_base: float = 0.7) -> None:
        try:
            extracted = parse_intent(self.db_name, text)
        except Exception as exc:
            logger.warning("Multi-slot extraction failed: %s", exc)
            return
        for field in self._fields:
            name  = field["name"]
            ftype = field["type"]
            value = extracted.get(name)
            if not _is_trivial(value, ftype):
                self._set_slot(name, value, confidence_base)
                logger.debug("Pre-filled '%s' = %r (conf=%.2f)", name, value, confidence_base)

    def _build_prefill_report(self) -> str:
        """
        Return a short sentence announcing what was extracted from the anchor.
        Example: "Got your mood and sleep from that."
        Empty string if nothing was pre-filled.
        """
        filled = [
            f["name"]
            for f in self._fields
            if f["name"] in self.slots
            and not _is_trivial(self.slots[f["name"]]["value"], f["type"])
            and self.slots[f["name"]]["confidence"] >= 0.6
        ]
        if not filled:
            return ""
        if len(filled) == 1:
            return f"Got your {filled[0].lower()} from that."
        names = " and ".join(
            [f.lower() for f in filled[:-1]]
        ) + f" and {filled[-1].lower()}"
        return f"Got your {names} from that."

    # ── Slot management ───────────────────────────────────────────────────────

    def _set_slot(self, field_name: str, value: Any, confidence: float) -> None:
        if value is not None:
            field = next((f for f in self._fields if f["name"] == field_name), None)
            if field and field.get("type") == "number":
                try:
                    value = validate_number(field_name, value, field_def=field)
                except ValidationError as exc:
                    logger.warning("Slot '%s': range violation — %s. Storing None.", field_name, exc)
                    value      = None
                    confidence = 0.0
        self.slots[field_name] = _make_slot(value, confidence)
        logger.debug("Slot set | field='%s' value=%r conf=%.2f", field_name, value, confidence)

    # ── Summary builder ───────────────────────────────────────────────────────

    def _build_summary_text(self) -> str:
        lines = [f"\n── Summary: {self.db_name} {'─' * max(0, 40 - len(self.db_name))}"]
        for i, field in enumerate(self._fields, 1):
            name = field["name"]
            slot = self.slots.get(name)
            if slot:
                v        = slot["value"]
                conf_tag = f"  [conf: {slot['confidence']:.0%}]" if slot["confidence"] < 0.8 else ""
                display  = (
                    ", ".join(str(x) for x in v) if isinstance(v, list)
                    else ("(empty)" if v is None else str(v))
                )
                lines.append(f"  {i:>2}. {name}: {display}{conf_tag}")
            else:
                lines.append(f"  {i:>2}. {name}: (not collected)")
        return "\n".join(lines)

    # ── Field resolver ────────────────────────────────────────────────────────

    def _resolve_field(self, text: str) -> dict[str, Any] | None:
        raw = text.strip().lower()
        if not raw:
            return None

        # Apply field aliases before matching
        for alias, canonical in _FIELD_ALIASES.items():
            if alias in raw:
                raw = raw.replace(alias, canonical)
                break

        m = re.search(r"\b(\d{1,2})\b", raw)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(self._fields):
                return self._fields[idx]

        cmd = parse_command(raw)
        if cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(self._fields):
                return self._fields[idx]

        for f in self._fields:
            if f["name"].lower() in raw or raw in f["name"].lower():
                return f

        return None
