# nlp/intent_parser.py

"""
NLP / Intent Parser — transcript text → structured field extraction.
(v0.5 — anchor multi-select fix)

Changes from v0.4
-----------------
- _extract_multi_select(): full-input fallback (Strategy 3) is now gated
  behind `allow_fallback=True`.  During the anchor pass (allow_fallback=False),
  this fallback was splitting the entire summary narrative — "Exhausted." —
  into Activities items, pre-filling the field with garbage and triggering
  a spurious REPAIR conflict on the next real answer.
- _extract_field(): passes allow_fallback through to _extract_multi_select().
- All v0.4 fixes preserved:
    - parse_command(): filler words removed from "yes" aliases.
    - validate_answer(): skip-intent checked first for all field types.
    - _extract_field(): DistilBERT gated on allow_fallback.
"""

import re
import logging
import threading
from typing import Any

from notion.schema_manager import load_schema

logger = logging.getLogger("chitragupta.intent_parser")


# ── DistilBERT singleton ──────────────────────────────────────────────────────

_qa_pipeline   = None
_pipeline_lock = threading.Lock()


def _get_pipeline():
    global _qa_pipeline
    if _qa_pipeline is not None:
        if _qa_pipeline is False:
            raise NLPLoadError("DistilBERT model failed to load (cached failure).")
        return _qa_pipeline
    with _pipeline_lock:
        if _qa_pipeline is not None:
            if _qa_pipeline is False:
                raise NLPLoadError("DistilBERT model failed to load (cached failure).")
            return _qa_pipeline
        try:
            from transformers import pipeline
            import config.settings as cfg
            logger.info("Loading DistilBERT pipeline ...")
            _qa_pipeline = pipeline(
                "question-answering",
                model=cfg.NLP_MODEL,
                tokenizer=cfg.NLP_MODEL,
            )
            logger.info("DistilBERT pipeline loaded.")
        except Exception as exc:
            _qa_pipeline = False
            raise NLPLoadError(f"Failed to load NLP model: {exc}") from exc
    return _qa_pipeline


# ── Input sanitisation ────────────────────────────────────────────────────────

def _sanitize(text: Any) -> str:
    if text is None:
        return ""
    try:
        return str(text).strip()
    except Exception:
        return ""


_GARBAGE_PATTERNS = re.compile(
    r"^[^a-zA-Z0-9]{3,}$|^\s*$|^(.)\1{4,}$"
)


def _is_garbage(text: str) -> bool:
    return bool(_GARBAGE_PATTERNS.match(text)) or len(text) < 1


# ── Command parser ────────────────────────────────────────────────────────────

def _word_in(word: str, text: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text))


def parse_command(transcript: str) -> str:
    """
    Map spoken transcript to canonical command string.
    Filler words (okay, ok, right, sure, good, proceed) removed from 'yes'
    aliases — they appear constantly as thinking pauses.
    """
    t = _sanitize(transcript).lower()
    if not t:
        return "unknown"

    _MENU_PHRASES: dict[str, str] = {
        "log entry": "2", "log an entry": "2", "add entry": "2",
        "new entry": "2", "add a log": "2",
        "create database": "1", "new database": "1", "add database": "1",
        "analyze data": "3", "analyse data": "3", "show analysis": "3",
        "link database": "4", "link databases": "4",
        "start scheduler": "5", "set scheduler": "5",
    }
    for phrase, cmd in _MENU_PHRASES.items():
        if phrase in t:
            return cmd

    if any(_word_in(w, t) for w in ("yes", "yeah", "yep", "yup", "confirm", "save", "correct")):
        return "yes"
    if "looks good" in t or "that's right" in t or "that's correct" in t:
        return "yes"

    if any(_word_in(w, t) for w in ("cancel", "discard", "abort", "stop", "no", "nope", "nah", "quit")):
        return "cancel"

    if any(_word_in(w, t) for w in ("edit", "change", "modify", "update", "fix", "wrong", "incorrect", "adjust")):
        return "edit"

    if any(_word_in(w, t) for w in ("foreground", "blocking")):
        return "1"
    if any(_word_in(w, t) for w in ("background", "daemon")):
        return "2"
    if "run now" in t or _word_in("immediately", t):
        return "3"

    number_map = {
        "one": "1", "first": "1", "create": "1",
        "two": "2", "second": "2", "log": "2",
        "three": "3", "third": "3", "analyse": "3", "analyze": "3", "analysis": "3",
        "four": "4", "fourth": "4",
        "five": "5", "fifth": "5", "schedule": "5", "scheduler": "5",
        "six": "6", "sixth": "6", "exit": "6", "leave": "6", "bye": "6", "goodbye": "6",
        "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    }
    for phrase, command in number_map.items():
        if _word_in(phrase, t):
            return command

    return "unknown"


# ── Heuristic extractors ──────────────────────────────────────────────────────

_WORD_NUMBERS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}


def _words_to_digits(text: str) -> str:
    tokens = text.lower().split()
    out = []
    for tok in tokens:
        stripped = tok.rstrip(".,!?;:")
        suffix   = tok[len(stripped):]
        out.append(str(_WORD_NUMBERS[stripped]) + suffix if stripped in _WORD_NUMBERS else tok)
    return " ".join(out)


def _extract_number(field_name: str, text: str) -> float | int | None:
    text = _words_to_digits(text)
    name_pattern = re.escape(field_name.lower())
    anchored = re.search(
        rf"{name_pattern}\s*(?:is|was|:|=|of)?\s*([\d]+(?:[.,]\d+)?)",
        text.lower(),
    )
    if anchored:
        return _as_number(anchored.group(1).replace(",", "."))
    all_numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    if all_numbers:
        return _as_number(all_numbers[0])
    return None


def _as_number(raw: str) -> float | int:
    f = float(raw)
    return int(f) if f.is_integer() else f


_SKIP_PHRASES: frozenset[str] = frozenset({
    "next", "skip", "pass", "move on", "next field", "next one",
    "go on", "proceed", "continue", "move", "move to next",
    "skip that", "skip this", "leave it", "leave blank", "no answer",
})


def _is_skip_intent(text: str) -> bool:
    clean = text.strip().lower().rstrip(".,!?")
    if clean in _SKIP_PHRASES:
        return True
    for phrase in _SKIP_PHRASES:
        if clean.startswith(phrase) and len(clean[len(phrase):].strip().split()) <= 2:
            return True
    return False


def _extract_text(field_name: str, text: str, allow_fallback: bool = True) -> str:
    if _is_skip_intent(text):
        return ""
    name_pattern = re.escape(field_name.lower())
    anchored = re.search(
        rf"(?:^|(?<=[.!?\n]))\s*{name_pattern}\s*(?:is|was|:|=)?\s*([^,.;!?\n]+)",
        text.lower(),
    )
    if anchored:
        start, end = anchored.span(1)
        return text[start:end].strip()
    if not allow_fallback:
        return ""
    cleaned = text.strip()
    return cleaned[:120] if cleaned else ""


def _extract_bool(field_name: str, text: str) -> bool | None:
    name_pattern = re.escape(field_name.lower())
    lowered = text.lower()
    if re.search(
        rf"(?:not|no|didn't|did not|wasn't)\s+{name_pattern}"
        rf"|{name_pattern}\s+(?:is|was)?\s*(?:not|no|false|off)", lowered
    ):
        return False
    if re.search(
        rf"{name_pattern}\s*(?:is|was|:|=)?\s*(?:yes|true|on|done|complete)"
        rf"|\b{name_pattern}\b", lowered
    ):
        return True
    return None


def _extract_select(field_name: str, text: str, options: list[str]) -> str:
    lowered = text.lower()
    negated: set[str] = {m.group(1) for m in _NEGATION_PRE_RE.finditer(lowered)}
    for opt in options:
        opt_lower = opt.lower()
        if set(opt_lower.split()) & negated:
            continue
        if opt_lower in lowered:
            return opt
    return ""


def _extract_multi_select(
    field_name: str,
    text: str,
    options: list[str],
    allow_fallback: bool = True,
) -> list[str]:
    """
    Strategy 1: known options — scan text for option keywords.
    Strategy 2: field-anchored regex match.
    Strategy 3: full-input split (ONLY when allow_fallback=True).

    FIX: Strategy 3 is now gated on allow_fallback.  During the anchor pass
    (allow_fallback=False) the entire anchor narrative was being split into
    Activities items — "Exhausted." became an activity.
    """
    # Strategy 1
    if options:
        lowered = text.lower()
        return [opt for opt in options if opt.lower() in lowered]

    # Strategy 2: field-anchored
    name_pattern = re.escape(field_name.lower())
    match = re.search(
        rf"{name_pattern}\s*(?:is|are|was|:|=)?\s*([^.!?\n]+)",
        text.lower(),
    )
    if match:
        items = [i.strip().title() for i in re.split(r",|\band\b|\bor\b|[&+]", match.group(1)) if i.strip()]
        if items:
            return items

    # Strategy 3: full-input split — gated on allow_fallback
    if not allow_fallback:
        return []

    raw_items = re.split(r",|\band\b|\bor\b|[&+]", text)
    items = [i.strip().title() for i in raw_items if i.strip() and len(i.strip()) > 1]
    if items:
        logger.debug("_extract_multi_select: full-input fallback for '%s' → %r", field_name, items)
    return items


def _extract_date(field_name: str, text: str) -> str:
    from datetime import date, timedelta
    lowered = text.lower()
    if "today" in lowered:
        return date.today().isoformat()
    if "yesterday" in lowered:
        return (date.today() - timedelta(days=1)).isoformat()
    if "tomorrow" in lowered:
        return (date.today() + timedelta(days=1)).isoformat()
    iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso:
        return iso.group(1)
    slash = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", text)
    if slash:
        d, m, y = slash.group(1), slash.group(2), slash.group(3)
        y = f"20{y}" if len(y) == 2 else y
        try:
            return date(int(y), int(m), int(d)).isoformat()
        except ValueError:
            try:
                return date(int(y), int(d), int(m)).isoformat()
            except ValueError:
                pass
    return ""


def _distilbert_extract(field_name: str, text: str) -> str:
    if not text.strip():
        return ""
    try:
        qa     = _get_pipeline()
        result = qa(question=f"What is {field_name}?", context=text)
        answer = result.get("answer", "").strip()
        if result.get("score", 0.0) >= 0.15 and answer:
            return answer
    except NLPLoadError:
        logger.debug("DistilBERT unavailable for '%s' (cached).", field_name)
    except Exception as exc:
        logger.warning("DistilBERT error for '%s': %s", field_name, exc)
    return ""


def _extract_field(
    field: dict[str, Any],
    text: str,
    allow_fallback: bool = True,
) -> Any:
    name    = field["name"]
    ftype   = field["type"]
    options = field.get("options", [])

    if ftype == "number":
        return _extract_number(name, text)
    if ftype in ("title", "rich_text", "url", "email", "phone_number"):
        value = _extract_text(name, text, allow_fallback=allow_fallback)
        if not value and allow_fallback:
            value = _distilbert_extract(name, text)
        return value
    if ftype == "checkbox":
        result = _extract_bool(name, text)
        return result if result is not None else False
    if ftype == "select":
        return _extract_select(name, text, options)
    if ftype == "multi_select":
        # FIX: pass allow_fallback so anchor pass doesn't trigger full-input split
        return _extract_multi_select(name, text, options, allow_fallback=allow_fallback)
    if ftype == "date":
        return _extract_date(name, text)
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def parse_intent(database_name: str, text: str) -> dict[str, Any]:
    clean = _sanitize(text)
    if not clean or _is_garbage(clean):
        logger.warning("parse_intent: rejected empty/garbage input for db='%s'.", database_name)
        return {}
    schema = load_schema(database_name)
    fields: list[dict[str, Any]] = schema["fields"]
    extracted: dict[str, Any] = {}
    for field in fields:
        name  = field["name"]
        value = _extract_field(field, clean, allow_fallback=False)
        extracted[name] = value
        logger.debug("parse_intent | field='%s' value=%r", name, value)
    logger.info("parse_intent complete | db='%s' fields=%d", database_name, len(extracted))
    return extracted


_VAGUE_SINGLE: frozenset[str] = frozenset({
    "okay", "ok", "fine", "good", "bad", "some", "maybe", "yes",
    "no", "stuff", "things", "something", "nothing", "everything",
    "a lot", "not much", "idk", "dunno", "well", "alright",
})

_NO_NUMBER_RE  = re.compile(r"\d")
_DATE_SIGNALS  = frozenset({"today", "yesterday", "tomorrow"})
_DATE_RE       = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[-/]\d{1,2}")
_NEGATION_PRE_RE = re.compile(
    r"\b(?:not|no|never|without|wasn't|isn't|didn't|don't|hardly|barely)\b\s+(\w+)"
)
_BOOL_YES = frozenset({"yes", "yeah", "yep", "true", "done", "completed", "1", "on", "did"})
_BOOL_NO  = frozenset({"no", "nope", "nah", "false", "not", "0", "off", "didn't", "did not"})


def validate_answer(field: dict[str, Any], raw: str) -> tuple[bool, str]:
    ftype = field["type"]
    raw_s = _sanitize(raw)

    if not raw_s or _is_garbage(raw_s):
        return False, "empty"

    # Skip intent checked first — for ALL field types
    if _is_skip_intent(raw_s):
        return False, "empty"

    raw_lower = raw_s.lower()
    words     = raw_lower.split()

    if ftype == "number":
        if not _NO_NUMBER_RE.search(_words_to_digits(raw_s)):
            return False, "no_number"
        return True, ""

    if ftype == "checkbox":
        word_set = set(words)
        if word_set & _BOOL_YES or word_set & _BOOL_NO:
            return True, ""
        if any(w in raw_lower for w in _BOOL_YES | _BOOL_NO):
            return True, ""
        return False, "ambiguous_bool"

    if ftype == "select":
        options = field.get("options", [])
        if options:
            negated: set[str] = {m.group(1) for m in _NEGATION_PRE_RE.finditer(raw_lower)}
            if not any(
                opt.lower() in raw_lower and not (set(opt.lower().split()) & negated)
                for opt in options
            ):
                return False, "no_option_match"
        return True, ""

    if ftype == "multi_select":
        return True, ""

    if ftype == "date":
        if not any(s in raw_lower for s in _DATE_SIGNALS) and not _DATE_RE.search(raw_s):
            return False, "no_date"
        return True, ""

    if ftype in ("title", "rich_text", "url", "email", "phone_number"):
        if len(words) == 1 and words[0] in _VAGUE_SINGLE:
            return False, "too_vague"
        if words and all(w in _VAGUE_SINGLE for w in words):
            return False, "too_vague"
        return True, ""

    return True, ""


def extract_field_answer(field: dict[str, Any], raw: str) -> Any:
    clean = _sanitize(raw)
    if not clean:
        return None
    return _extract_field(field, clean)


def detect_context(text: str, database_names: list[str]) -> str | None:
    clean = _sanitize(text)
    if not clean or not database_names:
        return None
    lowered = clean.lower()
    scores: list[tuple[int, str]] = []
    for name in database_names:
        score = sum(1 for w in re.findall(r"\w+", name.lower()) if w in lowered)
        if score > 0:
            scores.append((score, name))
    if not scores:
        return None
    scores.sort(key=lambda x: x[0], reverse=True)
    if len(scores) > 1 and scores[1][0] == scores[0][0]:
        return None
    return scores[0][1]


class NLPLoadError(Exception):
    """Raised when the DistilBERT model/pipeline cannot be loaded."""
