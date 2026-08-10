# nlp/schema_inferencer.py

"""
Schema Inferencer  (v0.4 — Jarvis question quality + natural schema read-back)

Changes from v0.3
-----------------
- generate_field_question(): "Title" field no longer produces "…for this
  Title entry" — the word appeared twice.  Special-cased to a cleaner form.
- generate_field_question(): all templates shortened and made more
  conversational — form-style phrasing replaced with natural speech.
- describe_schema_naturally(): new public function that returns a single
  spoken sentence describing the schema, used by main.py before the
  confirmation prompt so the user hears what they're about to create
  instead of reading a table.
- All v0.3 fixes preserved: Notes always last, rich_text contextual
  questions, 1-to-10 scale hints for subjective numeric fields.
"""

import re
import logging
from typing import Any

logger = logging.getLogger("chitragupta.schema_inferencer")


# ── Field inference rules ─────────────────────────────────────────────────────

_FIELD_RULES: list[tuple[str, str, str, dict]] = [
    (r"\btitle|name|topic|subject|entry|log\b",       "Title",              "title",        {}),
    (r"\bmood|happiness|emotion|feeling\b",            "Mood",               "number",       {}),
    (r"\benergy|vitality|stamina\b",                   "Energy",             "number",       {}),
    (r"\bsleep|rest|hours? slept\b",                   "Sleep Hours",        "number",       {}),
    (r"\bcalori|kcal|food intake\b",                   "Calories",           "number",       {}),
    (r"\bweight|mass|kg|lbs?\b",                       "Weight",             "number",       {}),
    (r"\bsteps?|walk|distance|km|miles?\b",            "Steps",              "number",       {}),
    (r"\bheartrate|heart rate|bpm|pulse\b",            "Heart Rate",         "number",       {}),
    (r"\bwater|hydration|liters?\b",                   "Water Intake",       "number",       {}),
    (r"\bscore|rating|point|rank\b",                   "Score",              "number",       {}),
    (r"\bproductiv|focus|deep work|hours? worked\b",   "Productivity",       "number",       {}),
    (r"\bscreen ?time|phone usage\b",                  "Screen Time",        "number",       {}),
    (r"\bstress|anxiety|tension\b",                    "Stress Level",       "number",       {}),
    (r"\bpain|discomfort|ache\b",                      "Pain Level",         "number",       {}),
    (r"\bmeditat|mindful\b",                           "Meditation Minutes", "number",       {}),
    (r"\bworkout|exercise|gym|training|sport\b",
     "Workout Type", "select",
     {"options": ["Strength", "Cardio", "Yoga", "Sports", "Rest", "Other"]}),
    (r"\bmeal|diet|nutrition|eating\b",
     "Meal Type", "select",
     {"options": ["Breakfast", "Lunch", "Dinner", "Snack"]}),
    (r"\bpriority|urgency|importance\b",
     "Priority", "select",
     {"options": ["High", "Medium", "Low"]}),
    (r"\bstatus|progress|stage\b",
     "Status", "select",
     {"options": ["Not Started", "In Progress", "Done", "Blocked"]}),
    (r"\bmood type|emotion type\b",
     "Mood Type", "select",
     {"options": ["Happy", "Neutral", "Sad", "Anxious", "Excited"]}),
    (r"\btag|label|categor|topic|area\b",              "Tags",               "multi_select", {}),
    (r"\bactivit|task|thing[s]? done\b",               "Activities",         "multi_select", {}),
    (r"\bskill|learned|studied\b",                     "Skills",             "multi_select", {}),
    (r"\bsymptom|issue|problem\b",                     "Symptoms",           "multi_select", {}),
    (r"\bdate|day|when|timestamp\b",                   "Date",               "date",         {}),
    (r"\btime|hour|clock\b",                           "Time",               "rich_text",    {}),
    (r"\bmedication|medicine|pill|supplement\b",       "Took Medication",    "checkbox",     {}),
    (r"\bexercis|worked out|trained\b",                "Exercised",          "checkbox",     {}),
    (r"\bjournal|wrote|writing\b",                     "Journaled",          "checkbox",     {}),
    (r"\bnote|journal|thought|reflection|comment|detail|description\b",
     "Notes",      "rich_text", {}),
    (r"\bgoal|intention|target|aim\b",                 "Goals",              "rich_text",    {}),
    (r"\bgrateful|gratitude|thankful\b",               "Gratitude",          "rich_text",    {}),
    (r"\blearn|insight|takeaway\b",                    "Learnings",          "rich_text",    {}),
]

_TAIL_TYPES:           frozenset[str] = frozenset({"rich_text"})
_RICH_TEXT_STRUCTURAL: frozenset[str] = frozenset({"time"})


# ── Core inference ────────────────────────────────────────────────────────────

def infer_schema_from_description(description: str) -> list[dict[str, Any]]:
    """Rich_text fields sorted to tail so Notes is always last."""
    lowered = description.lower()
    seen_names: set[str] = set()
    structured: list[dict[str, Any]] = []
    tail:       list[dict[str, Any]] = []

    structured.append({"name": "Title", "type": "title"})
    seen_names.add("title")
    structured.append({"name": "Date",  "type": "date"})
    seen_names.add("date")

    for pattern, name, ftype, extras in _FIELD_RULES:
        if name.lower() in seen_names:
            continue
        if re.search(pattern, lowered):
            field: dict[str, Any] = {"name": name, "type": ftype, **extras}
            seen_names.add(name.lower())
            is_tail = ftype in _TAIL_TYPES and name.lower() not in _RICH_TEXT_STRUCTURAL
            (tail if is_tail else structured).append(field)

    if "notes" not in seen_names:
        tail.append({"name": "Notes", "type": "rich_text"})

    fields = structured + tail
    logger.info(
        "infer_schema_from_description | %d fields (%d structured, %d tail)",
        len(fields), len(structured), len(tail),
    )
    return fields


def format_schema_preview(fields: list[dict[str, Any]]) -> str:
    lines = []
    for f in fields:
        opts = f.get("options", [])
        line = f"  • {f['name']}  ({f['type']})"
        if opts:
            line += f"  [{', '.join(opts[:4])}{'…' if len(opts) > 4 else ''}]"
        lines.append(line)
    return "\n".join(lines)


def generate_followup_questions(
    fields: list[dict[str, Any]],
    description: str,
) -> list[str]:
    questions: list[str] = []
    field_names = [f["name"].lower() for f in fields]
    lowered     = description.lower()

    if not any(w in lowered for w in ("daily", "weekly", "every day", "each day")):
        questions.append(
            "How often do you plan to log — daily, weekly, or after specific events?"
        )

    number_fields = [f for f in fields if f["type"] == "number"]
    if number_fields and not any(w in lowered for w in ("scale", "out of", "1 to", "1-10")):
        names = " and ".join(f["name"] for f in number_fields[:2])
        questions.append(
            f"For {names} — should I use a 1-to-10 scale, or actual values like hours or calories?"
        )

    if not any(w in lowered for w in ("goal", "target", "aim", "benchmark")):
        questions.append("Do you want to track any goals or daily targets alongside your data?")

    if "notes" not in field_names and "reflection" not in field_names:
        questions.append("Want a free-text notes field for thoughts alongside your tracked data?")

    if not any(w in field_names for w in ("mood", "energy", "feeling")):
        questions.append("Should I add a mood or energy level field?")

    return questions[:3]


# ── Anchor question ───────────────────────────────────────────────────────────

_ANCHOR_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdaily|day\b"),
     "How did your day go? Give me a quick summary for {name}."),
    (re.compile(r"\bhealth|fitness|workout|body|exercise\b"),
     "How are you feeling physically? Walk me through today's {name}."),
    (re.compile(r"\bmood|emotion|mental|feeling|wellbeing\b"),
     "How are you feeling right now? Let's fill in your {name}."),
    (re.compile(r"\bproject|work|task|sprint|productivity\b"),
     "How did work go today? Let's log your {name}."),
    (re.compile(r"\bjournal|diary|reflection\b"),
     "What's on your mind? I'll help you fill in your {name}."),
    (re.compile(r"\bfood|meal|diet|nutrition|eating|calori\b"),
     "Tell me what you ate and drank today for your {name}."),
    (re.compile(r"\bsleep|rest|recovery|nap\b"),
     "How did you sleep last night? Let's log your {name}."),
    (re.compile(r"\bfinance|money|expense|budget|spend\b"),
     "Tell me about your finances today for your {name}."),
    (re.compile(r"\blearn|study|book|course|read\b"),
     "What did you learn today? Let's log your {name}."),
    (re.compile(r"\bhabit|routine|tracker\b"),
     "How did your habits go today? Let's log your {name}."),
]

_ANCHOR_DEFAULT = "Let's log an entry for {name}. Give me a quick summary of how things went."


def generate_anchor_question(database_name: str, fields: list[dict[str, Any]]) -> str:
    lowered = database_name.lower()
    for pattern, template in _ANCHOR_RULES:
        if pattern.search(lowered):
            return template.format(name=database_name)
    field_names_joined = " ".join(f["name"].lower() for f in fields)
    for pattern, template in _ANCHOR_RULES:
        if pattern.search(field_names_joined):
            return template.format(name=database_name)
    return _ANCHOR_DEFAULT.format(name=database_name)


# ── Field question generation ─────────────────────────────────────────────────

_SCALE_FIELDS: frozenset[str] = frozenset({
    "mood", "energy", "stress level", "pain level",
    "productivity", "score", "rating",
})

_RICH_TEXT_QUESTIONS: dict[str, str] = {
    "notes":      "Any notes to add? Speak freely or say 'skip'.",
    "goals":      "What were your goals today? Say 'skip' to leave blank.",
    "gratitude":  "What are you grateful for? Say 'skip' if nothing comes to mind.",
    "learnings":  "What did you learn or take away today? Say 'skip' to move on.",
    "reflection": "Any reflections to capture? Say 'skip' to finish.",
    "time":       "What time is this entry for?",
}

# Punchy, conversational field question templates
_FIELD_QUESTIONS: dict[str, str] = {
    "title":        "What should I call this entry?",
    "rich_text":    "Tell me about {name}. Say 'skip' to leave it blank.",
    "number":       "What's your {name}?",
    "number_scale": "How's your {name} today — 1 to 10?",
    "select":       "For {name}: {options}. Which one?",
    "multi_select": "What {name} today? {options_hint}.",
    "date":         "What date? Say today, yesterday, or a specific date.",
    "checkbox":     "Did you {name_lower}?",
    "url":          "URL for {name}?",
    "email":        "Email for {name}?",
    "phone_number": "Phone number for {name}?",
}

_FIELD_QUESTION_DEFAULT = "What's {name}?"


def generate_field_question(field: dict[str, Any]) -> str:
    """
    Punchy, type-adapted question for a single field.

    FIX: "Title" field no longer produces "What should I call this Title entry?"
    — when name is "Title" the generic 'title' template is used directly.
    FIX: rich_text fields get contextual questions.
    FIX: subjective numeric fields get a 1-to-10 scale hint.
    """
    name    = field["name"]
    ftype   = field["type"]
    options = field.get("options", [])

    # Contextual rich_text
    if ftype == "rich_text":
        return _RICH_TEXT_QUESTIONS.get(name.lower(), f"Tell me about {name}. Say 'skip' to leave it blank.")

    # Scale hint for subjective numerics
    if ftype == "number" and name.lower() in _SCALE_FIELDS:
        return f"How's your {name} today — 1 to 10?"

    # FIX: "Title" would produce "for this Title entry" double-word
    if ftype == "title":
        return "What should I call this entry?"

    template  = _FIELD_QUESTIONS.get(ftype, _FIELD_QUESTION_DEFAULT)
    opts_str  = ", ".join(options[:6]) if options else "choose one"
    opts_hint = f"options: {', '.join(options[:6])}" if options else "name them, separated by commas"

    return (
        template
        .replace("{name}",         name)
        .replace("{options}",      opts_str)
        .replace("{options_hint}", opts_hint)
        .replace("{name_lower}",   name.lower())
    )


# ── Clarification questions ───────────────────────────────────────────────────

_CLARIFICATIONS: dict[str, str] = {
    "empty":           "Didn't catch {name}. Give me something, or say 'skip'.",
    "no_number":       "I need a number for {name}. How many exactly?",
    "too_vague":       "Can you be more specific about {name}?",
    "no_date":         "I need a date for {name}. Say today, yesterday, or 15/4.",
    "ambiguous_bool":  "For {name} — just say yes or no.",
    "no_option_match": "For {name}, pick one of: {options}.",
}

_CLARIFICATION_DEFAULT = "Could you clarify {name}? You said: '{answer}'."


def generate_clarification_question(field: dict[str, Any], answer: str, reason: str) -> str:
    name     = field["name"]
    options  = field.get("options", [])
    opts_str = ", ".join(options[:6]) if options else "the listed options"
    template = _CLARIFICATIONS.get(reason, _CLARIFICATION_DEFAULT)
    return (
        template
        .replace("{name}",    name)
        .replace("{answer}",  answer[:80])
        .replace("{options}", opts_str)
    )


# ── Natural language schema description ───────────────────────────────────────

def describe_schema_naturally(fields: list[dict[str, Any]], db_name: str) -> str:
    """
    Return a single spoken sentence describing the schema.

    Example:
    "I'll create 'Daily Log' with 6 fields — a title, today's date,
    your mood on a scale of 1 to 10, energy, activities, and notes."

    Used by main.py to speak the schema before the visual show_schema()
    confirmation, so the user hears what they're about to create.
    """
    if not fields:
        return f"I'll create an empty database called '{db_name}'."

    parts: list[str] = []
    for f in fields:
        name  = f["name"]
        ftype = f["type"]
        opts  = f.get("options", [])

        if ftype == "title":
            parts.append("a title")
        elif ftype == "date":
            parts.append("today's date")
        elif ftype == "number" and name.lower() in _SCALE_FIELDS:
            parts.append(f"your {name.lower()} on a scale of 1 to 10")
        elif ftype == "number":
            parts.append(f"your {name.lower()}")
        elif ftype == "multi_select":
            parts.append(f"your {name.lower()}")
        elif ftype == "select" and opts:
            parts.append(f"{name.lower()} ({' / '.join(opts[:3])})")
        elif ftype == "checkbox":
            parts.append(f"whether you {name.lower()}")
        elif ftype == "rich_text":
            parts.append(name.lower())
        else:
            parts.append(name.lower())

    n = len(parts)
    if n == 1:
        field_str = parts[0]
    elif n == 2:
        field_str = f"{parts[0]} and {parts[1]}"
    else:
        field_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return f"I'll create '{db_name}' with {n} fields — {field_str}."
