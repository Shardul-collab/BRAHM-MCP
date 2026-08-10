# main.py

"""
Chitragupta — Voice-First Entry Point  (v0.7 — Jarvis UX layer)

Changes from v0.6
-----------------
- Time-aware greeting: "Good evening. Chitragupta ready." at startup.
- DB name validation: rejects garbage like "okay man?" with a retry prompt
  before building a schema for a nonsense name.
- Natural schema read-back: before the confirmation loop, Jarvis speaks a
  single sentence describing what will be created — no table reading.
- Spoken save progress: "Saving to Notion now." before the API call,
  "Done." after, so there's no eerie silence during the network call.
- Closing line: "Anything else?" after a successful entry save.
- Immediate first-entry offer: after DB creation, asks if user wants to
  log their first entry right now instead of dropping to the main menu.
- import datetime added for time-aware greeting.
"""

import sys
import logging
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

for _noisy in ("transformers", "torch", "whisper", "httpx", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from config.settings import APP_NAME, APP_VERSION, NOTION_PAGE_ID
import core.conversation_ui as ui

from notion.schema_manager import (
    create_schema, load_schema, list_schemas, update_notion_id,
    schema_to_notion_properties,
    SchemaValidationError, SchemaMissingError, SchemaAlreadyExistsError,
)
from notion.notion_client import create_database, create_page, NotionAPIError, check_schema_drift
from notion.relation_manager import create_relation, RelationError
from notion.write_journal import pending_for_database
from core.json_builder import build_json, json_to_notion_properties, ValidationError
from core.conversation_engine import run_conversation
from voice.whisper_handler import manual_input
from nlp.intent_parser import parse_command
from nlp.schema_inferencer import (
    infer_schema_from_description,
    generate_followup_questions,
    describe_schema_naturally,
)
from analysis.pattern_analyzer import analyze_database, print_report
from scheduler.reminder import run_scheduler, run_now, run_in_background, SchedulerConfigError

logger = logging.getLogger("chitragupta.main")


# ── DB name validation ────────────────────────────────────────────────────────

_DB_NOISE: frozenset[str] = frozenset({
    "okay", "ok", "yes", "yeah", "no", "what", "huh",
    "um", "uh", "hey", "hi", "hello", "done", "next",
})


def _is_valid_db_name(name: str) -> tuple[bool, str]:
    """Returns (valid, reason). Reason is empty when valid."""
    stripped = name.strip().rstrip("?!.,")
    if len(stripped) < 2:
        return False, "That name is too short."
    if stripped.lower() in _DB_NOISE:
        return False, f"'{stripped}' doesn't sound like a database name."
    alpha_count = sum(c.isalpha() for c in stripped)
    if alpha_count < 2:
        return False, "That doesn't look like a valid name — try something more descriptive."
    return True, ""


# ── Shared picker ─────────────────────────────────────────────────────────────

def _pick_database(action: str = "use") -> str | None:
    schemas = list_schemas()
    if not schemas:
        ui.speak("No databases yet. Create one first.")
        return None
    return ui.pick_from_list(schemas, prompt=f"Which database do you want to {action}?")


# ── Feature: Create Database ──────────────────────────────────────────────────

def setup_database() -> None:
    ui.section("Create Database")

    # ── 1. Name — with validation + retry ────────────────────────────────────
    for _attempt in range(3):
        db_name = ui.ask_text("What should we call this database?")
        if not db_name:
            ui.error("A database name is required.")
            return
        valid, reason = _is_valid_db_name(db_name)
        if valid:
            db_name = db_name.strip().rstrip("?!.,")
            break
        ui.speak(f"{reason} What should we call this database?")
    else:
        ui.error("Couldn't get a valid database name. Cancelled.")
        return

    ui.acknowledge(f"'{db_name}'.")

    # ── 2. Parent page ID ─────────────────────────────────────────────────────
    env_page_id = NOTION_PAGE_ID.strip()
    if env_page_id:
        ui.info(f"Using NOTION_PAGE_ID from .env: {env_page_id}")
        ui.speak("Parent page ID loaded from your .env. Press Enter to use it, or type a different ID.")
        typed    = manual_input(f"Notion parent page ID [{env_page_id}]: ").strip()
        parent_id = typed if typed else env_page_id
    else:
        ui.speak("NOTION_PAGE_ID is not set in your .env. Please type your Notion parent page ID.")
        parent_id = manual_input("Notion parent page ID: ").strip()

    if not parent_id:
        ui.error("Parent page ID is required. Set NOTION_PAGE_ID in your .env.")
        return

    # ── 3. Conversational schema building ─────────────────────────────────────
    fields = _build_schema_conversationally(db_name)
    if not fields:
        ui.error("Setup cancelled.")
        return

    # ── 4. Natural language read-back + visual confirmation ───────────────────
    while True:
        ui.show_schema(fields, title=f"Schema for '{db_name}'")
        # Speak a single natural sentence instead of reading the table
        ui.speak(describe_schema_naturally(fields, db_name))
        ui.speak("Does this look right? Say 'yes' to create, 'edit' to change fields, or 'cancel'.")

        cmd = ui.ask_command("yes / edit / cancel", timeout=6)

        if cmd == "yes":
            break
        if cmd == "edit":
            fields = _edit_fields_conversationally(fields)
            if fields is None:
                ui.error("Cancelled.")
                return
            continue
        if cmd == "cancel":
            ui.error("Cancelled.")
            return
        ui.speak("Say 'yes', 'edit', or 'cancel'.")

    # ── 5. Save schema locally ────────────────────────────────────────────────
    try:
        create_schema(db_name, fields)
        ui.acknowledge("Schema saved.")
    except SchemaValidationError as exc:
        ui.error(str(exc))
        return
    except SchemaAlreadyExistsError:
        try:
            existing = load_schema(db_name)
            if existing.get("notion_database_id"):
                ui.error(
                    f"'{db_name}' already exists with Notion ID "
                    f"'{existing['notion_database_id']}'. "
                    "Delete the local schema file to start over."
                )
                return
            ui.acknowledge(f"Recovering orphaned schema for '{db_name}'.")
            create_schema(db_name, fields, overwrite=True)
        except Exception:
            if not ui.confirm("That database already exists. Overwrite?", default=False):
                return
            create_schema(db_name, fields, overwrite=True)

    # ── 6. Create in Notion ───────────────────────────────────────────────────
    ui.speak(f"Creating '{db_name}' in Notion now.")
    try:
        schema       = load_schema(db_name)
        notion_props = schema_to_notion_properties(schema)
        response     = create_database(parent_id, db_name, notion_props)
        notion_id    = response["id"]
    except (NotionAPIError, ValidationError) as exc:
        ui.error(f"Error creating database in Notion: {exc}")
        return

    try:
        update_notion_id(db_name, notion_id)
    except Exception as exc:
        ui.error(
            f"Database created in Notion (ID: {notion_id}) but the local schema "
            f"could not be updated: {exc}\n"
            f"Fix manually: set notion_database_id = '{notion_id}' in your schema file."
        )
        return

    ui.speak(f"'{db_name}' is live.")
    ui.info(f"Notion ID: {notion_id}")
    logger.info("setup_database complete | db='%s' id=%s", db_name, notion_id)

    # ── 7. Immediate first-entry offer ────────────────────────────────────────
    if ui.confirm("Want to log your first entry right now?", default=False):
        _log_entry_for(db_name)


def _build_schema_conversationally(db_name: str) -> list[dict] | None:
    desc = ui.ask(f"What do you mainly want to track in '{db_name}'?", timeout=10)
    if not desc:
        ui.error("Got nothing — aborting.")
        return None
    ui.acknowledge()

    fields       = infer_schema_from_description(desc)
    followups    = generate_followup_questions(fields, desc)
    full_context = desc

    for q in followups[:3]:
        ui.speak(q)
        answer = ui.ask_text("Your answer", timeout=9)
        if not answer:
            ui.acknowledge("Skipping that.")
            continue
        ui.acknowledge()
        full_context = f"{full_context}. {answer}"
        fields = infer_schema_from_description(full_context)

    return fields


def _edit_fields_conversationally(fields: list[dict]) -> list[dict] | None:
    current = list(fields)
    while True:
        ui.show_schema(current, title="Current fields")
        ui.speak("Say 'add', 'remove', 'done', or 'cancel'.")
        cmd_raw = ui.ask("add / remove / done / cancel", timeout=7)
        cmd     = parse_command(cmd_raw)

        if cmd == "yes" or any(w in cmd_raw.lower() for w in ("done", "finish", "looks good")):
            return current
        if cmd == "cancel":
            return None
        if "add" in cmd_raw.lower():
            name = ui.ask_text("Name of the new field?", timeout=8)
            if not name:
                continue
            ftype = ui.ask_text(
                f"Type for '{name}'? (number, rich_text, select, multi_select, checkbox, date)",
                timeout=8,
            ).strip().lower()
            if not ftype:
                continue
            new_field: dict = {"name": name, "type": ftype}
            if ftype in ("select", "multi_select"):
                raw_opts = ui.ask_text("Options? Separate by commas, or skip.", timeout=8)
                if raw_opts and raw_opts.lower() not in ("skip", "no", "none"):
                    new_field["options"] = [o.strip() for o in raw_opts.split(",") if o.strip()]
            current.append(new_field)
            ui.acknowledge(f"'{name}' ({ftype}) added.")
        elif any(w in cmd_raw.lower() for w in ("remove", "delete")):
            raw_target = ui.ask("Which field? Say its name or number.", timeout=6)
            removed    = None
            try:
                idx = int(parse_command(raw_target)) - 1
                if 0 <= idx < len(current):
                    removed = current.pop(idx)
            except (ValueError, TypeError):
                match = next((f for f in current if f["name"].lower() in raw_target.lower()), None)
                if match:
                    current.remove(match)
                    removed = match
            if removed:
                ui.acknowledge(f"'{removed['name']}' removed.")
            else:
                ui.speak("Couldn't find that field. Try its number.")
        else:
            ui.speak("Say 'add', 'remove', 'done', or 'cancel'.")


# ── Feature: Log Entry ────────────────────────────────────────────────────────

def _log_entry_for(db_name: str) -> None:
    """Core logging logic for a named database."""
    try:
        schema = load_schema(db_name)
    except SchemaMissingError as exc:
        ui.error(str(exc))
        return

    notion_db_id = schema.get("notion_database_id", "").strip()
    if not notion_db_id:
        ui.error(f"No Notion ID for '{db_name}'. Run 'Create Database' first.")
        return

    drift = check_schema_drift(notion_db_id, schema.get("fields", []))
    if drift:
        drift_lines = "\n".join(f"  • {d}" for d in drift)
        ui.error(f"Schema drift detected for '{db_name}':\n{drift_lines}")
        if not ui.confirm("Log entry anyway?", default=False):
            return

    pending = pending_for_database(notion_db_id)
    if pending:
        ui.error(
            f"Warning: {len(pending)} unconfirmed write(s) for '{db_name}'. "
            "A previous entry may have been saved. Check Notion before continuing."
        )
        if not ui.confirm("Save anyway?", default=False):
            ui.info("Entry discarded.")
            return

    approved = run_conversation(db_name, schema)
    if approved is None:
        ui.info("Entry discarded.")
        return

    try:
        validated      = build_json(db_name, approved)
        notion_payload = json_to_notion_properties(db_name, validated)
    except ValidationError as exc:
        ui.error(f"Entry contains invalid data and cannot be saved:\n{exc}")
        return

    # Spoken progress during Notion API call
    ui.speak("Saving to Notion now.")
    try:
        response = create_page(notion_db_id, notion_payload)
        page_id  = response.get("id", "unknown")
        ui.speak("Done. Entry saved.")
        ui.info(f"Page ID: {page_id}")
        logger.info("log_entry complete | db='%s' page=%s", db_name, page_id)
        # Closing line
        ui.speak("Anything else?")
    except ValidationError as exc:
        ui.error(f"Payload validation failed — entry not saved:\n{exc}")
    except NotionAPIError as exc:
        ui.error(f"Notion API error: {exc}")
        ui.speak("Couldn't reach Notion. Your session data is in the write journal.")
    except Exception as exc:
        ui.error(f"Unexpected error: {exc}")
        logger.exception("log_entry: save failed")


def log_entry() -> None:
    ui.section("Log Entry")
    db_name = _pick_database("log to")
    if not db_name:
        return
    _log_entry_for(db_name)


# ── Feature: Analyze ─────────────────────────────────────────────────────────

def run_analysis() -> None:
    ui.section("Analyze Data")
    db_name = _pick_database("analyze")
    if not db_name:
        return
    ui.speak(f"Fetching and analyzing '{db_name}'.")
    try:
        insights = analyze_database(db_name)
        print_report(insights)
    except SchemaMissingError as exc:
        ui.error(str(exc))
    except Exception as exc:
        ui.error(f"Analysis error: {exc}")
        logger.exception("run_analysis failed")


# ── Feature: Link Databases ───────────────────────────────────────────────────

def link_databases() -> None:
    ui.section("Link Two Databases")
    schemas = list_schemas()
    if len(schemas) < 2:
        ui.speak("You need at least two databases to create a relation.")
        return
    ui.speak("First, pick the source database.")
    db_a = _pick_database("use as source")
    if not db_a:
        return
    ui.speak("Now pick the target database.")
    db_b = _pick_database("use as target")
    if not db_b:
        return
    if db_a == db_b:
        ui.error("Source and target must be different databases.")
        return
    relation_name = ui.ask_text(f"What should the relation field be called in '{db_a}'?")
    if not relation_name:
        ui.error("Relation name cannot be empty.")
        return
    bidirectional = ui.confirm("Add a reverse relation in the target too?", default=False)
    reverse_name  = ""
    if bidirectional:
        reverse_name = ui.ask_text(f"Name of the reverse relation in '{db_b}'?")
        if not reverse_name:
            ui.error("Reverse relation name cannot be empty.")
            return
    try:
        create_relation(
            database_a=db_a, database_b=db_b,
            relation_name=relation_name,
            bidirectional=bidirectional,
            reverse_relation_name=reverse_name,
        )
        ui.speak(f"Relation '{relation_name}' created.")
    except (RelationError, SchemaMissingError) as exc:
        ui.error(str(exc))
    except Exception as exc:
        ui.error(f"Unexpected error: {exc}")
        logger.exception("link_databases failed")


# ── Feature: Scheduler ────────────────────────────────────────────────────────

def start_scheduler_mode() -> None:
    ui.section("Scheduler")
    db_name = _pick_database("schedule daily logging for")
    if not db_name:
        return

    def scheduled_task() -> None:
        logger.info("Scheduled task triggered | db='%s'", db_name)
        _log_entry_for(db_name)

    cmd = ui.menu(
        "How do you want to run it?",
        {"1": "Foreground — block this terminal", "2": "Background — keep CLI active",
         "3": "Run once now", "0": "Cancel"},
    )
    if cmd == "1":
        try:
            run_scheduler(scheduled_task)
        except SchedulerConfigError as exc:
            ui.error(str(exc))
    elif cmd == "2":
        try:
            run_in_background(scheduled_task)
            ui.speak("Scheduler running in the background.")
        except SchedulerConfigError as exc:
            ui.error(str(exc))
    elif cmd == "3":
        run_now(scheduled_task)
    elif cmd in ("cancel", "0"):
        return
    else:
        ui.speak("Didn't catch that.")


# ── Main loop ─────────────────────────────────────────────────────────────────

_MENU_OPTIONS = {
    "1": "Create database",
    "2": "Log entry",
    "3": "Analyze data",
    "4": "Link databases",
    "5": "Start scheduler",
    "6": "Exit",
}
_MENU_ACTIONS = {
    "1": setup_database,
    "2": log_entry,
    "3": run_analysis,
    "4": link_databases,
    "5": start_scheduler_mode,
}
_ALIASES: dict[str, str] = {
    "create": "1", "log": "2", "analyse": "3", "analyze": "3",
    "link": "4", "schedule": "5", "exit": "6", "quit": "6", "bye": "6",
}


def main() -> None:
    ui.header(APP_NAME, APP_VERSION)

    # Time-aware greeting
    hour = datetime.now().hour
    if hour < 12:
        greeting = "Good morning."
    elif hour < 17:
        greeting = "Good afternoon."
    elif hour < 21:
        greeting = "Good evening."
    else:
        greeting = "Working late? Good night."

    ui.speak(f"{greeting} Chitragupta ready. Say a number or keyword.")
    print("  💡  Tip: you'll hear a beep before each listening window opens.\n")

    while True:
        ui.speak("Main menu.")
        cmd = ui.menu("What would you like to do?", _MENU_OPTIONS)
        cmd = _ALIASES.get(cmd, cmd)

        if cmd == "6":
            ui.speak("Goodbye.")
            logger.info("User exited.")
            sys.exit(0)

        if cmd not in _MENU_ACTIONS:
            ui.speak("Didn't catch that. Say a number from 1 to 6, or a keyword like 'log'.")
            continue

        try:
            _MENU_ACTIONS[cmd]()
        except KeyboardInterrupt:
            ui.speak("Returning to the main menu.")
            logger.info("KeyboardInterrupt — back to menu.")


if __name__ == "__main__":
    main()
