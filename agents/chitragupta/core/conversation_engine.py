# core/conversation_engine.py

"""
Conversation Engine  (v0.6 — session memory + spoken progress)

Changes from v0.5
-----------------
- run_conversation(): saves confirmed session to session_memory after a
  successful confirm_data() call so the FSM can offer "same as last time?"
  on the next session.
- run_conversation(): spoken progress line ("Saving…") moved here from
  main.py since this module owns the confirmation handoff.  main.py still
  handles the actual Notion API call and its own spoken feedback.
- already_confirmed=True passed to confirm_data() (v0.5 fix preserved).
- ui.capture() public alias used (v0.5 fix preserved).
"""

import logging
from typing import Any

from core.dialogue_fsm import DialogueFSM, State
from core.json_builder import build_json
from core.confirmation import confirm_data
import core.conversation_ui as ui

logger = logging.getLogger("chitragupta.conversation_engine")


def _ask(prompt: str, timeout: int = 10) -> str:
    if not prompt:
        return ui.capture(timeout)
    return ui.ask(prompt, timeout=timeout)


def run_conversation(database_name: str, schema: dict) -> dict[str, Any] | None:
    """
    Run a full conversational logging session.

    Returns confirmed, schema-aligned JSON dict or None if cancelled.
    """
    fields: list[dict[str, Any]] = schema.get("fields", [])
    if not fields:
        logger.warning("run_conversation: no fields in schema for '%s'", database_name)
        ui.error(f"No fields found in the '{database_name}' schema.")
        return None

    ui.section(f"Logging — {database_name}")

    fsm    = DialogueFSM(database_name, schema)
    prompt = fsm.start()

    _TERMINAL = {State.STORE, State.ERROR}

    while fsm.state not in _TERMINAL:
        user_input = _ask(prompt, timeout=12)
        prompt     = fsm.step(user_input)
        if prompt and "?" not in prompt and not prompt.lower().startswith("got it"):
            ui.info(prompt)

    if fsm.state == State.ERROR:
        ui.speak(prompt or "Session ended.")
        return None

    if prompt:
        ui.speak(prompt)

    raw_collected: dict[str, Any] = fsm.export_json()
    json_data = build_json(database_name, raw_collected)

    approved = confirm_data(json_data, schema, already_confirmed=True)

    # Save session for "same as last time?" on next run
    if approved:
        try:
            from core.session_memory import save_session
            save_session(database_name, approved)
        except Exception as exc:
            logger.debug("session_memory save failed (non-fatal): %s", exc)

    return approved
