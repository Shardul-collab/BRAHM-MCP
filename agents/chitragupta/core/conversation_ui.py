# core/conversation_ui.py

"""
Conversational UI Layer — v2.1 (Jarvis polish)

Single choke-point for ALL terminal output and ALL user input.

Changes from v2.0
-----------------
- acknowledge(): rotates through a pool of 10 natural responses instead of
  always saying "Got it."  Jarvis never repeats itself every turn.
- All v2.0 fixes preserved (TTS deadlock fix, token-scored list match,
  duplicate print removal, public capture(), error spoken, etc.)
"""

from __future__ import annotations

import logging
import random
import re
import shutil
import threading
from typing import Any

logger = logging.getLogger("chitragupta.ui")

# ── ANSI palette ──────────────────────────────────────────────────────────────

_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)

# ── Terminal width ────────────────────────────────────────────────────────────

def _term_width(fallback: int = 60) -> int:
    try:
        return min(shutil.get_terminal_size(fallback=(fallback, 24)).columns, 80)
    except Exception:
        return fallback

# ── TTS singleton ─────────────────────────────────────────────────────────────

_tts_engine = None
_tts_lock   = threading.Lock()
_tts_failed = False

_TTS_MAX_CHARS: int = 120   # truncate before speaking, full text still printed


def _get_tts():
    global _tts_engine, _tts_failed
    if _tts_failed:
        return None
    if _tts_engine is not None:
        return _tts_engine
    with _tts_lock:
        if _tts_engine is not None:
            return _tts_engine
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 0.9)
            _tts_engine = engine
            logger.debug("TTS engine initialised.")
        except Exception as exc:
            logger.warning("TTS unavailable: %s — print-only mode.", exc)
            _tts_failed = True
    return _tts_engine


def _say(text: str) -> None:
    """
    Synchronous TTS with character cap.
    pyttsx3 / SAPI5 is single-threaded COM — no concurrent calls.
    Text > _TTS_MAX_CHARS is truncated at the last word boundary.
    Full text is always printed unchanged.
    """
    engine = _get_tts()
    if engine is None:
        return
    clean = _strip_ansi(text).strip()
    if not clean:
        return
    if len(clean) > _TTS_MAX_CHARS:
        spoken = clean[:_TTS_MAX_CHARS].rsplit(" ", 1)[0]
    else:
        spoken = clean
    try:
        engine.say(spoken)
        engine.runAndWait()
    except Exception as exc:
        logger.debug("TTS say() failed: %s", exc)

# ── Core output ───────────────────────────────────────────────────────────────

def speak(text: str) -> None:
    """Print full text; TTS only the first line (preamble before \\n)."""
    print(f"\n  {_CYAN}{_BOLD}🤖  {_RESET}{_CYAN}{text}{_RESET}")
    first_line = text.split("\n")[0].strip()
    if first_line:
        _say(first_line)


# Varied acknowledgements — Jarvis never says "Got it." every single time.
_ACK_POOL: list[str] = [
    "Got it.", "Noted.", "Perfect.", "Understood.", "Done.",
    "Right.", "Copy that.", "Sure.", "Good.", "Okay.",
]

def acknowledge(text: str | None = None) -> None:
    """Short spoken acknowledgement — rotates through natural responses."""
    msg = text if text is not None else random.choice(_ACK_POOL)
    print(f"\n  {_DIM}✓  {msg}{_RESET}")
    _say(msg)


def error(text: str) -> None:
    """Print error in full; speak a short audio cue."""
    print(f"\n  {_RED}✗  {text}{_RESET}")
    _say("Something went wrong.")


def info(text: str) -> None:
    """Visual-only neutral info line."""
    print(f"  {_DIM}{text}{_RESET}")

# ── Input helpers ─────────────────────────────────────────────────────────────

def _capture(timeout: int, _retry_on_silence: bool = True) -> str:
    """
    Internal: voice → text with keyboard fallback after two silent attempts.
    No redundant 'Listening…' print — whisper_handler owns that line.
    No '→ transcript' echo — capture_command's '✓ Heard:' is the one source.
    """
    try:
        from voice.whisper_handler import capture_command, manual_input
        response = capture_command(timeout=timeout)
        if response:
            return response.strip()
        if _retry_on_silence:
            response = capture_command(timeout=timeout)
            if response:
                return response.strip()
        print(f"  {_YELLOW}No voice detected — type your answer:{_RESET}")
        return manual_input("").strip()
    except Exception:
        try:
            return input(f"  {_BOLD}>{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""


def capture(timeout: int = 8) -> str:
    """Public alias for _capture() — used by conversation_engine."""
    return _capture(timeout)


def ask(prompt: str, timeout: int = 8) -> str:
    speak(prompt)
    return _capture(timeout)


def ask_command(prompt: str, timeout: int = 7) -> str:
    from nlp.intent_parser import parse_command
    raw = ask(prompt, timeout)
    if not raw:
        logger.debug("ask_command: silence.")
        return "unknown"
    return parse_command(raw)


def ask_text(prompt: str, timeout: int = 8) -> str:
    return ask(prompt, timeout)


def confirm(prompt: str, default: bool = True) -> bool:
    from nlp.intent_parser import parse_command
    for attempt in range(2):
        raw = ask(prompt + " (yes / no)", timeout=6)
        cmd = parse_command(raw)
        if cmd == "yes":
            return True
        if cmd in ("no", "cancel"):
            return False
        if attempt == 0:
            speak("Sorry, I didn't catch that. Please say yes or no.")
        else:
            direction = "yes" if default else "no"
            speak(f"Still unclear — treating that as '{direction}'.")
    return default

# ── Menu helpers ──────────────────────────────────────────────────────────────

def menu(title: str, options: dict[str, str]) -> str:
    print(f"\n{_BOLD}  {title}{_RESET}")
    print(f"  {'─' * 46}")
    for key, label in options.items():
        print(f"  {_CYAN}{key}{_RESET}  {label}")
    print()
    _say("Listening for your choice.")
    from nlp.intent_parser import parse_command
    return parse_command(_capture(timeout=7))


def pick_from_list(
    items: list[str],
    prompt: str = "Which one?",
    allow_cancel: bool = True,
) -> str | None:
    """
    Numbered list picker with layered resolution:
    1. Exact digit → index
    2. parse_command digit
    3. Normalised exact name
    4. Token Jaccard (floor 0.4, margin 0.2)
    5. Ambiguity → clarification + digit-only retry
    """
    if not items:
        return None

    print(f"\n  {_BOLD}{prompt}{_RESET}")
    print(f"  {'─' * 36}")
    for i, name in enumerate(items, 1):
        print(f"  {_CYAN}{i}{_RESET}.  {name}")
    if allow_cancel:
        print(f"  {_CYAN}0{_RESET}.  Cancel")
    print()

    if len(items) <= 4:
        numbered = ", ".join(f"{i}: {n}" for i, n in enumerate(items, 1))
        _say(f"{prompt} Options: {numbered}.")
    else:
        _say(prompt)

    raw = _capture(timeout=6)
    if not raw:
        return None

    raw_stripped = raw.strip()
    raw_lower    = raw_stripped.lower().rstrip(".,!?")

    from nlp.intent_parser import parse_command

    if allow_cancel and (raw_stripped == "0" or parse_command(raw) == "cancel"):
        return None

    if raw_stripped.isdigit():
        idx = int(raw_stripped) - 1
        if 0 <= idx < len(items):
            return items[idx]

    cmd = parse_command(raw)
    if cmd.isdigit():
        idx = int(cmd) - 1
        if 0 <= idx < len(items):
            return items[idx]

    for name in items:
        if name.lower() == raw_lower:
            return name

    def _jaccard(a: str, b: str) -> float:
        ta = set(re.findall(r"\w+", a.lower()))
        tb = set(re.findall(r"\w+", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    scores = sorted(
        [(name, _jaccard(name, raw_lower)) for name in items],
        key=lambda x: x[1], reverse=True,
    )
    best_name, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else 0.0

    if best_score >= 0.4 and (best_score - second_score) >= 0.2:
        return best_name

    if best_score > 0.0:
        candidates = [n for n, s in scores if s >= 0.2][:2]
        if len(candidates) >= 2:
            speak(f"Did you mean {candidates[0]} or {candidates[1]}? Say the number.")
        else:
            speak("I'm not sure which one. Please say the number.")
    else:
        speak("I didn't catch that. Please say the number.")

    raw2 = _capture(timeout=5, _retry_on_silence=False)
    if raw2.strip().isdigit():
        idx = int(raw2.strip()) - 1
        if 0 <= idx < len(items):
            return items[idx]

    return None

# ── Schema / data display ─────────────────────────────────────────────────────

def show_schema(fields: list[dict[str, Any]], title: str = "Schema") -> None:
    """Visual only — never calls speak()."""
    width = _term_width()
    bar   = "─" * max(0, width - len(title) - 4)
    print(f"\n  {_BOLD}── {title} {bar}{_RESET}")
    for f in fields:
        name  = f["name"]
        ftype = f["type"]
        opts  = f.get("options", [])
        opt_str = (
            f"  [{_DIM}{', '.join(opts[:4])}{'…' if len(opts) > 4 else ''}{_RESET}]"
            if opts else ""
        )
        print(f"  {_CYAN}•{_RESET} {name}  {_DIM}({ftype}){_RESET}{opt_str}")
    print()


def show_entry(
    data: dict[str, Any],
    title: str = "Entry",
    confidences: dict[str, float] | None = None,
) -> None:
    """Visual only. Low-confidence fields marked [?] when confidences dict provided."""
    width = _term_width()
    bar   = "─" * max(0, width - len(title) - 4)
    print(f"\n  {_BOLD}── {title} {bar}{_RESET}")
    if not data:
        print(f"  {_YELLOW}(empty){_RESET}")
    else:
        max_k = max(len(k) for k in data)
        for k, v in data.items():
            val      = _fmt(v)
            conf_tag = ""
            if confidences and confidences.get(k, 1.0) < 0.7:
                conf_tag = f"  {_YELLOW}[?]{_RESET}"
            print(f"  {_CYAN}{k.ljust(max_k)}{_RESET}  {val}{conf_tag}")
    print()


def _fmt(value: Any) -> str:
    if value is None:
        return f"{_YELLOW}—{_RESET}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else f"{_YELLOW}—{_RESET}"
    if isinstance(value, bool):
        return f"{_GREEN}yes{_RESET}" if value else f"{_RED}no{_RESET}"
    return str(value)

# ── Header / section ──────────────────────────────────────────────────────────

def header(title: str, version: str = "") -> None:
    width = _term_width(fallback=54)
    bar   = "═" * width
    tag   = f"  v{version}" if version else ""
    print(f"\n{_BOLD}{bar}{_RESET}")
    print(f"{_BOLD}  {title}{tag}{_RESET}")
    print(f"{_BOLD}{bar}{_RESET}\n")
    _say(f"{title}, version {version}." if version else title)


def section(title: str) -> None:
    width = _term_width(fallback=52)
    pad   = max(0, width - len(title) - 4)
    print(f"\n{_BOLD}── {title} {'─' * pad}{_RESET}")
    _say(f"Starting: {title}.")
