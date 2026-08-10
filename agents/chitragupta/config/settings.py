# config/settings.py

"""
Central configuration for Chitragupta.
All paths, constants, and environment variables are loaded here.
Other modules import from this file — never from os.environ directly.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
SCHEMA_DIR = DATA_DIR / "schemas"
DB_PATH    = DATA_DIR / "chitragupta.db"
LOG_PATH   = BASE_DIR / "logs" / "chitragupta.log"

SESSION_MEMORY_PATH = DATA_DIR / "session_memory.json"

SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv(BASE_DIR / ".env")

# AUDIT FIX C-001 — Strip NOTION_TOKEN before use.
#
# os.getenv() returns the raw string from .env, including any trailing
# whitespace, carriage returns, or newlines that an editor may have added.
# A token like "secret_abc \n" produces an invalid Authorization header and
# Notion returns 401, even though the token itself is correct.
#
# .strip() is applied here at the single point where the value is read so
# every downstream consumer (notion_client._headers, startup guard below)
# sees the clean value without needing to sanitise it themselves.
NOTION_TOKEN:   str = os.getenv("NOTION_TOKEN",   "").strip()
NOTION_VERSION: str = os.getenv("NOTION_VERSION", "2022-06-28").strip()
NOTION_PAGE_ID: str = os.getenv("NOTION_PAGE_ID", "").strip()

if not NOTION_TOKEN:
    raise EnvironmentError(
        "NOTION_TOKEN is not set. "
        "Copy .env.example → .env and add your Notion integration token."
    )

# AUDIT FIX C-001 — Token format sanity check.
#
# Notion internal integration tokens always start with "secret_".
# Notion OAuth access tokens always start with "ntn_".
# Anything else is almost certainly a misconfigured value (e.g. a database ID,
# a page URL, or a copy-paste error).  We log a warning here at startup so the
# operator sees it in the logs before the first API call fails with 401.
# This is a warning — not a hard failure — because Notion may introduce new
# token prefixes in future API versions.
_logger_pre = logging.getLogger("chitragupta.config")
if not (NOTION_TOKEN.startswith("secret_") or NOTION_TOKEN.startswith("ntn_")):
    _logger_pre.warning(
        "NOTION_TOKEN does not start with 'secret_' or 'ntn_'. "
        "Verify your integration token format in .env. "
        "An incorrectly formatted token will cause 401 errors on all Notion requests."
    )

# ── Whisper ───────────────────────────────────────────────────────────────────

WHISPER_MODEL:        str = os.getenv("WHISPER_MODEL", "base")
AUDIO_SAMPLE_RATE:    int = 16_000
AUDIO_RECORD_SECONDS: int = 10
AUDIO_CHANNELS:       int = 1

# ── NLP / DistilBERT ──────────────────────────────────────────────────────────

NLP_MODEL:      str = os.getenv("NLP_MODEL", "distilbert-base-uncased-distilled-squad")
NLP_MAX_LENGTH: int = 512

# ── Scheduler ─────────────────────────────────────────────────────────────────

SCHEDULE_TIME: str = os.getenv("SCHEDULE_TIME", "09:00")

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("chitragupta.config")
logger.debug("Settings loaded. BASE_DIR=%s", BASE_DIR)

# ── Notion API ────────────────────────────────────────────────────────────────

NOTION_BASE_URL: str = "https://api.notion.com/v1"

SUPPORTED_PROPERTY_TYPES: list[str] = [
    "title", "rich_text", "number", "select", "multi_select",
    "date", "checkbox", "url", "email", "phone_number",
    "relation", "people", "files", "formula",
]

# ── Misc ──────────────────────────────────────────────────────────────────────

APP_NAME:    str = "Chitragupta"
APP_VERSION: str = "0.1.0"
