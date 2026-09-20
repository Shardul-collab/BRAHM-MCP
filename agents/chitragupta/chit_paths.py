"""
chit_paths.py — where BRAHM lives, resolved so it cannot be wrong.

Every module in Chitragupta that touches a file on disk used to spell this
itself as `os.environ.get("BRAHM_ROOT", "/mnt/d/brahm")` — a WSL2 path from
the original dev box. The 2026-09-10 portability pass kept that literal as the
fallback, so the failure only moved: with BRAHM_ROOT exported (the MCP server's
environment) everything worked, and launching the API the documented way
(`python api_server.py`, no BRAHM_ROOT) died in init_brahm_db() with
`PermissionError: [Errno 13] Permission denied: '/mnt/d'`. On a host where
/mnt/d IS writable it is worse than a crash: get_connection() mkdir -p's the
directory and serves a fresh empty brahm.db while the real one sits untouched.

A default that names another machine's filesystem is never right. This file is
at <root>/agents/chitragupta/chit_paths.py, so parents[2] IS the root, on every
host, with no environment at all. BRAHM_ROOT still wins when it is set, for the
container (BRAHM_ROOT=/app) and for tests that point at a fixture tree.
"""

from __future__ import annotations

import os
from pathlib import Path

#: <root>/agents/chitragupta/chit_paths.py -> parents[2] == <root>
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

BRAHM_ROOT = Path(os.environ.get("BRAHM_ROOT") or _DEFAULT_ROOT)

#: The central BRAHM database (Projects, results, documents, papers).
BRAHM_DB_PATH = Path(
    os.environ.get("BRAHM_DB_PATH") or (BRAHM_ROOT / "data/brahm.db")
)

#: SHANI's workflow database. Read-only from Chitragupta.
SHANI_DB_PATH = BRAHM_ROOT / "agents/shani/database/research_workflow.db"

#: Chitragupta's own legacy store (brahm_activity, ganesh_documents, ...).
KNOWLEDGE_DB_PATH = Path(__file__).resolve().parent / "database/brahm_knowledge.db"
