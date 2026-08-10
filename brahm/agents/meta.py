"""
brahm/agents/meta.py
=====================
Group META — BRAHM self-description and health tools.
brahm_overview is generated from live registry + health checks — never stale.
"""

import asyncio
import logging
from datetime import datetime, timezone

from brahm.brahm_registry import brahm_tool, registry
from brahm.shared.constants import AGENTS, DB_PATH, SHANI_STAGE_SEQUENCE, GANESH_STAGE_SEQUENCE
from brahm.shared.helpers import _ok, _err
from brahm.shared.http import (
    _check_shani, SHANI_START_HINT,
    _check_ganesh, GANESH_START_HINT,
)

log = logging.getLogger("mcp.brahm.meta")


async def _health_shani() -> dict:
    ok = await _check_shani()
    return {"agent": "SHANI", "status": "online" if ok else "offline",
            "type": "http_api", "hint": "" if ok else SHANI_START_HINT}

async def _health_ganesh() -> dict:
    ok = await _check_ganesh()
    return {"agent": "GANESH", "status": "online" if ok else "offline",
            "type": "http_api", "hint": "" if ok else GANESH_START_HINT}

def _health_db() -> dict:
    try:
        import sqlite3
        con = sqlite3.connect(DB_PATH)
        con.execute("SELECT 1")
        con.close()
        return {"component": "SQLite DB", "status": "ok", "path": DB_PATH}
    except Exception as exc:
        return {"component": "SQLite DB", "status": "error", "detail": str(exc)}

def _health_vidur() -> dict:
    try:
        import extractor, auto_detector  # noqa: F401
        return {"agent": "VIDUR", "status": "ok", "type": "local_import"}
    except ImportError as exc:
        return {"agent": "VIDUR", "status": "import_error", "detail": str(exc)}

def _health_vishwakarma() -> dict:
    try:
        import vishwakarma  # noqa: F401
        return {"agent": "Vishwakarma", "status": "ok", "type": "local_import"}
    except ImportError as exc:
        return {"agent": "Vishwakarma", "status": "import_error", "detail": str(exc)}

def _health_chitragupta() -> dict:
    try:
        from notion.notion_client import create_page  # noqa: F401
        return {"agent": "Chitragupta", "status": "ok", "type": "local_import"}
    except ImportError as exc:
        return {"agent": "Chitragupta", "status": "import_error", "detail": str(exc)}


@brahm_tool(
    name        = "brahm_overview",
    group       = "meta",
    description = (
        "Return a complete live map of the entire BRAHM system: "
        "all agents and their status, all registered tools by group, "
        "pipeline stage sequences, and infrastructure health. "
        "Always call this first to orient yourself before any task."
    ),
    input_schema = {"type": "object", "properties": {}, "required": []},
)
async def brahm_overview(args: dict) -> dict:
    shani_health, ganesh_health = await asyncio.gather(
        _health_shani(), _health_ganesh()
    )
    infra_health = await asyncio.to_thread(lambda: [
        _health_db(), _health_vidur(),
        _health_vishwakarma(), _health_chitragupta(),
    ])
    tool_groups = registry.summary()
    total_tools = len(registry)
    agent_summary = []
    for name, info in AGENTS.items():
        entry = {"agent": name, **info}
        if name == "SHANI":
            entry["health"] = shani_health["status"]
        elif name == "GANESH":
            entry["health"] = ganesh_health["status"]
        else:
            matched = next((h for h in infra_health if h.get("agent") == name), None)
            entry["health"] = matched["status"] if matched else "unknown"
        agent_summary.append(entry)
    return _ok({
        "brahm_version":   "2.0",
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "total_tools":     total_tools,
        "agents":          agent_summary,
        "pipeline_stages": {
            "shani":  list(SHANI_STAGE_SEQUENCE),
            "ganesh": list(GANESH_STAGE_SEQUENCE),
        },
        "tool_groups":     tool_groups,
        "infrastructure": {
            "http_agents":  [shani_health, ganesh_health],
            "local_agents": infra_health,
        },
        "notes": [
            "S6 and S7 removed from SHANI — handled by GANESH.",
            "Use ganesh_write_review / ganesh_synthesize for scientific writing.",
        ],
    })


@brahm_tool(
    name        = "brahm_health",
    group       = "meta",
    description = (
        "Quick health check of all BRAHM infrastructure: "
        "API services (SHANI, GANESH), SQLite DB, and local agent imports. "
        "Faster than brahm_overview — use when you just need to verify "
        "that services are running before a task."
    ),
    input_schema = {"type": "object", "properties": {}, "required": []},
)
async def brahm_health(args: dict) -> dict:
    shani_health, ganesh_health = await asyncio.gather(
        _health_shani(), _health_ganesh()
    )
    local_health = await asyncio.to_thread(lambda: [
        _health_db(), _health_vidur(),
        _health_vishwakarma(), _health_chitragupta(),
    ])
    all_ok = (
        shani_health["status"] == "online"
        and all(h.get("status") in ("ok", "online") for h in local_health)
    )
    return _ok({
        "overall":     "healthy" if all_ok else "degraded",
        "http_agents": [shani_health, ganesh_health],
        "local":       local_health,
    })
