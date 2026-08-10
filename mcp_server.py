"""
mcp_server.py  —  BRAHM MCP Server
=====================================
Entry point only. Contains zero tool logic.
All tool definitions, schemas, and handlers live in brahm/agents/<agent>.py

sys.path ordering note (READ BEFORE CHANGING)
----------------------------------------------
Chitragupta MUST be index 0 — it has its own core/ package that must NOT
be shadowed by SHANI's core/. SHANI added first (lower priority),
Chitragupta added last (index 0 = wins on conflict).
"""

import asyncio
import json
import logging
import sys

# ─── sys.path — DO NOT REORDER ────────────────────────────────────────────────
_A = "/mnt/d/brahm/agents"
sys.path.insert(0, "/mnt/d/brahm")                  # brahm package
sys.path.insert(0, f"{_A}/shani")                   # SHANI        (lower priority)
sys.path.insert(0, f"{_A}/chitragupta/analysis")    # analysis module
sys.path.insert(0, f"{_A}/vidur")                   # VIDUR
sys.path.insert(0, f"{_A}/vishwakarma")             # Vishwakarma root
sys.path.insert(0, f"{_A}/ganesh")                  # GANESH
sys.path.insert(0, f"{_A}/chitragupta")             # Chitragupta  (index 0 = wins)

# ─── Environment ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv("/mnt/d/brahm/agents/chitragupta/.env")

# ─── MCP SDK ──────────────────────────────────────────────────────────────────
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ─── Registry (must be imported before any agent module) ──────────────────────
from brahm.brahm_registry import registry
from brahm.shared.helpers import _err

# ─── Agent modules — importing registers all @brahm_tool handlers ─────────────
import brahm.agents.shani           # Group A — SHANI pipeline
import brahm.agents.chitragupta     # Group B — Notion / Chitragupta
import brahm.agents.research        # Group C — Research DB queries
import brahm.agents.analysis        # Group D — Analysis tools
import brahm.agents.db_tools        # Group E — DB write + audit
import brahm.agents.vidur           # Group G — VIDUR classifier
import brahm.agents.vishwakarma     # Group H — Quantum ESPRESSO
import brahm.agents.ganesh          # Group I — GANESH writing
import brahm.agents.meta            # Meta     — brahm_overview, brahm_health

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("mcp.brahm")
server = Server("brahm-mcp")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return registry.all_tools()


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        result = await registry.dispatch(name, arguments or {})
    except Exception as exc:
        result = _err(f"Unhandled exception in {name}", str(exc))
        log.exception("Unhandled error in tool %s", name)
    return [
        types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False),
        )
    ]


async def main() -> None:
    log.info("BRAHM MCP starting — %d tools registered", len(registry))
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
