"""
mcp_server.py  —  BRAHM MCP
=============================
Deploy to: /mnt/d/SQL_IMP_AI_Project/mcp_server.py
Run under: /mnt/d/chitragupta/.venv/bin/python

Exposes tools across 7 groups:
  Group A  — SHANI Pipeline Tools       (shani_*)
  Group B  — Notion / Chitragupta Tools (notion_*)
  Group C  — Research Query Tools       (research_*)
  Group D  — Analysis Tools             (analysis_*)
  Group E  — Correction Tools           (db_*)
  Group F  — Review Generation Tools    (review_*)
  Group G  — VIDUR Classifier Tools     (vidur_*)

Architecture contract:
  - Intelligence lives in the agent; this server provides clean primitives
  - Every tool returns {"status": "success"|"error", ...} — never raises
  - Fresh Repository() per tool call — thread-safe
  - SHANI API calls via httpx; Notion via Chitragupta imports; DB via Repository
  - Group D analysis is read-only on SQLite
  - Group E writes append to mcp_corrections.jsonl audit log
  - Group G (VIDUR) runs fully locally — no HTTP, no cloud
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ─── Path injection ───────────────────────────────────────────────────────────
# Chitragupta MUST be index 0 (highest priority) so its internal
# `from core.validator` resolves to Chitragupta's core/, not SHANI's core/.
sys.path.insert(0, "/mnt/d/SQL_IMP_AI_Project")   # SHANI  — added first = lower priority
sys.path.insert(0, "/mnt/d/chitragupta/analysis")  # analysis module
sys.path.insert(0, "/mnt/d/chitragupta")            # Chitragupta — added last = index 0 = wins

# VIDUR — Characterization Classifier (local-only, no API deps)
# Adjust VIDUR_ROOT if the project lives elsewhere on your machine.
VIDUR_ROOT = "/mnt/d/vidur"
sys.path.insert(0, VIDUR_ROOT)

# VISHWAKARMA — Quantum ESPRESSO DFT agent
# Adjust VISHWAKARMA_ROOT to where vishwakarma/ package lives.
VISHWAKARMA_ROOT = "/mnt/d/vishwakarma"
sys.path.insert(0, VISHWAKARMA_ROOT)

from dotenv import load_dotenv
load_dotenv("/mnt/d/chitragupta/.env")

# ─── MCP SDK ──────────────────────────────────────────────────────────────────
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ─── Chitragupta imports (before SHANI core/ is in play) ─────────────────────
from notion.notion_client import (
    create_page,
    update_page,
    query_database_page,
    get_database,
    NotionAPIError,
)
from notion.schema_manager import (
    load_schema,
    create_schema,
    update_notion_id,
    SchemaAlreadyExistsError,
    SchemaMissingError,
)
from config.settings import NOTION_PAGE_ID
import notion_exporter as _ne
from research_analyzer import ResearchAnalyzer

# ─── SHANI imports ────────────────────────────────────────────────────────────
from repositories.repository import Repository

# Orchestrator.STAGE_SEQUENCE is the only thing needed from core/orchestrator.
# Hardcoded here to avoid the core/ package collision between SHANI and Chitragupta.
STAGE_SEQUENCE = (
    "S1", "S2", "S2_75", "S2_5",
    "S3", "S4", "S5", "S5_5", "S6", "S7"
)

# ─── Constants ────────────────────────────────────────────────────────────────
SHANI_BASE     = "http://localhost:8000"
DB_PATH        = "/mnt/d/SQL_IMP_AI_Project/database/research_workflow.db"
AUDIT_LOG_PATH = "/mnt/d/SQL_IMP_AI_Project/database/mcp_corrections.jsonl"
SHANI_VENV_PY  = "/mnt/d/SQL_IMP_AI_Project/venv/bin/python"
CHIT_VENV_PY   = "/mnt/d/chitragupta/.venv/bin/python"
RATE_LIMIT_SEC = 0.38

PAPER_WRITABLE_FIELDS = {
    "title", "doi", "abstract", "pdf_url", "pdf_status",
    "status", "created_at", "failed_candidates", "last_error",
}
PAPER_IMMUTABLE_FIELDS = {
    "id", "workflow_id", "source", "raw_text", "file_path",
}
CONFIG_WRITABLE_FIELDS = {
    "material", "focus", "structure", "method",
    "properties", "characterization", "domain",
}

VALID_STAGES = {
    "S1", "S2", "S2_75", "S2_5", "S3", "S4", "S5", "S5_5", "S6", "S7",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("mcp.brahm")

server = Server("brahm-mcp")

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _ok(data: dict) -> dict:
    return {"status": "success", **data}


def _err(msg: str, detail: str = "") -> dict:
    return {"status": "error", "error": msg, "detail": detail}


def _repo() -> Repository:
    """Fresh Repository per tool call — thread-safe."""
    return Repository(DB_PATH)


def _analyzer() -> ResearchAnalyzer:
    return ResearchAnalyzer(DB_PATH)


async def _shani_get(path: str) -> dict:
    """Async GET to SHANI API."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{SHANI_BASE}{path}")
            if r.status_code >= 400:
                return _err(
                    f"SHANI API error {r.status_code}",
                    r.text[:500],
                )
            return r.json()
    except Exception as exc:
        return _err("SHANI API unreachable", str(exc))


async def _shani_post(path: str, body: dict) -> dict:
    """Async POST to SHANI API."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{SHANI_BASE}{path}", json=body)
            if r.status_code >= 400:
                return _err(
                    f"SHANI API error {r.status_code}",
                    r.text[:500],
                )
            return r.json()
    except Exception as exc:
        return _err("SHANI API unreachable", str(exc))


async def _check_shani() -> bool:
    """Returns True if SHANI API is reachable."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{SHANI_BASE}/docs")
            return r.status_code == 200
    except Exception:
        return False


def _shani_required(result: dict) -> dict:
    """Wraps a pipeline tool call with a SHANI availability guard."""
    # caller checks _check_shani() before calling — this is a pass-through helper
    return result


def _audit_log(tool: str, record: dict) -> None:
    """Append a JSON-lines audit record for Group E write operations."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        **record,
    }
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("Audit log write failed: %s", exc)


# ─── Notion property builders ─────────────────────────────────────────────────

def _title(text: str) -> dict:
    return {"title": [{"type": "text", "text": {"content": str(text)[:500]}}]}

def _rtext(text: str) -> dict:
    return {"rich_text": [{"type": "text", "text": {"content": str(text)[:2000]}}]}

def _select(value: str) -> dict:
    return {"select": {"name": str(value)}}

def _number(value) -> dict:
    return {"number": value}

def _url(value: Optional[str]) -> dict:
    return {"url": value if value else None}

def _date(value: str) -> dict:
    return {"date": {"start": value}}


def _extract_notion_page(page: dict) -> dict:
    """Flatten a raw Notion page API object into a clean dict."""
    props = page.get("properties", {})
    result: dict = {"notion_page_id": page.get("id", "")}
    for key, prop in props.items():
        ptype = prop.get("type")
        if ptype == "title":
            texts = prop.get("title", [])
            result[key] = "".join(t.get("text", {}).get("content", "") for t in texts)
        elif ptype == "rich_text":
            texts = prop.get("rich_text", [])
            result[key] = "".join(t.get("text", {}).get("content", "") for t in texts)
        elif ptype == "select":
            sel = prop.get("select")
            result[key] = sel["name"] if sel else None
        elif ptype == "number":
            result[key] = prop.get("number")
        elif ptype == "url":
            result[key] = prop.get("url")
        elif ptype == "date":
            d = prop.get("date")
            result[key] = d["start"] if d else None
        else:
            result[key] = str(prop)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS (list_tools)
# ═════════════════════════════════════════════════════════════════════════════

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [

        # ── Group A ──────────────────────────────────────────────────────────

        types.Tool(
            name="shani_create_workflow",
            description=(
                "Create a new research workflow with topic configuration. "
                "Returns workflow_id needed for all subsequent operations. "
                "Starts in 'paused' state — call shani_run_workflow to begin. "
                "The 'focus' field is the most critical: space-separated keywords "
                "like 'defects vacancies interstitials traps deep levels'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Workflow name"},
                    "material": {"type": "string", "description": "Primary material, e.g. 'ZnSe'"},
                    "focus": {"type": "string", "description": "Space-separated research focus keywords (most important)"},
                    "structure": {"type": "string", "description": "Structural forms, e.g. 'thin film nanoparticle'"},
                    "method": {"type": "string", "description": "Synthesis methods, e.g. 'MBE sputtering ALD'"},
                    "properties": {"type": "string", "description": "Properties of interest, e.g. 'bandgap conductivity'"},
                    "characterization": {"type": "string", "description": "Techniques, e.g. 'XRD SEM DLTS'"},
                    "use_local": {"type": "boolean", "default": False, "description": "Use PDFs from papers/ folder, skip S1-S3"},
                },
                "required": ["name"],
            },
        ),

        types.Tool(
            name="shani_run_workflow",
            description=(
                "Start a paused workflow. Runs asynchronously — returns immediately. "
                "Poll shani_get_status to monitor. Default stop is S4 (paper extraction). "
                "Workflow must be in 'paused' state."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                    "stop_after_stage": {
                        "type": "string",
                        "enum": ["S1", "S2", "S2_75", "S2_5", "S3", "S4", "S5", "S5_5", "S6", "S7"],
                        "default": "S4",
                        "description": "Stage to stop at. S4 for data collection, S7 for full review generation.",
                    },
                },
                "required": ["workflow_id"],
            },
        ),

        types.Tool(
            name="shani_batch_run",
            description=(
                "Create and run multiple workflows concurrently (1-20). "
                "Ideal for launching 10-15 themed workflows covering a material system. "
                "All run in parallel threads. Returns all workflow IDs immediately — "
                "poll shani_get_all_status to track progress."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflows": {
                        "type": "array",
                        "maxItems": 20,
                        "description": "List of workflow configurations",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "material": {"type": "string"},
                                "focus": {"type": "string"},
                                "structure": {"type": "string"},
                                "method": {"type": "string"},
                                "properties": {"type": "string"},
                                "characterization": {"type": "string"},
                                "use_local": {"type": "boolean", "default": False},
                            },
                            "required": ["name"],
                        },
                    },
                    "stop_after_stage": {
                        "type": "string",
                        "enum": ["S1", "S2", "S2_75", "S2_5", "S3", "S4", "S5", "S5_5", "S6", "S7"],
                        "default": "S4",
                    },
                },
                "required": ["workflows"],
            },
        ),

        types.Tool(
            name="shani_get_status",
            description=(
                "Get full status of a workflow: all stage records + latest execution attempt. "
                "A workflow is 'done' when status='paused' and current_stage matches stop_after_stage, "
                "or status='completed' for full pipeline runs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                },
                "required": ["workflow_id"],
            },
        ),

        types.Tool(
            name="shani_get_all_status",
            description=(
                "Summary of ALL workflows: ID, name, status, current stage, paper counts. "
                "Use during batch monitoring to check overall progress at a glance."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),

        types.Tool(
            name="shani_get_papers",
            description="Get all papers collected by a workflow, with optional status filter.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                    "status_filter": {
                        "type": "string",
                        "enum": ["all", "extracted", "pending", "failed", "knowledge_ready", "completed"],
                        "default": "all",
                    },
                },
                "required": ["workflow_id"],
            },
        ),

        types.Tool(
            name="shani_get_paper_content",
            description="Get all extracted text sections for a specific paper (section_name → content).",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                    "paper_id": {"type": "integer"},
                },
                "required": ["workflow_id", "paper_id"],
            },
        ),

        types.Tool(
            name="shani_extract_workflow_data",
            description=(
                "Full dump of all papers + extracted content for a workflow. "
                "Use after S4 completes. papers_with_content is the usable subset."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                },
                "required": ["workflow_id"],
            },
        ),

        types.Tool(
            name="shani_clear_database",
            description=(
                "DESTRUCTIVE: Delete all workflows, papers, content, and knowledge. "
                "Irreversible. Use before a fresh research campaign. "
                "Must pass confirm=true explicitly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "enum": [True],
                        "description": "Must be true. Prevents accidental deletion.",
                    },
                },
                "required": ["confirm"],
            },
        ),

        types.Tool(
            name="shani_reset_workflow",
            description=(
                "Reset a failed or stuck workflow back to 'paused' so it can be restarted. "
                "Optionally specify from_stage to retry from a specific stage "
                "(stages from that point onwards are deleted). "
                "Use when shani_get_status shows status='failed'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                    "from_stage": {
                        "type": "string",
                        "enum": ["S1", "S2", "S2_75", "S2_5", "S3", "S4", "S5", "S5_5", "S6", "S7"],
                        "description": "Delete stages from this point onwards and retry. Omit to just reset status.",
                    },
                },
                "required": ["workflow_id"],
            },
        ),

        types.Tool(
            name="queue_add_workflow",
            description=(
                "Add a single workflow config to the local queue file WITHOUT running it. "
                "Call this once per workflow topic. After all topics are queued, "
                "the user will call run-queue to trigger them one by one. "
                "ALWAYS include: name, material, focus, structure, method, properties, characterization."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name":             {"type": "string"},
                    "material":         {"type": "string"},
                    "focus":            {"type": "string"},
                    "structure":        {"type": "string"},
                    "method":           {"type": "string"},
                    "properties":       {"type": "string"},
                    "characterization": {"type": "string"},
                },
                "required": ["name", "material", "focus"],
            },
        ),

        # ── Group B ──────────────────────────────────────────────────────────

        types.Tool(
            name="notion_export_research",
            description=(
                "Export top-ranked research papers from SHANI to Notion. "
                "Scores papers by material relevance + knowledge density + content completeness. "
                "Selects top N per workflow theme. Creates the Notion DB if it doesn't exist. "
                "Synchronous — takes ~2 minutes for 300 papers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "top_n_per_workflow": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 5,
                        "maximum": 50,
                        "description": "Top papers to select per workflow theme",
                    },
                    "min_relevance_score": {
                        "type": "number",
                        "default": 0.0,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Minimum relevance threshold (0.0–1.0)",
                    },
                },
                "required": [],
            },
        ),

        types.Tool(
            name="notion_query_papers",
            description=(
                "Query the Notion research database. "
                "Filter by workflow theme or minimum relevance score. "
                "Returns structured paper entries with all extracted knowledge fields."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_theme": {
                        "type": "string",
                        "enum": [
                            "ZnSe Fundamentals", "ZnO Fundamentals", "ZnSe vs ZnO Comparative",
                            "ZnSeO Alloy Formation", "Thermodynamics & Phase Stability",
                            "Oxygen Incorporation", "Defect Chemistry", "Bandgap Engineering",
                            "Optical Properties", "Charge Transport", "Thin Film Synthesis",
                            "Post-Deposition Treatments", "Characterization Techniques",
                            "Applications", "Challenges & Research Gaps",
                        ],
                        "description": "Filter by workflow theme",
                    },
                    "min_relevance_score": {
                        "type": "number",
                        "description": "Minimum relevance score (0.0–1.0)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "maximum": 100,
                        "description": "Max results to return",
                    },
                    "start_cursor": {
                        "type": "string",
                        "description": "Notion pagination cursor from previous call",
                    },
                },
                "required": [],
            },
        ),

        types.Tool(
            name="notion_get_database_info",
            description=(
                "Get metadata about the Notion research database: name, ID, property schema. "
                "Use after notion_export_research to verify the export completed successfully."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),

        types.Tool(
            name="notion_update_paper",
            description=(
                "Update specific fields on an existing Notion page (paper row). "
                "Use for correcting exported data without re-running the full export. "
                "page_id comes from a notion_query_papers result."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "Notion page ID"},
                    "fields": {
                        "type": "object",
                        "description": "Fields to update",
                        "properties": {
                            "Title": {"type": "string"},
                            "Year": {"type": "integer"},
                            "DOI": {"type": "string"},
                            "Abstract": {"type": "string"},
                            "Relevance Score": {"type": "number"},
                        },
                    },
                },
                "required": ["page_id", "fields"],
            },
        ),

        # ── Group C ──────────────────────────────────────────────────────────

        types.Tool(
            name="research_knowledge_summary",
            description=(
                "Statistical summary of extracted research knowledge: counts by category "
                "(material, synthesis_method, characterization, application) and "
                "top values per category. Check this before starting S5 to assess corpus quality."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "integer",
                        "description": "Restrict to one workflow. Omit for all workflows.",
                    },
                },
                "required": [],
            },
        ),

        types.Tool(
            name="research_find_papers_by_topic",
            description=(
                "Search extracted paper content for a topic using keyword matching "
                "against title, abstract, and/or PaperContent sections."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to search, e.g. ['bandgap', 'ZnSe', 'annealing']",
                    },
                    "search_in": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["title", "abstract", "content"]},
                        "default": ["title", "abstract"],
                        "description": "Where to search",
                    },
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["keywords"],
            },
        ),

        types.Tool(
            name="research_get_database_stats",
            description=(
                "Overall DB health metrics: workflow count, paper count, extraction rates, "
                "knowledge density. Use to assess data collection quality before export."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),

        # ── Group D ──────────────────────────────────────────────────────────

        types.Tool(
            name="analysis_technique_frequency",
            description=(
                "Count how often each synthesis method, characterization technique, "
                "material variant, or application appears across all papers. "
                "Use to understand which approaches dominate the collected corpus."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["all", "material", "synthesis_method", "characterization", "application", "computational_method"],
                        "description": "Which category to count",
                    },
                    "workflow_id": {"type": "integer", "description": "Restrict to one workflow"},
                    "top_n": {"type": "integer", "default": 20},
                    "min_count": {"type": "integer", "default": 2, "description": "Filter out singletons"},
                },
                "required": ["category"],
            },
        ),

        types.Tool(
            name="analysis_trend_report",
            description=(
                "Find co-occurrence patterns between two knowledge categories. "
                "e.g. 'what characterization methods appear with magnetron sputtering?' "
                "Returns ranked pairs with paper counts and sample titles."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "primary_category": {
                        "type": "string",
                        "enum": ["material", "synthesis_method", "characterization", "application", "computational_method"],
                    },
                    "secondary_category": {
                        "type": "string",
                        "enum": ["material", "synthesis_method", "characterization", "application", "computational_method"],
                        "description": "Co-occurrence axis",
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "Filter primary_category to this value, e.g. 'magnetron sputtering'",
                    },
                    "workflow_id": {"type": "integer"},
                    "min_co_occurrence": {"type": "integer", "default": 3},
                },
                "required": ["primary_category"],
            },
        ),

        types.Tool(
            name="analysis_find_gaps",
            description=(
                "Identify under-explored or unexplored A×B combinations. "
                "The most strategically valuable analysis tool. "
                "e.g. 'which material + synthesis_method pairings have no papers?'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "category_a": {
                        "type": "string",
                        "enum": ["material", "synthesis_method", "characterization", "application", "computational_method"],
                    },
                    "category_b": {
                        "type": "string",
                        "enum": ["material", "synthesis_method", "characterization", "application", "computational_method"],
                    },
                    "known_values_a": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Constrain A axis values. Omit to auto-discover from corpus.",
                    },
                    "known_values_b": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Constrain B axis values.",
                    },
                    "gap_threshold": {
                        "type": "integer", "default": 2,
                        "description": "Combinations with ≤ this many papers are flagged as gaps",
                    },
                },
                "required": ["category_a", "category_b"],
            },
        ),

        types.Tool(
            name="analysis_parameter_distribution",
            description=(
                "Extract and aggregate quantitative values from the knowledge base. "
                "e.g. 'what bandgap values have been reported for ZnSe?' "
                "Returns mean/min/max by material, plus source sentences."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "parameter_keywords": {
                        "type": "array", "items": {"type": "string"},
                        "description": "e.g. ['bandgap', 'band gap', 'Eg'] or ['annealing temperature']",
                    },
                    "workflow_id": {"type": "integer"},
                    "extract_numbers": {"type": "boolean", "default": True},
                    "group_by_material": {"type": "boolean", "default": True},
                },
                "required": ["parameter_keywords"],
            },
        ),

        types.Tool(
            name="analysis_workflow_comparison",
            description=(
                "Compare 2+ workflows to identify overlapping and unique coverage. "
                "compare_by: 'knowledge' (all values), 'materials', 'techniques', or 'papers'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_ids": {
                        "type": "array", "items": {"type": "integer"},
                        "minItems": 2,
                        "description": "IDs of workflows to compare",
                    },
                    "compare_by": {
                        "type": "string",
                        "enum": ["papers", "techniques", "materials", "knowledge"],
                        "default": "knowledge",
                    },
                },
                "required": ["workflow_ids"],
            },
        ),

        types.Tool(
            name="analysis_save_to_notion",
            description=(
                "Persist an analysis result to the 'ZnSe Analysis Results' Notion database. "
                "Creates the database if it doesn't exist. Builds a research audit trail "
                "across multiple campaigns."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Analysis name/title"},
                    "type": {
                        "type": "string",
                        "enum": ["Trend", "Gap", "Parameter", "Comparison", "Frequency"],
                    },
                    "key_finding": {"type": "string", "description": "1–2 sentence summary of the top finding"},
                    "results": {"type": "object", "description": "Full results dict to persist as JSON"},
                    "action_items": {"type": "string", "description": "Agent-generated next steps"},
                    "workflow_ids": {
                        "type": "array", "items": {"type": "integer"},
                        "description": "Workflows this analysis covered",
                    },
                },
                "required": ["name", "type", "key_finding"],
            },
        ),

        # ── Group E ──────────────────────────────────────────────────────────

        types.Tool(
            name="db_list_suspect_papers",
            description=(
                "Find papers with data quality issues: missing DOIs, malformed dates, "
                "empty abstracts, no content. Returns papers with issue flags and suggested fixes. "
                "Run this before db_bulk_fix to identify what needs correction."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer", "description": "Restrict to one workflow"},
                    "issue_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["missing_doi", "missing_abstract", "bad_date", "short_abstract", "no_content"],
                        },
                        "description": "Specific issues to check. Omit to check all.",
                    },
                    "limit": {"type": "integer", "default": 50},
                },
                "required": [],
            },
        ),

        types.Tool(
            name="db_update_paper",
            description=(
                "Direct field update on a Paper row. Corrects metadata issues. "
                "Blocked fields: id, workflow_id, source, raw_text, file_path. "
                "Always writes to the audit log. Use db_list_suspect_papers first to identify targets."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "integer"},
                    "fields": {
                        "type": "object",
                        "description": "Fields to update (title, doi, abstract, pdf_url, pdf_status, status, created_at)",
                    },
                },
                "required": ["paper_id", "fields"],
            },
        ),

        types.Tool(
            name="db_update_workflow_config",
            description=(
                "Correct WorkflowResearchConfig fields for a workflow. "
                "Useful when a config was created with wrong material/focus."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                    "fields": {
                        "type": "object",
                        "description": "Fields: material, focus, structure, method, properties, characterization, domain",
                    },
                },
                "required": ["workflow_id", "fields"],
            },
        ),

        types.Tool(
            name="db_bulk_fix",
            description=(
                "Apply a regex pattern-based fix across multiple papers in a workflow. "
                "ALWAYS call with dry_run=true first to preview affected rows. "
                "Only call with dry_run=false after confirming the preview is correct."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer", "description": "Required — bulk ops always scoped to one workflow"},
                    "field": {"type": "string", "description": "Which field to fix (must be in writable allowlist)"},
                    "match_pattern": {"type": "string", "description": "Python regex to identify affected rows"},
                    "replacement": {"type": "string", "description": "Replacement string (supports regex groups)"},
                    "dry_run": {"type": "boolean", "default": True, "description": "True = preview only, no writes"},
                },
                "required": ["workflow_id", "field", "match_pattern", "replacement"],
            },
        ),

        # ── Group F ──────────────────────────────────────────────────────────

        types.Tool(
            name="review_run_knowledge_extraction",
            description=(
                "Trigger S5 (extract_research_knowledge) on one or more workflows paused at S4. "
                "Validates preconditions: workflow must be paused with sufficient extracted papers. "
                "S5 is the most token-intensive stage — validate corpus quality first with "
                "research_get_database_stats."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_ids": {
                        "type": "array", "items": {"type": "integer"},
                        "description": "Workflows to run S5 on",
                    },
                    "min_extracted_papers": {
                        "type": "integer", "default": 10,
                        "description": "Skip workflows with fewer extracted papers",
                    },
                    "stop_after": {
                        "type": "string",
                        "enum": ["S5", "S5_5"],
                        "default": "S5",
                        "description": "S5_5 also runs review direction generation",
                    },
                },
                "required": ["workflow_ids"],
            },
        ),

        types.Tool(
            name="review_generate_direction",
            description=(
                "Trigger S5_5 (generate_review_direction) for a workflow. "
                "Generates the structure/outline for the review paper. "
                "Run after S5 completes. Idempotent — returns existing direction if already generated."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                },
                "required": ["workflow_id"],
            },
        ),

        types.Tool(
            name="review_draft_sections",
            description=(
                "Trigger S6 (draft_sections) — actual review section drafting. "
                "Run after S5_5 completes. Use force=true to re-draft after corpus changes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                    "force": {
                        "type": "boolean", "default": False,
                        "description": "Re-run even if S6 already completed",
                    },
                },
                "required": ["workflow_id"],
            },
        ),

        types.Tool(
            name="review_synthesize_final",
            description=(
                "Trigger S7 (synthesize_paper) — final synthesis stage. "
                "The terminal review generation step. "
                "Optionally auto-exports to Notion after completion."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                    "export_to_notion": {
                        "type": "boolean", "default": False,
                        "description": "Auto-export to Notion after S7 completes",
                    },
                },
                "required": ["workflow_id"],
            },
        ),

        types.Tool(
            name="review_get_draft",
            description=(
                "Retrieve the current draft output from a completed review stage. "
                "stage='S6' returns section drafts; stage='S7' returns final synthesis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "integer"},
                    "stage": {
                        "type": "string",
                        "enum": ["S5_5", "S6", "S7"],
                        "description": "Which stage's output to retrieve",
                    },
                },
                "required": ["workflow_id", "stage"],
            },
        ),

        # ── Group G — VIDUR Classifier Tools ─────────────────────────────────

        types.Tool(
            name="vidur_classify",
            description=(
                "Classify a scientific instrument file using VIDUR. "
                "Auto-detects the characterization technique (XRD, UV-Vis, SEM_EDX, Raman) "
                "and parses the data into a structured format. "
                "Runs fully locally — no cloud, no HTTP. "
                "Returns technique, confidence score, detection signals, and parsed data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Absolute path to the instrument data file. "
                            "Supported: .xrdml, .raw, .xy, .dat, .asc (XRD); "
                            ".sp, .abs, .spc (UV-Vis); "
                            ".emsa, .msa, .spx (SEM/EDS); "
                            ".wdf, .spc (Raman); "
                            ".pdf, .docx, .csv, .txt (generic text extraction)."
                        ),
                    },
                },
                "required": ["file_path"],
            },
        ),

        types.Tool(
            name="vidur_list_techniques",
            description=(
                "List all characterization techniques that VIDUR can detect and parse. "
                "Returns technique names, supported file extensions, and key detection keywords."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),

        types.Tool(
            name="vidur_health",
            description=(
                "Check VIDUR health: verify all parser modules load correctly "
                "and core imports (extractor, auto_detector, router) are available. "
                "Returns per-parser status and overall readiness."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),

        types.Tool(
            name="brahm_overview",
            description=(
                "Return a complete live map of the entire BRAHM system: "
                "all agents (SHANI, Chitragupta, VIDUR, Vishwakarma), their status, "
                "all tool groups with every tool name + one-line purpose, "
                "and infrastructure health (API, DB, Notion, VIDUR, QE binaries). "
                "Use this as your first call to orient yourself before any task."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),

        # ── Group H — Vishwakarma (Quantum ESPRESSO) ──────────────────────────

        types.Tool(
            name="vishwakarma_health",
            description=(
                "Check Vishwakarma health: verify QE binaries (pw.x, ph.x, pp.x, "
                "dos.x, bands.x, neb.x, etc.) are reachable, and all Vishwakarma "
                "Python modules import correctly. Returns per-binary path or missing status."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),

        types.Tool(
            name="vishwakarma_generate_input",
            description=(
                "Generate a Quantum ESPRESSO input file without running it. "
                "Supports: scf | nscf | relax | vc-relax | bands | phonon | dos | "
                "projwfc | pp | neb | hp | cp. "
                "Returns the input file as a string for review before execution."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "calc_type": {
                        "type": "string",
                        "enum": ["scf","nscf","relax","vc-relax","bands",
                                 "phonon","dos","projwfc","pp","neb","hp","cp"],
                        "description": "Calculation type",
                    },
                    "structure": {
                        "type": "object",
                        "description": (
                            "Crystal structure dict: prefix, ibrav, cell_parameters (3x3 Å), "
                            "nat, ntyp, atomic_species [{symbol,mass,pseudo}], "
                            "atomic_positions [{symbol,x,y,z}], kpoints {mode,mesh,shift}."
                        ),
                    },
                    "calc_params": {
                        "type": "object",
                        "description": (
                            "Calculation parameters: ecutwfc, ecutrho, occupations, smearing, "
                            "degauss, conv_thr, pseudo_dir, outdir, nspin, nbnd, "
                            "hubbard_u, dft_d3, input_dft, nstep, etc."
                        ),
                    },
                    "phonon_params": {
                        "type": "object",
                        "description": "Extra params for phonon: qpoints, ldisp, nq, epsil, lraman",
                    },
                },
                "required": ["calc_type", "structure", "calc_params"],
            },
        ),

        types.Tool(
            name="vishwakarma_run_scf",
            description=(
                "Run a pw.x SCF (self-consistent field) calculation. "
                "Returns job_id, convergence status, total energy (Ry and eV), "
                "Fermi energy, band gap (if insulator), and wall time."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "structure":   {"type": "object", "description": "Crystal structure dict"},
                    "calc_params": {"type": "object", "description": "Calculation parameters"},
                    "label":       {"type": "string", "default": "scf"},
                    "mpi_np":      {"type": "integer", "default": 1, "description": "MPI processes"},
                    "timeout":     {"type": "integer", "default": 3600, "description": "Timeout in seconds"},
                },
                "required": ["structure", "calc_params"],
            },
        ),

        types.Tool(
            name="vishwakarma_run_relax",
            description=(
                "Run ionic relaxation (relax) or variable-cell relaxation (vc-relax) "
                "followed by a final SCF on the optimised geometry. "
                "Returns optimised atomic positions, cell parameters, final energy, and forces."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "structure":   {"type": "object"},
                    "calc_params": {"type": "object"},
                    "vc_relax":    {"type": "boolean", "default": False, "description": "Also relax cell volume/shape"},
                    "label":       {"type": "string", "default": "relax"},
                    "mpi_np":      {"type": "integer", "default": 1},
                    "timeout":     {"type": "integer", "default": 7200},
                },
                "required": ["structure", "calc_params"],
            },
        ),

        types.Tool(
            name="vishwakarma_run_bands",
            description=(
                "Run band structure calculation: SCF → NSCF on k-path → bands.x post-processing. "
                "Provide kpath as list of high-symmetry points. "
                "Returns job IDs for each step and parsed eigenvalue data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "structure":   {"type": "object"},
                    "calc_params": {"type": "object"},
                    "kpath": {
                        "type": "array",
                        "description": "List of [kx,ky,kz,npoints] for k-path segments",
                        "items": {"type": "array"},
                    },
                    "label":   {"type": "string", "default": "bands"},
                    "mpi_np":  {"type": "integer", "default": 1},
                    "timeout": {"type": "integer", "default": 3600},
                },
                "required": ["structure", "calc_params"],
            },
        ),

        types.Tool(
            name="vishwakarma_run_dos",
            description=(
                "Run density of states: SCF → dense NSCF → dos.x. "
                "Returns Fermi energy and job IDs. "
                "Actual DOS data is written to the fildos file in the job directory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "structure":    {"type": "object"},
                    "calc_params":  {"type": "object"},
                    "dense_kmesh":  {"type": "array", "description": "[nk1,nk2,nk3] for NSCF"},
                    "emin":         {"type": "number", "default": -20.0},
                    "emax":         {"type": "number", "default":  20.0},
                    "label":        {"type": "string", "default": "dos"},
                    "mpi_np":       {"type": "integer", "default": 1},
                    "timeout":      {"type": "integer", "default": 7200},
                },
                "required": ["structure", "calc_params"],
            },
        ),

        types.Tool(
            name="vishwakarma_run_phonon",
            description=(
                "Run DFPT phonon calculation: SCF → ph.x. "
                "Supports q-point mesh (ldisp=true) or specific q-points. "
                "Can compute dielectric tensor + Born effective charges (epsil=true) "
                "and Raman tensors (lraman=true). "
                "To resume a previously interrupted/timed-out run instead of restarting "
                "from scratch, pass recover=true plus existing_scf_job_id (the job_id "
                "of the SCF step from the original run)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "structure":    {"type": "object"},
                    "calc_params":  {"type": "object"},
                    "ldisp":   {"type": "boolean", "default": True, "description": "Use q-mesh (true) or explicit q-points"},
                    "nq":      {"type": "array", "default": [4,4,4], "description": "[nq1,nq2,nq3] q-mesh"},
                    "qpoints": {"type": "array", "description": "Explicit q-points [[qx,qy,qz],...]"},
                    "epsil":   {"type": "boolean", "default": True, "description": "Compute dielectric tensor + Born charges"},
                    "lraman":  {"type": "boolean", "default": False, "description": "Compute Raman tensors"},
                    "label":   {"type": "string", "default": "phonon"},
                    "mpi_np":  {"type": "integer", "default": 1},
                    "timeout": {"type": "integer", "default": 14400},
                    "recover": {"type": "boolean", "default": False, "description": "Resume from a prior interrupted run's .recover scratch file instead of restarting"},
                    "existing_scf_job_id": {"type": "string", "description": "job_id of the SCF step from the original run (required when recover=true)"},
                },
                "required": ["structure", "calc_params"],
            },
        ),

        types.Tool(
            name="vishwakarma_run_neb",
            description=(
                "Run nudged elastic band (NEB) calculation to find transition states "
                "and minimum energy paths between two structures. "
                "Provide initial and final structures; intermediate images are interpolated."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "initial_structure": {"type": "object", "description": "Reactant structure"},
                    "final_structure":   {"type": "object", "description": "Product structure"},
                    "calc_params":       {"type": "object"},
                    "num_images":    {"type": "integer", "default": 7},
                    "ci_scheme":     {"type": "string", "enum": ["no-CI","auto","manual"], "default": "auto"},
                    "opt_scheme":    {"type": "string", "enum": ["broyden","sd","lbfgs"], "default": "broyden"},
                    "nstep_path":    {"type": "integer", "default": 200},
                    "label":         {"type": "string", "default": "neb"},
                    "mpi_np":        {"type": "integer", "default": 1},
                    "timeout":       {"type": "integer", "default": 28800},
                },
                "required": ["initial_structure", "final_structure", "calc_params"],
            },
        ),

        types.Tool(
            name="vishwakarma_run_hp",
            description=(
                "Compute Hubbard U parameters from linear response theory using hp.x. "
                "Run after a converged SCF. Returns recommended U values per species "
                "for use in DFT+U calculations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prefix":    {"type": "string"},
                    "outdir":    {"type": "string"},
                    "nq":        {"type": "array", "default": [2,2,2], "description": "q-mesh for HP"},
                    "job_label": {"type": "string", "default": "hp"},
                    "mpi_np":    {"type": "integer", "default": 1},
                    "timeout":   {"type": "integer", "default": 7200},
                },
                "required": ["prefix", "outdir"],
            },
        ),

        types.Tool(
            name="vishwakarma_parse_output",
            description=(
                "Parse a Quantum ESPRESSO output file that already exists on disk "
                "or from a completed job. Returns structured data: energy, forces, "
                "stress, positions, convergence, Fermi energy, band gap, phonon frequencies, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["job_id", "file_path"],
                        "description": "job_id: read from job directory; file_path: read from absolute path",
                    },
                    "job_id":    {"type": "string", "description": "Job ID (if source=job_id)"},
                    "file_path": {"type": "string", "description": "Absolute path to .out file (if source=file_path)"},
                    "code":      {"type": "string", "enum": ["pw","ph","dos","bands","neb"], "default": "pw"},
                },
                "required": ["source", "code"],
            },
        ),

        types.Tool(
            name="vishwakarma_list_pseudopotentials",
            description=(
                "Discover and list all UPF pseudopotential files in the configured pseudo_dir. "
                "Shows element, functional (PBE/LDA/PBEsol), type (NC/US/PAW), and library. "
                "Optionally cross-check against a structure to flag missing pseudopotentials."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pseudo_dirs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Directories to scan. Defaults to QE_PSEUDO_DIR env var.",
                    },
                    "structure": {
                        "type": "object",
                        "description": "Optional — if provided, checks which elements have pseudopotentials",
                    },
                    "preferred_functional": {"type": "string", "default": "pbe"},
                    "preferred_type":       {"type": "string", "default": "us"},
                },
                "required": [],
            },
        ),

        types.Tool(
            name="vishwakarma_get_job_status",
            description="Get the status of a specific Vishwakarma job by job_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                },
                "required": ["job_id"],
            },
        ),

        types.Tool(
            name="vishwakarma_list_jobs",
            description=(
                "List all Vishwakarma calculation jobs, newest first. "
                "Filter by status: created | running | completed | failed | timeout. "
                "Shows label, code, status, timing for each job."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "enum": ["all","created","running","completed","failed","timeout"],
                        "default": "all",
                    },
                    "limit": {"type": "integer", "default": 20},
                },
                "required": [],
            },
        ),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# TOOL DISPATCH
# ═════════════════════════════════════════════════════════════════════════════

@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent]:
    try:
        result = await _dispatch(name, arguments or {})
    except Exception as exc:
        result = _err(f"Unhandled exception in {name}", str(exc))
        log.exception("Unhandled error in tool %s", name)

    return [
        types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False),
        )
    ]


async def _dispatch(name: str, args: dict) -> dict:
    """Route tool name → handler."""
    handlers = {
        # Group A
        "shani_create_workflow":          _shani_create_workflow,
        "shani_run_workflow":             _shani_run_workflow,
        "shani_batch_run":               _shani_batch_run,
        "shani_get_status":              _shani_get_status,
        "shani_get_all_status":          _shani_get_all_status,
        "shani_get_papers":              _shani_get_papers,
        "shani_get_paper_content":       _shani_get_paper_content,
        "shani_extract_workflow_data":   _shani_extract_workflow_data,
        "shani_clear_database":          _shani_clear_database,
        "shani_reset_workflow":          _shani_reset_workflow,
        # Group A extra
        "queue_add_workflow":            _queue_add_workflow,
        # Group B
        "notion_export_research":        _notion_export_research,
        "notion_query_papers":           _notion_query_papers,
        "notion_get_database_info":      _notion_get_database_info,
        "notion_update_paper":           _notion_update_paper,
        # Group C
        "research_knowledge_summary":    _research_knowledge_summary,
        "research_find_papers_by_topic": _research_find_papers_by_topic,
        "research_get_database_stats":   _research_get_database_stats,
        # Group D
        "analysis_technique_frequency":  _analysis_technique_frequency,
        "analysis_trend_report":         _analysis_trend_report,
        "analysis_find_gaps":            _analysis_find_gaps,
        "analysis_parameter_distribution": _analysis_parameter_distribution,
        "analysis_workflow_comparison":  _analysis_workflow_comparison,
        "analysis_save_to_notion":       _analysis_save_to_notion,
        # Group E
        "db_list_suspect_papers":        _db_list_suspect_papers,
        "db_update_paper":               _db_update_paper,
        "db_update_workflow_config":     _db_update_workflow_config,
        "db_bulk_fix":                   _db_bulk_fix,
        # Group F
        "review_run_knowledge_extraction": _review_run_knowledge_extraction,
        "review_generate_direction":     _review_generate_direction,
        "review_draft_sections":         _review_draft_sections,
        "review_synthesize_final":       _review_synthesize_final,
        "review_get_draft":              _review_get_draft,
        # Group G — VIDUR
        "vidur_classify":                _vidur_classify,
        "vidur_list_techniques":         _vidur_list_techniques,
        "vidur_health":                  _vidur_health,
        # Group H — Vishwakarma
        "vishwakarma_health":               _vishwakarma_health,
        "vishwakarma_generate_input":       _vishwakarma_generate_input,
        "vishwakarma_run_scf":              _vishwakarma_run_scf,
        "vishwakarma_run_relax":            _vishwakarma_run_relax,
        "vishwakarma_run_bands":            _vishwakarma_run_bands,
        "vishwakarma_run_dos":              _vishwakarma_run_dos,
        "vishwakarma_run_phonon":           _vishwakarma_run_phonon,
        "vishwakarma_run_neb":              _vishwakarma_run_neb,
        "vishwakarma_run_hp":               _vishwakarma_run_hp,
        "vishwakarma_parse_output":         _vishwakarma_parse_output,
        "vishwakarma_list_pseudopotentials":_vishwakarma_list_pseudopotentials,
        "vishwakarma_get_job_status":       _vishwakarma_get_job_status,
        "vishwakarma_list_jobs":            _vishwakarma_list_jobs,
        # Meta
        "brahm_overview":                _brahm_overview,
    }
    handler = handlers.get(name)
    if handler is None:
        return _err(f"Unknown tool: {name}")
    return await handler(args)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP A — SHANI PIPELINE TOOLS
# ═════════════════════════════════════════════════════════════════════════════

QUEUE_PATH = "/mnt/d/SQL_IMP_AI_Project/workflow_queue.json"

async def _queue_add_workflow(args: dict) -> dict:
    """Save one workflow config to queue file. Does not trigger SHANI."""
    import json as _json
    from pathlib import Path as _Path
    try:
        existing = []
        if _Path(QUEUE_PATH).exists():
            with open(QUEUE_PATH) as f:
                existing = _json.load(f)
        existing.append(args)
        with open(QUEUE_PATH, "w") as f:
            _json.dump(existing, f, indent=2)
        return _ok({
            "queued": args.get("name"),
            "queue_length": len(existing),
            "message": f"Added '{args.get('name')}' to queue. Total queued: {len(existing)}.",
        })
    except Exception as exc:
        return _err("Queue write failed", str(exc))


async def _shani_create_workflow(args: dict) -> dict:
    if not await _check_shani():
        return _err(
            "SHANI API not running.",
            "Start with: python -m uvicorn api:app --host 0.0.0.0 --port 8000"
        )
    body = {k: v for k, v in args.items() if v is not None}
    result = await _shani_post("/workflows", body)
    if result.get("status") == "error":
        return result
    # SHANI returns {"workflow_id": N, "status": "paused"} on success
    # Normalise to our standard envelope
    if "workflow_id" in result:
        return _ok({"workflow_id": result["workflow_id"], "shani_status": result.get("status")})
    return _ok(result)


async def _shani_run_workflow(args: dict) -> dict:
    if not await _check_shani():
        return _err("SHANI API not running.",
                    "Start with: python -m uvicorn api:app --host 0.0.0.0 --port 8000")
    wf_id = args["workflow_id"]
    stop_after = args.get("stop_after_stage", "S4")
    result = await _shani_post(
        f"/workflows/{wf_id}/run",
        {"stop_after_stage": stop_after},
    )
    if result.get("status") == "error":
        return result
    return _ok(result)


async def _shani_batch_run(args: dict) -> dict:
    if not await _check_shani():
        return _err("SHANI API not running.",
                    "Start with: python -m uvicorn api:app --host 0.0.0.0 --port 8000")
    result = await _shani_post("/workflows/batch", args)
    if result.get("status") == "error":
        return result
    return _ok(result)


async def _shani_get_status(args: dict) -> dict:
    if not await _check_shani():
        return _err("SHANI API not running.")
    result = await _shani_get(f"/workflows/{args['workflow_id']}/status")
    if result.get("status") == "error":
        return result
    return _ok(result)


async def _shani_get_all_status(args: dict) -> dict:
    """Direct SQLite — no API endpoint for listing all workflows."""
    def _query() -> dict:
        repo = _repo()
        try:
            rows = repo.fetch_all(
                """
                SELECT w.id, w.name, w.status, w.current_stage,
                  (SELECT COUNT(*) FROM Paper p WHERE p.workflow_id = w.id) AS papers,
                  (SELECT COUNT(*) FROM Paper p WHERE p.workflow_id = w.id
                   AND p.status IN ('extracted','knowledge_ready','completed')) AS extracted,
                  (SELECT COUNT(*) FROM Paper p WHERE p.workflow_id = w.id
                   AND p.status = 'failed') AS failed
                FROM Workflow w ORDER BY w.id
                """
            )
            workflows = [dict(r) for r in rows]
            running = sum(1 for w in workflows if w["status"] == "running")
            paused  = sum(1 for w in workflows if w["status"] == "paused")
            return _ok({
                "total_workflows": len(workflows),
                "running": running,
                "paused": paused,
                "workflows": workflows,
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_query)


async def _shani_get_papers(args: dict) -> dict:
    if not await _check_shani():
        return _err("SHANI API not running.")
    wf_id = args["workflow_id"]
    status_filter = args.get("status_filter", "all")
    result = await _shani_get(f"/workflows/{wf_id}/papers")
    if result.get("status") == "error":
        return result
    papers = result.get("papers", [])
    if status_filter != "all":
        papers = [p for p in papers if p.get("status") == status_filter]
    return _ok({
        "workflow_id": wf_id,
        "status_filter": status_filter,
        "count": len(papers),
        "papers": papers,
    })


async def _shani_get_paper_content(args: dict) -> dict:
    if not await _check_shani():
        return _err("SHANI API not running.")
    wf_id = args["workflow_id"]
    paper_id = args["paper_id"]
    result = await _shani_get(f"/workflows/{wf_id}/papers/{paper_id}/content")
    if result.get("status") == "error":
        return result
    return _ok(result)


async def _shani_extract_workflow_data(args: dict) -> dict:
    if not await _check_shani():
        return _err("SHANI API not running.")
    wf_id = args["workflow_id"]
    result = await _shani_get(f"/workflows/{wf_id}/extract")
    if result.get("status") == "error":
        return result
    return _ok(result)


async def _shani_clear_database(args: dict) -> dict:
    if not args.get("confirm"):
        return _err("confirm must be true to proceed. This operation is irreversible.")

    def _run_clear() -> dict:
        try:
            result = subprocess.run(
                [SHANI_VENV_PY, "-m", "core.shani", "del_r"],
                cwd="/mnt/d/SQL_IMP_AI_Project",
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return _err("clear_database subprocess failed", result.stderr[:500])
            return _ok({
                "message": "Database cleared successfully.",
                "stdout": result.stdout[-1000:],
            })
        except subprocess.TimeoutExpired:
            return _err("clear_database timed out after 60 s")
        except Exception as exc:
            return _err("clear_database failed", str(exc))

    return await asyncio.to_thread(_run_clear)


async def _shani_reset_workflow(args: dict) -> dict:
    """Direct SQLite: set workflow to paused, optionally delete stages from from_stage."""
    wf_id = args["workflow_id"]
    from_stage: Optional[str] = args.get("from_stage")

    def _reset() -> dict:
        repo = _repo()
        try:
            wf = repo.fetch_one("SELECT id, status, name FROM Workflow WHERE id = ?", (wf_id,))
            if not wf:
                return _err(f"Workflow {wf_id} not found")

            if from_stage:
                if from_stage not in VALID_STAGES:
                    return _err(f"Invalid from_stage '{from_stage}'")
                # Determine stages to delete (from_stage and everything after it)
                seq = STAGE_SEQUENCE
                try:
                    start_idx = seq.index(from_stage)
                except ValueError:
                    return _err(f"Stage '{from_stage}' not in STAGE_SEQUENCE")
                stages_to_delete = seq[start_idx:]

                # Delete Stage rows for these stages
                with repo.transaction() as cursor:
                    for stage_name in stages_to_delete:
                        cursor.execute(
                            "DELETE FROM Stage WHERE workflow_id = ? AND stage_name = ?",
                            (wf_id, stage_name),
                        )
                    # Reset workflow
                    cursor.execute(
                        "UPDATE Workflow SET status = 'paused', current_stage = ?, "
                        "updated_at = ? WHERE id = ?",
                        (from_stage, datetime.utcnow().isoformat(), wf_id),
                    )
            else:
                with repo.transaction() as cursor:
                    cursor.execute(
                        "UPDATE Workflow SET status = 'paused', updated_at = ? WHERE id = ?",
                        (datetime.utcnow().isoformat(), wf_id),
                    )

            return _ok({
                "workflow_id": wf_id,
                "name": dict(wf)["name"],
                "previous_status": dict(wf)["status"],
                "new_status": "paused",
                "from_stage": from_stage,
                "message": (
                    f"Reset to 'paused'. Deleted stages from {from_stage} onwards."
                    if from_stage else
                    "Status set to 'paused'. No stages deleted."
                ),
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_reset)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP B — NOTION / CHITRAGUPTA TOOLS
# ═════════════════════════════════════════════════════════════════════════════

async def _notion_export_research(args: dict) -> dict:
    top_n = args.get("top_n_per_workflow", 20)
    min_score = args.get("min_relevance_score", 0.0)

    def _export() -> dict:
        try:
            repo = _repo()
            try:
                # Monkey-patch TOP_N so load_and_rank_papers uses our value
                original_top_n = _ne.TOP_N
                _ne.TOP_N = top_n

                papers = _ne.load_and_rank_papers(repo)
            finally:
                _ne.TOP_N = original_top_n
                repo.close()

            if not papers:
                return _err("No papers found to export. "
                            "Ensure S4 has completed on at least one workflow.")

            # Apply minimum score filter
            if min_score > 0.0:
                papers = [p for p in papers if p.get("relevance_score", 0) >= min_score]

            if not papers:
                return _err(f"No papers passed min_relevance_score={min_score}")

            db_id = _ne.setup_database()
            success, failed, skipped = 0, 0, 0

            for i, paper in enumerate(papers):
                try:
                    props = _ne.build_row(paper)
                    create_page(db_id, props)
                    success += 1
                except NotionAPIError as exc:
                    log.error("Notion error paper %d: %s", paper["id"], exc)
                    failed += 1
                    if exc.notion_status == 429:
                        time.sleep(5)
                except Exception as exc:
                    log.error("Error paper %d: %s", paper["id"], exc)
                    skipped += 1
                time.sleep(_ne.RATE_LIMIT_SEC)

            notion_url = f"https://notion.so/{db_id.replace('-', '')}"
            return _ok({
                "total_selected": len(papers),
                "pushed": success,
                "failed": failed,
                "skipped": skipped,
                "notion_url": notion_url,
                "database_id": db_id,
            })
        except Exception as exc:
            return _err("Export failed", str(exc))

    return await asyncio.to_thread(_export)


async def _notion_query_papers(args: dict) -> dict:
    workflow_theme = args.get("workflow_theme")
    min_score      = args.get("min_relevance_score")
    limit          = min(args.get("limit", 20), 100)
    start_cursor   = args.get("start_cursor")

    def _query() -> dict:
        try:
            schema = load_schema("ZnSe Research Knowledge Base")
            db_id  = schema.get("notion_database_id", "")
            if not db_id:
                return _err("Notion database not set up yet. Run notion_export_research first.")

            filters: list = []
            if workflow_theme:
                filters.append({
                    "property": "Workflow Theme",
                    "select": {"equals": workflow_theme},
                })
            if min_score is not None:
                filters.append({
                    "property": "Relevance Score",
                    "number": {"greater_than_or_equal_to": min_score},
                })

            filter_payload = None
            if len(filters) == 1:
                filter_payload = filters[0]
            elif len(filters) > 1:
                filter_payload = {"and": filters}

            items, next_cursor = query_database_page(
                db_id,
                filter_payload=filter_payload,
                page_size=limit,
                start_cursor=start_cursor,
            )

            papers = [_extract_notion_page(p) for p in items]
            return _ok({
                "count": len(papers),
                "next_cursor": next_cursor,
                "has_more": next_cursor is not None,
                "papers": papers,
            })
        except SchemaMissingError:
            return _err("Notion database schema not found. Run notion_export_research first.")
        except NotionAPIError as exc:
            return _err(f"Notion API error [{exc.notion_status}]", str(exc))
        except Exception as exc:
            return _err("Query failed", str(exc))

    return await asyncio.to_thread(_query)


async def _notion_get_database_info(args: dict) -> dict:
    def _info() -> dict:
        try:
            schema = load_schema("ZnSe Research Knowledge Base")
            db_id  = schema.get("notion_database_id", "")
            if not db_id:
                return _err("Notion database not set up yet. Run notion_export_research first.")
            info = get_database(db_id)
            title = info.get("title", [{}])[0].get("plain_text", "")
            props = list(info.get("properties", {}).keys())
            url   = info.get("url", f"https://notion.so/{db_id.replace('-', '')}")
            return _ok({
                "database_id": db_id,
                "title": title,
                "url": url,
                "property_names": props,
                "property_count": len(props),
                "local_schema_fields": len(schema.get("fields", [])),
            })
        except SchemaMissingError:
            return _err("Notion schema not found. Run notion_export_research first.")
        except NotionAPIError as exc:
            return _err(f"Notion API error [{exc.notion_status}]", str(exc))
        except Exception as exc:
            return _err("Database info fetch failed", str(exc))

    return await asyncio.to_thread(_info)


async def _notion_update_paper(args: dict) -> dict:
    page_id = args["page_id"]
    fields  = args.get("fields", {})

    if not fields:
        return _err("fields cannot be empty")

    FIELD_BUILDERS = {
        "Title":           lambda v: _title(v),
        "Year":            lambda v: _number(int(v)),
        "DOI":             lambda v: _url(v),
        "Abstract":        lambda v: _rtext(v),
        "Relevance Score": lambda v: _number(float(v)),
    }

    def _update() -> dict:
        try:
            props: dict = {}
            for field, value in fields.items():
                builder = FIELD_BUILDERS.get(field)
                if builder is None:
                    return _err(
                        f"Field '{field}' is not updatable via notion_update_paper. "
                        f"Allowed: {list(FIELD_BUILDERS.keys())}"
                    )
                props[field] = builder(value)

            result = update_page(page_id, props)
            return _ok({
                "page_id": page_id,
                "updated_fields": list(fields.keys()),
                "notion_id": result.get("id", page_id),
            })
        except NotionAPIError as exc:
            return _err(f"Notion API error [{exc.notion_status}]", str(exc))
        except Exception as exc:
            return _err("Update failed", str(exc))

    return await asyncio.to_thread(_update)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP C — RESEARCH QUERY TOOLS
# ═════════════════════════════════════════════════════════════════════════════

async def _research_knowledge_summary(args: dict) -> dict:
    wf_id: Optional[int] = args.get("workflow_id")

    def _query() -> dict:
        repo = _repo()
        try:
            wf_filter = "JOIN Paper p ON p.id = rk.paper_id AND p.workflow_id = ?" if wf_id else ""
            params = (wf_id,) if wf_id else ()

            # Category counts
            cat_rows = repo.fetch_all(
                f"""
                SELECT rk.category,
                       COUNT(*) as total_rows,
                       COUNT(DISTINCT rk.paper_id) as papers
                FROM ResearchKnowledge rk
                {wf_filter}
                GROUP BY rk.category
                ORDER BY total_rows DESC
                """,
                params,
            )
            by_category = [dict(r) for r in cat_rows]

            # Top values per category
            top_values: dict = {}
            for cat_row in by_category:
                cat = cat_row["category"]
                top = repo.fetch_all(
                    f"""
                    SELECT rk.value, COUNT(*) as cnt
                    FROM ResearchKnowledge rk
                    {wf_filter}
                    WHERE rk.category = ?
                    GROUP BY rk.value
                    ORDER BY cnt DESC LIMIT 10
                    """,
                    params + (cat,),
                )
                top_values[cat] = [{"value": r["value"], "count": r["cnt"]} for r in top]

            total_rows = sum(c["total_rows"] for c in by_category)
            total_papers = (
                repo.fetch_one(
                    "SELECT COUNT(DISTINCT paper_id) FROM ResearchKnowledge"
                    + (" JOIN Paper p ON p.id = ResearchKnowledge.paper_id WHERE p.workflow_id = ?" if wf_id else ""),
                    (wf_id,) if wf_id else (),
                )[0]
                if cat_rows else 0
            )

            return _ok({
                "workflow_id": wf_id,
                "total_knowledge_rows": total_rows,
                "papers_with_knowledge": total_papers,
                "by_category": by_category,
                "top_values_per_category": top_values,
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_query)


async def _research_find_papers_by_topic(args: dict) -> dict:
    keywords: list  = args["keywords"]
    search_in: list = args.get("search_in", ["title", "abstract"])
    limit: int      = args.get("limit", 20)

    if not keywords:
        return _err("keywords cannot be empty")

    def _search() -> dict:
        repo = _repo()
        try:
            matched_ids: set = set()
            results: list    = []

            for kw in keywords:
                kw_lower = kw.lower()
                if "title" in search_in:
                    rows = repo.fetch_all(
                        """SELECT DISTINCT id, workflow_id, title, abstract, status
                           FROM Paper WHERE LOWER(title) LIKE ? LIMIT ?""",
                        (f"%{kw_lower}%", limit),
                    )
                    for r in rows:
                        if r["id"] not in matched_ids:
                            matched_ids.add(r["id"])
                            results.append(dict(r) | {"matched_keyword": kw, "matched_in": "title"})

                if "abstract" in search_in:
                    rows = repo.fetch_all(
                        """SELECT DISTINCT id, workflow_id, title, abstract, status
                           FROM Paper WHERE LOWER(abstract) LIKE ? LIMIT ?""",
                        (f"%{kw_lower}%", limit),
                    )
                    for r in rows:
                        if r["id"] not in matched_ids:
                            matched_ids.add(r["id"])
                            results.append(dict(r) | {"matched_keyword": kw, "matched_in": "abstract"})

                if "content" in search_in:
                    rows = repo.fetch_all(
                        """SELECT DISTINCT p.id, p.workflow_id, p.title, p.abstract, p.status,
                                  pc.section_name
                           FROM Paper p JOIN PaperContent pc ON pc.paper_id = p.id
                           WHERE LOWER(pc.content) LIKE ? LIMIT ?""",
                        (f"%{kw_lower}%", limit),
                    )
                    for r in rows:
                        if r["id"] not in matched_ids:
                            matched_ids.add(r["id"])
                            results.append(dict(r) | {"matched_keyword": kw, "matched_in": f"content:{r['section_name']}"})

                if len(results) >= limit:
                    break

            return _ok({
                "keywords": keywords,
                "search_in": search_in,
                "count": len(results[:limit]),
                "papers": results[:limit],
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_search)


async def _research_get_database_stats(args: dict) -> dict:
    def _stats() -> dict:
        repo = _repo()
        try:
            row = repo.fetch_one(
                """
                SELECT
                  (SELECT COUNT(*) FROM Workflow)                        AS workflows,
                  (SELECT COUNT(*) FROM Workflow WHERE status = 'paused') AS paused_wf,
                  (SELECT COUNT(*) FROM Workflow WHERE status = 'running') AS running_wf,
                  (SELECT COUNT(*) FROM Paper)                           AS total_papers,
                  (SELECT COUNT(*) FROM Paper WHERE status = 'extracted') AS extracted,
                  (SELECT COUNT(*) FROM Paper WHERE status = 'knowledge_ready') AS knowledge_ready,
                  (SELECT COUNT(*) FROM Paper WHERE status = 'completed') AS completed_papers,
                  (SELECT COUNT(*) FROM Paper WHERE status = 'failed')    AS failed,
                  (SELECT COUNT(*) FROM PaperContent)                    AS content_sections,
                  (SELECT COUNT(*) FROM ResearchKnowledge)               AS knowledge_rows,
                  (SELECT COUNT(DISTINCT paper_id) FROM ResearchKnowledge) AS papers_with_knowledge
                FROM (SELECT 1)
                """
            )
            stats = dict(row)
            total = stats.get("total_papers", 0) or 1
            extracted = stats.get("extracted", 0) + stats.get("knowledge_ready", 0) + stats.get("completed_papers", 0)
            stats["extraction_rate_pct"] = round(extracted / total * 100, 1)
            stats["knowledge_density"] = (
                round(stats["knowledge_rows"] / max(stats["papers_with_knowledge"], 1), 1)
                if stats["knowledge_rows"] else 0
            )
            return _ok({"stats": stats})
        finally:
            repo.close()

    return await asyncio.to_thread(_stats)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP D — ANALYSIS TOOLS
# ═════════════════════════════════════════════════════════════════════════════

async def _analysis_technique_frequency(args: dict) -> dict:
    return await asyncio.to_thread(
        _analyzer().technique_frequency,
        args["category"],
        args.get("workflow_id"),
        args.get("top_n", 20),
        args.get("min_count", 2),
    )


async def _analysis_trend_report(args: dict) -> dict:
    return await asyncio.to_thread(
        _analyzer().trend_report,
        args["primary_category"],
        args.get("secondary_category"),
        args.get("filter_value"),
        args.get("workflow_id"),
        args.get("min_co_occurrence", 3),
    )


async def _analysis_find_gaps(args: dict) -> dict:
    return await asyncio.to_thread(
        _analyzer().find_gaps,
        args["category_a"],
        args["category_b"],
        args.get("known_values_a"),
        args.get("known_values_b"),
        args.get("gap_threshold", 2),
    )


async def _analysis_parameter_distribution(args: dict) -> dict:
    return await asyncio.to_thread(
        _analyzer().parameter_distribution,
        args["parameter_keywords"],
        args.get("workflow_id"),
        args.get("extract_numbers", True),
        args.get("group_by_material", True),
    )


async def _analysis_workflow_comparison(args: dict) -> dict:
    return await asyncio.to_thread(
        _analyzer().workflow_comparison,
        args["workflow_ids"],
        args.get("compare_by", "knowledge"),
    )


ANALYSIS_DB_NAME = "ZnSe Analysis Results"
ANALYSIS_SCHEMA  = [
    {"name": "Analysis Name",   "type": "title"},
    {"name": "Type",            "type": "select",
     "options": ["Trend", "Gap", "Parameter", "Comparison", "Frequency"]},
    {"name": "Date Run",        "type": "date"},
    {"name": "Workflows Scope", "type": "rich_text"},
    {"name": "Key Finding",     "type": "rich_text"},
    {"name": "Full Results",    "type": "rich_text"},
    {"name": "Action Items",    "type": "rich_text"},
]

async def _analysis_save_to_notion(args: dict) -> dict:
    name          = args["name"]
    analysis_type = args["type"]
    key_finding   = args["key_finding"]
    results       = args.get("results", {})
    action_items  = args.get("action_items", "")
    workflow_ids  = args.get("workflow_ids", [])

    def _save() -> dict:
        try:
            # Ensure analysis schema + DB exist
            try:
                schema = load_schema(ANALYSIS_DB_NAME)
            except SchemaMissingError:
                schema = create_schema(ANALYSIS_DB_NAME, ANALYSIS_SCHEMA)

            db_id = schema.get("notion_database_id", "")
            if not db_id:
                from notion.notion_client import create_database
                from notion.schema_manager import schema_to_notion_properties
                notion_props = schema_to_notion_properties(schema)
                result = create_database(NOTION_PAGE_ID, ANALYSIS_DB_NAME, notion_props)
                db_id = result["id"]
                update_notion_id(ANALYSIS_DB_NAME, db_id)

            today = datetime.utcnow().strftime("%Y-%m-%d")
            results_json = json.dumps(results, ensure_ascii=False)[:2000]
            scope_str = (
                f"Workflows: {', '.join(str(w) for w in workflow_ids)}"
                if workflow_ids else "All workflows"
            )

            props = {
                "Analysis Name":   _title(name),
                "Type":            _select(analysis_type),
                "Date Run":        _date(today),
                "Workflows Scope": _rtext(scope_str),
                "Key Finding":     _rtext(key_finding),
                "Full Results":    _rtext(results_json),
                "Action Items":    _rtext(action_items),
            }
            page = create_page(db_id, props)
            return _ok({
                "saved": True,
                "page_id": page.get("id"),
                "database_id": db_id,
                "notion_url": f"https://notion.so/{db_id.replace('-', '')}",
            })
        except NotionAPIError as exc:
            return _err(f"Notion API error [{exc.notion_status}]", str(exc))
        except Exception as exc:
            return _err("Save to Notion failed", str(exc))

    return await asyncio.to_thread(_save)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP E — CORRECTION TOOLS
# ═════════════════════════════════════════════════════════════════════════════

async def _db_list_suspect_papers(args: dict) -> dict:
    wf_id       = args.get("workflow_id")
    issue_types = set(args.get("issue_types") or [
        "missing_doi", "missing_abstract", "bad_date", "short_abstract", "no_content"
    ])
    limit       = args.get("limit", 50)

    def _list() -> dict:
        repo = _repo()
        try:
            wf_filter = "AND p.workflow_id = ?" if wf_id else ""
            params    = (wf_id,) if wf_id else ()

            rows = repo.fetch_all(
                f"""
                SELECT p.id, p.workflow_id, p.title, p.abstract, p.doi,
                       p.status, p.created_at, p.updated_at
                FROM Paper p
                WHERE 1=1 {wf_filter}
                ORDER BY p.id LIMIT ?
                """,
                params + (limit * 3,),
            )

            # Content count per paper
            content_map = {
                r["paper_id"]: r["cnt"]
                for r in repo.fetch_all(
                    "SELECT paper_id, COUNT(*) as cnt FROM PaperContent GROUP BY paper_id"
                )
            }

            suspect = []
            for r in rows:
                d = dict(r)
                issues: list = []

                if "missing_doi" in issue_types and not d.get("doi"):
                    issues.append("missing_doi")
                if "missing_abstract" in issue_types and not d.get("abstract"):
                    issues.append("missing_abstract")
                if "short_abstract" in issue_types:
                    ab = (d.get("abstract") or "")
                    if 0 < len(ab) < 80:
                        issues.append("short_abstract")
                if "no_content" in issue_types and content_map.get(d["id"], 0) == 0:
                    issues.append("no_content")
                if "bad_date" in issue_types:
                    ca = d.get("created_at", "")
                    if ca and (ca.startswith("1970") or ca.startswith("0000")):
                        issues.append("bad_date")

                if issues:
                    d["issues"] = issues
                    d["abstract_length"] = len(d.get("abstract") or "")
                    d["content_sections"] = content_map.get(d["id"], 0)
                    suspect.append(d)
                    if len(suspect) >= limit:
                        break

            return _ok({
                "workflow_id": wf_id,
                "issue_types_checked": list(issue_types),
                "suspect_count": len(suspect),
                "papers": suspect,
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_list)


async def _db_update_paper(args: dict) -> dict:
    paper_id = args["paper_id"]
    fields   = args.get("fields", {})

    if not fields:
        return _err("fields cannot be empty")

    immutable = set(fields.keys()) & PAPER_IMMUTABLE_FIELDS
    if immutable:
        return _err(f"Fields {sorted(immutable)} are immutable and cannot be updated.")

    invalid = set(fields.keys()) - PAPER_WRITABLE_FIELDS
    if invalid:
        return _err(
            f"Fields {sorted(invalid)} are not in the writable allowlist.",
            f"Writable: {sorted(PAPER_WRITABLE_FIELDS)}",
        )

    def _update() -> dict:
        repo = _repo()
        try:
            paper = repo.fetch_one(
                "SELECT id, workflow_id, " + ", ".join(PAPER_WRITABLE_FIELDS) + " FROM Paper WHERE id = ?",
                (paper_id,),
            )
            if not paper:
                return _err(f"Paper {paper_id} not found")

            old_values = {f: dict(paper).get(f) for f in fields.keys()}

            set_clauses = ", ".join(f"{f} = ?" for f in fields.keys())
            values      = list(fields.values())

            with repo.transaction() as cursor:
                cursor.execute(
                    f"UPDATE Paper SET {set_clauses}, updated_at = ? WHERE id = ?",
                    values + [datetime.utcnow().isoformat(), paper_id],
                )

            _audit_log("db_update_paper", {
                "paper_id":      paper_id,
                "workflow_id":   dict(paper)["workflow_id"],
                "fields_changed": list(fields.keys()),
                "old_values":    old_values,
                "new_values":    fields,
            })

            return _ok({
                "paper_id":      paper_id,
                "fields_updated": list(fields.keys()),
                "old_values":    old_values,
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_update)


async def _db_update_workflow_config(args: dict) -> dict:
    wf_id  = args["workflow_id"]
    fields = args.get("fields", {})

    if not fields:
        return _err("fields cannot be empty")

    invalid = set(fields.keys()) - CONFIG_WRITABLE_FIELDS
    if invalid:
        return _err(
            f"Fields {sorted(invalid)} are not writable on WorkflowResearchConfig.",
            f"Writable: {sorted(CONFIG_WRITABLE_FIELDS)}",
        )

    def _update() -> dict:
        repo = _repo()
        try:
            config = repo.fetch_one(
                "SELECT * FROM WorkflowResearchConfig WHERE workflow_id = ?", (wf_id,)
            )
            if not config:
                return _err(f"WorkflowResearchConfig not found for workflow {wf_id}")

            old_values = {f: dict(config).get(f) for f in fields.keys()}
            set_clauses = ", ".join(f"{f} = ?" for f in fields.keys())

            with repo.transaction() as cursor:
                cursor.execute(
                    f"UPDATE WorkflowResearchConfig SET {set_clauses} WHERE workflow_id = ?",
                    list(fields.values()) + [wf_id],
                )

            _audit_log("db_update_workflow_config", {
                "workflow_id":   wf_id,
                "fields_changed": list(fields.keys()),
                "old_values":    old_values,
                "new_values":    fields,
            })

            return _ok({
                "workflow_id":   wf_id,
                "fields_updated": list(fields.keys()),
                "old_values":    old_values,
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_update)


async def _db_bulk_fix(args: dict) -> dict:
    wf_id         = args["workflow_id"]
    field         = args["field"]
    match_pattern = args["match_pattern"]
    replacement   = args["replacement"]
    dry_run       = args.get("dry_run", True)

    if field not in PAPER_WRITABLE_FIELDS:
        return _err(
            f"Field '{field}' is not in the writable allowlist.",
            f"Writable: {sorted(PAPER_WRITABLE_FIELDS)}",
        )

    try:
        compiled = re.compile(match_pattern)
    except re.error as exc:
        return _err(f"Invalid regex pattern: {exc}")

    def _bulk() -> dict:
        repo = _repo()
        try:
            rows = repo.fetch_all(
                f"SELECT id, {field} FROM Paper WHERE workflow_id = ? AND {field} IS NOT NULL",
                (wf_id,),
            )

            affected = []
            for r in rows:
                current_val = r[field] or ""
                if compiled.search(current_val):
                    new_val = compiled.sub(replacement, current_val)
                    affected.append({
                        "paper_id":  r["id"],
                        "old_value": current_val[:200],
                        "new_value": new_val[:200],
                    })

            if dry_run:
                return _ok({
                    "dry_run":       True,
                    "workflow_id":   wf_id,
                    "field":         field,
                    "match_pattern": match_pattern,
                    "replacement":   replacement,
                    "would_affect":  len(affected),
                    "preview":       affected[:20],
                    "message": (
                        f"DRY RUN: {len(affected)} rows would be updated. "
                        "Call with dry_run=false to apply."
                    ),
                })

            # Apply the fix
            with repo.transaction() as cursor:
                for item in affected:
                    cursor.execute(
                        f"UPDATE Paper SET {field} = ?, updated_at = ? WHERE id = ?",
                        (item["new_value"], datetime.utcnow().isoformat(), item["paper_id"]),
                    )

            _audit_log("db_bulk_fix", {
                "workflow_id":   wf_id,
                "field":         field,
                "match_pattern": match_pattern,
                "replacement":   replacement,
                "rows_updated":  len(affected),
            })

            return _ok({
                "dry_run":      False,
                "workflow_id":  wf_id,
                "field":        field,
                "rows_updated": len(affected),
                "message": f"Applied: {len(affected)} rows updated.",
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_bulk)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP F — REVIEW GENERATION TOOLS
# ═════════════════════════════════════════════════════════════════════════════

async def _review_run_knowledge_extraction(args: dict) -> dict:
    if not await _check_shani():
        return _err("SHANI API not running.")

    workflow_ids     = args["workflow_ids"]
    min_papers       = args.get("min_extracted_papers", 10)
    stop_after       = args.get("stop_after", "S5")

    if stop_after not in ("S5", "S5_5"):
        return _err("stop_after must be 'S5' or 'S5_5'")

    dispatched: list = []
    skipped:    list = []

    for wf_id in workflow_ids:
        # Validate preconditions
        status_result = await _shani_get(f"/workflows/{wf_id}/status")
        if status_result.get("status") == "error":
            skipped.append({"id": wf_id, "reason": status_result.get("error", "API error")})
            continue

        wf = status_result.get("workflow", {})
        if wf.get("status") != "paused":
            skipped.append({
                "id": wf_id,
                "reason": f"status='{wf.get('status')}' — must be 'paused' to run",
            })
            continue

        # Check extracted paper count
        def _count_extracted(wid):
            repo = _repo()
            try:
                row = repo.fetch_one(
                    """SELECT COUNT(*) FROM Paper
                       WHERE workflow_id = ? AND status IN ('extracted','knowledge_ready','completed')""",
                    (wid,),
                )
                return row[0]
            finally:
                repo.close()

        extracted_count = await asyncio.to_thread(_count_extracted, wf_id)
        if extracted_count < min_papers:
            skipped.append({
                "id": wf_id,
                "reason": (
                    f"Only {extracted_count} extracted papers — "
                    f"minimum is {min_papers}"
                ),
            })
            continue

        run_result = await _shani_post(
            f"/workflows/{wf_id}/run",
            {"stop_after_stage": stop_after},
        )
        if run_result.get("status") == "error":
            skipped.append({"id": wf_id, "reason": run_result.get("error", "dispatch failed")})
        else:
            dispatched.append({"id": wf_id, "stop_after": stop_after})

    return _ok({
        "dispatched":   dispatched,
        "skipped":      skipped,
        "dispatched_count": len(dispatched),
        "skipped_count":    len(skipped),
        "message": (
            f"Dispatched S5 on {len(dispatched)} workflows. "
            f"Poll shani_get_all_status to monitor."
        ),
    })


async def _review_generate_direction(args: dict) -> dict:
    """Trigger S5_5. Returns existing direction if already generated."""
    if not await _check_shani():
        return _err("SHANI API not running.")

    wf_id = args["workflow_id"]
    status_result = await _shani_get(f"/workflows/{wf_id}/status")
    if status_result.get("status") == "error":
        return status_result

    # Check if S5_5 already completed
    stages = status_result.get("stages", [])
    s5_5_stage = next((s for s in stages if s["stage_name"] == "S5_5"), None)
    if s5_5_stage and s5_5_stage.get("status") == "completed":
        return _ok({
            "workflow_id": wf_id,
            "s5_5_status": "already_completed",
            "message": "S5_5 already completed. Use review_get_draft to retrieve direction.",
        })

    # Check S5 completed first
    s5_stage = next((s for s in stages if s["stage_name"] == "S5"), None)
    if not s5_stage or s5_stage.get("status") != "completed":
        return _err(
            "S5 must be completed before generating review direction.",
            "Run review_run_knowledge_extraction first.",
        )

    # Verify workflow is paused before triggering
    wf = status_result.get("workflow", {})
    if wf.get("status") != "paused":
        return _err(
            f"Workflow must be 'paused' to run S5_5. Current: {wf.get('status')}",
        )

    result = await _shani_post(f"/workflows/{wf_id}/run", {"stop_after_stage": "S5_5"})
    if result.get("status") == "error":
        return result
    return _ok({
        "workflow_id": wf_id,
        "dispatched": "S5_5",
        "message": "Review direction generation (S5_5) started. Poll shani_get_status.",
    })


async def _review_draft_sections(args: dict) -> dict:
    """Trigger S6 draft_sections."""
    if not await _check_shani():
        return _err("SHANI API not running.")

    wf_id = args["workflow_id"]
    force = args.get("force", False)

    status_result = await _shani_get(f"/workflows/{wf_id}/status")
    if status_result.get("status") == "error":
        return status_result

    stages = status_result.get("stages", [])
    wf     = status_result.get("workflow", {})

    s6_stage = next((s for s in stages if s["stage_name"] == "S6"), None)
    if s6_stage and s6_stage.get("status") == "completed" and not force:
        return _ok({
            "workflow_id": wf_id,
            "s6_status": "already_completed",
            "message": "S6 already completed. Use force=true to re-draft.",
        })

    s5_5_stage = next((s for s in stages if s["stage_name"] == "S5_5"), None)
    if not s5_5_stage or s5_5_stage.get("status") != "completed":
        return _err(
            "S5_5 must be completed before drafting sections.",
            "Run review_generate_direction first.",
        )

    if wf.get("status") != "paused":
        return _err(f"Workflow must be 'paused' to run S6. Current: {wf.get('status')}")

    result = await _shani_post(f"/workflows/{wf_id}/run", {"stop_after_stage": "S6"})
    if result.get("status") == "error":
        return result
    return _ok({
        "workflow_id": wf_id,
        "dispatched": "S6",
        "message": "Section drafting (S6) started. Poll shani_get_status.",
    })


async def _review_synthesize_final(args: dict) -> dict:
    """Trigger S7 final synthesis, optionally export to Notion."""
    if not await _check_shani():
        return _err("SHANI API not running.")

    wf_id           = args["workflow_id"]
    export_to_notion = args.get("export_to_notion", False)

    status_result = await _shani_get(f"/workflows/{wf_id}/status")
    if status_result.get("status") == "error":
        return status_result

    stages = status_result.get("stages", [])
    wf     = status_result.get("workflow", {})

    s6_stage = next((s for s in stages if s["stage_name"] == "S6"), None)
    if not s6_stage or s6_stage.get("status") != "completed":
        return _err(
            "S6 must be completed before final synthesis.",
            "Run review_draft_sections first.",
        )

    if wf.get("status") != "paused":
        return _err(f"Workflow must be 'paused' to run S7. Current: {wf.get('status')}")

    result = await _shani_post(f"/workflows/{wf_id}/run", {"stop_after_stage": "S7"})
    if result.get("status") == "error":
        return result

    response = _ok({
        "workflow_id":     wf_id,
        "dispatched":      "S7",
        "export_to_notion": export_to_notion,
        "message": (
            "Final synthesis (S7) started. Poll shani_get_status. "
            + ("Will auto-export to Notion when completed — poll and then call notion_export_research."
               if export_to_notion else "")
        ),
    })
    return response


async def _review_get_draft(args: dict) -> dict:
    """Retrieve draft output from a completed review stage via SQLite."""
    wf_id = args["workflow_id"]
    stage = args["stage"]  # S5_5 | S6 | S7

    def _fetch() -> dict:
        repo = _repo()
        try:
            # Verify stage completed
            stage_row = repo.fetch_one(
                "SELECT id, status, started_at, ended_at FROM Stage "
                "WHERE workflow_id = ? AND stage_name = ? ORDER BY id DESC LIMIT 1",
                (wf_id, stage),
            )
            if not stage_row:
                return _err(
                    f"Stage {stage} has not run for workflow {wf_id}.",
                    f"Run the appropriate review_ tool first.",
                )
            if stage_row["status"] != "completed":
                return _err(
                    f"Stage {stage} status is '{stage_row['status']}' — not yet completed."
                )

            # Retrieve content from PaperContent with review-type section names
            # SHANI stores review output in PaperContent with synthetic paper entries
            # or in dedicated tables depending on version. We check both.

            # Strategy 1: look for PaperContent rows created after stage started
            content_rows = repo.fetch_all(
                """
                SELECT pc.section_name, pc.content, p.title AS paper_title
                FROM PaperContent pc
                JOIN Paper p ON p.id = pc.paper_id
                WHERE p.workflow_id = ?
                  AND pc.section_name IN (
                    'review_direction','review_outline','draft_introduction',
                    'draft_methodology','draft_results','draft_discussion',
                    'draft_conclusion','final_synthesis','review_abstract',
                    'direction','outline','synthesis'
                  )
                ORDER BY p.id DESC, pc.id DESC
                LIMIT 50
                """,
                (wf_id,),
            )

            sections = {}
            for r in content_rows:
                sections[r["section_name"]] = r["content"]

            if not sections:
                # Strategy 2: return the most recent raw content sections
                # as a fallback, noting the user should check SHANI internals
                return _ok({
                    "workflow_id":  wf_id,
                    "stage":        stage,
                    "stage_status": stage_row["status"],
                    "sections":     {},
                    "note": (
                        "No review-specific section names found in PaperContent. "
                        "The stage completed successfully but output may be stored in a "
                        "SHANI-internal table not yet exposed. "
                        "Check the SHANI project for ReviewDraft or SynthesizedPaper tables."
                    ),
                })

            return _ok({
                "workflow_id":  wf_id,
                "stage":        stage,
                "stage_status": stage_row["status"],
                "started_at":   stage_row["started_at"],
                "ended_at":     stage_row["ended_at"],
                "section_count": len(sections),
                "sections":     sections,
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_fetch)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP G — VIDUR CLASSIFIER TOOLS
# ═════════════════════════════════════════════════════════════════════════════

async def _vidur_classify(args: dict) -> dict:
    """
    Run the full VIDUR pipeline on a local file.
    extractor → auto_detector → router → parser
    All I/O is synchronous; offloaded to thread to avoid blocking the event loop.
    """
    file_path = args.get("file_path", "").strip()
    if not file_path:
        return _err("Missing required argument: file_path")

    def _run() -> dict:
        try:
            import os as _os
            if not _os.path.isfile(file_path):
                return _err(f"File not found: {file_path}")

            # Step 1 — extract
            from extractor import extract
            data = extract(file_path)

            # Step 2 — detect
            from auto_detector import detect
            detection = detect(data)

            # Step 3 — route + parse
            from router import route
            result = route(detection, data)

            # Sanitise numpy arrays that can't be JSON-serialised
            # (parsers return plain Python lists already, but be defensive)
            parsed = result.get("parsed_data")
            if parsed:
                for key in ("axis", "intensity"):
                    if key in parsed:
                        parsed[key] = [
                            float(v) for v in parsed[key]
                            if v != "..." and v is not None
                        ]

            return _ok({
                "technique":   result.get("technique", "Unknown"),
                "confidence":  round(result.get("confidence", 0.0), 4),
                "signals":     result.get("signals", []),
                "parsed_data": parsed,
                "error":       result.get("error"),
            })

        except ImportError as exc:
            return _err(
                "VIDUR import failed — check VIDUR_ROOT in mcp_server.py",
                str(exc),
            )
        except Exception as exc:
            return _err("VIDUR pipeline error", str(exc))

    return await asyncio.to_thread(_run)


async def _vidur_list_techniques(_args: dict) -> dict:
    """Return metadata about every parser VIDUR registers."""
    techniques = [
        {
            "technique":  "XRD",
            "description": "X-Ray Diffraction — powder/single-crystal patterns",
            "extensions": [".xrdml", ".raw", ".xy", ".dat", ".asc"],
            "axis":        "2Theta (degrees, 5–90°)",
            "strong_keywords": ["2theta", "xrd", "diffraction", "bragg", "d-spacing"],
        },
        {
            "technique":  "UV-Vis",
            "description": "UV-Visible Spectroscopy — absorbance/transmittance",
            "extensions": [".sp", ".abs", ".dsp", ".spc", ".csv", ".txt"],
            "axis":        "Wavelength_nm (200–1100 nm)",
            "strong_keywords": ["absorbance", "wavelength", "uv-vis", "transmittance", "nm"],
        },
        {
            "technique":  "SEM_EDX",
            "description": "Scanning Electron Microscopy / Energy Dispersive X-ray",
            "extensions": [".emsa", ".msa", ".spx", ".eds", ".spc"],
            "axis":        "Energy_keV (0–20 keV)",
            "strong_keywords": ["keV", "eds", "edx", "sem", "weight %", "atomic %"],
        },
        {
            "technique":  "Raman",
            "description": "Raman Spectroscopy — vibrational/rotational modes",
            "extensions": [".wdf", ".spc", ".txt", ".csv", ".dat"],
            "axis":        "RamanShift_cm-1 (100–3500 cm⁻¹)",
            "strong_keywords": ["raman", "cm-1", "wavenumber", "raman shift", "stokes"],
        },
    ]
    return _ok({
        "count":      len(techniques),
        "techniques": techniques,
    })


async def _vidur_health(_args: dict) -> dict:
    """
    Verify that VIDUR modules and all parsers import correctly.
    Does not touch the filesystem beyond the import.
    """
    def _check() -> dict:
        results = {}
        overall = True

        # Core modules
        for module_name in ("extractor", "auto_detector", "router"):
            try:
                __import__(module_name)
                results[module_name] = "ok"
            except Exception as exc:
                results[module_name] = f"FAILED: {exc}"
                overall = False

        # Parser modules
        for parser in ("parsers.xrd", "parsers.uvvis", "parsers.sem_eds", "parsers.raman"):
            short = parser.split(".")[-1]
            try:
                __import__(parser)
                results[f"parser:{short}"] = "ok"
            except Exception as exc:
                results[f"parser:{short}"] = f"FAILED: {exc}"
                overall = False

        return _ok({
            "ready":   overall,
            "modules": results,
            "note": (
                "All modules healthy — VIDUR is ready."
                if overall else
                "One or more modules failed to import. "
                "Check VIDUR_ROOT path in mcp_server.py and ensure dependencies "
                "(numpy, pypdf, Pillow) are installed in the active venv."
            ),
        })

    return await asyncio.to_thread(_check)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP H — VISHWAKARMA (QUANTUM ESPRESSO)
# ═════════════════════════════════════════════════════════════════════════════

_QE_WORKDIR  = os.environ.get("VISHWAKARMA_WORKDIR", "/tmp/vishwakarma_jobs")
_QE_BIN_DIR  = os.environ.get("QE_BIN_DIR",          "/usr/local/bin")
_QE_PSEUDO   = os.environ.get("QE_PSEUDO_DIR",        "/mnt/d/pseudo")


async def _vishwakarma_health(_args: dict) -> dict:
    def _check() -> dict:
        try:
            from vishwakarma import runner as _r
            from vishwakarma import input_generator   # noqa: F401
            from vishwakarma import output_parser      # noqa: F401
            from vishwakarma import pseudo_manager     # noqa: F401
            from vishwakarma import workflow           # noqa: F401
        except ImportError as exc:
            return _err("Vishwakarma modules failed to import", str(exc))

        binaries = _r.check_binaries(_QE_BIN_DIR)
        any_found = any(v is not None for v in binaries.values())
        return _ok({
            "ready":    any_found,
            "bin_dir":  _QE_BIN_DIR,
            "workdir":  _QE_WORKDIR,
            "pseudo_dir": _QE_PSEUDO,
            "binaries": binaries,
            "note": (
                "Set QE_BIN_DIR env var if binaries are in a non-standard location."
                if not any_found else
                f"{sum(1 for v in binaries.values() if v)} / {len(binaries)} QE binaries found."
            ),
        })
    return await asyncio.to_thread(_check)


async def _vishwakarma_generate_input(args: dict) -> dict:
    def _gen() -> dict:
        try:
            from vishwakarma import input_generator as ig
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))

        calc_type   = args.get("calc_type", "scf")
        structure   = args.get("structure", {})
        calc_params = args.get("calc_params", {})
        ph_params   = args.get("phonon_params", {})

        try:
            if calc_type == "scf":
                text = ig.scf(structure, calc_params)
            elif calc_type == "nscf":
                text = ig.nscf(structure, calc_params)
            elif calc_type == "relax":
                text = ig.relax(structure, calc_params, vc=False)
            elif calc_type == "vc-relax":
                text = ig.relax(structure, calc_params, vc=True)
            elif calc_type == "bands":
                text = ig.bands(structure, calc_params)
            elif calc_type == "dos":
                prefix = structure.get("prefix", "pwscf")
                outdir = calc_params.get("outdir", "./out")
                text = ig.dos(prefix, outdir, **{k: ph_params[k] for k in
                              ("emin","emax","deltaE","fildos") if k in ph_params})
            elif calc_type == "projwfc":
                prefix = structure.get("prefix", "pwscf")
                outdir = calc_params.get("outdir", "./out")
                text = ig.projwfc(prefix, outdir)
            elif calc_type == "pp":
                prefix = structure.get("prefix", "pwscf")
                outdir = calc_params.get("outdir", "./out")
                text = ig.pp(prefix, outdir,
                             plot_num=ph_params.get("plot_num", 0),
                             fileout=ph_params.get("fileout", "charge.xsf"))
            elif calc_type == "phonon":
                prefix = structure.get("prefix", "pwscf")
                outdir = calc_params.get("outdir", "./out")
                text = ig.phonon(prefix, outdir,
                                 qpoints=ph_params.get("qpoints"),
                                 ldisp=ph_params.get("ldisp", False),
                                 nq=tuple(ph_params.get("nq", [4,4,4])),
                                 epsil=ph_params.get("epsil", False),
                                 lraman=ph_params.get("lraman", False))
            elif calc_type == "hp":
                prefix = structure.get("prefix", "pwscf")
                outdir = calc_params.get("outdir", "./out")
                text = ig.hp(prefix, outdir,
                             nq=tuple(ph_params.get("nq", [2,2,2])))
            elif calc_type == "cp":
                text = ig.cp(structure, calc_params)
            else:
                return _err(f"Unknown calc_type: {calc_type}")

            return _ok({"calc_type": calc_type, "input_text": text,
                        "line_count": text.count("\n")})
        except Exception as exc:
            return _err(f"Input generation failed for {calc_type}", str(exc))

    return await asyncio.to_thread(_gen)


async def _vishwakarma_run_scf(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import workflow as wf
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        try:
            result = wf.scf_only(
                structure=args["structure"],
                calc_params=args["calc_params"],
                label=args.get("label", "scf"),
                workdir=_QE_WORKDIR,
                bin_dir=_QE_BIN_DIR,
                timeout=args.get("timeout", 3600),
                mpi_np=args.get("mpi_np", 1),
            )
            return _ok(result)
        except Exception as exc:
            return _err("SCF calculation failed", str(exc))
    return await asyncio.to_thread(_run)


async def _vishwakarma_run_relax(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import workflow as wf
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        try:
            result = wf.relax_then_scf(
                structure=args["structure"],
                calc_params=args["calc_params"],
                vc=args.get("vc_relax", False),
                label=args.get("label", "relax"),
                workdir=_QE_WORKDIR,
                bin_dir=_QE_BIN_DIR,
                timeout=args.get("timeout", 7200),
                mpi_np=args.get("mpi_np", 1),
            )
            return _ok(result)
        except Exception as exc:
            return _err("Relaxation failed", str(exc))
    return await asyncio.to_thread(_run)


async def _vishwakarma_run_bands(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import workflow as wf
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        try:
            result = wf.band_structure(
                structure=args["structure"],
                calc_params=args["calc_params"],
                kpath=args.get("kpath"),
                label=args.get("label", "bands"),
                workdir=_QE_WORKDIR,
                bin_dir=_QE_BIN_DIR,
                timeout=args.get("timeout", 3600),
                mpi_np=args.get("mpi_np", 1),
            )
            return _ok(result)
        except Exception as exc:
            return _err("Band structure calculation failed", str(exc))
    return await asyncio.to_thread(_run)


async def _vishwakarma_run_dos(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import workflow as wf
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        try:
            result = wf.dos_workflow(
                structure=args["structure"],
                calc_params=args["calc_params"],
                dense_kmesh=args.get("dense_kmesh"),
                emin=args.get("emin", -20.0),
                emax=args.get("emax",  20.0),
                label=args.get("label", "dos"),
                workdir=_QE_WORKDIR,
                bin_dir=_QE_BIN_DIR,
                timeout=args.get("timeout", 7200),
                mpi_np=args.get("mpi_np", 1),
            )
            return _ok(result)
        except Exception as exc:
            return _err("DOS calculation failed", str(exc))
    return await asyncio.to_thread(_run)


async def _vishwakarma_run_phonon(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import workflow as wf
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        try:
            nq = tuple(args.get("nq", [4, 4, 4]))
            result = wf.phonon_workflow(
                structure=args["structure"],
                calc_params=args["calc_params"],
                qpoints=args.get("qpoints"),
                ldisp=args.get("ldisp", True),
                nq=nq,
                epsil=args.get("epsil", True),
                label=args.get("label", "phonon"),
                workdir=_QE_WORKDIR,
                bin_dir=_QE_BIN_DIR,
                timeout=args.get("timeout", 14400),
                mpi_np=args.get("mpi_np", 1),
                recover=args.get("recover", False),
                existing_scf_job_id=args.get("existing_scf_job_id"),
            )
            return _ok(result)
        except Exception as exc:
            return _err("Phonon calculation failed", str(exc))
    return await asyncio.to_thread(_run)


async def _vishwakarma_run_neb(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import input_generator as ig
            from vishwakarma import runner as r
            from vishwakarma import output_parser as op
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        try:
            ini = args["initial_structure"]
            fin = args["final_structure"]
            cp  = args["calc_params"]
            images = [ini, fin]

            neb_input = ig.neb(
                images=images,
                calc_params=cp,
                num_of_images=args.get("num_images", 7),
                ci_scheme=args.get("ci_scheme", "auto"),
                opt_scheme=args.get("opt_scheme", "broyden"),
                nstep_path=args.get("nstep_path", 200),
            )
            jid = r.create_job(args.get("label","neb"), "neb", neb_input,
                               _QE_WORKDIR, args.get("mpi_np", 1))
            status = r.run_job(jid, _QE_WORKDIR, args.get("timeout", 28800), _QE_BIN_DIR)
            out    = r.get_output(jid, _QE_WORKDIR)
            parsed = op.parse_neb(out)
            return _ok({"job_id": jid, "status": status, "parsed": parsed})
        except Exception as exc:
            return _err("NEB calculation failed", str(exc))
    return await asyncio.to_thread(_run)


async def _vishwakarma_run_hp(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import input_generator as ig
            from vishwakarma import runner as r
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        try:
            hp_input = ig.hp(
                prefix=args["prefix"],
                outdir=args["outdir"],
                nq=tuple(args.get("nq", [2, 2, 2])),
            )
            jid = r.create_job(args.get("job_label", "hp"), "hp", hp_input,
                               _QE_WORKDIR, args.get("mpi_np", 1))
            status = r.run_job(jid, _QE_WORKDIR, args.get("timeout", 7200), _QE_BIN_DIR)
            out    = r.get_output(jid, _QE_WORKDIR)
            # Extract U values from output
            import re as _re
            u_vals = _re.findall(r"Hubbard U\s*\(\w+\)\s*=\s*([-\d.]+)", out)
            return _ok({
                "job_id":     jid,
                "status":     status,
                "u_values_ev": [float(u) for u in u_vals],
                "note": "U values in eV. Use in DFT+U via hubbard_u in calc_params.",
            })
        except Exception as exc:
            return _err("HP (Hubbard U) calculation failed", str(exc))
    return await asyncio.to_thread(_run)


async def _vishwakarma_parse_output(args: dict) -> dict:
    def _parse() -> dict:
        try:
            from vishwakarma import output_parser as op
            from vishwakarma import runner as r
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))

        source = args.get("source", "job_id")
        code   = args.get("code", "pw")

        if source == "job_id":
            job_id = args.get("job_id", "")
            if not job_id:
                return _err("job_id required when source=job_id")
            text = r.get_output(job_id, _QE_WORKDIR)
        else:
            fp = args.get("file_path", "")
            if not fp or not os.path.isfile(fp):
                return _err(f"File not found: {fp}")
            with open(fp, errors="replace") as f:
                text = f.read()

        if not text:
            return _err("Output file is empty or not found")

        try:
            parsed = op.parse(text, code)
            return _ok({"code": code, "parsed": parsed})
        except Exception as exc:
            return _err("Parse failed", str(exc))

    return await asyncio.to_thread(_parse)


async def _vishwakarma_list_pseudopotentials(args: dict) -> dict:
    def _list() -> dict:
        try:
            from vishwakarma import pseudo_manager as pm
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))

        dirs = args.get("pseudo_dirs") or [_QE_PSEUDO]
        pseudos = pm.discover(dirs)
        structure = args.get("structure")

        result = _ok({
            "pseudo_dirs": dirs,
            "total_found": len(pseudos),
            "pseudopotentials": pseudos[:100],   # cap output
        })

        if structure:
            recs = pm.list_for_structure(
                structure, dirs,
                preferred_functional=args.get("preferred_functional", "pbe"),
                preferred_type=args.get("preferred_type", "us"),
            )
            result["structure_check"] = recs

        return result

    return await asyncio.to_thread(_list)


async def _vishwakarma_get_job_status(args: dict) -> dict:
    def _get() -> dict:
        try:
            from vishwakarma import runner as r
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        job_id = args.get("job_id", "")
        status = r.get_job_status(job_id, _QE_WORKDIR)
        return _ok(status)
    return await asyncio.to_thread(_get)


async def _vishwakarma_list_jobs(args: dict) -> dict:
    def _list() -> dict:
        try:
            from vishwakarma import runner as r
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        sf = args.get("status_filter", "all")
        jobs = r.list_jobs(
            workdir=_QE_WORKDIR,
            limit=args.get("limit", 20),
            status_filter=None if sf == "all" else sf,
        )
        return _ok({"count": len(jobs), "jobs": jobs})
    return await asyncio.to_thread(_list)


# ═════════════════════════════════════════════════════════════════════════════
# META — BRAHM SYSTEM OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

async def _brahm_overview(_args: dict) -> dict:
    """
    Live system map: agents, tool groups, infra health — one call, full picture.
    Safe to call at any time; read-only, no side effects.
    """

    # ── 1. Infrastructure health probes ──────────────────────────────────────
    shani_up = await _check_shani()
    db_exists = Path(DB_PATH).exists()

    notion_status = "unknown"
    notion_db_id  = None
    try:
        schema = load_schema("ZnSe Research Knowledge Base")
        notion_db_id  = schema.get("notion_database_id", "")
        notion_status = "schema_loaded"
    except SchemaMissingError:
        notion_status = "schema_missing"
    except Exception as exc:
        notion_status = f"error: {exc}"

    vidur_ok = True
    vidur_detail: dict = {}
    def _probe_vidur():
        nonlocal vidur_ok
        for mod in ("extractor", "auto_detector", "router",
                    "parsers.xrd", "parsers.uvvis", "parsers.sem_eds", "parsers.raman"):
            try:
                __import__(mod)
                vidur_detail[mod] = "ok"
            except Exception as exc:
                vidur_detail[mod] = f"FAILED: {exc}"
                vidur_ok = False

    import asyncio as _asyncio
    await _asyncio.to_thread(_probe_vidur)

    # DB quick stats
    db_stats: dict = {}
    if db_exists:
        def _db_probe():
            try:
                repo = _repo()
                row = repo.fetch_one("SELECT COUNT(*) AS n FROM Paper", ())
                db_stats["papers"] = row["n"] if row else 0
                row2 = repo.fetch_one("SELECT COUNT(*) AS n FROM Workflow", ())
                db_stats["workflows"] = row2["n"] if row2 else 0
                repo.close()
            except Exception as exc:
                db_stats["error"] = str(exc)
        await _asyncio.to_thread(_db_probe)

    # ── 2. Static system map ──────────────────────────────────────────────────
    agents = [
        {
            "name":        "SHANI",
            "role":        "Scientific literature pipeline — search, ingest, extract, review",
            "status":      "api_up" if shani_up else "api_down",
            "api":         SHANI_BASE,
            "db_path":     DB_PATH,
            "db_exists":   db_exists,
            "db_stats":    db_stats,
            "note": (
                "API must be running for Group A and Group F tools. "
                "Groups C/D/E query SQLite directly and work without the API."
            ),
        },
        {
            "name":    "Chitragupta",
            "role":    "Voice-first Notion integration — logging, schema, export",
            "status":  "imported",
            "notion":  {
                "schema_status": notion_status,
                "database_id":   (notion_db_id[:8] + "...") if notion_db_id else None,
            },
            "note": "Notion API key must be in .env for Group B tools to succeed.",
        },
        {
            "name":    "VIDUR",
            "role":    "Local instrument file classifier — XRD, UV-Vis, SEM/EDS, Raman",
            "status":  "ready" if vidur_ok else "degraded",
            "root":    VIDUR_ROOT,
            "modules": vidur_detail,
            "note":    "Fully local. No API, no internet. Adjust VIDUR_ROOT if imports fail.",
        },
        {
            "name":    "Vishwakarma",
            "role":    "Quantum ESPRESSO DFT agent — SCF, relax, bands, DOS, phonon, NEB, HP",
            "status":  "unknown — call vishwakarma_health to probe",
            "bin_dir": _QE_BIN_DIR,
            "workdir": _QE_WORKDIR,
            "pseudo_dir": _QE_PSEUDO,
            "note": (
                "Set QE_BIN_DIR, VISHWAKARMA_WORKDIR, QE_PSEUDO_DIR env vars. "
                "All calculations run locally via subprocess — no internet."
            ),
        },
    ]

    tool_groups = [
        {
            "group":  "A — SHANI Pipeline",
            "prefix": "shani_* / queue_*",
            "requires": "SHANI API running",
            "tools": [
                {"name": "shani_create_workflow",        "purpose": "Create a new research workflow"},
                {"name": "shani_run_workflow",           "purpose": "Start a paused workflow"},
                {"name": "shani_batch_run",              "purpose": "Create + run up to 20 workflows concurrently"},
                {"name": "shani_get_status",             "purpose": "Full status of one workflow (stages + execution)"},
                {"name": "shani_get_all_status",         "purpose": "Summary of ALL workflows at a glance"},
                {"name": "shani_get_papers",             "purpose": "List papers collected by a workflow"},
                {"name": "shani_get_paper_content",      "purpose": "Extracted text sections for one paper"},
                {"name": "shani_extract_workflow_data",  "purpose": "Full dump: all papers + content for a workflow"},
                {"name": "shani_clear_database",         "purpose": "⚠ Wipe entire DB (confirm=true required)"},
                {"name": "shani_reset_workflow",         "purpose": "Reset failed/stuck workflow to paused"},
                {"name": "queue_add_workflow",           "purpose": "Append workflow config to local queue file"},
            ],
        },
        {
            "group":  "B — Notion / Chitragupta",
            "prefix": "notion_*",
            "requires": "NOTION_API_KEY in .env",
            "tools": [
                {"name": "notion_export_research",   "purpose": "Export top SHANI papers to Notion DB"},
                {"name": "notion_query_papers",      "purpose": "Query Notion research DB with filters"},
                {"name": "notion_get_database_info", "purpose": "Notion DB metadata + property schema"},
                {"name": "notion_update_paper",      "purpose": "Patch fields on one Notion page"},
            ],
        },
        {
            "group":  "C — Research Query",
            "prefix": "research_*",
            "requires": "SQLite DB exists (no API needed)",
            "tools": [
                {"name": "research_knowledge_summary",    "purpose": "Stats on extracted knowledge by category"},
                {"name": "research_find_papers_by_topic", "purpose": "Search papers by keyword / material"},
                {"name": "research_get_database_stats",   "purpose": "Row counts + completeness across tables"},
            ],
        },
        {
            "group":  "D — Analysis (read-only)",
            "prefix": "analysis_*",
            "requires": "SQLite DB exists",
            "tools": [
                {"name": "analysis_technique_frequency",    "purpose": "Which characterisation techniques appear most"},
                {"name": "analysis_trend_report",           "purpose": "Publication year / method trend over time"},
                {"name": "analysis_find_gaps",              "purpose": "Under-covered topics in the corpus"},
                {"name": "analysis_parameter_distribution", "purpose": "Distribution of numeric params (bandgap, etc.)"},
                {"name": "analysis_workflow_comparison",    "purpose": "Side-by-side stats across workflows"},
                {"name": "analysis_save_to_notion",         "purpose": "Push analysis result to a Notion page"},
            ],
        },
        {
            "group":  "E — Corrections (write)",
            "prefix": "db_*",
            "requires": "SQLite DB exists — writes appended to audit log",
            "tools": [
                {"name": "db_list_suspect_papers",    "purpose": "Flag papers with anomalous / missing data"},
                {"name": "db_update_paper",           "purpose": "Patch writable fields on a Paper row"},
                {"name": "db_update_workflow_config", "purpose": "Edit workflow focus / material / method"},
                {"name": "db_bulk_fix",               "purpose": "Batch-apply a correction rule across papers"},
            ],
        },
        {
            "group":  "F — Review Generation",
            "prefix": "review_*",
            "requires": "SHANI API running + S4/S5 completed",
            "tools": [
                {"name": "review_run_knowledge_extraction", "purpose": "Run S5 (parameter extraction) on workflows"},
                {"name": "review_generate_direction",       "purpose": "Run S5_5 (review outline generation)"},
                {"name": "review_draft_sections",           "purpose": "Run S6 (section drafting)"},
                {"name": "review_synthesize_final",         "purpose": "Run S7 (final synthesis)"},
                {"name": "review_get_draft",                "purpose": "Retrieve draft output from S5_5/S6/S7"},
            ],
        },
        {
            "group":  "G — VIDUR Classifier",
            "prefix": "vidur_*",
            "requires": "VIDUR_ROOT importable, numpy installed — no API",
            "tools": [
                {"name": "vidur_classify",         "purpose": "Classify + parse one instrument file (file_path)"},
                {"name": "vidur_list_techniques",  "purpose": "List all 4 detectable techniques + extensions"},
                {"name": "vidur_health",           "purpose": "Verify all 7 VIDUR modules import correctly"},
            ],
        },
        {
            "group":  "H — Vishwakarma (Quantum ESPRESSO)",
            "prefix": "vishwakarma_*",
            "requires": "QE binaries installed, QE_BIN_DIR set, pseudopotentials in QE_PSEUDO_DIR",
            "tools": [
                {"name": "vishwakarma_health",                "purpose": "Check QE binary availability + module imports"},
                {"name": "vishwakarma_generate_input",        "purpose": "Generate QE input file without running"},
                {"name": "vishwakarma_run_scf",               "purpose": "SCF single-point energy calculation"},
                {"name": "vishwakarma_run_relax",             "purpose": "Ionic/cell relaxation → final SCF"},
                {"name": "vishwakarma_run_bands",             "purpose": "Band structure: SCF → NSCF → bands.x"},
                {"name": "vishwakarma_run_dos",               "purpose": "Density of states: SCF → NSCF → dos.x"},
                {"name": "vishwakarma_run_phonon",            "purpose": "DFPT phonons via ph.x (+ Born charges/dielectric)"},
                {"name": "vishwakarma_run_neb",               "purpose": "Nudged elastic band — transition states"},
                {"name": "vishwakarma_run_hp",                "purpose": "Compute Hubbard U via hp.x (linear response)"},
                {"name": "vishwakarma_parse_output",          "purpose": "Parse any QE output file into structured data"},
                {"name": "vishwakarma_list_pseudopotentials", "purpose": "Discover UPF files + cross-check vs structure"},
                {"name": "vishwakarma_get_job_status",        "purpose": "Status of a specific job by job_id"},
                {"name": "vishwakarma_list_jobs",             "purpose": "List all QE jobs with status filter"},
            ],
        },
        {
            "group":  "Meta",
            "prefix": "brahm_*",
            "requires": "always available",
            "tools": [
                {"name": "brahm_overview", "purpose": "This call — full live system map"},
            ],
        },
    ]

    # ── 3. Stage pipeline reference ───────────────────────────────────────────
    pipeline_stages = [
        {"stage": "S1",    "name": "search_papers",           "description": "Fetch papers from Semantic Scholar / OpenAlex / arXiv"},
        {"stage": "S2",    "name": "download_pdfs",           "description": "Download PDFs for matched papers"},
        {"stage": "S2_75", "name": "filter_papers",           "description": "Domain-drift filter pass 1"},
        {"stage": "S2_5",  "name": "score_papers",            "description": "Multi-factor relevance scoring"},
        {"stage": "S3",    "name": "select_papers",           "description": "Threshold + ranking cutoff"},
        {"stage": "S4",    "name": "extract_paper_content",   "description": "Full text + table + equation extraction"},
        {"stage": "S5",    "name": "extract_research_knowledge", "description": "Parameter extraction + cross-paper aggregation"},
        {"stage": "S5_5",  "name": "generate_review_direction",  "description": "Outline + structure for the review"},
        {"stage": "S6",    "name": "draft_sections",          "description": "Section-by-section draft generation"},
        {"stage": "S7",    "name": "synthesize_paper",        "description": "Final synthesis — complete review document"},
    ]

    # ── 4. Quick-start action hints ───────────────────────────────────────────
    next_steps = []
    if not shani_up:
        next_steps.append("Start SHANI API before using Group A or F tools.")
    if not db_exists:
        next_steps.append("SQLite DB not found — run shani_create_workflow to initialise.")
    if notion_status == "schema_missing":
        next_steps.append("Run notion_export_research to create the Notion schema.")
    if not vidur_ok:
        next_steps.append(f"Fix VIDUR imports — check VIDUR_ROOT={VIDUR_ROOT}.")
    if not next_steps:
        next_steps.append("All systems nominal. Pick a tool group and begin.")

    total_tools = sum(len(g["tools"]) for g in tool_groups)

    return _ok({
        "system":          "BRAHM MCP",
        "total_tools":     total_tools,
        "agents":          agents,
        "tool_groups":     tool_groups,
        "pipeline_stages": pipeline_stages,
        "next_steps":      next_steps,
    })


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP + MAIN
# ═════════════════════════════════════════════════════════════════════════════

async def _startup_checks() -> None:
    """Log system readiness on server startup."""
    log.info("=" * 60)
    log.info("BRAHM MCP starting")
    log.info("=" * 60)

    # SHANI API check
    shani_up = await _check_shani()
    log.info("SHANI API (%s): %s", SHANI_BASE, "✓ reachable" if shani_up else "✗ NOT reachable")
    if not shani_up:
        log.warning(
            "Pipeline tools (Group A, F) will return errors until SHANI API is started. "
            "Query tools (C, D, E) and Notion tools (B) will still work."
        )

    # DB check
    db_exists = Path(DB_PATH).exists()
    log.info("SQLite DB (%s): %s", DB_PATH, "✓ exists" if db_exists else "✗ NOT found")

    # Notion check (schema presence, not API call)
    try:
        schema = load_schema("ZnSe Research Knowledge Base")
        db_id  = schema.get("notion_database_id", "")
        log.info("Notion schema: ✓ loaded | DB ID: %s", db_id[:8] + "..." if db_id else "not set")
    except SchemaMissingError:
        log.info("Notion schema: not yet created (run notion_export_research to set up)")
    except Exception as exc:
        log.warning("Notion schema check failed: %s", exc)

    log.info("Tools registered: 46 (29 existing + 3 VIDUR + 13 Vishwakarma + 1 meta)")

    # VIDUR check
    try:
        import extractor as _ext  # noqa: F401
        import auto_detector as _ad  # noqa: F401
        import router as _rt  # noqa: F401
        log.info("VIDUR (%s): ✓ core modules imported", VIDUR_ROOT)
    except Exception as exc:
        log.warning(
            "VIDUR modules not importable from VIDUR_ROOT=%s: %s  "
            "(vidur_classify will fail until path is corrected)",
            VIDUR_ROOT, exc,
        )

    # Vishwakarma check
    try:
        from vishwakarma import runner as _vr
        bins = _vr.check_binaries(_QE_BIN_DIR)
        found = sum(1 for v in bins.values() if v)
        log.info("Vishwakarma: ✓ modules imported | QE binaries: %d/%d found in %s",
                 found, len(bins), _QE_BIN_DIR)
    except Exception as exc:
        log.warning(
            "Vishwakarma modules not importable from VISHWAKARMA_ROOT=%s: %s  "
            "(vishwakarma_* tools will fail until path is corrected)",
            VISHWAKARMA_ROOT, exc,
        )

    log.info("=" * 60)


async def main() -> None:
    await _startup_checks()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
