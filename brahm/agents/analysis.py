"""
brahm/agents/analysis.py
=========================
Group D — Analysis tools. Read-only on SQLite. Uses ResearchAnalyzer.
"""

import asyncio
import json
import logging
from datetime import datetime

from brahm.brahm_registry import brahm_tool
from brahm.shared.helpers import _ok, _err, _analyzer, _title, _rtext, _select, _date

log = logging.getLogger("mcp.brahm.analysis")

KNOWLEDGE_CATS = ["material", "synthesis_method", "characterization", "application", "computational_method"]

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


@brahm_tool(
    name="analysis_technique_frequency", group="analysis",
    description=(
        "Count how often each synthesis method, characterization technique, "
        "material variant, or application appears across all papers."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "category":    {"type": "string", "enum": ["all"] + KNOWLEDGE_CATS},
            "workflow_id": {"type": "integer"},
            "top_n":       {"type": "integer", "default": 20},
            "min_count":   {"type": "integer", "default": 2},
        },
        "required": ["category"],
    },
)
async def analysis_technique_frequency(args: dict) -> dict:
    return await asyncio.to_thread(
        _analyzer().technique_frequency,
        args["category"], args.get("workflow_id"),
        args.get("top_n", 20), args.get("min_count", 2),
    )


@brahm_tool(
    name="analysis_trend_report", group="analysis",
    description=(
        "Find co-occurrence patterns between two knowledge categories. "
        "e.g. 'what characterization methods appear with magnetron sputtering?'"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "primary_category":   {"type": "string", "enum": KNOWLEDGE_CATS},
            "secondary_category": {"type": "string", "enum": KNOWLEDGE_CATS},
            "filter_value":       {"type": "string"},
            "workflow_id":        {"type": "integer"},
            "min_co_occurrence":  {"type": "integer", "default": 3},
        },
        "required": ["primary_category"],
    },
)
async def analysis_trend_report(args: dict) -> dict:
    return await asyncio.to_thread(
        _analyzer().trend_report,
        args["primary_category"], args.get("secondary_category"),
        args.get("filter_value"), args.get("workflow_id"),
        args.get("min_co_occurrence", 3),
    )


@brahm_tool(
    name="analysis_find_gaps", group="analysis",
    description=(
        "Identify under-explored or unexplored A x B combinations. "
        "e.g. 'which material + synthesis_method pairings have no papers?'"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "category_a":     {"type": "string", "enum": KNOWLEDGE_CATS},
            "category_b":     {"type": "string", "enum": KNOWLEDGE_CATS},
            "known_values_a": {"type": "array", "items": {"type": "string"}},
            "known_values_b": {"type": "array", "items": {"type": "string"}},
            "gap_threshold":  {"type": "integer", "default": 2},
        },
        "required": ["category_a", "category_b"],
    },
)
async def analysis_find_gaps(args: dict) -> dict:
    return await asyncio.to_thread(
        _analyzer().find_gaps,
        args["category_a"], args["category_b"],
        args.get("known_values_a"), args.get("known_values_b"),
        args.get("gap_threshold", 2),
    )


@brahm_tool(
    name="analysis_parameter_distribution", group="analysis",
    description=(
        "Extract and aggregate quantitative values from the knowledge base. "
        "e.g. 'what bandgap values have been reported for ZnSe?'"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "parameter_keywords": {"type": "array", "items": {"type": "string"}},
            "workflow_id":        {"type": "integer"},
            "extract_numbers":    {"type": "boolean", "default": True},
            "group_by_material":  {"type": "boolean", "default": True},
        },
        "required": ["parameter_keywords"],
    },
)
async def analysis_parameter_distribution(args: dict) -> dict:
    return await asyncio.to_thread(
        _analyzer().parameter_distribution,
        args["parameter_keywords"], args.get("workflow_id"),
        args.get("extract_numbers", True), args.get("group_by_material", True),
    )


@brahm_tool(
    name="analysis_workflow_comparison", group="analysis",
    description="Compare 2+ workflows to identify overlapping and unique coverage.",
    input_schema={
        "type": "object",
        "properties": {
            "workflow_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
            "compare_by":   {"type": "string",
                             "enum": ["papers", "techniques", "materials", "knowledge"],
                             "default": "knowledge"},
        },
        "required": ["workflow_ids"],
    },
)
async def analysis_workflow_comparison(args: dict) -> dict:
    return await asyncio.to_thread(
        _analyzer().workflow_comparison,
        args["workflow_ids"], args.get("compare_by", "knowledge"),
    )


@brahm_tool(
    name="analysis_save_to_notion", group="analysis",
    description=(
        "Persist an analysis result to the 'ZnSe Analysis Results' Notion database. "
        "Creates the database if it doesn't exist."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name":         {"type": "string"},
            "type":         {"type": "string",
                             "enum": ["Trend", "Gap", "Parameter", "Comparison", "Frequency"]},
            "key_finding":  {"type": "string"},
            "results":      {"type": "object"},
            "action_items": {"type": "string"},
            "workflow_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["name", "type", "key_finding"],
    },
)
async def analysis_save_to_notion(args: dict) -> dict:
    name          = args["name"]
    analysis_type = args["type"]
    key_finding   = args["key_finding"]
    results       = args.get("results", {})
    action_items  = args.get("action_items", "")
    workflow_ids  = args.get("workflow_ids", [])

    def _save() -> dict:
        try:
            from notion.notion_client import create_page, NotionAPIError
            from notion.schema_manager import (
                load_schema, create_schema, update_notion_id, SchemaMissingError
            )
            from config.settings import NOTION_PAGE_ID

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

            today      = datetime.utcnow().strftime("%Y-%m-%d")
            scope_str  = (f"Workflows: {', '.join(str(w) for w in workflow_ids)}"
                          if workflow_ids else "All workflows")
            props = {
                "Analysis Name":   _title(name),
                "Type":            _select(analysis_type),
                "Date Run":        _date(today),
                "Workflows Scope": _rtext(scope_str),
                "Key Finding":     _rtext(key_finding),
                "Full Results":    _rtext(json.dumps(results, ensure_ascii=False)[:2000]),
                "Action Items":    _rtext(action_items),
            }
            page = create_page(db_id, props)
            return _ok({
                "saved":       True,
                "page_id":     page.get("id"),
                "database_id": db_id,
                "notion_url":  f"https://notion.so/{db_id.replace('-', '')}",
            })
        except Exception as exc:
            return _err("Save to Notion failed", str(exc))

    return await asyncio.to_thread(_save)
