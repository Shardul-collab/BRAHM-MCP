"""
brahm/shared/helpers.py
========================
Shared utility functions used across all agent modules.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from brahm.shared.constants import DB_PATH, AUDIT_LOG_PATH

log = logging.getLogger("mcp.brahm")


def _ok(data: dict) -> dict:
    return {"status": "success", **data}

def _err(msg: str, detail: str = "") -> dict:
    return {"status": "error", "error": msg, "detail": detail}


def _repo():
    import importlib.util, sys
    _spec = importlib.util.spec_from_file_location(
        "shani_repository",
        "/mnt/d/brahm/agents/shani/repositories/repository.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.Repository(DB_PATH)

def _analyzer():
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "research_analyzer",
        "/mnt/d/brahm/agents/chitragupta/analysis/research_analyzer.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.ResearchAnalyzer(DB_PATH)


def _audit_log(tool: str, record: dict) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool":      tool,
        **record,
    }
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("Audit log write failed: %s", exc)


def _title(text: str) -> dict:
    return {"title": [{"type": "text", "text": {"content": str(text)[:500]}}]}

def _rtext(text: str) -> dict:
    return {"rich_text": [{"type": "text", "text": {"content": str(text)[:2000]}}]}

def _select(value: str) -> dict:
    return {"select": {"name": str(value)}}

def _number(value: Any) -> dict:
    return {"number": value}

def _url(value: Optional[str]) -> dict:
    return {"url": value if value else None}

def _date(value: str) -> dict:
    return {"date": {"start": value}}


def _extract_notion_page(page: dict) -> dict:
    props  = page.get("properties", {})
    result = {"notion_page_id": page.get("id", "")}
    for key, prop in props.items():
        ptype = prop.get("type")
        if ptype == "title":
            result[key] = "".join(
                t.get("text", {}).get("content", "") for t in prop.get("title", [])
            )
        elif ptype == "rich_text":
            result[key] = "".join(
                t.get("text", {}).get("content", "") for t in prop.get("rich_text", [])
            )
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
