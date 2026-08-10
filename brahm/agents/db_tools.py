"""
brahm/agents/db_tools.py
=========================
Group E — DB correction tools. All writes append to audit log.
"""

import asyncio
import re
from datetime import datetime

from brahm.brahm_registry import brahm_tool
from brahm.shared.helpers import _ok, _err, _repo, _audit_log
from brahm.shared.constants import (
    PAPER_WRITABLE_FIELDS, PAPER_IMMUTABLE_FIELDS, CONFIG_WRITABLE_FIELDS
)


@brahm_tool(
    name="db_list_suspect_papers", group="db_tools",
    description=(
        "Find papers with data quality issues: missing DOIs, malformed dates, "
        "empty abstracts, no content. Run before db_bulk_fix to identify targets."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workflow_id":  {"type": "integer"},
            "issue_types":  {
                "type": "array",
                "items": {"type": "string",
                          "enum": ["missing_doi","missing_abstract","bad_date",
                                   "short_abstract","no_content"]},
            },
            "limit": {"type": "integer", "default": 50},
        },
        "required": [],
    },
)
async def db_list_suspect_papers(args: dict) -> dict:
    wf_id       = args.get("workflow_id")
    issue_types = set(args.get("issue_types") or [
        "missing_doi","missing_abstract","bad_date","short_abstract","no_content"
    ])
    limit = args.get("limit", 50)

    def _list() -> dict:
        repo = _repo()
        try:
            wf_filter = "AND p.workflow_id = ?" if wf_id else ""
            params    = (wf_id,) if wf_id else ()
            rows = repo.fetch_all(
                f"""
                SELECT p.id, p.workflow_id, p.title, p.abstract, p.doi,
                       p.status, p.created_at, p.updated_at
                FROM Paper p WHERE 1=1 {wf_filter}
                ORDER BY p.id LIMIT ?
                """, params + (limit * 3,),
            )
            content_map = {
                r["paper_id"]: r["cnt"]
                for r in repo.fetch_all(
                    "SELECT paper_id, COUNT(*) as cnt FROM PaperContent GROUP BY paper_id"
                )
            }
            suspect = []
            for r in rows:
                d = dict(r)
                issues = []
                if "missing_doi"      in issue_types and not d.get("doi"):
                    issues.append("missing_doi")
                if "missing_abstract" in issue_types and not d.get("abstract"):
                    issues.append("missing_abstract")
                if "short_abstract"   in issue_types:
                    ab = d.get("abstract") or ""
                    if 0 < len(ab) < 80:
                        issues.append("short_abstract")
                if "no_content"       in issue_types and content_map.get(d["id"], 0) == 0:
                    issues.append("no_content")
                if "bad_date"         in issue_types:
                    ca = d.get("created_at", "")
                    if ca and (ca.startswith("1970") or ca.startswith("0000")):
                        issues.append("bad_date")
                if issues:
                    d["issues"]           = issues
                    d["abstract_length"]  = len(d.get("abstract") or "")
                    d["content_sections"] = content_map.get(d["id"], 0)
                    suspect.append(d)
                    if len(suspect) >= limit:
                        break
            return _ok({
                "workflow_id":         wf_id,
                "issue_types_checked": list(issue_types),
                "suspect_count":       len(suspect),
                "papers":              suspect,
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_list)


@brahm_tool(
    name="db_update_paper", group="db_tools",
    description=(
        "Direct field update on a Paper row. "
        "Blocked fields: id, workflow_id, source, raw_text, file_path. "
        "Always writes to audit log."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "paper_id": {"type": "integer"},
            "fields":   {
                "type": "object",
                "description": "Fields to update (title, doi, abstract, pdf_url, status, ...)",
            },
        },
        "required": ["paper_id", "fields"],
    },
)
async def db_update_paper(args: dict) -> dict:
    paper_id = args["paper_id"]
    fields   = args.get("fields", {})
    if not fields:
        return _err("fields cannot be empty")
    immutable = set(fields.keys()) & PAPER_IMMUTABLE_FIELDS
    if immutable:
        return _err(f"Fields {sorted(immutable)} are immutable.")
    invalid = set(fields.keys()) - PAPER_WRITABLE_FIELDS
    if invalid:
        return _err(f"Fields {sorted(invalid)} not in writable allowlist.",
                    f"Writable: {sorted(PAPER_WRITABLE_FIELDS)}")

    def _update() -> dict:
        repo = _repo()
        try:
            paper = repo.fetch_one(
                "SELECT id, workflow_id, " + ", ".join(PAPER_WRITABLE_FIELDS)
                + " FROM Paper WHERE id = ?", (paper_id,)
            )
            if not paper:
                return _err(f"Paper {paper_id} not found")
            old_values  = {f: dict(paper).get(f) for f in fields.keys()}
            set_clauses = ", ".join(f"{f} = ?" for f in fields.keys())
            with repo.transaction() as cursor:
                cursor.execute(
                    f"UPDATE Paper SET {set_clauses}, updated_at = ? WHERE id = ?",
                    list(fields.values()) + [datetime.utcnow().isoformat(), paper_id],
                )
            _audit_log("db_update_paper", {
                "paper_id":       paper_id,
                "workflow_id":    dict(paper)["workflow_id"],
                "fields_changed": list(fields.keys()),
                "old_values":     old_values,
                "new_values":     fields,
            })
            return _ok({"paper_id": paper_id, "fields_updated": list(fields.keys()),
                        "old_values": old_values})
        finally:
            repo.close()

    return await asyncio.to_thread(_update)


@brahm_tool(
    name="db_update_workflow_config", group="db_tools",
    description="Correct WorkflowResearchConfig fields for a workflow.",
    input_schema={
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
)
async def db_update_workflow_config(args: dict) -> dict:
    wf_id  = args["workflow_id"]
    fields = args.get("fields", {})
    if not fields:
        return _err("fields cannot be empty")
    invalid = set(fields.keys()) - CONFIG_WRITABLE_FIELDS
    if invalid:
        return _err(f"Fields {sorted(invalid)} not writable.",
                    f"Writable: {sorted(CONFIG_WRITABLE_FIELDS)}")

    def _update() -> dict:
        repo = _repo()
        try:
            config = repo.fetch_one(
                "SELECT * FROM WorkflowResearchConfig WHERE workflow_id = ?", (wf_id,)
            )
            if not config:
                return _err(f"WorkflowResearchConfig not found for workflow {wf_id}")
            old_values  = {f: dict(config).get(f) for f in fields.keys()}
            set_clauses = ", ".join(f"{f} = ?" for f in fields.keys())
            with repo.transaction() as cursor:
                cursor.execute(
                    f"UPDATE WorkflowResearchConfig SET {set_clauses} WHERE workflow_id = ?",
                    list(fields.values()) + [wf_id],
                )
            _audit_log("db_update_workflow_config", {
                "workflow_id":    wf_id,
                "fields_changed": list(fields.keys()),
                "old_values":     old_values,
                "new_values":     fields,
            })
            return _ok({"workflow_id": wf_id, "fields_updated": list(fields.keys()),
                        "old_values": old_values})
        finally:
            repo.close()

    return await asyncio.to_thread(_update)


@brahm_tool(
    name="db_bulk_fix", group="db_tools",
    description=(
        "Apply a regex pattern-based fix across multiple papers in a workflow. "
        "ALWAYS call with dry_run=true first to preview. "
        "Only call with dry_run=false after confirming the preview."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workflow_id":    {"type": "integer"},
            "field":          {"type": "string"},
            "match_pattern":  {"type": "string"},
            "replacement":    {"type": "string"},
            "dry_run":        {"type": "boolean", "default": True},
        },
        "required": ["workflow_id", "field", "match_pattern", "replacement"],
    },
)
async def db_bulk_fix(args: dict) -> dict:
    wf_id         = args["workflow_id"]
    field         = args["field"]
    match_pattern = args["match_pattern"]
    replacement   = args["replacement"]
    dry_run       = args.get("dry_run", True)

    if field not in PAPER_WRITABLE_FIELDS:
        return _err(f"Field '{field}' not in writable allowlist.",
                    f"Writable: {sorted(PAPER_WRITABLE_FIELDS)}")
    try:
        compiled = re.compile(match_pattern)
    except re.error as exc:
        return _err(f"Invalid regex: {exc}")

    def _bulk() -> dict:
        repo = _repo()
        try:
            rows = repo.fetch_all(
                f"SELECT id, {field} FROM Paper WHERE workflow_id=? AND {field} IS NOT NULL",
                (wf_id,),
            )
            affected = []
            for r in rows:
                current_val = r[field] or ""
                if compiled.search(current_val):
                    new_val = compiled.sub(replacement, current_val)
                    affected.append({"paper_id": r["id"],
                                     "old_value": current_val[:200],
                                     "new_value": new_val[:200]})
            if dry_run:
                return _ok({
                    "dry_run":       True,
                    "workflow_id":   wf_id,
                    "field":         field,
                    "would_affect":  len(affected),
                    "preview":       affected[:20],
                    "message":       f"DRY RUN: {len(affected)} rows would be updated. Call with dry_run=false to apply.",
                })
            with repo.transaction() as cursor:
                for item in affected:
                    cursor.execute(
                        f"UPDATE Paper SET {field}=?, updated_at=? WHERE id=?",
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
                "message":      f"Applied: {len(affected)} rows updated.",
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_bulk)
