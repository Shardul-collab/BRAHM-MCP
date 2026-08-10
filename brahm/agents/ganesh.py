"""
brahm/agents/ganesh.py
=======================
Group I — GANESH scientific writing + synthesis tools (G1-G5).
GANESH handles all document generation that was previously S6/S7 in SHANI.

Tools:
  ganesh_health          — check GANESH API status
  ganesh_write_review    — G1-G3: context load, plan, draft (requires API)
  ganesh_synthesize      — G4-G5: cross-section review + integration (requires API)
  ganesh_get_document    — retrieve document/draft from GANESH (requires API)
  ganesh_list_documents  — list all GANESH documents (requires API)
  ganesh_daily_report    — template-based daily report from CHITRAGUPTA events (no API needed)
"""

from brahm.brahm_registry import brahm_tool, requires_api
from brahm.shared.helpers import _ok, _err
from brahm.shared.http import (
    _ganesh_get, _ganesh_post, _check_ganesh, GANESH_START_HINT,
    _chitragupta_get, _chitragupta_post, _chitragupta_patch, _check_chitragupta,
    CHITRAGUPTA_START_HINT,
)


# ═════════════════════════════════════════════════════════════════════════════
# GANESH API TOOLS (require localhost:8001)
# ═════════════════════════════════════════════════════════════════════════════

@brahm_tool(
    name="ganesh_health", group="ganesh",
    description="Check if GANESH API is running and reachable at localhost:8001.",
    input_schema={"type": "object", "properties": {}, "required": []},
)
async def ganesh_health(args: dict) -> dict:
    ok = await _check_ganesh()
    return _ok({
        "status": "online" if ok else "offline",
        "hint":   "" if ok else GANESH_START_HINT,
    })


@brahm_tool(
    name="ganesh_write_review", group="ganesh",
    description=(
        "Trigger GANESH G1-G3: load research context from SHANI, "
        "plan the document structure, and draft all sections. "
        "Replaces the old review_draft_sections (S6). "
        "Requires GANESH API running at localhost:8001."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workflow_ids": {
                "type": "array", "items": {"type": "integer"},
                "description": "SHANI workflow IDs to pull research context from",
            },
            "document_type": {
                "type": "string",
                "enum": ["literature_review", "dft_report", "research_report",
                         "manuscript_draft", "technical_summary"],
                "default": "literature_review",
            },
            "title":      {"type": "string"},
            "project_id": {"type": "integer"},
        },
        "required": ["workflow_ids"],
    },
)
@requires_api(_check_ganesh, "GANESH", GANESH_START_HINT)
async def ganesh_write_review(args: dict) -> dict:
    result = await _ganesh_post("/documents/write", args)
    if result.get("status") == "error":
        return result
    import asyncio as _aio
    from brahm.shared.http import _chit_store_async
    _aio.ensure_future(_chit_store_async('/v1/store/ganesh', {
        'document_id':   result.get('document_id', ''),
        'workflow_ids':  args.get('workflow_ids', []),
        'document_type': args.get('document_type', ''),
        'status':        'drafting',
    }))
    return _ok(result)


@brahm_tool(
    name="ganesh_synthesize", group="ganesh",
    description=(
        "Trigger GANESH G4-G5: cross-section review and final document integration. "
        "Replaces the old review_synthesize_final (S7). "
        "Run after ganesh_write_review completes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
        },
        "required": ["document_id"],
    },
)
@requires_api(_check_ganesh, "GANESH", GANESH_START_HINT)
async def ganesh_synthesize(args: dict) -> dict:
    result = await _ganesh_post("/documents/synthesize", args)
    if result.get("status") == "error":
        return result
    import asyncio as _aio
    from brahm.shared.http import _chit_store_async
    _aio.ensure_future(_chit_store_async('/v1/store/ganesh', {
        'document_id':   args.get('document_id', ''),
        'workflow_ids':  [],
        'document_type': '',
        'status':        'complete',
    }))
    return _ok(result)


@brahm_tool(
    name="ganesh_get_document", group="ganesh",
    description=(
        "Retrieve a completed document or draft from GANESH. "
        "Replaces the old review_get_draft."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
            "stage": {
                "type": "string",
                "enum": ["G2", "G3", "G4", "G5"],
                "description": "Which stage output to retrieve",
            },
        },
        "required": ["document_id"],
    },
)
@requires_api(_check_ganesh, "GANESH", GANESH_START_HINT)
async def ganesh_get_document(args: dict) -> dict:
    result = await _ganesh_get(f"/documents/{args['document_id']}")
    if result.get("status") == "error":
        return result
    return _ok(result)


@brahm_tool(
    name="ganesh_list_documents", group="ganesh",
    description="List all documents generated by GANESH.",
    input_schema={
        "type": "object",
        "properties": {
            "status_filter": {
                "type": "string",
                "enum": ["all", "draft", "reviewing", "completed", "failed"],
                "default": "all",
            },
            "limit": {"type": "integer", "default": 20},
        },
        "required": [],
    },
)
@requires_api(_check_ganesh, "GANESH", GANESH_START_HINT)
async def ganesh_list_documents(args: dict) -> dict:
    result = await _ganesh_get("/documents")
    if result.get("status") == "error":
        return result
    return _ok(result)


# ═════════════════════════════════════════════════════════════════════════════
# DAILY REPORT — no GANESH API needed, reads from CHITRAGUPTA directly
# ═════════════════════════════════════════════════════════════════════════════

@brahm_tool(
    name="ganesh_daily_report", group="ganesh",
    description=(
        "Generate a markdown daily report for a project from today's events in brahm.db. "
        "Reads all ProjectEvents logged today from CHITRAGUPTA, groups by agent, "
        "and produces a structured markdown summary. No LLM required. "
        "Optionally saves the report as a GaneshDocument in brahm.db. "
        "Call this at end of session to create a human-readable record."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "integer"},
            "save": {
                "type": "boolean",
                "default": True,
                "description": "Save report as a daily_report document in brahm.db",
            },
        },
        "required": ["project_id"],
    },
)
async def ganesh_daily_report(args: dict) -> dict:
    from datetime import datetime, timezone

    if not await _check_chitragupta():
        return _err("CHITRAGUPTA API not running.", CHITRAGUPTA_START_HINT)

    project_id = args["project_id"]
    save       = args.get("save", True)
    today      = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── 1. Fetch project metadata ─────────────────────────────────────────────
    project_result = await _chitragupta_get(f"/v1/projects/{project_id}")
    if project_result.get("status") == "error":
        return project_result
    project = project_result.get("project", {})

    # ── 2. Fetch today's events ───────────────────────────────────────────────
    events_result = await _chitragupta_get(
        f"/v1/projects/{project_id}/events?today_only=true"
    )
    if events_result.get("status") == "error":
        return events_result
    events = events_result.get("events", [])

    # ── 3. Fetch DFT result summary ───────────────────────────────────────────
    dft_result = await _chitragupta_get(
        f"/v1/results/dft/summary?project_id={project_id}"
    )
    dft_summary = dft_result.get("summary", {}) if dft_result.get("status") != "error" else {}

    # ── 4. Fetch instrument results count ────────────────────────────────────
    instr_result = await _chitragupta_get(
        f"/v1/results/instrument?project_id={project_id}"
    )
    instr_count = instr_result.get("count", 0) if instr_result.get("status") != "error" else 0

    # ── 5. Paper stats ────────────────────────────────────────────────────────
    paper_stats_result = await _chitragupta_get("/v1/papers/stats")
    paper_stats = paper_stats_result.get("stats", {}) if paper_stats_result.get("status") != "error" else {}

    # ── 6. Group events by agent ──────────────────────────────────────────────
    by_agent: dict[str, list] = {}
    for ev in events:
        by_agent.setdefault(ev.get("agent", "Unknown"), []).append(ev)

    pending_decisions = project.get("pending_decisions", [])

    # ── 7. Build markdown ─────────────────────────────────────────────────────
    lines = [
        f"# BRAHM Daily Report — {today}",
        f"",
        f"**Project:** {project.get('name', f'Project {project_id}')}",
        f"**Objective:** {project.get('objective', '_not set_')}",
        f"**Status:** {project.get('status', 'active')}",
        f"",
        f"## Summary",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Events today | {len(events)} |",
        f"| Agents active | {', '.join(by_agent.keys()) or 'none'} |",
        f"| Papers in brahm.db | {paper_stats.get('total', 0)} |",
        f"| Instrument results | {instr_count} |",
        f"| DFT calculations | {dft_summary.get('total', 0)} |",
        f"| Pending decisions | {len(pending_decisions)} |",
        f"",
    ]

    # Events by agent
    if events:
        lines.append("## Agent Activity")
        for agent, agent_events in by_agent.items():
            lines.append(f"### {agent} ({len(agent_events)} events)")
            for ev in agent_events:
                ts     = (ev.get("started_at") or "")[:16].replace("T", " ")
                status = ev.get("status", "")
                badge  = " ✓" if status == "completed" else (" ✗" if status == "failed" else "")
                lines.append(
                    f"- `{ts}` **{ev.get('event_type', '')}**{badge}: {ev.get('summary', '')}"
                )
            lines.append("")
    else:
        lines.append("## Agent Activity\n_No events logged today._\n")

    # DFT breakdown
    if dft_summary:
        lines.append("## DFT Calculations (all time)")
        for calc_type, count in dft_summary.items():
            if calc_type != "total":
                lines.append(f"- **{calc_type}**: {count}")
        lines.append("")

    # Pending decisions
    if pending_decisions:
        lines.append("## ⚠ Pending Decisions")
        for d in pending_decisions:
            lines.append(f"- [{d.get('id')}] {d.get('question', '')}")
            if d.get("recommendation"):
                lines.append(f"  - *Recommendation:* {d['recommendation']}")
        lines.append("")

    lines.append(f"---\n_Generated by BRAHM GANESH · {today}_")

    report_md = "\n".join(lines)

    # ── 8. Save to brahm.db ───────────────────────────────────────────────────
    saved_document_id = None
    if save:
        doc_result = await _chitragupta_post("/v1/documents", {
            "project_id":    project_id,
            "document_type": "daily_report",
            "title":         f"Daily Report {today}",
        })
        if doc_result.get("status") != "error":
            doc_id = doc_result.get("document_id")
            if doc_id:
                sec_result = await _chitragupta_post(
                    f"/v1/documents/{doc_id}/sections",
                    {"section_name": "daily_summary", "order_index": 0},
                )
                if sec_result.get("status") != "error":
                    sid = sec_result.get("section_id")
                    if sid:
                        await _chitragupta_patch(
                            f"/v1/sections/{sid}/final",
                            {"content": report_md},
                        )
                        saved_document_id = doc_id

    return _ok({
        "project_id":        project_id,
        "date":              today,
        "total_events":      len(events),
        "agents_active":     list(by_agent.keys()),
        "pending_decisions": len(pending_decisions),
        "report_markdown":   report_md,
        "saved_document_id": saved_document_id,
    })
