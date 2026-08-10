"""
brahm/agents/chitragupta.py
============================
Group B — CHITRAGUPTA knowledge management tools.

Replaces the old Notion-based tools with direct calls to the
CHITRAGUPTA brahm_db API (localhost:8003).

Tools (11):
  chitragupta_create_project        — start a new research project
  chitragupta_get_project           — full project state + pending decisions
  chitragupta_log_event             — record an agent action to the timeline
  chitragupta_list_workloads        — list queued/running/done workloads
  chitragupta_check_paper           — deduplication check before download
  chitragupta_register_paper        — register a newly downloaded paper
  chitragupta_save_instrument_result — persist a VIDUR classify result
  chitragupta_save_dft_result       — persist a Vishwakarma QE result
  chitragupta_create_document       — create a GANESH document record
  chitragupta_get_document          — retrieve document + all sections
  chitragupta_daily_report          — today's event summary for a project
"""

import logging

from brahm.brahm_registry import brahm_tool, requires_api
from brahm.shared.helpers import _ok, _err
from brahm.shared.http import (
    _chitragupta_get,
    _chitragupta_post,
    _chitragupta_patch,
    _check_chitragupta,
    CHITRAGUPTA_START_HINT,
)

log = logging.getLogger("mcp.brahm.chitragupta")

# ─── convenience guard ────────────────────────────────────────────────────────
_api = lambda fn: requires_api(_check_chitragupta, "CHITRAGUPTA", CHITRAGUPTA_START_HINT)(fn)


# ═════════════════════════════════════════════════════════════════════════════
# PROJECTS
# ═════════════════════════════════════════════════════════════════════════════

@brahm_tool(
    name="chitragupta_create_project",
    group="chitragupta",
    description=(
        "Create a new research project in brahm.db. "
        "Every SHANI workflow, VIDUR result, DFT calculation, and GANESH document "
        "should be linked to a project. Returns project_id needed for all other tools."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name":      {"type": "string", "description": "Short project name, e.g. 'ZnSe Defect Study'"},
            "objective": {"type": "string", "description": "One-paragraph research objective"},
        },
        "required": ["name", "objective"],
    },
)
@_api
async def chitragupta_create_project(args: dict) -> dict:
    result = await _chitragupta_post("/v1/projects", {
        "name":      args["name"],
        "objective": args["objective"],
    })
    if result.get("status") == "error":
        return result
    return _ok({
        "project_id": result["project_id"],
        "name":       args["name"],
        "message":    f"Project '{args['name']}' created with id={result['project_id']}.",
    })


@brahm_tool(
    name="chitragupta_get_project",
    group="chitragupta",
    description=(
        "Get full project state: metadata, pending decisions, active research cycle. "
        "Call this at the start of each session to orient yourself before any task."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "integer"},
        },
        "required": ["project_id"],
    },
)
@_api
async def chitragupta_get_project(args: dict) -> dict:
    result = await _chitragupta_get(f"/v1/projects/{args['project_id']}")
    if result.get("status") == "error":
        return result
    return _ok({"project": result.get("project", result)})


# ═════════════════════════════════════════════════════════════════════════════
# EVENTS
# ═════════════════════════════════════════════════════════════════════════════

@brahm_tool(
    name="chitragupta_log_event",
    group="chitragupta",
    description=(
        "Record an agent action to the project timeline. "
        "Call this whenever an agent (SHANI, VIDUR, Vishwakarma, GANESH) "
        "starts or completes a significant action. "
        "event_type examples: workflow_started, papers_collected, "
        "classification_done, scf_completed, review_drafted."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "integer"},
            "agent":      {
                "type": "string",
                "enum": ["SHANI", "VIDUR", "Vishwakarma", "GANESH", "BRAHM", "Human"],
            },
            "event_type": {"type": "string", "description": "Snake_case event identifier"},
            "summary":    {"type": "string", "description": "One-line human-readable summary"},
            "payload":    {
                "type": "object",
                "description": "Optional structured data (workflow_id, paper counts, job_id, etc.)",
            },
        },
        "required": ["project_id", "agent", "event_type", "summary"],
    },
)
@_api
async def chitragupta_log_event(args: dict) -> dict:
    body = {
        "agent":      args["agent"],
        "event_type": args["event_type"],
        "summary":    args["summary"],
        "payload":    args.get("payload"),
    }
    result = await _chitragupta_post(f"/v1/projects/{args['project_id']}/events", body)
    if result.get("status") == "error":
        return result
    return _ok({
        "event_id":   result.get("event_id"),
        "project_id": args["project_id"],
        "logged":     args["summary"],
    })


# ═════════════════════════════════════════════════════════════════════════════
# WORKLOADS
# ═════════════════════════════════════════════════════════════════════════════

@brahm_tool(
    name="chitragupta_list_workloads",
    group="chitragupta",
    description=(
        "List workloads for a project, with optional filters. "
        "Workloads are queued tasks for specific agents. "
        "Use to check what SHANI/VIDUR/Vishwakarma work is pending or running."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "integer"},
            "status": {
                "type": "string",
                "enum": ["queued", "running", "completed", "failed", "all"],
                "default": "all",
                "description": "Filter by workload status",
            },
            "agent": {
                "type": "string",
                "enum": ["SHANI", "VIDUR", "Vishwakarma", "GANESH"],
                "description": "Filter by agent. Omit for all agents.",
            },
        },
        "required": ["project_id"],
    },
)
@_api
async def chitragupta_list_workloads(args: dict) -> dict:
    params = f"/v1/projects/{args['project_id']}/workloads"
    qs = []
    status = args.get("status", "all")
    agent  = args.get("agent")
    if status and status != "all":
        qs.append(f"status={status}")
    if agent:
        qs.append(f"agent={agent}")
    if qs:
        params += "?" + "&".join(qs)

    result = await _chitragupta_get(params)
    if result.get("status") == "error":
        return result
    workloads = result.get("workloads", [])
    return _ok({
        "project_id": args["project_id"],
        "count":      len(workloads),
        "workloads":  workloads,
    })


# ═════════════════════════════════════════════════════════════════════════════
# PAPERS (deduplication)
# ═════════════════════════════════════════════════════════════════════════════

@brahm_tool(
    name="chitragupta_check_paper",
    group="chitragupta",
    description=(
        "Check whether a paper already exists in brahm.db before downloading. "
        "Pass doi (preferred) or title. "
        "Returns exists=true + global_paper_id if found — SHANI should skip download "
        "and call chitragupta_register_paper with link_only=true instead. "
        "This is the deduplication gate — always call before a new download."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "doi":   {"type": "string", "description": "Paper DOI (preferred identifier)"},
            "title": {"type": "string", "description": "Paper title (fallback if no DOI)"},
        },
        "required": [],
    },
)
@_api
async def chitragupta_check_paper(args: dict) -> dict:
    if not args.get("doi") and not args.get("title"):
        return _err("Provide at least one of: doi, title")
    result = await _chitragupta_post("/v1/papers/check", {
        "doi":   args.get("doi"),
        "title": args.get("title"),
    })
    if result.get("status") == "error":
        return result
    return _ok({
        "exists":          result.get("exists", False),
        "global_paper_id": result.get("global_paper_id"),
        "shani_paper_id":  result.get("shani_paper_id"),
        "action":          "link_existing" if result.get("exists") else "proceed_with_download",
    })


@brahm_tool(
    name="chitragupta_register_paper",
    group="chitragupta",
    description=(
        "Register a paper in brahm.db after SHANI downloads it. "
        "Call this once per new paper after successful download (not for duplicates). "
        "Returns global_paper_id — the stable cross-workflow paper identifier."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "title":          {"type": "string"},
            "doi":            {"type": "string"},
            "abstract":       {"type": "string"},
            "project_id":     {"type": "integer", "description": "Associate with a project"},
            "workflow_id":    {"type": "integer", "description": "SHANI workflow that found this paper"},
            "shani_paper_id": {"type": "integer", "description": "Paper.id in SHANI's SQLite DB"},
        },
        "required": ["title"],
    },
)
@_api
async def chitragupta_register_paper(args: dict) -> dict:
    result = await _chitragupta_post("/v1/papers", {
        "title":          args["title"],
        "doi":            args.get("doi"),
        "abstract":       args.get("abstract"),
        "project_id":     args.get("project_id"),
        "workflow_id":    args.get("workflow_id"),
        "shani_paper_id": args.get("shani_paper_id"),
    })
    if result.get("status") == "error":
        return result
    return _ok({
        "global_paper_id": result.get("global_paper_id"),
        "title":           args["title"],
        "message":         f"Paper registered with global_paper_id={result.get('global_paper_id')}.",
    })


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═════════════════════════════════════════════════════════════════════════════

@brahm_tool(
    name="chitragupta_save_instrument_result",
    group="chitragupta",
    description=(
        "Persist a VIDUR instrument classification result to brahm.db. "
        "Call this immediately after vidur_classify succeeds. "
        "Stores technique, confidence, parsed_data, and any literature gaps identified."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id":  {"type": "integer"},
            "file_path":   {"type": "string", "description": "Absolute path to classified file"},
            "technique":   {"type": "string", "description": "e.g. XRD, UV-Vis, SEM_EDX, Raman"},
            "confidence":  {"type": "number", "description": "VIDUR confidence score 0–1"},
            "signals":     {"type": "array",  "description": "Detection signals from auto_detector"},
            "parsed_data": {"type": "object", "description": "Parsed axis/intensity arrays"},
            "cycle_id":    {"type": "integer", "description": "Current research cycle (optional)"},
            "comparison_result": {
                "type": "object",
                "description": "Optional comparison against literature values",
            },
            "match_score":      {"type": "number", "description": "Literature match score 0–1"},
            "gaps_identified":  {
                "type": "array",
                "description": "List of literature gaps found during comparison",
            },
        },
        "required": ["project_id", "file_path", "technique", "confidence", "signals", "parsed_data"],
    },
)
@_api
async def chitragupta_save_instrument_result(args: dict) -> dict:
    result = await _chitragupta_post("/v1/results/instrument", {
        "project_id":        args["project_id"],
        "file_path":         args["file_path"],
        "technique":         args["technique"],
        "confidence":        args["confidence"],
        "signals":           args["signals"],
        "parsed_data":       args["parsed_data"],
        "cycle_id":          args.get("cycle_id"),
        "comparison_result": args.get("comparison_result"),
        "match_score":       args.get("match_score"),
        "gaps_identified":   args.get("gaps_identified"),
    })
    if result.get("status") == "error":
        return result
    return _ok({
        "result_id":  result.get("result_id"),
        "project_id": args["project_id"],
        "technique":  args["technique"],
        "message":    f"{args['technique']} result saved with id={result.get('result_id')}.",
    })


@brahm_tool(
    name="chitragupta_save_dft_result",
    group="chitragupta",
    description=(
        "Persist a Vishwakarma Quantum ESPRESSO calculation result to brahm.db. "
        "Call this immediately after any vishwakarma_run_* tool completes successfully. "
        "Stores job_id, calc_type, structure, input params, parsed output, and wall time."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id":         {"type": "integer"},
            "job_id":             {"type": "string", "description": "Vishwakarma job ID"},
            "calc_type":          {
                "type": "string",
                "enum": ["scf", "nscf", "relax", "vc-relax", "bands", "dos",
                         "projwfc", "pp", "phonon", "neb", "hp", "cp"],
            },
            "structure":          {"type": "object", "description": "Crystal structure dict used"},
            "input_params":       {"type": "object", "description": "calc_params used"},
            "output_parsed":      {"type": "object", "description": "Parsed output from vishwakarma_parse_output"},
            "status":             {"type": "string", "enum": ["completed", "failed"], "default": "completed"},
            "wall_time_seconds":  {"type": "number"},
            "cycle_id":           {"type": "integer"},
        },
        "required": ["project_id", "job_id", "calc_type"],
    },
)
@_api
async def chitragupta_save_dft_result(args: dict) -> dict:
    result = await _chitragupta_post("/v1/results/dft", {
        "project_id":        args["project_id"],
        "job_id":            args["job_id"],
        "calc_type":         args["calc_type"],
        "structure":         args.get("structure"),
        "input_params":      args.get("input_params"),
        "output_parsed":     args.get("output_parsed"),
        "status":            args.get("status", "completed"),
        "wall_time_seconds": args.get("wall_time_seconds"),
        "cycle_id":          args.get("cycle_id"),
    })
    if result.get("status") == "error":
        return result
    return _ok({
        "result_id":  result.get("result_id"),
        "project_id": args["project_id"],
        "job_id":     args["job_id"],
        "calc_type":  args["calc_type"],
        "message":    f"{args['calc_type']} result saved with id={result.get('result_id')}.",
    })


# ═════════════════════════════════════════════════════════════════════════════
# DOCUMENTS
# ═════════════════════════════════════════════════════════════════════════════

@brahm_tool(
    name="chitragupta_create_document",
    group="chitragupta",
    description=(
        "Create a GANESH document record in brahm.db before writing begins. "
        "document_type options: literature_review, dft_report, research_report, "
        "manuscript_draft, technical_summary, daily_report. "
        "Returns document_id used by GANESH throughout G1–G5."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id":    {"type": "integer"},
            "document_type": {
                "type": "string",
                "enum": ["literature_review", "dft_report", "research_report",
                         "manuscript_draft", "technical_summary", "daily_report"],
            },
            "title":         {"type": "string"},
            "workflow_ids":  {
                "type": "array", "items": {"type": "integer"},
                "description": "SHANI workflow IDs whose knowledge feeds this document",
            },
            "dft_result_ids": {
                "type": "array", "items": {"type": "integer"},
                "description": "DFT result IDs to include",
            },
            "instrument_result_ids": {
                "type": "array", "items": {"type": "integer"},
                "description": "Instrument result IDs to include",
            },
            "llm_backend": {
                "type": "string",
                "enum": ["groq", "ollama"],
                "description": "LLM backend GANESH will use for this document",
            },
            "cycle_id": {"type": "integer"},
        },
        "required": ["project_id", "document_type", "title"],
    },
)
@_api
async def chitragupta_create_document(args: dict) -> dict:
    result = await _chitragupta_post("/v1/documents", {
        "project_id":            args["project_id"],
        "document_type":         args["document_type"],
        "title":                 args["title"],
        "workflow_ids":          args.get("workflow_ids"),
        "dft_result_ids":        args.get("dft_result_ids"),
        "instrument_result_ids": args.get("instrument_result_ids"),
        "llm_backend":           args.get("llm_backend"),
        "cycle_id":              args.get("cycle_id"),
    })
    if result.get("status") == "error":
        return result
    return _ok({
        "document_id":   result.get("document_id"),
        "project_id":    args["project_id"],
        "document_type": args["document_type"],
        "title":         args["title"],
        "message":       f"Document '{args['title']}' created with id={result.get('document_id')}.",
    })


@brahm_tool(
    name="chitragupta_get_document",
    group="chitragupta",
    description=(
        "Retrieve a GANESH document with all its sections (draft, critique, final content). "
        "Use to review writing progress or retrieve completed text."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id":       {"type": "integer"},
            "include_sections":  {
                "type": "boolean",
                "default": True,
                "description": "Set false to get metadata only (faster)",
            },
        },
        "required": ["document_id"],
    },
)
@_api
async def chitragupta_get_document(args: dict) -> dict:
    doc_id = args["document_id"]
    include = args.get("include_sections", True)
    result = await _chitragupta_get(
        f"/v1/documents/{doc_id}?include_sections={'true' if include else 'false'}"
    )
    if result.get("status") == "error":
        return result
    return _ok({"document": result.get("document", result)})


# ═════════════════════════════════════════════════════════════════════════════
# DAILY REPORT
# ═════════════════════════════════════════════════════════════════════════════

@brahm_tool(
    name="chitragupta_daily_report",
    group="chitragupta",
    description=(
        "Generate a markdown daily report for a project from today's events. "
        "Reads all ProjectEvents logged today, groups by agent, "
        "and returns a structured summary. No LLM required — pure data aggregation. "
        "Call this at end of session to create a human-readable record."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "integer"},
            "save":       {
                "type": "boolean",
                "default": True,
                "description": "Save the report as a daily_report GaneshDocument in brahm.db",
            },
        },
        "required": ["project_id"],
    },
)
@_api
async def chitragupta_daily_report(args: dict) -> dict:
    from datetime import datetime, timezone

    project_id = args["project_id"]
    save       = args.get("save", True)

    # Fetch project metadata
    project_result = await _chitragupta_get(f"/v1/projects/{project_id}")
    if project_result.get("status") == "error":
        return project_result
    project = project_result.get("project", {})

    # Fetch today's events
    events_result = await _chitragupta_get(
        f"/v1/projects/{project_id}/events?today_only=true"
    )
    if events_result.get("status") == "error":
        return events_result
    events = events_result.get("events", [])

    # Group events by agent
    by_agent: dict[str, list] = {}
    for ev in events:
        agent = ev.get("agent", "Unknown")
        by_agent.setdefault(agent, []).append(ev)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build markdown
    lines = [
        f"# BRAHM Daily Report — {today}",
        f"**Project:** {project.get('name', f'Project {project_id}')}",
        f"**Objective:** {project.get('objective', '')}",
        f"**Total events today:** {len(events)}",
        "",
    ]

    if not events:
        lines.append("_No events logged today._")
    else:
        for agent, agent_events in by_agent.items():
            lines.append(f"## {agent} ({len(agent_events)} events)")
            for ev in agent_events:
                ts = ev.get("started_at", "")[:16].replace("T", " ")
                status = ev.get("status", "")
                status_badge = f" ✓" if status == "completed" else (f" ✗" if status == "failed" else "")
                lines.append(f"- `{ts}` **{ev.get('event_type', '')}**{status_badge}: {ev.get('summary', '')}")
            lines.append("")

    # Pending decisions
    pending = project.get("pending_decisions", [])
    if pending:
        lines.append("## ⚠ Pending Decisions")
        for d in pending:
            lines.append(f"- [{d.get('id')}] {d.get('question', '')}")
        lines.append("")

    report_md = "\n".join(lines)

    saved_document_id = None
    if save and events:
        # Save as GaneshDocument type=daily_report
        doc_result = await _chitragupta_post("/v1/documents", {
            "project_id":    project_id,
            "document_type": "daily_report",
            "title":         f"Daily Report {today}",
        })
        if doc_result.get("status") != "error":
            saved_document_id = doc_result.get("document_id")
            # Create a single section with the full report
            if saved_document_id:
                sec_result = await _chitragupta_post(
                    f"/v1/documents/{saved_document_id}/sections",
                    {"section_name": "daily_summary", "order_index": 0},
                )
                if sec_result.get("status") != "error":
                    sid = sec_result.get("section_id")
                    if sid:
                        await _chitragupta_patch(
                            f"/v1/sections/{sid}/final",
                            {"content": report_md},
                        )

    return _ok({
        "project_id":        project_id,
        "date":              today,
        "total_events":      len(events),
        "agents_active":     list(by_agent.keys()),
        "pending_decisions": len(pending),
        "report_markdown":   report_md,
        "saved_document_id": saved_document_id,
    })
