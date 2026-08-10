"""
brahm/agents/research.py
=========================
Group C — Read-only research query tools. No API needed — direct SQLite.

Also includes the gap-detection MCP wrappers (research_run_gap_map_phase,
research_run_gap_reduce_phase). These were previously only callable as a
standalone script (brahm/mcp/gap_detection_batch.py) with zero MCP entry
point — the actual reasoning pipeline existed and worked, but nothing
could invoke it. These two tools are the fix.
"""

import asyncio
import os
from brahm.brahm_registry import brahm_tool
from brahm.shared.helpers import _ok, _err, _repo


def _default_gap_checkpoint_path(workflow_id: int) -> str:
    return f"/mnt/d/brahm/data/gap_checkpoints/workflow_{workflow_id}.jsonl"


@brahm_tool(
    name="research_run_gap_map_phase", group="research",
    description=(
        "MAP phase of gap detection over a workflow's full ResearchFinding "
        "corpus: splits everything into token-safe batches and asks a "
        "reasoning LLM to propose one candidate research gap per batch. "
        "Checkpointed to a JSONL file — safe to call again if it stops "
        "partway (already-completed batches are skipped, so a timeout or "
        "crash mid-run just means calling this again continues where it "
        "left off). This can take a long time (65s pacing between LLM "
        "calls x number of batches) for large corpora — that's expected, "
        "not a hang. Follow with research_run_gap_reduce_phase once done."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workflow_id": {"type": "integer"},
            "checkpoint_path": {
                "type": "string",
                "description": "Defaults to a per-workflow path under data/gap_checkpoints/ if omitted.",
            },
            "pacing_seconds": {
                "type": "integer",
                "description": "Seconds between LLM calls (Groq TPM pacing). Defaults to 65.",
            },
        },
        "required": ["workflow_id"],
    },
)
async def research_run_gap_map_phase(args: dict) -> dict:
    workflow_id     = args["workflow_id"]
    checkpoint_path = args.get("checkpoint_path") or _default_gap_checkpoint_path(workflow_id)
    pacing_seconds  = args.get("pacing_seconds")

    def _run() -> dict:
        from brahm.mcp.gap_detection_batch import run_map_phase
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        repo = _repo()
        try:
            kwargs = {}
            if pacing_seconds is not None:
                kwargs["pacing_seconds"] = pacing_seconds
            result = run_map_phase(repo, workflow_id, checkpoint_path, **kwargs)
            result["checkpoint_path"] = checkpoint_path
            return result
        finally:
            repo.close()

    return await asyncio.to_thread(_run)


@brahm_tool(
    name="research_run_gap_reduce_phase", group="research",
    description=(
        "REDUCE phase of gap detection: takes every candidate gap from the "
        "map phase's checkpoint file and cross-checks each against a "
        "corpus-wide tag index of every paper, silently dropping any gap "
        "the corpus elsewhere already answers. Optionally pass project_id "
        "to queue each surviving gap as a Vishwakarma Workload in a "
        "Chitragupta project (visible via chitragupta_list_workloads) — "
        "this does NOT auto-generate a structure or launch any DFT "
        "calculation, it only creates a human-reviewable queued item. "
        "Building a real atomic structure from a gap statement and "
        "deciding whether to run it is a deliberate human step."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workflow_id": {"type": "integer"},
            "checkpoint_path": {"type": "string"},
            "project_id": {
                "type": "integer",
                "description": "If given, queue a Vishwakarma Workload per surviving gap in this Chitragupta project.",
            },
        },
        "required": ["workflow_id"],
    },
)
async def research_run_gap_reduce_phase(args: dict) -> dict:
    workflow_id     = args["workflow_id"]
    checkpoint_path = args.get("checkpoint_path") or _default_gap_checkpoint_path(workflow_id)
    project_id      = args.get("project_id")

    def _run() -> dict:
        from brahm.mcp.gap_detection_batch import run_reduce_phase
        repo = _repo()
        try:
            return run_reduce_phase(repo, workflow_id, checkpoint_path)
        finally:
            repo.close()

    result = await asyncio.to_thread(_run)
    result["checkpoint_path"] = checkpoint_path

    if result.get("status") != "success":
        return result

    final_gaps = (result.get("data") or {}).get("final_gaps", [])

    if project_id is not None and final_gaps:
        from brahm.shared.http import _chitragupta_post
        created = []
        for gap in final_gaps:
            wl_result = await _chitragupta_post(
                f"/v1/projects/{project_id}/workloads",
                {
                    "agent": "VISHWAKARMA",
                    "workload_type": "gap_verification",
                    "config": {
                        "workflow_id":           workflow_id,
                        "gap_statement":         gap.get("gap_statement"),
                        "supporting_paper_ids":  gap.get("supporting_paper_ids"),
                        "reasoning":             gap.get("reasoning"),
                    },
                    "priority": 5,
                },
            )
            created.append(wl_result)
        result["workloads_created"] = created

    return result


@brahm_tool(
    name="research_knowledge_summary", group="research",
    description=(
        "Statistical summary of extracted research knowledge: counts by category "
        "(material, synthesis_method, characterization, application) and "
        "top values per category. Check this before starting S5 to assess corpus quality."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workflow_id": {"type": "integer", "description": "Restrict to one workflow. Omit for all."},
        },
        "required": [],
    },
)
async def research_knowledge_summary(args: dict) -> dict:
    wf_id = args.get("workflow_id")

    def _query() -> dict:
        repo = _repo()
        try:
            wf_filter = "JOIN Paper p ON p.id = rk.paper_id AND p.workflow_id = ?" if wf_id else ""
            params    = (wf_id,) if wf_id else ()

            cat_rows = repo.fetch_all(
                f"""
                SELECT rk.category,
                       COUNT(*) as total_rows,
                       COUNT(DISTINCT rk.paper_id) as papers
                FROM ResearchKnowledge rk
                {wf_filter}
                GROUP BY rk.category
                ORDER BY total_rows DESC
                """, params,
            )
            by_category = [dict(r) for r in cat_rows]

            top_values = {}
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
                    """, params + (cat,),
                )
                top_values[cat] = [{"value": r["value"], "count": r["cnt"]} for r in top]

            total_rows = sum(c["total_rows"] for c in by_category)
            total_papers = (
                repo.fetch_one(
                    "SELECT COUNT(DISTINCT paper_id) FROM ResearchKnowledge"
                    + (" JOIN Paper p ON p.id = ResearchKnowledge.paper_id WHERE p.workflow_id = ?" if wf_id else ""),
                    (wf_id,) if wf_id else (),
                )[0] if cat_rows else 0
            )

            return _ok({
                "workflow_id":            wf_id,
                "total_knowledge_rows":   total_rows,
                "papers_with_knowledge":  total_papers,
                "by_category":            by_category,
                "top_values_per_category": top_values,
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_query)


@brahm_tool(
    name="research_find_papers_by_topic", group="research",
    description=(
        "Search extracted paper content for a topic using keyword matching "
        "against title, abstract, and/or PaperContent sections."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "keywords":  {"type": "array", "items": {"type": "string"}},
            "search_in": {
                "type": "array",
                "items": {"type": "string", "enum": ["title", "abstract", "content"]},
                "default": ["title", "abstract"],
            },
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["keywords"],
    },
)
async def research_find_papers_by_topic(args: dict) -> dict:
    keywords  = args["keywords"]
    search_in = args.get("search_in", ["title", "abstract"])
    limit     = args.get("limit", 20)

    if not keywords:
        return _err("keywords cannot be empty")

    def _search() -> dict:
        repo = _repo()
        try:
            matched_ids = set()
            results     = []

            for kw in keywords:
                kw_lower = kw.lower()
                if "title" in search_in:
                    rows = repo.fetch_all(
                        "SELECT DISTINCT id, workflow_id, title, abstract, status "
                        "FROM Paper WHERE LOWER(title) LIKE ? LIMIT ?",
                        (f"%{kw_lower}%", limit),
                    )
                    for r in rows:
                        if r["id"] not in matched_ids:
                            matched_ids.add(r["id"])
                            results.append(dict(r) | {"matched_keyword": kw, "matched_in": "title"})

                if "abstract" in search_in:
                    rows = repo.fetch_all(
                        "SELECT DISTINCT id, workflow_id, title, abstract, status "
                        "FROM Paper WHERE LOWER(abstract) LIKE ? LIMIT ?",
                        (f"%{kw_lower}%", limit),
                    )
                    for r in rows:
                        if r["id"] not in matched_ids:
                            matched_ids.add(r["id"])
                            results.append(dict(r) | {"matched_keyword": kw, "matched_in": "abstract"})

                if "content" in search_in:
                    rows = repo.fetch_all(
                        "SELECT DISTINCT p.id, p.workflow_id, p.title, p.abstract, p.status, "
                        "pc.section_name FROM Paper p "
                        "JOIN PaperContent pc ON pc.paper_id = p.id "
                        "WHERE LOWER(pc.content) LIKE ? LIMIT ?",
                        (f"%{kw_lower}%", limit),
                    )
                    for r in rows:
                        if r["id"] not in matched_ids:
                            matched_ids.add(r["id"])
                            results.append(dict(r) | {"matched_keyword": kw, "matched_in": f"content:{r['section_name']}"})

                if len(results) >= limit:
                    break

            return _ok({
                "keywords":  keywords,
                "search_in": search_in,
                "count":     len(results[:limit]),
                "papers":    results[:limit],
            })
        finally:
            repo.close()

    return await asyncio.to_thread(_search)


@brahm_tool(
    name="research_get_database_stats", group="research",
    description=(
        "Overall DB health metrics: workflow count, paper count, extraction rates, "
        "knowledge density. Use to assess data collection quality before export."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
)
async def research_get_database_stats(args: dict) -> dict:
    def _stats() -> dict:
        repo = _repo()
        try:
            row = repo.fetch_one(
                """
                SELECT
                  (SELECT COUNT(*) FROM Workflow)                          AS workflows,
                  (SELECT COUNT(*) FROM Workflow WHERE status='paused')    AS paused_wf,
                  (SELECT COUNT(*) FROM Workflow WHERE status='running')   AS running_wf,
                  (SELECT COUNT(*) FROM Paper)                             AS total_papers,
                  (SELECT COUNT(*) FROM Paper WHERE status='extracted')    AS extracted,
                  (SELECT COUNT(*) FROM Paper WHERE status='knowledge_ready') AS knowledge_ready,
                  (SELECT COUNT(*) FROM Paper WHERE status='completed')    AS completed_papers,
                  (SELECT COUNT(*) FROM Paper WHERE status='failed')       AS failed,
                  (SELECT COUNT(*) FROM PaperContent)                      AS content_sections,
                  (SELECT COUNT(*) FROM ResearchKnowledge)                 AS knowledge_rows,
                  (SELECT COUNT(DISTINCT paper_id) FROM ResearchKnowledge) AS papers_with_knowledge
                FROM (SELECT 1)
                """
            )
            stats = dict(row)
            total     = stats.get("total_papers", 0) or 1
            extracted = (stats.get("extracted", 0)
                         + stats.get("knowledge_ready", 0)
                         + stats.get("completed_papers", 0))
            stats["extraction_rate_pct"] = round(extracted / total * 100, 1)
            stats["knowledge_density"]   = (
                round(stats["knowledge_rows"] / max(stats["papers_with_knowledge"], 1), 1)
                if stats["knowledge_rows"] else 0
            )
            return _ok({"stats": stats})
        finally:
            repo.close()

    return await asyncio.to_thread(_stats)
