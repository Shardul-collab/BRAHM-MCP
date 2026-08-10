"""
brahm/agents/shani.py
======================
Group A — SHANI pipeline tools (S1-S5_5).
S6, S7 (old draft-writing stages) removed — that writing is handled by
GANESH now. S5_5 (reconstruct_findings) is NOT redundant with GANESH and
is kept: GANESH's G1 reads raw ResearchKnowledge rows directly (routing
tags into document sections), while gap_detection.py / 
gap_detection_batch.py specifically need S5_5's synthesized, per-paper
ResearchFinding sentences to reason about corpus-wide gaps. Two different
consumers, two different representations — previously the STAGE_ENUM
below omitted 'S5_5' based on a mistaken assumption that it was only
for the now-removed writing stages.
"""

import asyncio
import subprocess
from datetime import datetime
from pathlib import Path

from brahm.brahm_registry import brahm_tool, requires_api
from brahm.shared.helpers import _ok, _err, _repo
from brahm.shared.http import _shani_get, _shani_post, _check_shani, SHANI_START_HINT
from brahm.shared.constants import (
    SHANI_VENV_PY, SHANI_STAGE_SEQUENCE, SHANI_VALID_STAGES, QUEUE_PATH
)

STAGE_ENUM = ["S1", "S2", "S2_75", "S2_5", "S3", "S4", "S5", "S5_5"]


@brahm_tool(
    name="shani_create_workflow", group="shani",
    description=(
        "Create a new research workflow with topic configuration. "
        "Returns workflow_id needed for all subsequent operations. "
        "Starts in 'paused' state — call shani_run_workflow to begin. "
        "The 'focus' field is the most critical: space-separated keywords."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name":             {"type": "string"},
            "material":         {"type": "string"},
            "focus":            {"type": "string"},
            "structure":        {"type": "string"},
            "method":           {"type": "string"},
            "properties":       {"type": "string"},
            "characterization": {"type": "string"},
            "use_local":        {"type": "boolean", "default": False},
        },
        "required": ["name"],
    },
)
@requires_api(_check_shani, "SHANI", SHANI_START_HINT)
async def shani_create_workflow(args: dict) -> dict:
    body = {k: v for k, v in args.items() if v is not None}
    result = await _shani_post("/workflows", body)
    if result.get("status") == "error":
        return result
    if "workflow_id" in result:
        return _ok({"workflow_id": result["workflow_id"], "shani_status": result.get("status")})
    return _ok(result)


@brahm_tool(
    name="shani_run_workflow", group="shani",
    description=(
        "Start a paused workflow. Runs asynchronously — returns immediately. "
        "Poll shani_get_status to monitor. Default stop is S4."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workflow_id":      {"type": "integer"},
            "stop_after_stage": {"type": "string", "enum": STAGE_ENUM, "default": "S4"},
        },
        "required": ["workflow_id"],
    },
)
@requires_api(_check_shani, "SHANI", SHANI_START_HINT)
async def shani_run_workflow(args: dict) -> dict:
    wf_id      = args["workflow_id"]
    stop_after = args.get("stop_after_stage", "S4")
    result = await _shani_post(f"/workflows/{wf_id}/run", {"stop_after_stage": stop_after})
    if result.get("status") == "error":
        return result
    return _ok(result)


@brahm_tool(
    name="shani_batch_run", group="shani",
    description=(
        "Create and run multiple workflows concurrently (1-20). "
        "All run in parallel. Returns all workflow IDs immediately."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workflows": {
                "type": "array", "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "name":             {"type": "string"},
                        "material":         {"type": "string"},
                        "focus":            {"type": "string"},
                        "structure":        {"type": "string"},
                        "method":           {"type": "string"},
                        "properties":       {"type": "string"},
                        "characterization": {"type": "string"},
                        "use_local":        {"type": "boolean", "default": False},
                    },
                    "required": ["name"],
                },
            },
            "stop_after_stage": {"type": "string", "enum": STAGE_ENUM, "default": "S4"},
        },
        "required": ["workflows"],
    },
)
@requires_api(_check_shani, "SHANI", SHANI_START_HINT)
async def shani_batch_run(args: dict) -> dict:
    result = await _shani_post("/workflows/batch", args)
    if result.get("status") == "error":
        return result
    return _ok(result)


@brahm_tool(
    name="shani_get_status", group="shani",
    description="Get full status of a workflow: all stage records + latest execution attempt.",
    input_schema={
        "type": "object",
        "properties": {"workflow_id": {"type": "integer"}},
        "required": ["workflow_id"],
    },
)
@requires_api(_check_shani, "SHANI", SHANI_START_HINT)
async def shani_get_status(args: dict) -> dict:
    result = await _shani_get(f"/workflows/{args['workflow_id']}/status")
    if result.get("status") == "error":
        return result
    return _ok(result)


@brahm_tool(
    name="shani_get_all_status", group="shani",
    description="Summary of ALL workflows: ID, name, status, current stage, paper counts.",
    input_schema={"type": "object", "properties": {}, "required": []},
)
async def shani_get_all_status(args: dict) -> dict:
    def _query() -> dict:
        repo = _repo()
        try:
            rows = repo.fetch_all(
                """
                SELECT w.id, w.name, w.status, w.current_stage,
                  (SELECT COUNT(*) FROM Paper p WHERE p.workflow_id=w.id) AS papers,
                  (SELECT COUNT(*) FROM Paper p WHERE p.workflow_id=w.id
                   AND p.status IN ('extracted','knowledge_ready','completed')) AS extracted,
                  (SELECT COUNT(*) FROM Paper p WHERE p.workflow_id=w.id
                   AND p.status='failed') AS failed
                FROM Workflow w ORDER BY w.id
                """
            )
            workflows = [dict(r) for r in rows]
            return _ok({
                "total_workflows": len(workflows),
                "running": sum(1 for w in workflows if w["status"] == "running"),
                "paused":  sum(1 for w in workflows if w["status"] == "paused"),
                "workflows": workflows,
            })
        finally:
            repo.close()
    return await asyncio.to_thread(_query)


@brahm_tool(
    name="shani_get_papers", group="shani",
    description="Get all papers collected by a workflow, with optional status filter.",
    input_schema={
        "type": "object",
        "properties": {
            "workflow_id":   {"type": "integer"},
            "status_filter": {
                "type": "string",
                "enum": ["all","extracted","pending","failed","knowledge_ready","completed"],
                "default": "all",
            },
        },
        "required": ["workflow_id"],
    },
)
@requires_api(_check_shani, "SHANI", SHANI_START_HINT)
async def shani_get_papers(args: dict) -> dict:
    wf_id         = args["workflow_id"]
    status_filter = args.get("status_filter", "all")
    result = await _shani_get(f"/workflows/{wf_id}/papers")
    if result.get("status") == "error":
        return result
    papers = result.get("papers", [])
    if status_filter != "all":
        papers = [p for p in papers if p.get("status") == status_filter]
    return _ok({"workflow_id": wf_id, "count": len(papers), "papers": papers})


@brahm_tool(
    name="shani_get_paper_content", group="shani",
    description="Get all extracted text sections for a specific paper.",
    input_schema={
        "type": "object",
        "properties": {
            "workflow_id": {"type": "integer"},
            "paper_id":    {"type": "integer"},
        },
        "required": ["workflow_id", "paper_id"],
    },
)
@requires_api(_check_shani, "SHANI", SHANI_START_HINT)
async def shani_get_paper_content(args: dict) -> dict:
    result = await _shani_get(
        f"/workflows/{args['workflow_id']}/papers/{args['paper_id']}/content"
    )
    if result.get("status") == "error":
        return result
    return _ok(result)


@brahm_tool(
    name="shani_extract_workflow_data", group="shani",
    description="Full dump of all papers + extracted content for a workflow.",
    input_schema={
        "type": "object",
        "properties": {"workflow_id": {"type": "integer"}},
        "required": ["workflow_id"],
    },
)
@requires_api(_check_shani, "SHANI", SHANI_START_HINT)
async def shani_extract_workflow_data(args: dict) -> dict:
    result = await _shani_get(f"/workflows/{args['workflow_id']}/extract")
    if result.get("status") == "error":
        return result
    return _ok(result)


@brahm_tool(
    name="shani_clear_database", group="shani",
    description="DESTRUCTIVE: Delete all workflows, papers, content. Must pass confirm=true.",
    input_schema={
        "type": "object",
        "properties": {"confirm": {"type": "boolean", "enum": [True]}},
        "required": ["confirm"],
    },
)
async def shani_clear_database(args: dict) -> dict:
    if not args.get("confirm"):
        return _err("confirm must be true. This operation is irreversible.")
    def _run() -> dict:
        try:
            r = subprocess.run(
                [SHANI_VENV_PY, "-m", "core.shani", "del_r"],
                cwd=str(Path(SHANI_VENV_PY).parents[2]),
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode != 0:
                return _err("clear_database failed", r.stderr[:500])
            return _ok({"message": "Database cleared.", "stdout": r.stdout[-1000:]})
        except subprocess.TimeoutExpired:
            return _err("clear_database timed out")
        except Exception as exc:
            return _err("clear_database failed", str(exc))
    return await asyncio.to_thread(_run)


@brahm_tool(
    name="shani_reset_workflow", group="shani",
    description=(
        "Reset a failed or stuck workflow back to 'paused'. "
        "Optionally specify from_stage to delete stages from that point onwards."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workflow_id": {"type": "integer"},
            "from_stage":  {"type": "string", "enum": STAGE_ENUM},
        },
        "required": ["workflow_id"],
    },
)
async def shani_reset_workflow(args: dict) -> dict:
    wf_id      = args["workflow_id"]
    from_stage = args.get("from_stage")
    def _reset() -> dict:
        repo = _repo()
        try:
            wf = repo.fetch_one("SELECT id, status, name FROM Workflow WHERE id=?", (wf_id,))
            if not wf:
                return _err(f"Workflow {wf_id} not found")
            if from_stage:
                seq       = SHANI_STAGE_SEQUENCE
                start_idx = seq.index(from_stage)
                with repo.transaction() as cursor:
                    for s in seq[start_idx:]:
                        cursor.execute(
                            "DELETE FROM Stage WHERE workflow_id=? AND stage_name=?", (wf_id, s)
                        )
                    # If resetting to or through S5, clear extracted knowledge and reset paper statuses
                    if "S5" in seq[start_idx:]:
                        cursor.execute(
                            "DELETE FROM ResearchKnowledge WHERE paper_id IN "
                            "(SELECT id FROM Paper WHERE workflow_id=?)", (wf_id,)
                        )
                        cursor.execute(
                            "UPDATE Paper SET status='extracted' WHERE workflow_id=? "
                            "AND status IN ('knowledge_ready','completed')", (wf_id,)
                        )
                    cursor.execute(
                        "UPDATE Workflow SET status='paused', current_stage=?, updated_at=? WHERE id=?",
                        (from_stage, datetime.utcnow().isoformat(), wf_id),
                    )
            else:
                with repo.transaction() as cursor:
                    cursor.execute(
                        "UPDATE Workflow SET status='paused', updated_at=? WHERE id=?",
                        (datetime.utcnow().isoformat(), wf_id),
                    )
            return _ok({
                "workflow_id":     wf_id,
                "name":            dict(wf)["name"],
                "previous_status": dict(wf)["status"],
                "new_status":      "paused",
                "from_stage":      from_stage,
            })
        finally:
            repo.close()
    return await asyncio.to_thread(_reset)


@brahm_tool(
    name="queue_add_workflow", group="shani",
    description="Add a workflow config to the local queue file without running it.",
    input_schema={
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
)
async def queue_add_workflow(args: dict) -> dict:
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
        return _ok({"queued": args.get("name"), "queue_length": len(existing)})
    except Exception as exc:
        return _err("Queue write failed", str(exc))


@brahm_tool(
    name="shani_run_knowledge_extraction", group="shani",
    description=(
        "Trigger S5 (extract_research_knowledge) on workflows paused at S4. "
        "Validates preconditions: workflow must be paused with sufficient extracted papers."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workflow_ids":         {"type": "array", "items": {"type": "integer"}},
            "min_extracted_papers": {"type": "integer", "default": 10},
        },
        "required": ["workflow_ids"],
    },
)
@requires_api(_check_shani, "SHANI", SHANI_START_HINT)
async def shani_run_knowledge_extraction(args: dict) -> dict:
    workflow_ids = args["workflow_ids"]
    min_papers   = args.get("min_extracted_papers", 10)
    dispatched, skipped = [], []
    for wf_id in workflow_ids:
        status_result = await _shani_get(f"/workflows/{wf_id}/status")
        if status_result.get("status") == "error":
            skipped.append({"id": wf_id, "reason": status_result.get("error")})
            continue
        wf = status_result.get("workflow", {})
        if wf.get("status") != "paused":
            skipped.append({"id": wf_id, "reason": f"status must be paused, got '{wf.get('status')}'"})
            continue
        def _count(wid=wf_id):
            repo = _repo()
            try:
                row = repo.fetch_one(
                    "SELECT COUNT(*) FROM Paper WHERE workflow_id=? "
                    "AND status IN ('extracted','knowledge_ready','completed')", (wid,)
                )
                return row[0]
            finally:
                repo.close()
        count = await asyncio.to_thread(_count)
        if count < min_papers:
            skipped.append({"id": wf_id, "reason": f"Only {count} papers (min {min_papers})"})
            continue
        run = await _shani_post(f"/workflows/{wf_id}/run", {"stop_after_stage": "S5"})
        if run.get("status") == "error":
            skipped.append({"id": wf_id, "reason": run.get("error")})
        else:
            dispatched.append({"id": wf_id})
    return _ok({
        "dispatched": dispatched, "skipped": skipped,
        "dispatched_count": len(dispatched), "skipped_count": len(skipped),
    })
