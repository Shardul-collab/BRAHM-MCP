"""
brahm/coordinator/app.py
=========================
BRAHM Coordinator — Layer 1.

A FastAPI HTTP layer over the SAME brahm_registry used by mcp_server.py.
This does NOT replace mcp_server.py — the stdio MCP entry point is
untouched and keeps working for Claude Desktop / other MCP clients.
This is an additional consumer of the same tool registry, for the
browser dashboard.

Scope (deliberately narrow, per project decision):
  - Trigger pipelines via the same registry.dispatch(name, args) calls the
    MCP layer already uses — no pipeline logic duplicated, no orchestrator
    files (SHANI's or GANESH's) read from or written to.
  - Provide input for those pipelines. v1 = structured form fields (below),
    matching the fields shani_create_workflow already accepts. The
    conversational LLM router discussed as "Layer 2" is a separate,
    later phase — not built here.
  - Aggregate status across agents into one polling surface (/v2/status)
    and a WebSocket push feed (/v2/ws) for the live dashboard.

Run:
  cd /mnt/d/brahm
  python3 -m uvicorn brahm.coordinator.app:app --host 0.0.0.0 --port 8010 --reload
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── sys.path — same ordering rule as mcp_server.py. DO NOT REORDER ──────────
# Chitragupta MUST be index 0 (its own core/ package must not be shadowed
# by SHANI's core/). Copied verbatim from mcp_server.py's documented order.
BRAHM_ROOT = os.environ.get("BRAHM_ROOT", "/mnt/d/brahm")
_A = f"{BRAHM_ROOT}/agents"
sys.path.insert(0, BRAHM_ROOT)
sys.path.insert(0, f"{_A}/shani")
sys.path.insert(0, f"{_A}/chitragupta/analysis")
sys.path.insert(0, f"{_A}/vidur")
sys.path.insert(0, f"{_A}/vishwakarma")
sys.path.insert(0, f"{_A}/ganesh")
sys.path.insert(0, f"{_A}/chitragupta")

from dotenv import load_dotenv
load_dotenv(f"{BRAHM_ROOT}/agents/chitragupta/.env")

from brahm.brahm_registry import registry

# Import agent modules — this registers all @brahm_tool handlers, exactly
# as mcp_server.py does. Required before any registry.dispatch call below.
import brahm.agents.shani           # noqa: F401
import brahm.agents.chitragupta     # noqa: F401
import brahm.agents.research        # noqa: F401
import brahm.agents.analysis        # noqa: F401
import brahm.agents.db_tools        # noqa: F401
import brahm.agents.vidur           # noqa: F401
import brahm.agents.vishwakarma     # noqa: F401
import brahm.agents.ganesh          # noqa: F401
import brahm.agents.meta            # noqa: F401

from brahm.coordinator.run_store import RunStore
from brahm.coordinator.orchestration import run_shani_then_ganesh
from brahm.coordinator.plan_store import PlanStore
from brahm.coordinator.planning import generate_plan, execute_plan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("coordinator")

run_store = RunStore()
plan_store = PlanStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("BRAHM Coordinator starting — %d tools available via registry", len(registry))
    yield
    log.info("BRAHM Coordinator shut down.")


app = FastAPI(
    title       = "BRAHM Coordinator",
    description = "HTTP layer over brahm_registry for the BRAHM dashboard.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    name:             str
    material:         Optional[str] = None
    focus:            Optional[str] = None
    structure:        Optional[str] = None
    method:           Optional[str] = None
    properties:       Optional[str] = None
    characterization: Optional[str] = None
    use_local:        bool = False
    max_papers:       Optional[int] = None  # None = SHANI's own default (500)
    document_type:    str  = "literature_review"
    auto_write:       bool = True   # chain into GANESH after SHANI finishes


class AttachRequest(BaseModel):
    workflow_id:   int
    document_type: str  = "literature_review"
    auto_write:    bool = True


# ═══════════════════════════════════════════════════════════════════════════
# /v2/run
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/v2/run")
async def start_run(req: RunRequest):
    """
    Trigger a SHANI workflow (full S1 -> S5_5) via registry.dispatch. If
    auto_write=True (default), a background watcher chains into GANESH's
    write + synthesize once SHANI reports workflow.status == 'completed'.
    Returns immediately — poll GET /v2/status/{run_id} or open the
    WebSocket at /v2/ws/{run_id} to follow progress.
    """
    create_args = {k: v for k, v in {
        "name":             req.name,
        "material":         req.material,
        "focus":            req.focus,
        "structure":        req.structure,
        "method":           req.method,
        "properties":       req.properties,
        "characterization": req.characterization,
        "use_local":        req.use_local,
        "max_papers":       req.max_papers,
    }.items() if v is not None}

    create_result = await registry.dispatch("shani_create_workflow", create_args)
    if create_result.get("status") == "error":
        raise HTTPException(status_code=502, detail=create_result)

    workflow_id = create_result.get("workflow_id")
    if workflow_id is None:
        raise HTTPException(
            status_code=502,
            detail={"message": "shani_create_workflow did not return workflow_id", "raw": create_result},
        )

    run_id = run_store.create(
        workflow_id=workflow_id,
        document_type=req.document_type,
        auto_write=req.auto_write,
    )

    run_result = await registry.dispatch("shani_run_workflow", {
        "workflow_id":      workflow_id,
        "stop_after_stage": "S5_5",
    })
    if run_result.get("status") == "error":
        run_store.set_error(run_id, run_result.get("error", "shani_run_workflow failed"))
        raise HTTPException(status_code=502, detail=run_result)

    run_store.set_phase(run_id, "shani_running")

    # Background watcher — polls SHANI's existing status endpoint, and on
    # completion (if auto_write) calls GANESH's existing write/synthesize
    # tools. No orchestrator file touched; only existing registered tools
    # are called, exactly as documented in orchestration.py.
    asyncio.create_task(run_shani_then_ganesh(run_id, run_store, registry))

    return {
        "ok":          True,
        "run_id":      run_id,
        "workflow_id": workflow_id,
        "phase":       "shani_running",
    }


@app.post("/v2/attach")
async def attach_run(req: AttachRequest):
    """
    Re-attach a coordinator watcher to a SHANI workflow that is ALREADY
    running (created outside this endpoint, or whose original watcher
    exited early — e.g. from a timeout). Does NOT call shani_create_workflow
    or shani_run_workflow again — it only starts a new background watcher
    against the existing workflow_id, exactly like the one /v2/run starts.

    Use this to recover a run without touching or restarting SHANI itself.
    """
    status = await registry.dispatch("shani_get_status", {"workflow_id": req.workflow_id})
    if status.get("status") == "error":
        raise HTTPException(status_code=502, detail=status)
    if not status.get("workflow"):
        raise HTTPException(status_code=404, detail=f"workflow_id {req.workflow_id} not found in SHANI")

    run_id = run_store.create(
        workflow_id=req.workflow_id,
        document_type=req.document_type,
        auto_write=req.auto_write,
    )
    run_store.set_phase(run_id, "shani_running")

    asyncio.create_task(run_shani_then_ganesh(run_id, run_store, registry))

    return {
        "ok":          True,
        "run_id":      run_id,
        "workflow_id": req.workflow_id,
        "phase":       "shani_running",
        "message":     "Watcher (re)attached. SHANI workflow was not restarted.",
    }


# ═══════════════════════════════════════════════════════════════════════════
# /v2/status/{run_id}
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/v2/status/{run_id}")
async def get_run_status(run_id: str):
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run_id {run_id} not found")

    result = {
        "run_id":      run_id,
        "workflow_id": run["workflow_id"],
        "document_id": run.get("document_id"),
        "phase":       run["phase"],
        "error":       run.get("error"),
        "created_at":  run["created_at"],
    }

    shani_status = await registry.dispatch("shani_get_status", {"workflow_id": run["workflow_id"]})
    result["shani"] = shani_status

    if run.get("document_id") is not None:
        ganesh_status = await registry.dispatch("ganesh_get_document", {"document_id": run["document_id"]})
        result["ganesh"] = ganesh_status

    return result


@app.get("/v2/runs")
async def list_runs():
    """All runs tracked by this coordinator process since it last started."""
    return {"ok": True, "runs": run_store.all_runs()}


# ═══════════════════════════════════════════════════════════════════════════
# /v2/tool/{tool_name} — generic passthrough to any registered tool
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/v2/tool/{tool_name}")
async def call_tool(tool_name: str, args: dict):
    """
    Generic escape hatch: calls registry.dispatch(tool_name, args) directly,
    for any tool already registered via @brahm_tool — e.g. admin/recovery
    actions like shani_reset_workflow that don't have a dedicated /v2/*
    endpoint of their own. This is the SAME dispatch call the MCP/stdio
    layer makes; no new logic, no orchestrator touched.

    Example:
      POST /v2/tool/shani_reset_workflow
      {"workflow_id": 2, "from_stage": "S5"}
    """
    if tool_name not in registry:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
    result = await registry.dispatch(tool_name, args or {})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# /v2/plan — Layer 2, conversational plan proposal + approve/reject
# ═══════════════════════════════════════════════════════════════════════════

class PlanRequest(BaseModel):
    prompt: str


class ApproveRequest(BaseModel):
    edited_steps: Optional[list] = None  # edit-and-resubmit support


@app.post("/v2/plan")
async def create_plan(req: PlanRequest):
    """
    Runs a state-check against real registry tools, then calls NIM
    (gpt-oss-120b) to propose a plan. Every proposed step is validated
    against planning.ALLOWED_ACTIONS before being stored — the LLM
    cannot select destructive tools regardless of what it proposes.
    Returns the plan with status=pending_approval. Nothing executes yet.
    """
    try:
        plan = await generate_plan(req.prompt, plan_store)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return plan


@app.get("/v2/plan/{plan_id}")
async def get_plan(plan_id: str):
    plan = plan_store.get(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"plan_id {plan_id} not found")
    return plan


@app.post("/v2/plan/{plan_id}/approve")
async def approve_plan(plan_id: str, req: ApproveRequest):
    """
    Approves a plan and executes only its non-skip steps, in order, via
    the same registry.dispatch calls /v2/run already uses. If
    edited_steps is provided, it replaces the stored steps first
    (researcher tweaked something before approving).
    """
    plan = plan_store.get(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"plan_id {plan_id} not found")
    if req.edited_steps is not None:
        plan_store.set_steps(plan_id, req.edited_steps)
    plan_store.set_status(plan_id, "approved")
    asyncio.create_task(execute_plan(plan_id, plan_store, run_store))
    return {"ok": True, "plan_id": plan_id, "status": "executing"}


@app.post("/v2/plan/{plan_id}/reject")
async def reject_plan(plan_id: str):
    plan = plan_store.get(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"plan_id {plan_id} not found")
    plan_store.set_status(plan_id, "rejected")
    return {"ok": True, "plan_id": plan_id, "status": "rejected"}


# ═══════════════════════════════════════════════════════════════════════════
# /v2/health, /v2/overview — thin wrappers, zero new logic
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/v2/health")
async def get_health():
    """Thin wrapper around the existing brahm_health tool."""
    return await registry.dispatch("brahm_health", {})


@app.get("/v2/overview")
async def get_overview():
    """Thin wrapper around the existing brahm_overview tool."""
    return await registry.dispatch("brahm_overview", {})


# ═══════════════════════════════════════════════════════════════════════════
# WS /v2/ws/{run_id} — push layer for the live dashboard
# ═══════════════════════════════════════════════════════════════════════════

@app.websocket("/v2/ws/{run_id}")
async def run_status_ws(websocket: WebSocket, run_id: str):
    """
    Polls internal + agent state every ~1.5s and pushes to the browser
    ONLY when something actually changed. This is the coordinator polling
    each agent's existing read endpoints (same calls get_run_status makes)
    — no agent orchestrator is touched, and no agent is modified to push
    on its own.
    """
    await websocket.accept()

    run = run_store.get(run_id)
    if run is None:
        await websocket.send_json({"ok": False, "error": f"run_id {run_id} not found"})
        await websocket.close()
        return

    last_payload = None
    try:
        while True:
            run = run_store.get(run_id)
            if run is None:
                break

            shani_status = await registry.dispatch("shani_get_status", {"workflow_id": run["workflow_id"]})
            ganesh_status = None
            if run.get("document_id") is not None:
                ganesh_status = await registry.dispatch("ganesh_get_document", {"document_id": run["document_id"]})

            payload = {
                "run_id":      run_id,
                "workflow_id": run["workflow_id"],
                "document_id": run.get("document_id"),
                "phase":       run["phase"],
                "error":       run.get("error"),
                "shani":       shani_status,
                "ganesh":      ganesh_status,
            }

            if payload != last_payload:
                await websocket.send_json(payload)
                last_payload = payload

            if run["phase"] in ("complete", "failed"):
                break

            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        log.info("WebSocket disconnected for run_id=%s", run_id)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("brahm.coordinator.app:app", host="0.0.0.0", port=8010, log_level="info")
