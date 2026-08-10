"""
brahm_db/api.py
================
CHITRAGUPTA central data API routes.

This module defines `router` (an APIRouter), which is imported and mounted
into the real, live Chitragupta process by api/app.py's create_app().
That merged app (started via api_server.py) is the only running instance —
this file no longer runs standalone; the previous direct-run fallback
(`python -m brahm_db.api` on its own port 8003) has been removed since it
would now just conflict with the real service on the same port.

Endpoints:
  /health                          — liveness probe
  /v1/projects                     — project CRUD
  /v1/projects/{id}/events         — project timeline
  /v1/projects/{id}/workloads      — workload management
  /v1/projects/{id}/decisions      — decision points
  /v1/projects/{id}/cycles         — research cycles
  /v1/papers/check                 — deduplication check
  /v1/papers                       — register + list papers
  /v1/results/instrument           — VIDUR results
  /v1/results/dft                  — Vishwakarma results
  /v1/documents                    — GANESH documents
  /v1/documents/{id}/sections      — document sections
"""

from __future__ import annotations

import json
import logging
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from brahm_db.repositories import (
    ProjectRepo, PaperRepo,
    InstrumentResultRepo, DFTResultRepo, DocumentRepo,
)

log = logging.getLogger("brahm_db.api")

router = APIRouter()


def _err(msg: str, code: int = 400) -> HTTPException:
    return HTTPException(status_code=code, detail={"ok": False, "message": msg})


# ═════════════════════════════════════════════════════════════════════════════
# HEALTH
# ═════════════════════════════════════════════════════════════════════════════

@router.get("/health", tags=["Health"])
async def health():
    return {"ok": True, "service": "CHITRAGUPTA brahm_db", "port": 8003}


# ═════════════════════════════════════════════════════════════════════════════
# PROJECTS
# ═════════════════════════════════════════════════════════════════════════════

class CreateProjectRequest(BaseModel):
    name: str
    objective: str

class EventRequest(BaseModel):
    agent: str
    event_type: str
    summary: str
    payload: dict | None = None

class CompleteEventRequest(BaseModel):
    status: str = "completed"

class WorkloadRequest(BaseModel):
    agent: str
    workload_type: str
    config: dict
    priority: int = 5

class WorkloadStatusRequest(BaseModel):
    status: str

class DecisionRequest(BaseModel):
    question: str
    options: list | None = None
    recommendation: str | None = None
    workload_id: int | None = None

class ResolveDecisionRequest(BaseModel):
    human_response: str

class CycleRequest(BaseModel):
    objective: str | None = None


@router.post("/v1/projects", tags=["Projects"])
async def create_project(body: CreateProjectRequest):
    with ProjectRepo() as r:
        pid = r.create_project(body.name, body.objective)
        return {"ok": True, "project_id": pid}


@router.get("/v1/projects", tags=["Projects"])
async def list_projects(status: str | None = None):
    with ProjectRepo() as r:
        return {"ok": True, "projects": r.list_projects(status)}


@router.get("/v1/projects/{project_id}", tags=["Projects"])
async def get_project(project_id: int):
    with ProjectRepo() as r:
        p = r.get_project(project_id)
        if not p:
            raise _err(f"Project {project_id} not found", 404)
        p["pending_decisions"] = r.get_pending_decisions(project_id)
        p["active_cycle"]      = r.get_active_cycle(project_id)
        return {"ok": True, "project": p}


@router.patch("/v1/projects/{project_id}/status", tags=["Projects"])
async def update_project_status(project_id: int, body: dict):
    status = body.get("status")
    if not status:
        raise _err("status required")
    with ProjectRepo() as r:
        r.update_project_status(project_id, status)
        return {"ok": True, "project_id": project_id, "status": status}


# ── Events ────────────────────────────────────────────────────────────────────

@router.post("/v1/projects/{project_id}/events", tags=["Events"])
async def log_event(project_id: int, body: EventRequest):
    with ProjectRepo() as r:
        eid = r.log_event(
            project_id, body.agent, body.event_type,
            body.summary, body.payload,
        )
        return {"ok": True, "event_id": eid}


@router.patch("/v1/events/{event_id}/complete", tags=["Events"])
async def complete_event(event_id: int, body: CompleteEventRequest):
    with ProjectRepo() as r:
        r.complete_event(event_id, body.status)
        return {"ok": True, "event_id": event_id, "status": body.status}


@router.get("/v1/projects/{project_id}/events", tags=["Events"])
async def get_events(
    project_id: int,
    agent: str | None = None,
    since: str | None = None,
    today_only: bool = False,
):
    with ProjectRepo() as r:
        if today_only:
            events = r.get_today_events(project_id)
        else:
            events = r.get_events(project_id, agent=agent, since=since)
        return {"ok": True, "count": len(events), "events": events}


# ── Workloads ─────────────────────────────────────────────────────────────────

@router.post("/v1/projects/{project_id}/workloads", tags=["Workloads"])
async def create_workload(project_id: int, body: WorkloadRequest):
    with ProjectRepo() as r:
        wid = r.create_workload(
            project_id, body.agent, body.workload_type,
            body.config, body.priority,
        )
        return {"ok": True, "workload_id": wid}


@router.get("/v1/projects/{project_id}/workloads", tags=["Workloads"])
async def list_workloads(
    project_id: int,
    status: str | None = None,
    agent: str | None = None,
):
    with ProjectRepo() as r:
        return {
            "ok": True,
            "workloads": r.list_workloads(project_id, status, agent),
        }


@router.patch("/v1/workloads/{workload_id}/status", tags=["Workloads"])
async def update_workload_status(workload_id: int, body: WorkloadStatusRequest):
    with ProjectRepo() as r:
        r.update_workload_status(workload_id, body.status)
        return {"ok": True, "workload_id": workload_id, "status": body.status}


@router.get("/v1/workloads/next/{agent}", tags=["Workloads"])
async def next_workload(agent: str):
    with ProjectRepo() as r:
        w = r.get_next_queued_workload(agent)
        return {"ok": True, "workload": w}


# ── Decisions ─────────────────────────────────────────────────────────────────

@router.post("/v1/projects/{project_id}/decisions", tags=["Decisions"])
async def create_decision(project_id: int, body: DecisionRequest):
    with ProjectRepo() as r:
        did = r.create_decision(
            project_id, body.question, body.options,
            body.recommendation, body.workload_id,
        )
        return {"ok": True, "decision_id": did}


@router.get("/v1/projects/{project_id}/decisions", tags=["Decisions"])
async def get_decisions(project_id: int, status: str | None = None):
    with ProjectRepo() as r:
        if status == "pending" or status is None:
            decisions = r.get_pending_decisions(project_id)
        else:
            decisions = []
        return {"ok": True, "decisions": decisions}


@router.post("/v1/decisions/{decision_id}/resolve", tags=["Decisions"])
async def resolve_decision(decision_id: int, body: ResolveDecisionRequest):
    with ProjectRepo() as r:
        d = r.get_decision(decision_id)
        if not d:
            raise _err(f"Decision {decision_id} not found", 404)
        r.resolve_decision(decision_id, body.human_response)
        return {"ok": True, "decision_id": decision_id, "resolved": True}


# ── Cycles ────────────────────────────────────────────────────────────────────

@router.post("/v1/projects/{project_id}/cycles", tags=["Cycles"])
async def create_cycle(project_id: int, body: CycleRequest):
    with ProjectRepo() as r:
        cid = r.create_cycle(project_id, body.objective)
        return {"ok": True, "cycle_id": cid}


@router.get("/v1/projects/{project_id}/cycles/active", tags=["Cycles"])
async def get_active_cycle(project_id: int):
    with ProjectRepo() as r:
        c = r.get_active_cycle(project_id)
        return {"ok": True, "cycle": c}


# ═════════════════════════════════════════════════════════════════════════════
# PAPERS (deduplication)
# ═════════════════════════════════════════════════════════════════════════════

class PaperCheckRequest(BaseModel):
    doi: str | None = None
    title: str | None = None

class RegisterPaperRequest(BaseModel):
    title: str
    doi: str | None = None
    abstract: str | None = None
    project_id: int | None = None
    workflow_id: int | None = None
    shani_paper_id: int | None = None


@router.post("/v1/papers/check", tags=["Papers"])
async def check_paper(body: PaperCheckRequest):
    """
    SHANI calls this before downloading any paper.
    Returns existing=True + global_paper_id if already registered.
    """
    with PaperRepo() as r:
        existing = r.check_paper(body.doi, body.title)
        if existing:
            return {
                "ok": True,
                "exists": True,
                "global_paper_id": existing["id"],
                "shani_paper_id":  existing.get("shani_paper_id"),
            }
        return {"ok": True, "exists": False}


@router.post("/v1/papers", tags=["Papers"])
async def register_paper(body: RegisterPaperRequest):
    """SHANI calls this after successfully downloading a new paper."""
    with PaperRepo() as r:
        gid = r.register_paper(
            title=body.title,
            doi=body.doi,
            abstract=body.abstract,
            project_id=body.project_id,
            workflow_id=body.workflow_id,
            shani_paper_id=body.shani_paper_id,
        )
        return {"ok": True, "global_paper_id": gid}


@router.post("/v1/papers/{global_paper_id}/link", tags=["Papers"])
async def link_paper(global_paper_id: int, body: dict):
    """Link existing paper to a new project/workflow."""
    with PaperRepo() as r:
        r.link_paper_to_project(
            global_paper_id,
            body["project_id"],
            body.get("workflow_id"),
        )
        return {"ok": True, "linked": True}


@router.get("/v1/papers/stats", tags=["Papers"])
async def paper_stats():
    with PaperRepo() as r:
        return {"ok": True, "stats": r.stats()}


@router.get("/v1/papers", tags=["Papers"])
async def list_papers(
    project_id: int | None = None,
    not_vector_indexed: bool = False,
    limit: int = Query(default=100, le=500),
):
    with PaperRepo() as r:
        papers = r.list_papers(
            project_id=project_id,
            not_vector_indexed=not_vector_indexed,
            limit=limit,
        )
        return {"ok": True, "count": len(papers), "papers": papers}


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═════════════════════════════════════════════════════════════════════════════

class InstrumentResultRequest(BaseModel):
    project_id: int
    file_path: str
    technique: str
    confidence: float
    signals: list
    parsed_data: dict
    cycle_id: int | None = None
    comparison_result: dict | None = None
    match_score: float | None = None
    gaps_identified: list | None = None

class DFTResultRequest(BaseModel):
    project_id: int
    job_id: str
    calc_type: str
    structure: dict | None = None
    input_params: dict | None = None
    output_parsed: dict | None = None
    status: str = "completed"
    wall_time_seconds: float | None = None
    cycle_id: int | None = None


@router.post("/v1/results/instrument", tags=["Results"])
async def save_instrument_result(body: InstrumentResultRequest):
    with InstrumentResultRepo() as r:
        rid = r.save(**body.model_dump())
        return {"ok": True, "result_id": rid}


@router.get("/v1/results/instrument", tags=["Results"])
async def list_instrument_results(
    project_id: int,
    technique: str | None = None,
    cycle_id: int | None = None,
):
    with InstrumentResultRepo() as r:
        results = r.list_for_project(project_id, technique, cycle_id)
        return {"ok": True, "count": len(results), "results": results}


@router.get("/v1/results/instrument/gaps", tags=["Results"])
async def get_instrument_gaps(project_id: int):
    """Returns all literature gaps identified by VIDUR — triggers new SHANI workloads."""
    with InstrumentResultRepo() as r:
        gaps = r.get_gaps(project_id)
        return {"ok": True, "gap_count": len(gaps), "gaps": gaps}


@router.post("/v1/results/dft", tags=["Results"])
async def save_dft_result(body: DFTResultRequest):
    with DFTResultRepo() as r:
        rid = r.save(**body.model_dump())
        return {"ok": True, "result_id": rid}


@router.get("/v1/results/dft", tags=["Results"])
async def list_dft_results(
    project_id: int,
    calc_type: str | None = None,
    cycle_id: int | None = None,
):
    with DFTResultRepo() as r:
        results = r.list_for_project(project_id, calc_type=calc_type, cycle_id=cycle_id)
        return {"ok": True, "count": len(results), "results": results}


@router.get("/v1/results/dft/summary", tags=["Results"])
async def dft_summary(project_id: int):
    with DFTResultRepo() as r:
        return {"ok": True, "summary": r.get_calculation_summary(project_id)}


# ═════════════════════════════════════════════════════════════════════════════
# DOCUMENTS
# ═════════════════════════════════════════════════════════════════════════════

class CreateDocumentRequest(BaseModel):
    project_id: int
    document_type: str
    title: str
    workflow_ids: list[int] | None = None
    dft_result_ids: list[int] | None = None
    instrument_result_ids: list[int] | None = None
    llm_backend: str | None = None
    cycle_id: int | None = None

class CreateSectionRequest(BaseModel):
    section_name: str
    order_index: int = 0

class SectionContentRequest(BaseModel):
    content: str


@router.post("/v1/documents", tags=["Documents"])
async def create_document(body: CreateDocumentRequest):
    with DocumentRepo() as r:
        did = r.create_document(**body.model_dump())
        return {"ok": True, "document_id": did}


@router.get("/v1/documents", tags=["Documents"])
async def list_documents(
    project_id: int,
    document_type: str | None = None,
    status: str | None = None,
):
    with DocumentRepo() as r:
        docs = r.list_documents(project_id, document_type, status)
        return {"ok": True, "count": len(docs), "documents": docs}


@router.get("/v1/documents/{document_id}", tags=["Documents"])
async def get_document(document_id: int, include_sections: bool = True):
    with DocumentRepo() as r:
        if include_sections:
            doc = r.get_document_with_sections(document_id)
        else:
            doc = r.get_document(document_id)
        if not doc:
            raise _err(f"Document {document_id} not found", 404)
        return {"ok": True, "document": doc}


@router.patch("/v1/documents/{document_id}/status", tags=["Documents"])
async def update_document_status(document_id: int, body: dict):
    with DocumentRepo() as r:
        r.update_document_status(document_id, body["status"])
        return {"ok": True, "document_id": document_id}


@router.post("/v1/documents/{document_id}/sections", tags=["Documents"])
async def create_section(document_id: int, body: CreateSectionRequest):
    with DocumentRepo() as r:
        sid = r.create_section(document_id, body.section_name, body.order_index)
        return {"ok": True, "section_id": sid}


@router.patch("/v1/sections/{section_id}/draft", tags=["Documents"])
async def save_draft(section_id: int, body: SectionContentRequest):
    with DocumentRepo() as r:
        r.save_draft(section_id, body.content)
        return {"ok": True, "section_id": section_id, "status": "drafted"}


@router.patch("/v1/sections/{section_id}/critique", tags=["Documents"])
async def save_critique(section_id: int, body: SectionContentRequest):
    with DocumentRepo() as r:
        r.save_critique(section_id, body.content)
        return {"ok": True, "section_id": section_id, "status": "critiqued"}


@router.patch("/v1/sections/{section_id}/final", tags=["Documents"])
async def save_final(section_id: int, body: SectionContentRequest):
    with DocumentRepo() as r:
        r.save_final(section_id, body.content)
        return {"ok": True, "section_id": section_id, "status": "final"}


# End of routes. This module exports `router` only — no standalone app.
# The live process is api_server.py -> api/app.py, which imports `router`
# from here and mounts it alongside the other Chitragupta routers.
