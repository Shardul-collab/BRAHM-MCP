"""
SHANI API Layer
===============
Exposes SHANI pipeline control via HTTP.
Each request creates its own Repository instance — thread-safe.

Endpoints
---------
POST   /workflows                          Create workflow + config
POST   /workflows/{id}/run                 Run workflow (stop_after_stage optional)
POST   /workflows/batch                    Create + run N workflows concurrently
GET    /workflows/{id}/status              Workflow + all stage statuses
GET    /workflows/{id}/papers              All Paper rows for a workflow
GET    /workflows/{id}/papers/{paper_id}/content   PaperContent sections
GET    /workflows/{id}/extract             Combined Paper + PaperContent dump
"""

import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from repositories.repository import Repository
from core.orchestrator import Orchestrator, WorkflowNotFoundError, InvalidTransitionError
import repositories.workflow_repo as workflow_repo
import repositories.stage_repo as stage_repo
import repositories.execution_repo as execution_repo
import repositories.paper_repo as paper_repo
import repositories.paper_content_repo as paper_content_repo

app = FastAPI(
    title="SHANI Research Pipeline API",
    version="1.0.0",
    description="Controlled execution of SHANI S1–S5 pipeline stages"
)


# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────

class WorkflowCreateRequest(BaseModel):
    name: str
    material: Optional[str] = None
    structure: Optional[str] = None
    focus: Optional[str] = None
    method: Optional[str] = None
    properties: Optional[str] = None
    characterization: Optional[str] = None
    use_local: bool = False
    max_papers: Optional[int] = None  # None = use search_papers.py's FINAL_PAPER_LIMIT default


class RunRequest(BaseModel):
    stop_after_stage: Optional[str] = "S4"


class BatchWorkflowItem(BaseModel):
    name: str
    material: Optional[str] = None
    structure: Optional[str] = None
    focus: Optional[str] = None
    method: Optional[str] = None
    properties: Optional[str] = None
    characterization: Optional[str] = None
    use_local: bool = False
    max_papers: Optional[int] = None


class BatchRunRequest(BaseModel):
    workflows: list[BatchWorkflowItem]
    stop_after_stage: Optional[str] = "S4"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def make_repo() -> Repository:
    """Each call gets a fresh Repository (own SQLite connection)."""
    return Repository()


def _create_workflow_with_config(repo: Repository, req: WorkflowCreateRequest) -> int:
    """
    Creates Workflow + WorkflowResearchConfig atomically (single transaction).
    Returns workflow_id.

    Previously these were two separate repo.transaction() calls, so a
    failure between them (e.g. the WorkflowResearchConfig insert erroring)
    left an orphaned Workflow row with no config — S1 depends on
    WorkflowResearchConfig and would have nothing to work with. Folding
    both inserts into one transaction means either both rows exist or
    neither does.
    """
    use_local_int = 1 if req.use_local else 0

    with repo.transaction() as cursor:
        workflow_id = workflow_repo.create_workflow(
            repo=repo,
            name=req.name,
            current_stage="S1",
            status="paused",
            cursor=cursor,
        )

        cursor.execute(
            """
            INSERT INTO WorkflowResearchConfig (
                workflow_id, material, structure, focus,
                method, properties, characterization, use_local, max_papers
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_id,
                req.material, req.structure, req.focus,
                req.method, req.properties, req.characterization,
                use_local_int, req.max_papers
            )
        )

    return workflow_id


def _run_workflow_thread(workflow_id: int, stop_after_stage: str):
    """Executed in a background thread. Own repo + orchestrator."""
    repo = make_repo()
    try:
        orch = Orchestrator(repo)
        orch.start_workflow(workflow_id, stop_after_stage=stop_after_stage)
    except Exception as e:
        print(f"[API] Workflow {workflow_id} failed: {e}")
    finally:
        repo.close()


# ─────────────────────────────────────────────
# ENDPOINTS

@app.get("/health")
def health():
    """Quick liveness check."""
    return {"status": "ok", "service": "shani"}

@app.get("/workflows")
def list_workflows():
    """List all workflows."""
    repo = make_repo()
    try:
        rows = workflow_repo.get_all_workflows(repo)
        return [{"id": r["id"], "name": r["name"], "status": r["status"], "current_stage": r["current_stage"]} for r in rows]
    finally:
        repo.close()

# ─────────────────────────────────────────────

@app.post("/workflows", status_code=201)
def create_workflow(req: WorkflowCreateRequest):
    """
    Create a new workflow with research config.
    Returns workflow_id. Workflow starts in 'paused' state.
    """
    repo = make_repo()
    try:
        workflow_id = _create_workflow_with_config(repo, req)
        return {"workflow_id": workflow_id, "status": "paused"}
    finally:
        repo.close()


@app.post("/workflows/{workflow_id}/run")
def run_workflow(workflow_id: int, req: RunRequest, background_tasks: BackgroundTasks):
    """
    Start a workflow in a background thread.
    stop_after_stage defaults to 'S4'.
    Valid values: S1, S2, S2_75, S2_5, S3, S4, S5, null (run to completion)
    Workflow must be in 'paused' state.
    """
    valid_stages = {None, "S1", "S2", "S2_75", "S2_5", "S3", "S4", "S5", "S5_5"}

    if req.stop_after_stage not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stop_after_stage '{req.stop_after_stage}'. "
                   f"Valid: {sorted(s for s in valid_stages if s)}"
        )

    # Verify workflow exists and is paused before spawning thread
    repo = make_repo()
    try:
        wf = workflow_repo.get_workflow(repo, workflow_id)
        if wf is None:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        if wf["status"] != "paused":
            raise HTTPException(
                status_code=409,
                detail=f"Workflow must be 'paused' to run. Current status: {wf['status']}"
            )
    finally:
        repo.close()

    background_tasks.add_task(
        _run_workflow_thread, workflow_id, req.stop_after_stage
    )

    return {
        "workflow_id": workflow_id,
        "message": "Workflow started in background",
        "stop_after_stage": req.stop_after_stage
    }


@app.post("/workflows/batch", status_code=202)
def batch_run(req: BatchRunRequest):
    """
    Create and run 10–15 workflows concurrently.
    Each workflow runs in its own thread with its own DB connection.
    Returns list of created workflow IDs immediately.
    All threads run in background — poll /workflows/{id}/status to track.
    """
    if not 1 <= len(req.workflows) <= 20:
        raise HTTPException(
            status_code=400,
            detail="Batch size must be between 1 and 20"
        )

    valid_stages = {None, "S1", "S2", "S2_75", "S2_5", "S3", "S4", "S5", "S5_5"}
    if req.stop_after_stage not in valid_stages:
        raise HTTPException(status_code=400, detail=f"Invalid stop_after_stage")

    created = []

    # Create all workflows first (sequential, fast)
    for item in req.workflows:
        repo = make_repo()
        try:
            wf_req = WorkflowCreateRequest(**item.model_dump())
            workflow_id = _create_workflow_with_config(repo, wf_req)
            created.append(workflow_id)
        finally:
            repo.close()

    # Launch all threads concurrently
    threads = []
    for workflow_id in created:
        t = threading.Thread(
            target=_run_workflow_thread,
            args=(workflow_id, req.stop_after_stage),
            daemon=True
        )
        t.start()
        threads.append(t)

    return {
        "workflow_ids": created,
        "count": len(created),
        "stop_after_stage": req.stop_after_stage,
        "message": f"{len(created)} workflows running in background"
    }


@app.get("/workflows/{workflow_id}/status")
def get_status(workflow_id: int):
    """
    Returns workflow status + all stage records + latest execution attempt per stage.
    """
    repo = make_repo()
    try:
        wf = workflow_repo.get_workflow(repo, workflow_id)
        if wf is None:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        from core.orchestrator import Orchestrator as Orch
        stages = []
        for stage_name in Orch.STAGE_SEQUENCE:
            stage = stage_repo.get_stage_by_workflow_and_name(repo, workflow_id, stage_name)
            if stage:
                stage_dict = dict(stage)
                attempt = execution_repo.get_latest_attempt_for_stage(repo, stage_dict["id"])
                stage_dict["latest_attempt"] = dict(attempt) if attempt else None
                stages.append(stage_dict)

        return {
            "workflow": wf,
            "stages": stages
        }
    finally:
        repo.close()


@app.get("/workflows/{workflow_id}/papers")
def get_papers(workflow_id: int):
    """
    Returns all Paper rows for a workflow.
    """
    repo = make_repo()
    try:
        wf = workflow_repo.get_workflow(repo, workflow_id)
        if wf is None:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        rows = repo.fetch_all(
            """
            SELECT id, workflow_id, title, source, pdf_url, file_path,
                   abstract, pdf_status, pdf_path, doi, status,
                   created_at, updated_at
            FROM Paper
            WHERE workflow_id = ?
            ORDER BY id ASC
            """,
            (workflow_id,)
        )

        return {"workflow_id": workflow_id, "count": len(rows), "papers": [dict(r) for r in rows]}
    finally:
        repo.close()


@app.get("/workflows/{workflow_id}/papers/{paper_id}/content")
def get_paper_content(workflow_id: int, paper_id: int):
    """
    Returns all PaperContent sections for a specific paper.
    Verifies the paper belongs to the requested workflow.
    """
    repo = make_repo()
    try:
        paper = repo.fetch_one(
            "SELECT id, workflow_id, title, status FROM Paper WHERE id = ?",
            (paper_id,)
        )
        if paper is None:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")
        if paper["workflow_id"] != workflow_id:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} not in workflow {workflow_id}")

        content = paper_content_repo.get_paper_content(repo, paper_id)

        return {
            "paper_id": paper_id,
            "workflow_id": workflow_id,
            "title": paper["title"],
            "status": paper["status"],
            "sections": content or {}
        }
    finally:
        repo.close()


@app.get("/workflows/{workflow_id}/extract")
def extract_all(workflow_id: int):
    """
    Full extraction dump: every Paper + its PaperContent sections.
    Optimized for bulk data export after S4 completion.
    Only returns papers that have content (status = extracted or beyond).
    """
    repo = make_repo()
    try:
        wf = workflow_repo.get_workflow(repo, workflow_id)
        if wf is None:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        rows = repo.fetch_all(
            """
            SELECT id, title, source, pdf_url, abstract,
                   pdf_status, status, created_at, updated_at
            FROM Paper
            WHERE workflow_id = ?
            ORDER BY id ASC
            """,
            (workflow_id,)
        )

        results = []
        for row in rows:
            p = dict(row)
            content = paper_content_repo.get_paper_content(repo, p["id"])
            p["content"] = content or {}
            results.append(p)

        papers_with_content = [p for p in results if p["content"]]

        return {
            "workflow_id": workflow_id,
            "workflow_status": wf["status"],
            "total_papers": len(results),
            "papers_with_content": len(papers_with_content),
            "papers": results
        }
    finally:
        repo.close()



# ─────────────────────────────────────────────
# KNOWLEDGE SEARCH
# ─────────────────────────────────────────────
class KnowledgeSearchRequest(BaseModel):
    query:       str
    material:    Optional[str] = None
    category:    Optional[str] = None
    workflow_id: Optional[int] = None
    top_k:       int = 10

@app.post("/knowledge/search")
def knowledge_search(req: KnowledgeSearchRequest):
    from services.vector_db_service import VectorDBService
    vs = VectorDBService()
    if vs.index.ntotal == 0:
        return {"results": [], "message": "Vector index is empty — run S5 first"}
    query = req.query
    if req.material:
        query = f"{req.material} {query}"
    raw = vs.search(query, top_k=req.top_k * 3, workflow_id=req.workflow_id)
    results = []
    for r in raw:
        if req.category and r.get("category") != req.category:
            continue
        results.append({
            "knowledge_id": r.get("knowledge_id"),
            "category":     r.get("category"),
            "value":        r.get("value"),
            "sentence":     r.get("sentence"),
            "doi":          r.get("doi"),
            "title":        r.get("title"),
            "year":         r.get("year"),
            "score":        round(r.get("score", 0), 4),
            "paper_id":     r.get("paper_id"),
            "workflow_id":  r.get("workflow_id"),
        })
        if len(results) >= req.top_k:
            break
    return {"query": req.query, "total": len(results), "results": results}

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
