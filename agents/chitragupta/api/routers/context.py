# api/routers/context.py
"""
Context router — serve curated SHANI knowledge to GANESH.

Endpoints
---------
GET  /context/workflows          — list SHANI workflows with knowledge_ready/completed papers
POST /context/load               — return filtered knowledge package for GANESH G1
GET  /context/knowledge_summary  — lightweight category summary for document planning

Reads SHANI's research_workflow.db directly (read-only).
Never writes to SHANI's DB.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi import status as http_status

from api.dependencies import api_key_auth, http_error
from api.models import (
    WorkflowsResponse,
    WorkflowSummary,
    ContextPackage,
    KnowledgeEntry,
    KnowledgeSummaryResponse,
)

logger = logging.getLogger("chitragupta.api.context")

SHANI_DB = "/mnt/d/brahm/agents/shani/database/research_workflow.db"
READY_STATUSES = ("knowledge_ready", "completed")

router = APIRouter(
    prefix="/context",
    tags=["Context"],
    dependencies=[Depends(api_key_auth)],
)


def _shani_conn() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{SHANI_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


# ── GET /context/workflows ────────────────────────────────────────────────────

@router.get("/workflows", response_model=WorkflowsResponse)
async def list_ready_workflows() -> WorkflowsResponse:
    """
    List SHANI workflows that have at least one knowledge_ready or completed paper.
    These are the workflows eligible to supply context to GANESH.
    """
    try:
        con = _shani_conn()
        rows = con.execute("""
            SELECT
                w.id            AS workflow_id,
                w.name          AS name,
                w.status        AS status,
                w.current_stage AS current_stage,
                COUNT(DISTINCT p.id) AS papers,
                COUNT(DISTINCT rk.id) AS knowledge_rows
            FROM Workflow w
            JOIN Paper p ON p.workflow_id = w.id
                AND p.status IN ('knowledge_ready', 'completed')
            LEFT JOIN ResearchKnowledge rk ON rk.paper_id = p.id
            GROUP BY w.id
            ORDER BY knowledge_rows DESC
        """).fetchall()
        con.close()
    except Exception:
        logger.exception("Failed to query SHANI workflows")
        raise http_error(http_status.HTTP_502_BAD_GATEWAY,
                         "Could not read SHANI database.")

    workflows = [WorkflowSummary(**dict(r)) for r in rows]
    return WorkflowsResponse(ready_count=len(workflows), workflows=workflows)


# ── POST /context/load ────────────────────────────────────────────────────────

class ContextLoadRequest:
    def __init__(
        self,
        workflow_ids:      list[int],
        document_type:     str = "literature_review",
        max_per_category:  int = 50,
        min_relevance:     float = 0.0,
    ):
        self.workflow_ids     = workflow_ids
        self.document_type    = document_type
        self.max_per_category = max_per_category
        self.min_relevance    = min_relevance


from pydantic import BaseModel as _BaseModel

class ContextLoadBody(_BaseModel):
    workflow_ids:     list[int]
    document_type:    str  = "literature_review"
    max_per_category: int  = 50
    min_relevance:    float = 0.0


@router.post("/load", response_model=ContextPackage)
async def load_context(body: ContextLoadBody) -> ContextPackage:
    """
    Return a curated knowledge package for GANESH G1.

    Filters:
    - Papers must have status knowledge_ready or completed
    - Only workflows in workflow_ids
    - Deduplicated on (category, value, paper_id)
    - Balanced across categories
    - Spread across papers within each category
    """
    if not body.workflow_ids:
        raise http_error(http_status.HTTP_422_UNPROCESSABLE_ENTITY,
                         "workflow_ids must not be empty.")

    placeholders = ",".join("?" * len(body.workflow_ids))

    try:
        con = _shani_conn()

        # Paper count
        paper_count = con.execute(f"""
            SELECT COUNT(DISTINCT id) FROM Paper
            WHERE workflow_id IN ({placeholders})
              AND status IN ('knowledge_ready','completed')
        """, body.workflow_ids).fetchone()[0]

        # Deduplicated knowledge rows — one row per (category, value, paper_id)
        rk_rows = con.execute(f"""
            SELECT rk.category, rk.value, rk.sentence, rk.paper_id
            FROM ResearchKnowledge rk
            JOIN Paper p ON p.id = rk.paper_id
            WHERE p.workflow_id IN ({placeholders})
              AND p.status IN ('knowledge_ready','completed')
            GROUP BY rk.category, rk.value, rk.paper_id
            ORDER BY rk.category, rk.paper_id, rk.id
        """, body.workflow_ids).fetchall()

        # Top papers (title, doi, abstract)
        top_papers = con.execute(f"""
            SELECT p.id, p.title, p.doi, p.abstract, p.status,
                   COUNT(rk.id) AS knowledge_count
            FROM Paper p
            LEFT JOIN ResearchKnowledge rk ON rk.paper_id = p.id
            WHERE p.workflow_id IN ({placeholders})
              AND p.status IN ('knowledge_ready','completed')
            GROUP BY p.id
            ORDER BY knowledge_count DESC
            LIMIT 100
        """, body.workflow_ids).fetchall()

        con.close()
    except Exception:
        logger.exception("Failed to load context from SHANI DB")
        raise http_error(http_status.HTTP_502_BAD_GATEWAY,
                         "Could not read SHANI database.")

    # --- Group by category, spread across papers, cap per category ---
    from collections import defaultdict

    # First group by (category, paper_id) to interleave papers
    by_cat_paper: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple] = set()
    for r in rk_rows:
        key = (r["category"], r["value"].strip().lower(), r["paper_id"])
        if key in seen:
            continue
        seen.add(key)
        by_cat_paper[r["category"]][r["paper_id"]].append(
            KnowledgeEntry(
                category=r["category"],
                value=r["value"],
                sentence=r["sentence"],
                paper_id=r["paper_id"],
            )
        )

    # Interleave entries across papers within each category, cap at max_per_category
    knowledge: dict[str, list[KnowledgeEntry]] = {}
    for cat, papers_dict in by_cat_paper.items():
        entries: list[KnowledgeEntry] = []
        paper_lists = list(papers_dict.values())
        i = 0
        while len(entries) < body.max_per_category:
            added = False
            for pl in paper_lists:
                if i < len(pl) and len(entries) < body.max_per_category:
                    entries.append(pl[i])
                    added = True
            if not added:
                break
            i += 1
        knowledge[cat] = entries

    total_knowledge = sum(len(v) for v in knowledge.values())

    logger.info("Context loaded | workflows=%s doc_type=%s papers=%d knowledge=%d categories=%d",
                body.workflow_ids, body.document_type, paper_count, total_knowledge, len(knowledge))

    return ContextPackage(
        workflow_ids=body.workflow_ids,
        document_type=body.document_type,
        total_papers=paper_count,
        total_knowledge=total_knowledge,
        knowledge=knowledge,
        top_papers=[dict(r) for r in top_papers],
    )



@router.get("/knowledge_summary", response_model=KnowledgeSummaryResponse)
async def knowledge_summary(
    workflow_ids: str = Query(..., description="Comma-separated workflow IDs, e.g. 1,2,3"),
) -> KnowledgeSummaryResponse:
    """
    Lightweight category summary for GANESH document planning (G2).
    Returns counts per knowledge category — no full text, fast to call.
    """
    try:
        ids = [int(x.strip()) for x in workflow_ids.split(",") if x.strip()]
    except ValueError:
        raise http_error(http_status.HTTP_422_UNPROCESSABLE_ENTITY,
                         "workflow_ids must be comma-separated integers.")

    if not ids:
        raise http_error(http_status.HTTP_422_UNPROCESSABLE_ENTITY,
                         "workflow_ids must not be empty.")

    placeholders = ",".join("?" * len(ids))

    try:
        con = _shani_conn()

        by_cat = con.execute(f"""
            SELECT rk.category, COUNT(*) AS total_rows,
                   COUNT(DISTINCT rk.paper_id) AS papers
            FROM ResearchKnowledge rk
            JOIN Paper p ON p.id = rk.paper_id
            WHERE p.workflow_id IN ({placeholders})
              AND p.status IN ('knowledge_ready','completed')
            GROUP BY rk.category
            ORDER BY total_rows DESC
        """, ids).fetchall()

        ready_papers = con.execute(f"""
            SELECT COUNT(DISTINCT id) FROM Paper
            WHERE workflow_id IN ({placeholders})
              AND status IN ('knowledge_ready','completed')
        """, ids).fetchone()[0]

        con.close()
    except Exception:
        logger.exception("Failed to query knowledge summary")
        raise http_error(http_status.HTTP_502_BAD_GATEWAY,
                         "Could not read SHANI database.")

    by_category = [dict(r) for r in by_cat]
    total_rows  = sum(r["total_rows"] for r in by_category)

    return KnowledgeSummaryResponse(
        workflow_ids=ids,
        by_category=by_category,
        total_rows=total_rows,
        ready_papers=ready_papers,
    )
