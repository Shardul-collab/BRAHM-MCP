# api/routers/store.py
"""
Chitragupta Store Router
Receives results from GANESH, VIDUR, and Vishwakarma and persists them
to brahm_knowledge.db + logs brahm_activity.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import api_key_auth

logger = logging.getLogger("chitragupta.store")

DB_PATH       = "/mnt/d/brahm/agents/chitragupta/database/brahm_knowledge.db"
GANESH_BASE   = "http://localhost:8001"
VISHWAKARMA_BASE = "http://localhost:8004"

router = APIRouter(prefix="/store", tags=["Store"])


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _log_activity(cur: sqlite3.Cursor, agent: str, action: str,
                  triggered_by: str, status: str) -> None:
    cur.execute(
        """INSERT INTO brahm_activity (agent, action, triggered_by, status, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (agent, action, triggered_by, status,
         datetime.now(timezone.utc).isoformat()),
    )


# ── /store/ganesh ─────────────────────────────────────────────────────────────

class StoreGaneshRequest(BaseModel):
    document_id: str
    triggered_by: str = "brahm_llm"


@router.post("/ganesh")
async def store_ganesh(req: StoreGaneshRequest, _=Depends(api_key_auth)):
    """
    Pull document + sections from GANESH :8001 and write to
    ganesh_documents + ganesh_sections + brahm_activity.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{GANESH_BASE}/documents/{req.document_id}")
            if r.status_code != 200:
                raise HTTPException(502, f"GANESH returned {r.status_code}: {r.text[:200]}")
            data = r.json()
    except httpx.RequestError as exc:
        raise HTTPException(502, f"GANESH unreachable: {exc}")

    doc      = data.get("document", data)
    sections = data.get("sections", [])

    con = _db()
    try:
        cur = con.cursor()

        # Upsert document
        cur.execute(
            """INSERT INTO ganesh_documents
               (id, workflow_ids, document_type, status, final_document, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 status=excluded.status,
                 final_document=excluded.final_document""",
            (
                str(doc.get("id", req.document_id)),
                json.dumps(doc.get("workflow_ids", [])),
                doc.get("document_type", "unknown"),
                doc.get("status", "complete"),
                doc.get("final_document") or doc.get("content", ""),
                doc.get("created_at", datetime.now(timezone.utc).isoformat()),
            ),
        )
        doc_row_id = str(doc.get("id", req.document_id))

        # Insert sections
        for sec in sections:
            cur.execute(
                """INSERT INTO ganesh_sections
                   (document_id, section_name, draft_text, created_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    doc_row_id,
                    sec.get("section_name", "unknown"),
                    sec.get("draft_text") or sec.get("content", ""),
                    sec.get("created_at", datetime.now(timezone.utc).isoformat()),
                ),
            )

        _log_activity(cur, "ganesh", f"store_document:{doc_row_id}",
                      req.triggered_by, "success")
        con.commit()
    except Exception as exc:
        con.rollback()
        logger.exception("store_ganesh DB write failed")
        raise HTTPException(500, f"DB write failed: {exc}")
    finally:
        con.close()

    return {"ok": True, "document_id": doc_row_id, "sections_stored": len(sections)}


# ── /store/vidur ──────────────────────────────────────────────────────────────

class StoreVidurRequest(BaseModel):
    file_path:   str
    technique:   str
    confidence:  float
    signals:     list[str] = []
    parsed_data: dict[str, Any] = {}
    triggered_by: str = "brahm_llm"


@router.post("/vidur")
async def store_vidur(req: StoreVidurRequest, _=Depends(api_key_auth)):
    """Write a VIDUR classification result to vidur_classifications + brahm_activity."""
    con = _db()
    try:
        cur = con.cursor()
        cur.execute(
            """INSERT INTO vidur_classifications
               (file_path, technique, confidence, signals, parsed_data, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                req.file_path,
                req.technique,
                req.confidence,
                json.dumps(req.signals),
                json.dumps(req.parsed_data),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        row_id = cur.lastrowid
        _log_activity(cur, "vidur",
                      f"classify:{req.technique}:{req.file_path}",
                      req.triggered_by, "success")
        con.commit()
    except Exception as exc:
        con.rollback()
        logger.exception("store_vidur DB write failed")
        raise HTTPException(500, f"DB write failed: {exc}")
    finally:
        con.close()

    return {"ok": True, "classification_id": row_id}


# ── /store/vishwakarma ────────────────────────────────────────────────────────

class StoreVishwakarmaRequest(BaseModel):
    job_id:       str
    triggered_by: str = "brahm_llm"


@router.post("/vishwakarma")
async def store_vishwakarma(req: StoreVishwakarmaRequest, _=Depends(api_key_auth)):
    """
    Pull job result from Vishwakarma :8004 and write to
    vishwakarma_calculations + brahm_activity.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{VISHWAKARMA_BASE}/jobs/{req.job_id}")
            if r.status_code != 200:
                raise HTTPException(502, f"Vishwakarma returned {r.status_code}: {r.text[:200]}")
            job = r.json()
    except httpx.RequestError as exc:
        raise HTTPException(502, f"Vishwakarma unreachable: {exc}")

    parsed = job.get("parsed_output") or job.get("result") or {}

    con = _db()
    try:
        cur = con.cursor()
        cur.execute(
            """INSERT INTO vishwakarma_calculations
               (calculation_type, material_name, output_file_path,
                scf_iterations, converged, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                job.get("calc_type") or job.get("code", "unknown"),
                job.get("material_name", ""),
                job.get("output_file_path") or job.get("outfile", ""),
                parsed.get("scf_iterations") or parsed.get("n_scf_steps"),
                bool(parsed.get("converged", False)),
                job.get("created_at", datetime.now(timezone.utc).isoformat()),
            ),
        )
        row_id = cur.lastrowid
        _log_activity(cur, "vishwakarma",
                      f"store_job:{req.job_id}",
                      req.triggered_by, "success")
        con.commit()
    except Exception as exc:
        con.rollback()
        logger.exception("store_vishwakarma DB write failed")
        raise HTTPException(500, f"DB write failed: {exc}")
    finally:
        con.close()

    return {"ok": True, "calculation_id": row_id, "job_id": req.job_id}
