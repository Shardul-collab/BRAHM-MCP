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

import os
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import api_key_auth

logger = logging.getLogger("chitragupta.store")

# BRAHM_ROOT-relative, matching brahm/shared/constants.py. This was a
# hardcoded "/mnt/d/brahm/..." literal until 2026-09-10 — the v1.1.1
# "fix hardcoded BRAHM_ROOT paths" pass missed Chitragupta entirely. On any
# host that is not the WSL2 dev box (the Linux machine, or the container
# where BRAHM_ROOT=/app) sqlite3.connect() would either create a stray DB
# under a path nobody reads, or fail outright because the parent directory
# does not exist.
# 2026-09-20: the fallback below was `/mnt/d/brahm` -- see chit_paths.py.
from chit_paths import BRAHM_ROOT, KNOWLEDGE_DB_PATH  # noqa: E402

DB_PATH       = str(KNOWLEDGE_DB_PATH)
GANESH_BASE   = os.environ.get("GANESH_BASE",      "http://localhost:8001")
VISHWAKARMA_BASE = os.environ.get("VISHWAKARMA_BASE", "http://localhost:8004")

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

        # Re-storing a document is an EXPECTED operation — the document
        # INSERT above is an explicit upsert (ON CONFLICT DO UPDATE), and
        # GANESH documents get stored repeatedly as sections are drafted.
        # But sections were blind-INSERTed underneath it, so every re-store
        # appended a second full copy of every section: store twice, get
        # each section twice, with no unique constraint on the table to stop
        # it (ganesh_sections has only an AUTOINCREMENT id). Replace the
        # document's sections rather than accumulating them.
        #
        # Worth knowing: GANESH is independently known to return duplicate
        # sections in its own API responses (worked around client-side in
        # RunProgress.tsx by deduping on id), so this path could double an
        # already-doubled list.
        cur.execute("DELETE FROM ganesh_sections WHERE document_id = ?", (doc_row_id,))

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
    """
    The caller (brahm/agents/vishwakarma.py's _persist_run) already knows
    everything worth storing and sends it. Until 2026-09-10 this model
    declared only job_id and triggered_by, so pydantic silently DROPPED
    calculation_type, material_name, output_file_path, scf_iterations and
    converged — every field with any content in it — and the handler then
    tried to recover them by re-fetching the job. See store_vishwakarma.

    All five stay Optional so an older caller that sends only job_id still
    works; the handler falls back to the re-fetch path for anything absent.
    """
    job_id:           str
    calculation_type: str | None   = None
    material_name:    str | None   = None
    output_file_path: str | None   = None
    scf_iterations:   int | None   = None
    converged:        bool | None  = None
    triggered_by:     str          = "brahm_llm"


@router.post("/vishwakarma")
async def store_vishwakarma(req: StoreVishwakarmaRequest, _=Depends(api_key_auth)):
    """
    Write a Vishwakarma calculation to vishwakarma_calculations + brahm_activity.

    Prefers what the caller sent; falls back to Vishwakarma :8004 for anything
    missing.

    THE BUG THIS REPLACES (found 2026-09-10, confirmed against the live job
    store): the handler ignored the request body entirely and rebuilt every
    column from `GET /jobs/{job_id}`. That endpoint returns runner's
    status.json — job_id, label, code, status, created_at, started_at,
    ended_at, exit_code, error, mpi_np, extra_args, workdir — and nothing
    else. So:

      parsed = job.get("parsed_output") or job.get("result") or {}   -> {} always
      material_name    -> ""     (key does not exist)
      output_file_path -> ""     (key does not exist)
      scf_iterations   -> None   (parsed is empty)
      converged        -> False  (parsed is empty)
      calculation_type -> falls back to job["code"], the BINARY name ("pw"),
                          not the calculation type ("scf")

    The parsed output lives at a different endpoint — `/jobs/{id}/output` —
    which the handler never called. Every row written this way was blank.

    This is a regression, not an original gap: the single row sitting in
    vishwakarma_calculations (id=2, 2026-05-28) has material_name="In2Se3",
    a real output path, scf_iterations=10, converged=1 — values the code
    above cannot produce. An earlier payload-driven version of this handler
    wrote it.
    """
    job: dict = {}
    parsed: dict = {}

    # Only reach out to Vishwakarma if the caller left something out.
    needs_fetch = any(
        getattr(req, f) is None
        for f in ("calculation_type", "material_name", "output_file_path",
                  "scf_iterations", "converged")
    )
    if needs_fetch:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(f"{VISHWAKARMA_BASE}/jobs/{req.job_id}")
                if r.status_code != 200:
                    raise HTTPException(502, f"Vishwakarma returned {r.status_code}: {r.text[:200]}")
                job = r.json()
                # The parsed physics is on /output, not on the job status.
                ro = await client.get(f"{VISHWAKARMA_BASE}/jobs/{req.job_id}/output")
                if ro.status_code == 200:
                    parsed = (ro.json() or {}).get("parsed") or {}
        except httpx.RequestError as exc:
            # A partial payload is still worth storing — do not lose a real
            # calculation because the agent that produced it went away.
            logger.warning("Vishwakarma unreachable for %s (%s); storing payload only",
                           req.job_id, exc)

    def pick(sent, *fallbacks):
        if sent is not None:
            return sent
        for f in fallbacks:
            if f not in (None, ""):
                return f
        return None

    calculation_type = pick(req.calculation_type,
                            job.get("calc_type"), job.get("label"), job.get("code"))
    material_name    = pick(req.material_name,    job.get("material_name")) or ""
    output_file_path = pick(req.output_file_path,
                            job.get("output_file_path"), job.get("outfile"),
                            (job.get("workdir") and f"{job['workdir']}/output.out")) or ""
    scf_iterations   = pick(req.scf_iterations,
                            parsed.get("scf_iterations"), parsed.get("n_scf_steps"))
    converged        = pick(req.converged, parsed.get("converged"))

    con = _db()
    try:
        cur = con.cursor()
        cur.execute(
            """INSERT INTO vishwakarma_calculations
               (calculation_type, material_name, output_file_path,
                scf_iterations, converged, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                calculation_type or "unknown",
                material_name,
                output_file_path,
                scf_iterations,
                bool(converged) if converged is not None else False,
                job.get("created_at") or datetime.now(timezone.utc).isoformat(),
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
