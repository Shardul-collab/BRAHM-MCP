"""
brahm/coordinator/orchestration.py
====================================
Background watcher for a single coordinator "run".

Confirmed against real data (workflow_id=1, MoS2 FET, completed run):
  - SHANI's terminal success signal is workflow["status"] == "completed".
    Individual Stage rows can be stale/misleading (e.g. S4 and S5 showed
    status="failed" with a dangling "running" attempt on workflow_id=1,
    even though the workflow completed successfully through S5_5) — so
    this watcher checks workflow-level status only, never per-stage status.
  - GANESH's post-G1-G3 state is document.status == "reviewing" (confirmed
    empirically against document_id=29 in this session).
  - GANESH's post-G4-G5 terminal state is inferred as "completed" from
    ganesh_list_documents' status_filter enum (["all","draft","reviewing",
    "completed","failed"]) — NOT yet confirmed by an actual synthesize run.
    Flagged clearly below; verify the first time synthesize actually
    completes and adjust GANESH_DONE_STATUSES if it differs.

This module only calls tools already registered in brahm_registry — the
exact same calls the MCP/stdio layer already makes. No SHANI or GANESH
orchestrator file is read from or reasoned about beyond its documented
API contract, and none is modified.
"""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("coordinator.orchestration")

SHANI_POLL_INTERVAL_S  = 3.0
GANESH_POLL_INTERVAL_S = 3.0

# NOTE ON THESE NUMBERS: the original 1-hour SHANI ceiling was a guess made
# without checking real historical stage durations first — it was wrong.
# workflow_id=1 (MoS2 FET, real run) shows S5_5 ALONE took ~3 hours
# (2026-06-30T04:21 -> 07:18). Raised to a much more generous ceiling below.
# This is still an estimate, not a hard guarantee — if a real run legitimately
# needs longer than this, raise it again rather than assume this is final.
SHANI_TIMEOUT_S        = 8 * 60 * 60   # 8 hour ceiling for S1 -> S5_5
GANESH_TIMEOUT_S       = 60 * 60        # 1 hour ceiling per GANESH phase

# Confirmed terminal states
SHANI_DONE_STATUS   = "completed"
SHANI_FAILED_STATUS = "failed"

GANESH_G1G3_DONE_STATUS = "reviewing"   # confirmed empirically (doc_id=29)
GANESH_FAILED_STATUS    = "failed"

# NOT YET CONFIRMED by an actual end-to-end synthesize run — inferred from
# ganesh_list_documents' status_filter enum. Verify and correct if wrong.
GANESH_DONE_STATUSES = ("completed",)


async def run_shani_then_ganesh(run_id: str, run_store, registry) -> None:
    run = run_store.get(run_id)
    if run is None:
        log.error("run_id=%s vanished before watcher started", run_id)
        return

    workflow_id = run["workflow_id"]

    # ─── Phase 1: wait for SHANI (S1 -> S5_5) ────────────────────────────────
    elapsed = 0.0
    reached_shani_done = False
    while elapsed < SHANI_TIMEOUT_S:
        status = await registry.dispatch("shani_get_status", {"workflow_id": workflow_id})
        if status.get("status") == "error":
            run_store.set_error(run_id, f"shani_get_status failed: {status.get('error')}")
            return

        wf_status = status.get("workflow", {}).get("status")

        if wf_status == SHANI_DONE_STATUS:
            log.info("run_id=%s: SHANI workflow %d completed", run_id, workflow_id)
            run_store.set_phase(run_id, "shani_done")
            reached_shani_done = True
            break

        if wf_status == SHANI_FAILED_STATUS:
            run_store.set_error(run_id, f"SHANI workflow {workflow_id} reported status=failed")
            return

        await asyncio.sleep(SHANI_POLL_INTERVAL_S)
        elapsed += SHANI_POLL_INTERVAL_S

    if not reached_shani_done:
        run_store.set_error(
            run_id,
            f"Timed out waiting for SHANI workflow {workflow_id} after {SHANI_TIMEOUT_S}s"
        )
        return

    if not run.get("auto_write", True):
        run_store.set_phase(run_id, "complete")
        return

    # ─── Phase 2: GANESH write (G1-G3) ────────────────────────────────────────
    run_store.set_phase(run_id, "ganesh_writing")
    write_result = await registry.dispatch("ganesh_write_review", {
        "workflow_ids":  [workflow_id],
        "document_type": run.get("document_type", "literature_review"),
    })
    if write_result.get("status") == "error":
        run_store.set_error(run_id, f"ganesh_write_review failed: {write_result.get('error')}")
        return

    document_id = write_result.get("document_id")
    if document_id is None:
        run_store.set_error(run_id, "ganesh_write_review did not return document_id")
        return
    run_store.set_document_id(run_id, document_id)

    if not await _wait_for_ganesh_status(
        registry, run_store, run_id, document_id,
        target_status=GANESH_G1G3_DONE_STATUS,
        timeout_s=GANESH_TIMEOUT_S,
    ):
        return

    # ─── Phase 3: GANESH synthesize (G4-G5) ───────────────────────────────────
    run_store.set_phase(run_id, "ganesh_reviewing")
    synth_result = await registry.dispatch("ganesh_synthesize", {"document_id": document_id})
    if synth_result.get("status") == "error":
        run_store.set_error(run_id, f"ganesh_synthesize failed: {synth_result.get('error')}")
        return

    if not await _wait_for_ganesh_status(
        registry, run_store, run_id, document_id,
        target_status=None,  # any status in GANESH_DONE_STATUSES
        timeout_s=GANESH_TIMEOUT_S,
    ):
        return

    run_store.set_phase(run_id, "complete")
    log.info("run_id=%s: fully complete (workflow_id=%d, document_id=%s)",
              run_id, workflow_id, document_id)


async def _wait_for_ganesh_status(
    registry, run_store, run_id: str, document_id: int,
    target_status: str | None, timeout_s: float,
) -> bool:
    """
    Polls ganesh_get_document until doc.status matches target_status
    (or, if target_status is None, any value in GANESH_DONE_STATUSES).
    Returns True on success, False on failure/timeout (and sets the
    run's error state before returning False).
    """
    elapsed = 0.0
    while elapsed < timeout_s:
        doc_result = await registry.dispatch("ganesh_get_document", {"document_id": document_id})
        if doc_result.get("status") == "error":
            run_store.set_error(run_id, f"ganesh_get_document failed: {doc_result.get('error')}")
            return False

        doc_status = doc_result.get("document", {}).get("status")

        if doc_status == GANESH_FAILED_STATUS:
            run_store.set_error(run_id, f"GANESH document {document_id} reported status=failed")
            return False

        if target_status is not None and doc_status == target_status:
            return True
        if target_status is None and doc_status in GANESH_DONE_STATUSES:
            return True

        await asyncio.sleep(GANESH_POLL_INTERVAL_S)
        elapsed += GANESH_POLL_INTERVAL_S

    run_store.set_error(
        run_id,
        f"Timed out waiting for GANESH document {document_id} "
        f"(wanted status={target_status or GANESH_DONE_STATUSES}) after {timeout_s}s"
    )
    return False
