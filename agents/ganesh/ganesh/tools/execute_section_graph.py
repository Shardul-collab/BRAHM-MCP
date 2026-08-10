"""
ganesh/tools/execute_section_graph.py
======================================
G3 — Execute the section dependency graph.

Runs the SectionGraph DAG: for each READY section, calls SectionExecutor
which runs the write → critique → revise loop until approved or max iterations.

This tool is the most time-intensive stage. It runs entirely synchronously
inside a thread (called via asyncio.to_thread from the API).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

GANESH_ROOT = Path("/mnt/d/brahm/agents/ganesh")
if str(GANESH_ROOT) not in sys.path:
    sys.path.insert(0, str(GANESH_ROOT))

from section_graph    import SectionGraph, SectionStatus
from section_executor import SectionExecutor


def execute_section_graph(repo, document_id: int, config: dict) -> dict:
    """
    G3 tool function.

    Drives the SectionGraph DAG until all sections are APPROVED.
    Each section goes through SectionExecutor: write/critique/revise loop.

    Parameters
    ----------
    repo        : Repository
    document_id : int
    config      : dict — from GaneshDocument row (document_type, source_ids etc.)
    """

    print(f"[G3] Starting section graph execution for document_id={document_id}")

    # ── Load context bundle ───────────────────────────────────────────────────
    ctx_row = repo.fetch_one(
        "SELECT context_json FROM GaneshContext WHERE document_id = ? ORDER BY id DESC LIMIT 1",
        (document_id,),
    )
    if not ctx_row:
        raise ValueError(f"No GaneshContext for document_id={document_id}. Run G1 first.")

    context_bundle = json.loads(ctx_row["context_json"])

    # ── Build section graph ───────────────────────────────────────────────────
    graph    = SectionGraph.from_document(repo, document_id)
    executor = SectionExecutor(repo, document_id, context_bundle)

    if graph.is_complete():
        print("[G3] All sections already approved — nothing to do.")
        return {
            "status":           "success",
            "document_id":      document_id,
            "sections_approved": len(graph.get_approved_sections_ordered()),
            "note":             "All sections were already approved.",
        }

    approved_count = 0
    failed_sections = []
    iteration_limit = 50   # safety valve — prevents infinite loops

    for _ in range(iteration_limit):
        if graph.is_complete():
            break

        if graph.has_deadlock():
            status_summary = graph.get_status_summary()
            raise RuntimeError(
                f"[G3] Section graph deadlock detected. Status: {status_summary}"
            )

        ready = graph.get_ready_sections()
        if not ready:
            break

        for section in ready:
            print(f"[G3] Executing section: {section.section_name}")
            graph.mark_drafting(section.section_name)

            try:
                result = executor.run(section)
                if result.get("approved"):
                    graph.mark_approved(section.section_name)
                    approved_count += 1
                    print(f"[G3] ✓ Approved: {section.section_name} "
                          f"(score={result.get('final_score', '?')}, "
                          f"iterations={result.get('iterations', '?')})")
                else:
                    # Section hit max iterations without approval — soft fail
                    # Force-approve with quality flag so document can continue
                    graph.mark_approved(section.section_name)
                    approved_count += 1
                    failed_sections.append(section.section_name)
                    print(f"[G3] ⚠ Force-approved (max iterations): {section.section_name}")

            except Exception as exc:
                print(f"[G3] ✗ Section failed: {section.section_name} — {exc}")
                # Force-approve so graph can continue — document gets quality flag
                graph.mark_approved(section.section_name)
                failed_sections.append(section.section_name)

    # ── Update document status ────────────────────────────────────────────────
    now          = datetime.utcnow().isoformat()
    quality_flag = "below_threshold" if failed_sections else None

    with repo.transaction() as cursor:
        cursor.execute(
            """
            UPDATE GaneshDocument
            SET status = 'reviewing', quality_flag = ?, updated_at = ?
            WHERE id = ?
            """,
            (quality_flag, now, document_id),
        )

    final_summary = graph.get_status_summary()
    print(f"[G3] Complete. {approved_count} sections approved. "
          f"Failed: {failed_sections or 'none'}")

    return {
        "status":            "success",
        "document_id":       document_id,
        "sections_approved": approved_count,
        "sections_failed":   failed_sections,
        "quality_flag":      quality_flag,
        "section_statuses":  final_summary,
    }
