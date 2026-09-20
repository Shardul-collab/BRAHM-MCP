"""
ganesh/tools/integrate_document.py
====================================
G5 — Final document integration.

Assembles all approved sections in exec_order, calls LLM for a final
synthesis pass (abstract, transitions, references placeholder),
writes GaneshDocument.final_output, and sets status='completed'.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
import os
from pathlib import Path

# Was hardcoded to a WSL path (/mnt/d/brahm/...) until 2026-09-11 - the v1.1.1
# path-portability pass never reached GANESH's tools. BRAHM_ROOT wins if set.
_BRAHM_ROOT = Path(os.environ["BRAHM_ROOT"]) if os.environ.get("BRAHM_ROOT") else Path(__file__).resolve().parents[4]
GANESH_ROOT = _BRAHM_ROOT / "agents" / "ganesh"
if str(GANESH_ROOT) not in sys.path:
    sys.path.insert(0, str(GANESH_ROOT))

from section_graph import SectionGraph
from ganesh.llm_client import call_llm, LLMError


INTEGRATION_PROMPT = """You are assembling a final scientific document.

Document title: {title}
Document type: {document_type}

The following sections have been drafted and reviewed. Your task:
1. Write a concise Abstract (150-250 words) summarising the entire document
2. Add smooth transition sentences between sections where marked [TRANSITION NEEDED]
3. Do NOT rewrite section content — only add transitions and the abstract

Sections in order:
{sections_preview}

Return ONLY the assembled document in this format:
## Abstract
<abstract text>

---SECTIONS_FOLLOW---
<sections with transitions inserted>
"""


def integrate_document(repo, document_id: int, config: dict) -> dict:
    """
    G5 tool function.

    Assembles all approved sections into the final document,
    generates abstract via LLM, writes GaneshDocument.final_output.
    """

    print(f"[G5] Integrating document_id={document_id}")

    # ── Load document ─────────────────────────────────────────────────────────
    doc = repo.fetch_one(
        "SELECT title, document_type, outline_json FROM GaneshDocument WHERE id = ?",
        (document_id,),
    )
    if not doc:
        raise ValueError(f"GaneshDocument {document_id} not found")

    title         = doc["title"]
    document_type = doc["document_type"]

    # ── Load approved sections in order ──────────────────────────────────────
    graph = SectionGraph.from_document(repo, document_id)
    ordered_sections = graph.get_approved_sections_ordered()

    if not ordered_sections:
        raise ValueError(f"No approved sections for document_id={document_id}")

    # ── Load latest draft for each section ───────────────────────────────────
    assembled_sections: list[dict] = []
    for node in ordered_sections:
        draft_row = repo.fetch_one(
            """
            SELECT content, version FROM GaneshDraft
            WHERE section_id = ?
            ORDER BY version DESC LIMIT 1
            """,
            (node.section_id,),
        )
        content = draft_row["content"] if draft_row else ""
        assembled_sections.append({
            "section_name": node.section_name,
            "section_type": node.section_type,
            "content":      content,
            "exec_order":   node.exec_order,
        })

    # ── Assemble verbatim; the LLM writes only the abstract (2026-09-11) ─────
    # The old path sent the LLM each section's first 600 chars and then used
    # its reply AS the final sections, so any section longer than 600 chars
    # reached the document as a rewrite of its own opening. See
    # ganesh/writing/assemble.py.
    from ganesh.writing.assemble import grounded_abstract, render_document, evidence_lookup
    from ganesh.writing.grounding import _CITE_RE

    body_sections = [s for s in assembled_sections if s["section_type"] != "abstract"]
    failed = graph.get_failed_sections() if hasattr(graph, "get_failed_sections") else []

    print(f"[G5] Writing abstract from {len(body_sections)} full sections...")
    abstract_text, abstract_dropped = "", []
    try:
        abstract_text, abstract_dropped = grounded_abstract(
            lambda prompt, max_tokens=500: call_llm(prompt, max_tokens=max_tokens, role="polish"),
            title, body_sections)
        if abstract_dropped:
            print(f"[G5] Abstract: removed {len(abstract_dropped)} sentence(s) with values not in the body")
    except LLMError as e:
        print(f"[G5] Abstract generation failed: {e} — document has no abstract")

    eids = {e.strip() for s in body_sections for grp in _CITE_RE.findall(s["content"])
            for e in grp.replace(";", ",").split(",")}
    lookup = evidence_lookup(repo, eids)
    final_output, provenance = render_document(title, abstract_text, body_sections, failed, lookup)

    # ── Mark sections as integrated ───────────────────────────────────────────
    now = datetime.utcnow().isoformat()
    with repo.transaction() as cursor:
        for sec in assembled_sections:
            cursor.execute(
                """
                UPDATE GaneshSection
                SET status = 'integrated', updated_at = ?
                WHERE document_id = ? AND section_name = ?
                """,
                (now, document_id, sec["section_name"]),
            )

        # Write final output + mark completed
        word_count = len(final_output.split())
        cursor.execute(
            """
            UPDATE GaneshDocument
            SET final_output     = ?,
                status           = 'completed',
                total_iterations = (SELECT COALESCE(SUM(iteration_count), 0)
                                    FROM GaneshSection WHERE document_id = ?),
                updated_at       = ?
            WHERE id = ?
            """,
            (final_output, document_id, now, document_id),
        )

    print(f"[G5] Document complete: {word_count} words, "
          f"{len(assembled_sections)} sections integrated")

    return {
        "status":           "success",
        "document_id":      document_id,
        "word_count":       word_count,
        "sections_count":   len(assembled_sections),
        "has_abstract":     bool(abstract_text),
        "references":       len(provenance["references"]),
        "sections_failed":  failed,
        "final_output":     final_output,
    }
