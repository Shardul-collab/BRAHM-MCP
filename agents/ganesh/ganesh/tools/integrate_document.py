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
from pathlib import Path

GANESH_ROOT = Path("/mnt/d/brahm/agents/ganesh")
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

    # ── Build sections preview for LLM prompt ────────────────────────────────
    # Skip abstract section from preview (we're generating it)
    non_abstract = [s for s in assembled_sections if s["section_type"] != "abstract"]
    sections_preview_parts = []
    for i, sec in enumerate(non_abstract):
        preview = sec["content"][:600] + "..." if len(sec["content"]) > 600 else sec["content"]
        transition_marker = "\n[TRANSITION NEEDED]\n" if i < len(non_abstract) - 1 else ""
        sections_preview_parts.append(
            f"### {sec['section_name']}\n{preview}{transition_marker}"
        )

    sections_preview = "\n\n".join(sections_preview_parts)

    # ── LLM integration call ──────────────────────────────────────────────────
    print(f"[G5] Calling LLM for abstract + transitions ({len(assembled_sections)} sections)...")

    abstract_text   = ""
    final_assembled = ""

    try:
        prompt = INTEGRATION_PROMPT.format(
            title           = title,
            document_type   = document_type,
            sections_preview = sections_preview,
        )
        llm_output = call_llm(prompt, max_tokens=4096)

        # Parse abstract from LLM output
        if "## Abstract" in llm_output:
            parts = llm_output.split("---SECTIONS_FOLLOW---", 1)
            abstract_text = parts[0].replace("## Abstract", "").strip()
            llm_sections  = parts[1].strip() if len(parts) > 1 else ""
        else:
            abstract_text = ""
            llm_sections  = llm_output

    except LLMError as e:
        print(f"[G5] LLM integration failed: {e} — assembling without abstract/transitions")
        abstract_text = ""
        llm_sections  = ""

    # ── Assemble final document ───────────────────────────────────────────────
    doc_parts = [f"# {title}\n"]

    # Abstract
    if abstract_text:
        doc_parts.append(f"## Abstract\n\n{abstract_text}\n")

    # If LLM returned enhanced sections, use them; else use raw drafts
    if llm_sections:
        doc_parts.append(llm_sections)
    else:
        for sec in assembled_sections:
            if sec["section_type"] == "abstract":
                continue
            doc_parts.append(f"\n## {sec['section_name']}\n\n{sec['content']}\n")

    # References placeholder
    doc_parts.append(
        "\n## References\n\n"
        "_[References extracted from source papers — to be formatted per journal style]_\n"
    )

    final_output = "\n".join(doc_parts)

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
        "final_output":     final_output,
    }
