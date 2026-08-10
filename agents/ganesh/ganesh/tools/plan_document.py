"""
ganesh/tools/plan_document.py
==============================
G2 — LLM-driven document planning.

Reads the context bundle from GaneshContext, calls the LLM to produce
a DocumentPlan (outline + section specs + argument flow), then writes:
  - GaneshDocument.outline_json  — the full plan
  - GaneshSection rows           — one per planned section
"""

from __future__ import annotations

import json
from datetime import datetime

from ganesh.llm_client import call_llm_json, LLMError
from ganesh.document_types.literature_review import LITERATURE_REVIEW_SECTIONS
from ganesh.document_types.dft_report        import DFT_REPORT_SECTIONS
from ganesh.document_types.research_report   import RESEARCH_REPORT_SECTIONS
from ganesh.document_types.technical_summary import TECHNICAL_SUMMARY_SECTIONS
from ganesh.document_types.manuscript_draft  import MANUSCRIPT_DRAFT_SECTIONS

DOCUMENT_TYPE_SECTIONS = {
    "literature_review": LITERATURE_REVIEW_SECTIONS,
    "dft_report":        DFT_REPORT_SECTIONS,
    "research_report":   RESEARCH_REPORT_SECTIONS,
    "technical_summary": TECHNICAL_SUMMARY_SECTIONS,
    "manuscript_draft":  MANUSCRIPT_DRAFT_SECTIONS,
    "daily_report":      [],   # daily reports skip G2
}

PLAN_PROMPT = """You are a scientific document planner. 
Given research context, produce a detailed DocumentPlan as a JSON object.

Document type: {document_type}
Title: {title}

Research context summary:
- Total papers: {total_papers}
- Total knowledge rows: {total_knowledge_rows}
- Key materials: {materials}
- Key methods: {methods}
- Key characterization: {characterization}

Sections to plan (in order):
{sections_list}

For each section produce a spec with:
  - section_name: exact name from the list above
  - section_type: one of intro|body|conclusion|abstract|refs
  - brief: 2-3 sentence description of what this section covers
  - key_topics: list of 3-5 specific topics to cover
  - evidence_focus: which materials/methods/papers to emphasize
  - target_word_count: integer (300-1500 depending on section importance)
  - depends_on: list of section_names that must be written first ([] for intro)
  - quality_criteria: list of 2-3 specific quality checks for this section

Return ONLY a JSON object with this structure:
{{
  "title": "{title}",
  "document_type": "{document_type}",
  "argument_flow": "2-3 sentence description of the document's narrative arc",
  "sections": [ {{section spec}}, ... ]
}}
"""


def plan_document(repo, document_id: int, config: dict) -> dict:
    """
    G2 tool function.

    Reads context from GaneshContext, calls LLM to plan sections,
    writes GaneshSection rows and updates GaneshDocument.outline_json.
    """

    print(f"[G2] Planning document_id={document_id}")

    # ── Load document metadata ────────────────────────────────────────────────
    doc = repo.fetch_one(
        "SELECT title, document_type, source_ids FROM GaneshDocument WHERE id = ?",
        (document_id,),
    )
    if not doc:
        raise ValueError(f"GaneshDocument {document_id} not found")

    document_type = doc["document_type"]
    title         = doc["title"]

    # ── Load context bundle ───────────────────────────────────────────────────
    ctx_row = repo.fetch_one(
        "SELECT context_json FROM GaneshContext WHERE document_id = ? ORDER BY id DESC LIMIT 1",
        (document_id,),
    )
    if not ctx_row:
        raise ValueError(f"No GaneshContext found for document_id={document_id}. Run G1 first.")

    context = json.loads(ctx_row["context_json"])
    ks      = context.get("knowledge_summary", {})

    # ── Get section template for this document type ───────────────────────────
    section_templates = DOCUMENT_TYPE_SECTIONS.get(document_type, LITERATURE_REVIEW_SECTIONS)

    if not section_templates:
        raise ValueError(f"No section template defined for document_type='{document_type}'")

    sections_list = "\n".join(
        f"  {i+1}. {s['section_name']} ({s['section_type']})"
        for i, s in enumerate(section_templates)
    )

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = PLAN_PROMPT.format(
        document_type    = document_type,
        title            = title,
        total_papers     = context.get("total_papers", 0),
        total_knowledge_rows = context.get("total_knowledge_rows", 0),
        materials        = ", ".join(ks.get("material", [])[:8]),
        methods          = ", ".join(ks.get("synthesis_method", [])[:8]),
        characterization = ", ".join(ks.get("characterization", [])[:8]),
        sections_list    = sections_list,
    )

    print(f"[G2] Calling LLM for document plan ({len(section_templates)} sections)...")

    try:
        plan = call_llm_json(prompt, max_tokens=3000)
    except LLMError as e:
        raise RuntimeError(f"G2 LLM call failed: {e}")

    planned_sections = plan.get("sections", [])
    if not planned_sections:
        # Fall back to template sections without LLM briefs
        planned_sections = [
            {
                "section_name":      s["section_name"],
                "section_type":      s["section_type"],
                "brief":             s.get("description", ""),
                "key_topics":        [],
                "depends_on":        s.get("depends_on", []),
                "target_word_count": s.get("target_word_count", 500),
                "quality_criteria":  [],
            }
            for s in section_templates
        ]

    # ── Write GaneshSection rows ──────────────────────────────────────────────
    now = datetime.utcnow().isoformat()

    with repo.transaction() as cursor:
        # Clear any existing sections (idempotent G2)
        cursor.execute(
            "DELETE FROM GaneshSection WHERE document_id = ?",
            (document_id,),
        )

        for i, sec in enumerate(planned_sections):
            brief_json = json.dumps({
                "brief":             sec.get("brief", ""),
                "key_topics":        sec.get("key_topics", []),
                "evidence_focus":    sec.get("evidence_focus", ""),
                "target_word_count": sec.get("target_word_count", 500),
                "quality_criteria":  sec.get("quality_criteria", []),
            })
            depends_on = json.dumps(sec.get("depends_on", []))

            cursor.execute(
                """
                INSERT INTO GaneshSection
                    (document_id, section_name, section_type, brief_json,
                     depends_on, exec_order, status, iteration_count,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
                """,
                (
                    document_id,
                    sec.get("section_name", f"Section_{i+1}"),
                    sec.get("section_type", "body"),
                    brief_json,
                    depends_on,
                    i,
                    now, now,
                ),
            )

        # Update document outline + status
        cursor.execute(
            """
            UPDATE GaneshDocument
            SET outline_json = ?, status = 'drafting', updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(plan), now, document_id),
        )

    print(f"[G2] Plan complete: {len(planned_sections)} sections created")

    return {
        "status":           "success",
        "document_id":      document_id,
        "sections_planned": len(planned_sections),
        "argument_flow":    plan.get("argument_flow", ""),
        "section_names":    [s.get("section_name") for s in planned_sections],
    }
