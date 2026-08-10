"""
ganesh/tools/cross_section_review.py
======================================
G4 — Cross-section coherence review.

After all sections are approved (G3), this stage:
1. Reads all approved section drafts
2. Calls LLM to identify coherence issues across sections
   (contradictions, redundancy, argument flow breaks, missing transitions)
3. For sections flagged as needing revision, calls SectionExecutor.revise()
4. Stores a cross-section GaneshCritique

Failure policy: SOFT_CONTINUE — if G4 fails, document is still usable
with quality_flag='human_review_requested'.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

GANESH_ROOT = Path("/mnt/d/brahm/agents/ganesh")
if str(GANESH_ROOT) not in sys.path:
    sys.path.insert(0, str(GANESH_ROOT))

from section_graph    import SectionGraph
from section_executor import SectionExecutor
from ganesh.llm_client import call_llm_json, LLMError


CROSS_REVIEW_PROMPT = """You are a scientific document editor performing a cross-section review.

Document title: {title}
Document type: {document_type}

Below are all approved section drafts in narrative order:

{sections_text}

Review for:
1. CONTRADICTIONS — factual conflicts between sections
2. REDUNDANCY — significant content repeated across sections  
3. FLOW BREAKS — abrupt transitions or missing logical connections
4. ARGUMENT GAPS — claims made without supporting evidence elsewhere

Return ONLY a JSON object:
{{
  "overall_coherence_score": <float 0-10>,
  "issues": [
    {{
      "type": "contradiction|redundancy|flow_break|argument_gap",
      "sections_involved": ["Section A", "Section B"],
      "description": "specific issue description",
      "severity": "low|medium|high",
      "suggested_fix": "concrete suggestion"
    }}
  ],
  "sections_needing_revision": ["section_name", ...],
  "transition_suggestions": {{
    "from_section → to_section": "suggested transition text"
  }},
  "summary": "2-3 sentence overall assessment"
}}
"""


def cross_section_review(repo, document_id: int, config: dict) -> dict:
    """
    G4 tool function.

    Reads all approved sections, runs LLM cross-section critique,
    applies targeted revisions where needed.
    """

    print(f"[G4] Cross-section review for document_id={document_id}")

    # ── Load document ─────────────────────────────────────────────────────────
    doc = repo.fetch_one(
        "SELECT title, document_type FROM GaneshDocument WHERE id = ?",
        (document_id,),
    )
    if not doc:
        raise ValueError(f"GaneshDocument {document_id} not found")

    # ── Load context bundle ───────────────────────────────────────────────────
    ctx_row = repo.fetch_one(
        "SELECT context_json FROM GaneshContext WHERE document_id = ? ORDER BY id DESC LIMIT 1",
        (document_id,),
    )
    context_bundle = json.loads(ctx_row["context_json"]) if ctx_row else {}

    # ── Load approved sections in order ──────────────────────────────────────
    graph = SectionGraph.from_document(repo, document_id)
    ordered_sections = graph.get_approved_sections_ordered()

    if not ordered_sections:
        print("[G4] No approved sections found — skipping cross-section review.")
        return {
            "status":      "success",
            "document_id": document_id,
            "note":        "No approved sections to review.",
        }

    # Build sections text for prompt
    sections_parts = []
    section_drafts: dict[str, str] = {}

    for node in ordered_sections:
        draft_row = repo.fetch_one(
            """
            SELECT content FROM GaneshDraft
            WHERE section_id = ?
            ORDER BY version DESC LIMIT 1
            """,
            (node.section_id,),
        )
        content = draft_row["content"] if draft_row else "(no draft found)"
        section_drafts[node.section_name] = content
        # Truncate for prompt to avoid token overflow
        preview = content[:800] + "..." if len(content) > 800 else content
        sections_parts.append(f"=== {node.section_name} ===\n{preview}")

    sections_text = "\n\n".join(sections_parts)

    # ── LLM cross-section critique ────────────────────────────────────────────
    prompt = CROSS_REVIEW_PROMPT.format(
        title         = doc["title"],
        document_type = doc["document_type"],
        sections_text = sections_text,
    )

    print(f"[G4] Calling LLM for cross-section review ({len(ordered_sections)} sections)...")

    try:
        critique_data = call_llm_json(prompt, max_tokens=2000)
    except LLMError as e:
        print(f"[G4] LLM failed: {e} — marking for human review")
        _flag_human_review(repo, document_id)
        return {
            "status":      "success",
            "document_id": document_id,
            "quality_flag": "human_review_requested",
            "note":        f"Cross-section LLM call failed: {e}",
        }

    overall_score        = critique_data.get("overall_coherence_score", 0.0)
    issues               = critique_data.get("issues", [])
    needs_revision       = critique_data.get("sections_needing_revision", [])
    transition_suggestions = critique_data.get("transition_suggestions", {})

    print(f"[G4] Coherence score: {overall_score}/10, "
          f"issues: {len(issues)}, revisions needed: {len(needs_revision)}")

    # ── Store cross-section GaneshCritique ────────────────────────────────────
    now = datetime.utcnow().isoformat()
    # Store cross-section critique summary in GaneshDocument outline_json
    with repo.transaction() as cursor:
        cursor.execute(
            "UPDATE GaneshDocument SET quality_flag = ?, updated_at = ? WHERE id = ?",
            (
                f"coherence:{overall_score:.1f}" if overall_score < 7 else None,
                now,
                document_id,
            ),
        )

    # ── Apply targeted revisions ──────────────────────────────────────────────
    revised_sections = []
    if needs_revision:
        executor = SectionExecutor(repo, document_id, context_bundle)

        # Filter to sections that actually exist and are approved
        approved_names = {n.section_name for n in ordered_sections}
        to_revise = [s for s in needs_revision if s in approved_names]

        for section_name in to_revise[:3]:   # cap at 3 revisions in G4
            section_node = next(
                (n for n in ordered_sections if n.section_name == section_name),
                None,
            )
            if not section_node:
                continue

            # Find relevant issues for this section
            section_issues = [
                iss for iss in issues
                if section_name in iss.get("sections_involved", [])
            ]
            revision_instruction = "; ".join(
                iss.get("suggested_fix", "") for iss in section_issues
            )

            print(f"[G4] Revising section: {section_name}")
            try:
                executor.revise_for_coherence(
                    section_node,
                    revision_instruction,
                    transition_suggestions,
                )
                revised_sections.append(section_name)
            except Exception as exc:
                print(f"[G4] Revision failed for {section_name}: {exc}")

    # ── Update document status ────────────────────────────────────────────────
    quality_flag = None
    if overall_score < 5.0:
        quality_flag = "human_review_requested"
    elif overall_score < 7.0 and len(issues) > 3:
        quality_flag = "below_threshold"

    with repo.transaction() as cursor:
        cursor.execute(
            """
            UPDATE GaneshDocument
            SET status = 'integrating', quality_flag = ?, updated_at = ?
            WHERE id = ?
            """,
            (quality_flag, now, document_id),
        )

    return {
        "status":              "success",
        "document_id":         document_id,
        "coherence_score":     overall_score,
        "issues_found":        len(issues),
        "sections_revised":    revised_sections,
        "quality_flag":        quality_flag,
        "summary":             critique_data.get("summary", ""),
    }


def _flag_human_review(repo, document_id: int) -> None:
    now = datetime.utcnow().isoformat()
    with repo.transaction() as cursor:
        cursor.execute(
            """
            UPDATE GaneshDocument
            SET quality_flag = 'human_review_requested',
                status = 'integrating',
                updated_at = ?
            WHERE id = ?
            """,
            (now, document_id),
        )
