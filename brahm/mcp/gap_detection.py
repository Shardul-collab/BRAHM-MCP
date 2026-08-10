"""
brahm/mcp/gap_detection.py
============================
Coarse MCP tool: find_research_gap(workflow_id)

Joins ResearchFinding -> Paper -> PaperFigure for an entire workflow,
builds a digest, and asks a reasoning LLM (Groq gpt-oss-120b) to
propose a research gap / problem statement grounded in that corpus.

Design constraints (from architecture session, do not violate):
  - Operates over the WHOLE workflow's findings, not one paper.
  - Read-only. No confirmation gate needed (matches decision: only
    agent-handoffs and Chitragupta writes require confirmation).
  - Does not touch ResearchRelation (stub-quality, excluded for now).

This module is split into pure functions (digest building, prompt
building) and one orchestration function (find_research_gap) so the
pure parts can be unit-tested with mock data before ever touching the
real SHANI database.
"""

from __future__ import annotations
import json
import os
from typing import Any

try:
    from groq import Groq
except ImportError:
    Groq = None  # allows digest/prompt functions to be tested without the groq package


REASONING_MODEL = "openai/gpt-oss-120b"

MAX_FINDINGS_PER_DIGEST = 15  # lowered to fit Groq gpt-oss-120b TPM=8000 limit; batching strategy is a follow-on design item


def _truncate(text: str | None, limit: int = 400) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "..."


def build_finding_digest(
    findings: list[dict],
    papers_by_id: dict[int, dict],
    figures_by_paper_id: dict[int, list[dict]],
) -> str:
    """
    Pure function. Builds the text digest handed to the reasoning LLM.
    Groups by paper so the LLM sees "what this paper did" alongside
    "what was found", per the architecture session's requirement.
    """

    if len(findings) > MAX_FINDINGS_PER_DIGEST:
        findings = findings[:MAX_FINDINGS_PER_DIGEST]

    grouped: dict[int, list[dict]] = {}
    for f in findings:
        grouped.setdefault(f["paper_id"], []).append(f)

    blocks = []
    for paper_id, paper_findings in grouped.items():
        paper = papers_by_id.get(paper_id, {})
        title = paper.get("title", f"Paper {paper_id}")
        abstract = _truncate(paper.get("abstract"), 300)

        block_lines = [f'Paper [{paper_id}]: "{title}"']
        if abstract:
            block_lines.append(f"  Abstract: {abstract}")

        for f in paper_findings:
            block_lines.append(f"  - Finding: {f.get('finding_text', '').strip()}")

        figs = figures_by_paper_id.get(paper_id, [])
        if figs:
            cap_list = "; ".join(
                _truncate(fig.get("caption"), 100) for fig in figs if fig.get("caption")
            )
            if cap_list:
                block_lines.append(f"  Figures: {cap_list}")

        blocks.append("\n".join(block_lines))

    return "\n\n".join(blocks)


def build_gap_prompt(digest: str) -> str:
    """
    Pure function. Builds the reasoning prompt.
    """
    return (
        "You are a materials science research assistant helping identify "
        "a genuine research gap from an entire literature corpus below.\n\n"
        "RULES:\n"
        "1. Base your answer ONLY on the findings, abstracts, and figure "
        "captions given below. Do not invent facts not present here.\n"
        "2. A 'gap' means: a material/method/property combination that is "
        "under-studied, contradictory across papers, or a natural next step "
        "implied by what HAS been studied. It must be something the corpus "
        "as a whole does not already answer.\n"
        "3. If a finding elsewhere in this same corpus already answers a "
        "candidate gap, do NOT propose that gap — cite which paper answers "
        "it instead and pick a different candidate.\n"
        "4. Cite the paper IDs (in brackets, e.g. [12]) that inform your "
        "reasoning for the gap you propose.\n\n"
        "Return ONLY a JSON object with this shape:\n"
        '{"gap_statement": "...", '
        '"supporting_paper_ids": [12, 45], '
        '"reasoning": "one paragraph explaining why this is a real gap, '
        'referencing what has and has not been covered"}\n\n'
        f"Literature corpus:\n\n{digest}\n\n"
        "JSON object:"
    )


def _parse_gap_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def call_reasoning_llm(prompt: str, api_key: str | None = None) -> dict:
    """
    Calls Groq's reasoning model. Isolated so it can be mocked/skipped
    in pure-logic tests.

    NOTE: gpt-oss-120b puts chain-of-thought in a separate `.reasoning`
    field and the final answer in `.content`. Both draw from the same
    max_tokens budget, so a high reasoning_effort on a large prompt can
    exhaust the budget before any `.content` is written, leaving it
    empty. max_tokens must be generous enough to cover both.
    """
    if Groq is None:
        raise RuntimeError("groq package not installed — pip install groq")

    client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    completion = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        reasoning_effort="low",
        max_tokens=2000,
    )
    message = completion.choices[0].message
    raw = message.content
    finish_reason = completion.choices[0].finish_reason

    if not raw or not raw.strip():
        reasoning_preview = getattr(message, "reasoning", None)
        reasoning_preview = (reasoning_preview or "")[:500]
        raise RuntimeError(
            f"Empty .content from {REASONING_MODEL} "
            f"(finish_reason={finish_reason}). "
            f"Reasoning field was{' present' if reasoning_preview else ' also empty'}. "
            f"This usually means max_tokens was exhausted by reasoning "
            f"before the final answer was written — try raising max_tokens "
            f"or lowering reasoning_effort. "
            f"Reasoning preview: {reasoning_preview!r}"
        )

    try:
        return _parse_gap_response(raw)
    except Exception as e:
        raise RuntimeError(
            f"JSON parse failed: {e}. "
            f"finish_reason={finish_reason}, raw_length={len(raw)} chars. "
            f"finish_reason='length' means the response was cut off before "
            f"completing - max_tokens needs to be higher. "
        )


def find_research_gap(repo: Any, workflow_id: int) -> dict:
    """
    Real DB version. `repo` expected to expose fetch_all(sql, params)
    matching the existing BRAHM Repository interface (see
    agents/shani/repositories/repository.py).
    """

    findings = repo.fetch_all(
        """
        SELECT id, paper_id, finding_text, material, synthesis_method,
               characterization, property_name, property_value, condition_text
        FROM ResearchFinding
        WHERE workflow_id = ?
        """,
        (workflow_id,),
    )
    findings = [dict(f) for f in findings]

    if not findings:
        return {
            "status": "error",
            "data": None,
            "error": f"No ResearchFinding rows for workflow {workflow_id}",
        }

    paper_ids = sorted({f["paper_id"] for f in findings})
    placeholders = ",".join("?" * len(paper_ids))

    papers = repo.fetch_all(
        f"SELECT id, title, abstract FROM Paper WHERE id IN ({placeholders})",
        tuple(paper_ids),
    )
    papers_by_id = {p["id"]: dict(p) for p in papers}

    figures = repo.fetch_all(
        f"SELECT paper_id, image_path, caption, section_hint "
        f"FROM PaperFigure WHERE paper_id IN ({placeholders})",
        tuple(paper_ids),
    )
    figures_by_paper_id: dict[int, list[dict]] = {}
    for fig in figures:
        fig = dict(fig)
        figures_by_paper_id.setdefault(fig["paper_id"], []).append(fig)

    digest = build_finding_digest(findings, papers_by_id, figures_by_paper_id)
    prompt = build_gap_prompt(digest)

    try:
        result = call_reasoning_llm(prompt)
    except Exception as e:
        return {"status": "error", "data": None, "error": str(e)}

    return {"status": "success", "data": result, "error": None}
