"""
brahm/mcp/gap_detection_batch.py
===================================
Map-reduce batching for find_research_gap over a FULL workflow corpus.

Why this exists: gap_detection.py's find_research_gap() works but is
capped at MAX_FINDINGS_PER_DIGEST=15 to fit Groq gpt-oss-120b's TPM=8000
limit in a single call. The MoS2 workflow has 415 findings across 195
papers — far more than one call can hold, and the requirement (per
architecture session) is that a gap must hold against the WHOLE corpus,
not a slice.

Design: map/reduce.
  MAP:    split all findings into token-safe batches (grouped by paper,
          since abstracts cost tokens once per paper regardless of how
          many findings that paper has). One find_research_gap-style
          call per batch -> one candidate gap each. Checkpointed to a
          JSONL file after every batch (resumable — a crash at batch 20
          doesn't lose batches 1-19).
  REDUCE: one final call takes the compact list of candidate gaps (not
          full findings) plus a LIGHTWEIGHT index of every finding in
          the corpus (material/method/property/value tags only, no
          abstracts/captions/sentences) and cross-checks each candidate
          against the whole corpus, dropping any the corpus elsewhere
          already answers.

Rate limit handling: Groq enforces TPM=8000 at the ORGANIZATION level,
not per-key (confirmed against official Groq docs — round-robining
keys does not help). So: single key, sequential calls, paced with a
sleep between them.
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any

from brahm.mcp.gap_detection import (
    build_finding_digest,
    build_gap_prompt,
    call_reasoning_llm,
)

MAX_ESTIMATED_TOKENS_PER_BATCH = 2500
PACING_SECONDS = 65


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def chunk_papers_by_token_budget(
    findings_by_paper: dict[int, list[dict]],
    papers_by_id: dict[int, dict],
    figures_by_paper_id: dict[int, list[dict]],
    max_tokens: int = MAX_ESTIMATED_TOKENS_PER_BATCH,
) -> list[list[int]]:
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_tokens = 0

    for paper_id in sorted(findings_by_paper.keys()):
        paper = papers_by_id.get(paper_id, {})
        findings = findings_by_paper[paper_id]
        figs = figures_by_paper_id.get(paper_id, [])

        block_text = (paper.get("title", "") or "") + (paper.get("abstract", "") or "")
        block_text += "".join(f.get("finding_text", "") or "" for f in findings)
        block_text += "".join(fig.get("caption", "") or "" for fig in figs)
        paper_tokens = _estimate_tokens(block_text)

        if current_batch and current_tokens + paper_tokens > max_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(paper_id)
        current_tokens += paper_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def _load_checkpoint(checkpoint_path: str) -> list[dict]:
    path = Path(checkpoint_path)
    if not path.exists():
        return []
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _append_checkpoint(checkpoint_path: str, entry: dict) -> None:
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_map_phase(
    repo: Any,
    workflow_id: int,
    checkpoint_path: str,
    pacing_seconds: int = PACING_SECONDS,
) -> dict:
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
            "status": "error", "data": None,
            "error": f"No ResearchFinding rows for workflow {workflow_id}",
        }

    findings_by_paper: dict[int, list[dict]] = {}
    for f in findings:
        findings_by_paper.setdefault(f["paper_id"], []).append(f)

    paper_ids = sorted(findings_by_paper.keys())
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

    batches = chunk_papers_by_token_budget(
        findings_by_paper, papers_by_id, figures_by_paper_id
    )

    print(f"[gap_detection_batch] {len(findings)} findings across "
          f"{len(paper_ids)} papers -> {len(batches)} batches")

    already_done = _load_checkpoint(checkpoint_path)
    done_batch_indices = {e["batch_index"] for e in already_done}

    for i, batch_paper_ids in enumerate(batches):
        if i in done_batch_indices:
            print(f"[gap_detection_batch] batch {i+1}/{len(batches)} "
                  f"already checkpointed, skipping")
            continue

        batch_findings = [
            f for pid in batch_paper_ids for f in findings_by_paper[pid]
        ]
        digest = build_finding_digest(batch_findings, papers_by_id, figures_by_paper_id)
        prompt = build_gap_prompt(digest)

        print(f"[gap_detection_batch] batch {i+1}/{len(batches)} "
              f"({len(batch_paper_ids)} papers) -> calling LLM...")

        try:
            result = call_reasoning_llm(prompt)
            entry = {
                "batch_index": i,
                "paper_ids": batch_paper_ids,
                "gap_statement": result.get("gap_statement"),
                "supporting_paper_ids": result.get("supporting_paper_ids"),
                "reasoning": result.get("reasoning"),
            }
        except Exception as e:
            entry = {
                "batch_index": i,
                "paper_ids": batch_paper_ids,
                "error": str(e),
            }
            print(f"[gap_detection_batch] batch {i+1} FAILED: {e}")

        _append_checkpoint(checkpoint_path, entry)

        if i < len(batches) - 1:
            print(f"[gap_detection_batch] pacing {pacing_seconds}s before next batch...")
            time.sleep(pacing_seconds)

    return {
        "status": "success",
        "data": {"batches_total": len(batches), "batches_done": len(batches)},
        "error": None,
    }


def build_lightweight_index(repo, workflow_id):
    """
    Compact, deduped tag index of every PAPER in the workflow.
    Aggregates unique values PER PAPER in Python (not SQL DISTINCT across
    all columns together, which produced one line per distinct row
    combination instead of one line per paper - the original bug that
    caused a 10937-token request against an 8000 TPM limit).
    """
    findings = repo.fetch_all(
        """
        SELECT paper_id, material, synthesis_method,
               characterization, property_name
        FROM ResearchFinding
        WHERE workflow_id = ?
        """,
        (workflow_id,),
    )

    materials_by_paper = {}
    methods_by_paper = {}
    chars_by_paper = {}
    props_by_paper = {}

    for f in findings:
        f = dict(f)
        pid = f["paper_id"]
        if f.get("material"):
            materials_by_paper.setdefault(pid, set()).add(f["material"])
        if f.get("synthesis_method"):
            methods_by_paper.setdefault(pid, set()).add(f["synthesis_method"])
        if f.get("characterization"):
            chars_by_paper.setdefault(pid, set()).add(f["characterization"])
        if f.get("property_name"):
            props_by_paper.setdefault(pid, set()).add(f["property_name"])

    all_paper_ids = sorted(
        set(materials_by_paper) | set(methods_by_paper)
        | set(chars_by_paper) | set(props_by_paper)
    )

    MAX_VALUES_PER_FIELD = 3

    def _cap(values):
        values = sorted(values)
        shown = values[:MAX_VALUES_PER_FIELD]
        suffix = " (+{} more)".format(len(values) - MAX_VALUES_PER_FIELD) if len(values) > MAX_VALUES_PER_FIELD else ""
        return ", ".join(shown) + suffix

    lines = []
    for pid in all_paper_ids:
        parts = []
        if pid in materials_by_paper:
            parts.append("materials: " + _cap(materials_by_paper[pid]))
        if pid in methods_by_paper:
            parts.append("methods: " + _cap(methods_by_paper[pid]))
        if pid in chars_by_paper:
            parts.append("characterization: " + _cap(chars_by_paper[pid]))
        if pid in props_by_paper:
            parts.append("properties: " + _cap(props_by_paper[pid]))
        if parts:
            lines.append("[{}] ".format(pid) + "; ".join(parts))

    return "\n".join(lines)


def build_reduce_prompt(candidates: list[dict], lightweight_index: str) -> str:
    # NOTE: deliberately omitting each candidate's full `reasoning` text here.
    # Original attempt included it and blew the 8000 TPM limit (29 candidates'
    # reasoning paragraphs alone requested 15325 tokens). The reduce model
    # doesn't need the OLD reasoning — it re-derives its own check against
    # the tag index, so only gap_statement + supporting_paper_ids are needed.
    candidates_text = "\n".join(
        f"Candidate {i+1}: {(c.get('gap_statement') or '')[:180]} "
        f"(from papers {c.get('supporting_paper_ids')})"
        for i, c in enumerate(candidates)
        if c.get("gap_statement")
    )
    return (
        "You are reviewing candidate research gaps proposed from BATCHES of "
        "a larger literature corpus. Each candidate was proposed seeing only "
        "a subset of the papers. Below is a compact tag index of EVERY paper "
        "in the full corpus (material, synthesis method, characterization, "
        "property studied per paper).\n\n"
        "TASK: For each candidate, check the full tag index — if another "
        "paper's tags suggest that paper already covers this exact "
        "combination, DROP that candidate silently (do not explain why). "
        "Otherwise, keep it, with a reasoning of 20 words or fewer. "
        "Return only the surviving candidates, ranked by how well-grounded "
        "they are.\n\n"
        "Return ONLY a JSON object, no other text:\n"
        '{"final_gaps": [{"gap_statement": "...", "supporting_paper_ids": [...], '
        '"reasoning": "20 words or fewer"}]}\n\n'
        f"CANDIDATES:\n\n{candidates_text}\n\n"
        f"FULL CORPUS TAG INDEX:\n\n{lightweight_index}\n\n"
        "JSON object:"
    )


def run_reduce_phase(repo: Any, workflow_id: int, checkpoint_path: str) -> dict:
    entries = _load_checkpoint(checkpoint_path)
    candidates = [e for e in entries if "gap_statement" in e and e.get("gap_statement")]

    if not candidates:
        return {"status": "error", "data": None,
                "error": "No successful candidates in checkpoint file"}

    lightweight_index = build_lightweight_index(repo, workflow_id)
    prompt = build_reduce_prompt(candidates, lightweight_index)

    try:
        result = call_reasoning_llm(prompt)
    except Exception as e:
        return {"status": "error", "data": None, "error": str(e)}

    return {"status": "success", "data": result, "error": None}
