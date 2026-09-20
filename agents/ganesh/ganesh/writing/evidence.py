"""
Evidence packets for GANESH sections (G1 rework, 2026-09-11).

Replaces three G1 behaviours measured on workflow 1:
  * papers were chosen per VECTOR hit (8 slots per section), so a section drew
    on 4-5 of 9 papers and papers 10, 13, 14 appeared in none;
  * valid papers required status extracted|knowledge_ready|completed, so the 11
    abstract-only papers were excluded outright;
  * rows were fetched with LIMIT 30 and no ORDER BY, carried no paper id, and
    SECTION_CATEGORY_MAP was keyed "Methodology"/"Results" while the literature
    review's sections are "Synthesis Methods"/"Properties & Results", so the
    routing never applied.

Every Evidence item is identified as E<ResearchKnowledge.id>, so any citation
in the review resolves to one DB row, one paper and one source sentence.
"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from typing import Dict, List

from ganesh.writing.grounding import Evidence, fold, formulas_in

# Literature-review section -> knowledge categories. None = all categories.
SECTION_CATEGORIES = {
    "Introduction":         {"material", "application", "synthesis_method"},
    "Background":           {"material", "application", "ferroelectric_property", "optical_property"},
    "Materials Overview":   {"material", "optical_property", "electrical_property", "ferroelectric_property"},
    "Synthesis Methods":    {"synthesis_method", "growth_temperature", "growth_duration", "chamber_pressure",
                             "gas_flow", "growth_flux", "annealing_condition", "doping_parameter"},
    "Characterization":     {"characterization"},
    "Properties & Results": {"optical_property", "electrical_property", "ferroelectric_property",
                             "photodetector_metric", "photoresponsivity", "field_effect_mobility",
                             "on_off_ratio", "threshold_voltage", "subthreshold_swing",
                             "contact_resistance", "defect_type"},
    "Discussion":           None,
    "Research Gaps":        None,
    "Conclusion":           None,
}

_NUM = re.compile(r"\d")

# D13 (2026-09-11 pm). Measured on the rebuilt corpus with budget 36 / 6 per paper: 152 of 566
# rows (27%) reached any section, and the three all-category sections got identical packets.
TOPICAL_BUDGET, TOPICAL_PER_PAPER = 60, 10
ALL_CATEGORY_BUDGET = 36
ALL_CATEGORY_SECTIONS = ("Discussion", "Research Gaps", "Conclusion")


def paper_catalogue(conn: sqlite3.Connection, workflow_id: int) -> Dict[int, dict]:
    """Every paper with knowledge, labelled full text vs abstract-only."""
    rows = conn.execute(
        """
        SELECT p.id, p.title, p.year, COALESCE(p.doi, ''),
               SUM(k.source_type IN ('llm', 'rule', 'pattern')) AS fulltext_rows,
               COUNT(k.id) AS rows_total, COALESCE(p.abstract, '')
        FROM Paper p JOIN ResearchKnowledge k ON k.paper_id = p.id
        WHERE p.workflow_id = ?
        GROUP BY p.id ORDER BY p.id
        """, (workflow_id,)).fetchall()
    return {pid: {"paper_id": pid, "title": t or "", "year": y, "doi": d,
                  "source": "full text" if ft else "abstract", "rows": n,
                  "formulas": " ".join(sorted(formulas_in(fold(f"{t or ''} {ab}"))))}
            for pid, t, y, d, ft, n, ab in rows}


def _row_priority(category: str, value: str, sentence: str) -> tuple:
    # numbers first (they are what a review cites and what the verifier can
    # check), then rows with a real evidence sentence, then longer values
    return (0 if _NUM.search(value or "") else 1, 0 if sentence else 1, -len(value or ""))


def section_evidence(conn: sqlite3.Connection, workflow_id: int, section: str,
                     budget: int = 36, per_paper_cap: int = 6,
                     exclude_papers=(), exclude_ids=frozenset()) -> List[Evidence]:
    """
    Evidence for one section, spread across papers by round-robin so no single
    paper dominates and every contributing paper gets a turn.
    """
    cats = SECTION_CATEGORIES.get(section)
    papers = paper_catalogue(conn, workflow_id)
    q = ("SELECT k.id, k.paper_id, k.category, k.value, COALESCE(k.sentence, ''), k.source_type "
         "FROM ResearchKnowledge k JOIN Paper p ON p.id = k.paper_id WHERE p.workflow_id = ?")
    args = [workflow_id]
    if cats:
        q += f" AND k.category IN ({','.join('?' * len(cats))})"
        args += sorted(cats)
    by_paper = defaultdict(list)
    seen = set()
    for kid, pid, cat, val, sent, st in conn.execute(q, args):
        if pid in exclude_papers or pid not in papers or f"E{kid}" in exclude_ids:
            continue
        key = (pid, cat, (val or "").lower().strip())
        if key in seen:
            continue
        seen.add(key)
        by_paper[pid].append((_row_priority(cat, val, sent), kid, pid, cat, val, sent))
    for pid in by_paper:
        by_paper[pid].sort()
        by_paper[pid] = by_paper[pid][:per_paper_cap]

    chosen: List[Evidence] = []
    # full-text papers first in each round, then abstract-only
    order = sorted(by_paper, key=lambda p: (papers[p]["source"] != "full text", p))
    rnd = 0
    while len(chosen) < budget and any(len(by_paper[p]) > rnd for p in order):
        for pid in order:
            if len(chosen) >= budget:
                break
            if len(by_paper[pid]) > rnd:
                _, kid, _, cat, val, sent = by_paper[pid][rnd]
                meta = papers[pid]
                chosen.append(Evidence(f"E{kid}", pid, cat, val, sent, meta["source"],
                                       meta["title"], meta["year"], meta["doi"], meta["formulas"]))
        rnd += 1
    return chosen


def document_packets(conn: sqlite3.Connection, workflow_id: int, section_names) -> Dict[str, List[Evidence]]:
    """Packets for a whole document, in section order. Topical sections get TOPICAL_BUDGET; the
    all-category sections get only evidence that no earlier section (topical or all-category) used."""
    packets, used = {}, set()
    for s in section_names:
        if s in ALL_CATEGORY_SECTIONS:
            ev = section_evidence(conn, workflow_id, s, budget=ALL_CATEGORY_BUDGET,
                                  per_paper_cap=TOPICAL_PER_PAPER, exclude_ids=frozenset(used))
        else:
            ev = section_evidence(conn, workflow_id, s, budget=TOPICAL_BUDGET, per_paper_cap=TOPICAL_PER_PAPER)
        packets[s] = ev
        used |= {x.eid for x in ev}
    return packets


def format_for_prompt(evidence: List[Evidence]) -> str:
    lines = []
    for e in evidence:
        tag = "abstract only" if e.source == "abstract" else "full text"
        ctx = (e.sentence or "").replace("\n", " ")[:220]
        lines.append(f"[{e.eid}] (paper P{e.paper_id}, {tag}) {e.category}: {e.value}"
                     + (f' — "{ctx}"' if ctx else ""))
    return "\n".join(lines)
