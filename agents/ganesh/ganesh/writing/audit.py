"""
Claim audit for the TRL-5 bar (2026-09-11): >=90% of cited claims supported by
their source, no fabricated values.

The verifier checks that numbers/formulas appear in the cited evidence. The
audit checks what it cannot: that the evidence row says what the sentence
claims, and that the row itself matches the paper. For each sampled sentence
it lays out: the claim, each cited evidence row (value + extracted sentence),
and a window of the paper's own text around that sentence, for a human (or an
independent reviewer) to mark SUPPORTED / PARTIAL / UNSUPPORTED.

  python -m ganesh.writing.audit sample --text review.md --n 50 --out audit.md
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path

from ganesh.writing.grounding import _CITE_RE, split_sentences

ROOT = Path(__file__).resolve().parents[4]
DB = ROOT / "agents" / "shani" / "database" / "research_workflow.db"
_norm = lambda s: re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _window(raw: str, sentence: str, width: int = 450) -> str:
    if not raw or not sentence:
        return ""
    key = _norm(sentence)[:50]
    nraw, idx_map = [], []
    for i, ch in enumerate(raw):
        n = _norm(ch)
        if n:
            nraw.append(n); idx_map.append(i)
    pos = "".join(nraw).find(key)
    if pos < 0:
        return "(source sentence not located verbatim in the paper text)"
    i = idx_map[pos]
    return re.sub(r"\s+", " ", raw[max(0, i - width // 3): i + width])


def sample(sections_text: str, n: int, seed: int = 11) -> list:
    cited = [s for s in split_sentences(sections_text) if _CITE_RE.search(s)]
    random.Random(seed).shuffle(cited)
    return cited[:n]


def document_text(conn, document_id: int) -> str:
    """Latest draft of every approved/integrated section, in document order, with E-ids intact."""
    rows = conn.execute(
        "SELECT s.section_name, (SELECT d.content FROM GaneshDraft d WHERE d.section_id = s.id "
        "ORDER BY d.version DESC LIMIT 1) FROM GaneshSection s WHERE s.document_id = ? "
        "AND s.status IN ('approved', 'integrated') ORDER BY s.exec_order", (document_id,)).fetchall()
    return "\n\n".join((content or "").strip() for _, content in rows)


def build(text: str, n: int, out: Path):
    conn = sqlite3.connect(DB)
    lines = ["# Claim audit\n", "Mark each claim SUPPORTED / PARTIAL / UNSUPPORTED against the paper text.\n"]
    items = []
    for k, sent in enumerate(sample(text, n), 1):
        eids = [e.strip() for g in _CITE_RE.findall(sent) for e in re.split(r"[,;]", g)]
        ev = []
        for eid in eids:
            row = conn.execute("SELECT k.paper_id, k.category, k.value, COALESCE(k.sentence,''), k.source_type, "
                               "COALESCE(p.raw_text, p.abstract, ''), p.title FROM ResearchKnowledge k "
                               "JOIN Paper p ON p.id=k.paper_id WHERE k.id=?", (int(eid[1:]),)).fetchone()
            if row:
                pid, cat, val, s, st, raw, title = row
                ev.append({"eid": eid, "paper": pid, "title": title, "category": cat, "value": val,
                           "sentence": s, "source": st, "window": _window(raw, s)})
        items.append({"n": k, "claim": sent, "evidence": ev, "verdict": None, "note": None})
        lines.append(f"\n## {k}. {sent}\n")
        for e in ev:
            lines.append(f"- **{e['eid']}** P{e['paper']} ({e['source']}) {e['category']}: `{e['value']}`\n"
                         f"  - extracted sentence: \"{e['sentence'][:400]}\"\n"
                         f"  - paper text: \"{e['window']}\"")
        lines.append("\n**Verdict:** ____  **Note:** ____\n")
    out.write_text("\n".join(lines))
    out.with_suffix(".json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample"])
    ap.add_argument("--text")
    ap.add_argument("--document-id", type=int)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.document_id:
        txt = document_text(sqlite3.connect(DB), a.document_id)
    elif a.text:
        txt = Path(a.text).read_text()
    else:
        ap.error("--text or --document-id is required")
    if not _CITE_RE.search(txt):
        ap.error("no [E#] citations in the input - the final G5 document is renumbered; use --document-id")
    build(txt, a.n, Path(a.out))
