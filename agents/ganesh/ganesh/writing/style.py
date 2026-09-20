"""
Measured style properties of a paragraph (L5, and scoring for all variants).

Motivated by the structural findings Shardul collected on AI vs human prose:
uniform sentence length, lexical repetition and stock transitions. Every
number here is a measurement; the limits in STYLE_LIMITS are provisional
starting points to be read against the distribution on the pilot, not tuned
to make a variant win.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import List

from ganesh.writing.grounding import split_sentences, _CITE_RE

CLICHES = [
    "furthermore", "moreover", "additionally", "in addition", "it is worth noting",
    "it is important to note", "notably", "in conclusion", "overall", "in summary",
    "plays a crucial role", "plays a pivotal role", "a key role", "delve", "shed light",
    "paving the way", "pave the way", "in the realm of", "landscape", "tapestry",
    "significant attention", "garnered", "highlighting the", "underscores", "a testament",
    "cutting-edge", "state-of-the-art", "holds great promise", "promising candidate",
]

STYLE_LIMITS = {
    "sentence_len_cv_min": 0.30,   # human academic prose varies sentence length
    "cliches_max": 1,              # per paragraph
    "mtld_min": 50.0,              # lexical diversity
}

_LABEL_RE = re.compile(r"\b(?:[Pp]aper\s+)?P\d{1,3}\b|\bE\d{1,5}\b")
_FIG_RE = re.compile(r"\b(?:Fig(?:ure)?s?\.?|Tables?)\s*\(?\d", re.I)
_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(_CITE_RE.sub(" ", text))]


def mtld(words: List[str], ttr_threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis), forward+backward mean."""
    def one_pass(ws):
        factors, types, count = 0.0, set(), 0
        for w in ws:
            count += 1
            types.add(w)
            if len(types) / count <= ttr_threshold:
                factors += 1
                types, count = set(), 0
        if count:
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - ttr_threshold) if ttr < 1 else 0
        return len(ws) / factors if factors else float(len(ws))
    if len(words) < 10:
        return float(len(words))
    return (one_pass(words) + one_pass(list(reversed(words)))) / 2


def paragraph_style(text: str) -> dict:
    sents = split_sentences(text)
    lens = [len(_words(s)) for s in sents if _words(s)]
    low = text.lower()
    cliches = [c for c in CLICHES if c in low]
    cv = (statistics.pstdev(lens) / statistics.mean(lens)) if len(lens) >= 2 and statistics.mean(lens) else 0.0
    ws = _words(text)
    return {
        "sentences": len(sents),
        "words": len(ws),
        "sentence_len_cv": round(cv, 3),
        "mtld": round(mtld(ws), 1),
        "cliches": cliches,
        # internal labels and source figure numbers mean nothing to a reader of the review
        "label_leaks": len(_LABEL_RE.findall(_CITE_RE.sub(" ", text))),
        "figure_refs": len(_FIG_RE.findall(text)),
    }


def style_failures(style: dict) -> List[str]:
    out = []
    if style["sentences"] >= 3 and style["sentence_len_cv"] < STYLE_LIMITS["sentence_len_cv_min"]:
        out.append(f"sentence lengths are too uniform (CV {style['sentence_len_cv']})")
    if len(style["cliches"]) > STYLE_LIMITS["cliches_max"]:
        out.append("stock phrases: " + ", ".join(style["cliches"]))
    if style.get("label_leaks"):
        out.append("internal labels such as P13 written as words - refer to sources only by bracketed IDs")
    if style.get("figure_refs"):
        out.append("mentions figure/table numbers of the source papers - remove them")
    if style["words"] >= 60 and style["mtld"] < STYLE_LIMITS["mtld_min"]:
        out.append(f"repetitive vocabulary (MTLD {style['mtld']})")
    return out


def repeated_openings(paragraphs: List[str]) -> int:
    """How many paragraphs share their first two words with an earlier one."""
    seen, n = set(), 0
    for p in paragraphs:
        key = tuple(_words(p)[:2])
        if key in seen:
            n += 1
        seen.add(key)
    return n


def synthesis_metrics(paragraph_checks: List[list], evidence_by_id: dict) -> dict:
    """Cross-paper synthesis, counted: papers cited per paragraph."""
    per_para = []
    for checks in paragraph_checks:
        papers = {evidence_by_id[e].paper_id for c in checks for e in c.cited if e in evidence_by_id}
        per_para.append(len(papers))
    n = len(per_para) or 1
    return {
        "paragraphs": len(per_para),
        "multi_paper_paragraph_share": round(sum(1 for x in per_para if x >= 2) / n, 2),
        "mean_papers_per_paragraph": round(sum(per_para) / n, 2),
    }
