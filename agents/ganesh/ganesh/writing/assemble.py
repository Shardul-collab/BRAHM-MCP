"""
G5 assembly without rewriting the sections (2026-09-11).

The old G5 sent the LLM the first 600 characters of each section and, when it
answered, used the returned "sections with transitions inserted" AS the final
sections - so every section longer than 600 characters reached the final
document as a rewrite of its own opening. (A literature-review section of
~900 words is ~5,500 characters.) Its reference list was a placeholder.

Here: sections are assembled verbatim from their drafts; the LLM writes only
the abstract, from the full sections, and any abstract sentence whose number
or formula is not in the body is removed; [E#] citations are renumbered per
paper into a real reference list, and an evidence trail maps every citation
back to the ResearchKnowledge row and source sentence it came from.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Tuple

from ganesh.writing.grounding import claims_in, fold, formulas_in, mask_formulas, split_sentences, _CITE_RE

ABSTRACT_PROMPT = """Write the abstract (150-250 words) of the scientific review below.
Use only facts stated in the sections. Do not add numbers, materials or methods that the sections do not
contain. No citations, no headings, no bullet points.

Title: {title}

{body}

ABSTRACT:
"""


def grounded_abstract(llm: Callable, title: str, sections: List[dict], per_section_chars: int = 2500):
    body = "\n\n".join(f"## {s['section_name']}\n{_CITE_RE.sub('', s['content'])[:per_section_chars]}"
                       for s in sections)
    raw = llm(ABSTRACT_PROMPT.format(title=title, body=body), max_tokens=500).strip()
    full_body = fold(" ".join(s["content"] for s in sections))
    num_body, body_formulas = mask_formulas(full_body), formulas_in(full_body)
    kept, dropped = [], []
    for sent in split_sentences(raw):
        nums, forms = claims_in(sent)
        ok = all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", num_body) for n in nums) and \
             all(f in body_formulas for f in forms)
        (kept if ok else dropped).append(sent)
    return " ".join(kept), dropped


def renumber_citations(text: str, lookup: Dict[str, dict]) -> Tuple[str, List[dict], List[dict]]:
    """
    [E583, E12] -> [3, 7] numbered per PAPER in order of first appearance.
    lookup: eid -> {paper_id, title, year, doi, value, sentence, source}
    Returns (text, references, evidence_trail).
    """
    paper_no: Dict[int, int] = {}
    references, trail = [], []

    def repl(m):
        nums = []
        for eid in re.split(r"\s*[,;]\s*", m.group(1)):
            info = lookup.get(eid)
            if not info:
                continue
            pid = info["paper_id"]
            if pid not in paper_no:
                paper_no[pid] = len(paper_no) + 1
                references.append({"n": paper_no[pid], **{k: info.get(k) for k in ("paper_id", "title", "year", "doi", "source")}})
            trail.append({"n": paper_no[pid], "eid": eid, "value": info.get("value"),
                          "sentence": info.get("sentence"), "source": info.get("source")})
            if paper_no[pid] not in nums:
                nums.append(paper_no[pid])
        return "[" + ", ".join(str(n) for n in sorted(nums)) + "]" if nums else ""

    return _CITE_RE.sub(repl, text), references, trail


def render_document(title: str, abstract: str, sections: List[dict], failed: List[str],
                    lookup: Dict[str, dict]) -> Tuple[str, dict]:
    parts = [f"# {title}\n"]
    if abstract:
        parts.append(f"## Abstract\n\n{abstract}\n")
    body = []
    for s in sections:
        body.append(f"\n## {s['section_name']}\n\n{s['content'].strip()}\n")
    for name in failed:
        body.append(f"\n## {name}\n\n_[This section could not be generated; see GANESH logs.]_\n")
    text, refs, trail = renumber_citations("".join(body), lookup)
    parts.append(text)
    parts.append("\n## References\n")
    for r in refs:
        tag = " (abstract only)" if r.get("source") == "abstract" else ""
        doi = f" doi:{r['doi']}" if r.get("doi") else ""
        parts.append(f"[{r['n']}] {r.get('title') or 'Untitled'}{' (' + str(r['year']) + ')' if r.get('year') else ''}.{doi}{tag}")
    parts.append("\n## Evidence trail\n")
    parts.append("Every citation above resolves to a ResearchKnowledge row (E-id) and the sentence it was extracted from.\n")
    for t in trail:
        parts.append(f"- [{t['n']}] {t['eid']}: {t['value']} — \"{(t.get('sentence') or '')[:200]}\"")
    return "\n".join(parts), {"references": refs, "trail": trail}


def evidence_lookup(repo, eids) -> Dict[str, dict]:
    ids = sorted({int(e[1:]) for e in eids if e.startswith("E") and e[1:].isdigit()})
    if not ids:
        return {}
    rows = repo.fetch_all(
        f"SELECT k.id, k.paper_id, k.value, COALESCE(k.sentence,'') AS sentence, k.source_type, "
        f"p.title, p.year, COALESCE(p.doi,'') AS doi FROM ResearchKnowledge k JOIN Paper p ON p.id = k.paper_id "
        f"WHERE k.id IN ({','.join('?' * len(ids))})", tuple(ids))
    return {f"E{r['id']}": {"paper_id": r["paper_id"], "value": r["value"], "sentence": r["sentence"],
                            "source": "abstract" if r["source_type"] == "abstract" else "full text",
                            "title": r["title"], "year": r["year"], "doi": r["doi"]} for r in rows}
