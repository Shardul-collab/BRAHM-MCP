"""
L3 / L4 / L5 writing variants (2026-09-11). All share: evidence IDs, mandatory
citations, one verifier-driven repair pass, and removal of whatever still fails.

  L3  cluster evidence by category -> small writer paragraphs -> 14B flow pass
  L4  claim plan computed from the evidence -> small writer paragraphs -> 14B flow pass
  L5  L4 + measured style check per paragraph, rewrite only failing paragraphs
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Dict, List

from ganesh.writing.grounding import Evidence, check_text, failing, normalise_output, strip_failing, summarise
from ganesh.writing.evidence import format_for_prompt
from ganesh.writing.claims import build_claims
from ganesh.writing.style import paragraph_style, style_failures
from ganesh.llm_client import max_output_for

RULES = """RULES
- Use only the evidence listed below. Do not add knowledge from outside it.
- Every sentence that states a number, a chemical formula, a method or a finding must end with the ID(s)
  of the evidence it comes from, in square brackets, e.g. [E214] or [E214, E311].
- Copy numbers exactly as written in the evidence, with the same unit. Never round, convert or combine them.
- Where the evidence allows, compare papers: where results agree, where they differ, and under what
  conditions each was obtained.
- Evidence marked "abstract only" comes from a paper's abstract: write "reports" or "describes", not
  detailed claims about how it was measured.
- Do not say that a finding agrees with, contrasts with or is not corroborated by other work unless the
  evidence for both sides is cited in the same sentence.
- Give each value the material and phase its evidence names (InSe, Bi2Se3, Mn2In2Se5 ...). Do not attribute
  a value to the review's subject material unless its evidence does.
- Never write internal labels such as P13 or E214 as words; the bracketed IDs are the only way to refer to
  a source ("one study reports ... [E214]"). Do not mention figure or table numbers of the source papers.
- Write in the register of a peer-reviewed review article. No headings, no bullet points, no reference list.
"""


def _writer_prompt(topic, section, task, evidence):
    return (f"You are writing one paragraph of the section \"{section}\" of a scientific literature review "
            f"on {topic}.\n\n{RULES}\nTASK\n{task}\nLength: 90-150 words.\n\nEVIDENCE\n"
            f"{format_for_prompt(evidence)}\n\nPARAGRAPH:\n")


def _repair_prompt(topic, section, paragraph, bad, evidence):
    why = "\n".join(
        f"- \"{c.text}\" -> " + (
            "has no evidence ID" if c.status == "uncited" else
            f"cites unknown ID(s) {', '.join(c.missing)}" if c.status == "bad_citation" else
            f"{', '.join(c.missing)} not found in the evidence it cites ({', '.join(c.cited)})")
        for c in bad)
    return (f"This paragraph from the section \"{section}\" of a review on {topic} failed an automatic "
            f"evidence check.\n\nFAILING SENTENCES\n{why}\n\nRewrite the paragraph. For each failing sentence, "
            f"either cite the evidence that contains that exact number or formula, or remove the claim. Keep the "
            f"other sentences as they are.\n\n{RULES}\nEVIDENCE\n{format_for_prompt(evidence)}\n\n"
            f"PARAGRAPH\n{paragraph}\n\nREWRITTEN PARAGRAPH:\n")


def _style_prompt(topic, section, paragraph, problems, evidence):
    return (f"Revise this paragraph from the section \"{section}\" of a review on {topic}. An automatic style "
            f"check found: {'; '.join(problems)}.\nVary sentence length and structure, prefer precise verbs to "
            f"stock transitions, and do not repeat wording. Keep every fact, every number and every bracketed "
            f"evidence ID exactly as they are; add nothing new.\n\n{RULES}\nEVIDENCE\n{format_for_prompt(evidence)}\n\n"
            f"PARAGRAPH\n{paragraph}\n\nREVISED PARAGRAPH:\n")


def _polish_prompt(topic, section, paragraphs):
    body = "\n\n".join(paragraphs)
    return (f"Below are the paragraphs of the section \"{section}\" of a scientific review on {topic}.\n"
            f"Edit them into one coherent section: order the paragraphs logically, add short transitions, and "
            f"remove repetition between paragraphs. Do NOT add any fact, number, material or method. Keep every "
            f"bracketed evidence ID, e.g. [E214], attached to the claim it supports; do not invent IDs. "
            f"No headings, no bullet points.\n\nPARAGRAPHS\n{body}\n\nEDITED SECTION:\n")


class Writer:
    def __init__(self, llm: Callable, writer_model: str, polish_model: str | None, topic: str,
                 always_allowed=(), repair_model: str | None = None):
        self.llm, self.wm, self.pm, self.topic = llm, writer_model, polish_model, topic
        # hybrid setup (2026-09-11 pm): the verifier-driven repair is a small, constrained edit and
        # can run on the local 7B while the writer / flow pass run on a larger hosted model
        self.rm = repair_model or writer_model
        self.always_allowed = set(always_allowed)
        self.log: List[dict] = []

    def _call(self, prompt, model, max_tokens, step):
        t0 = time.time()
        out = normalise_output(self.llm(prompt, max_tokens=max_tokens, model=model, temperature=0.3)).strip()
        try:
            from ganesh.llm_client import LAST_CALL
            done, n_out = LAST_CALL.get("done_reason"), LAST_CALL.get("eval_count")
        except Exception:
            done, n_out = None, None
        self.log.append({"step": step, "model": model, "seconds": round(time.time() - t0, 1),
                         "prompt_chars": len(prompt), "out_chars": len(out), "max_tokens": max_tokens,
                         "done_reason": done, "eval_count": n_out})
        return out

    def grounded_paragraph(self, section, task, evidence: List[Evidence]) -> dict:
        ev = {e.eid: e for e in evidence}
        text = self._call(_writer_prompt(self.topic, section, task, evidence), self.wm, 400, "write")
        checks = check_text(text, ev, self.always_allowed)
        first = summarise(checks)
        repaired = False
        if failing(checks):
            text = self._call(_repair_prompt(self.topic, section, text, failing(checks), evidence),
                              self.rm, 400, "repair")
            checks = check_text(text, ev, self.always_allowed)
            repaired = True
        after_repair = summarise(checks)
        removed = [c.text for c in failing(checks)]
        total_sentences = len(checks)
        text = strip_failing(text, checks)
        return {"text": text, "evidence": evidence, "first_check": first, "after_repair": after_repair,
                "repaired": repaired, "removed": removed, "sentences_before_strip": total_sentences}

    def restyle(self, section, para: dict) -> dict:
        problems = style_failures(paragraph_style(para["text"]))
        if not problems:
            return para
        ev = {e.eid: e for e in para["evidence"]}
        new = self._call(_style_prompt(self.topic, section, para["text"], problems, para["evidence"]),
                         self.wm, 400, "style")
        checks = check_text(new, ev, self.always_allowed)
        # a style pass must not cost grounding: keep it only if nothing fails
        if failing(checks) or not new:
            para["style_pass"] = "rejected (broke grounding)"
            return para
        para = dict(para, text=new, style_pass=f"applied: {'; '.join(problems)}")
        return para

    def polish(self, section, paragraphs: List[dict]) -> dict:
        texts = [p["text"] for p in paragraphs if p["text"]]
        all_ev = {e.eid: e for p in paragraphs for e in p["evidence"]}
        before = "\n\n".join(texts)
        before_checks = check_text(before, all_ev, self.always_allowed)
        if not self.pm:
            return {"text": before, "checks": before_checks, "polish": "none"}
        cap = min(6000, max(1800, int(len(before.split()) * 1.6) + 300))
        if cap > max_output_for(self.pm):
            # e.g. Groq free tier: 1,000 output tokens/minute, a section needs ~1,800-3,600
            return {"text": before, "checks": before_checks,
                    "polish": f"skipped ({cap} output tokens needed, {max_output_for(self.pm)} allowed)",
                    "polish_detail": {"supported_before": summarise(before_checks)["supported"]}}
        out = self._call(_polish_prompt(self.topic, section, texts), self.pm, cap, "polish")
        if self.log[-1].get("done_reason") == "length":
            return {"text": before, "checks": before_checks, "polish": f"reverted (truncated at {cap} tokens)"}
        checks = check_text(out, all_ev, self.always_allowed)
        s_b, s_a = summarise(before_checks), summarise(checks)
        detail = {"supported_before": s_b["supported"], "supported_after": s_a["supported"],
                  "failing_after": s_a["unsupported"] + s_a["uncited"] + s_a["bad_citation"]}
        # the flow pass may not lose citations or introduce unsupported claims
        if s_a["supported"] < s_b["supported"] or failing(checks):
            kept = strip_failing(out, checks)
            kept_checks = check_text(kept, all_ev, self.always_allowed)
            detail["supported_after_strip"] = summarise(kept_checks)["supported"]
            if detail["supported_after_strip"] < s_b["supported"]:
                return {"text": before, "checks": before_checks, "polish": "reverted (lost supported claims)",
                        "polish_detail": detail}
            return {"text": kept, "checks": kept_checks, "polish": "applied, failing sentences removed",
                    "polish_detail": detail}
        return {"text": out, "checks": checks, "polish": "applied", "polish_detail": detail}


def plan_L3(evidence: List[Evidence]) -> List[tuple]:
    by_cat = defaultdict(list)
    for e in evidence:
        by_cat[e.category].append(e)
    tasks = []
    for cat, evs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        for i in range(0, len(evs), 8):
            tasks.append((f"Synthesise what the evidence says about {cat.replace('_', ' ')}.", evs[i:i + 8]))
    return tasks


def plan_L4(evidence: List[Evidence], gaps: dict | None = None) -> List[tuple]:
    tasks = []
    for c in build_claims(evidence, gaps):
        if not c.evidence:            # gap claims: no numbers, written as one closing sentence set
            tasks.append((f"State this gap plainly in one or two sentences, without citations: {c.statement}", []))
            continue
        task = f"Write about this claim, supporting it from the evidence: {c.statement}"
        if c.note:
            task += "\n" + c.note
        tasks.append((task, c.evidence))
    return tasks


def write_section(writer: Writer, variant: str, section: str, evidence: List[Evidence],
                  gaps: dict | None = None) -> dict:
    """gaps: corpus-wide {property: papers} - pass only for sections that discuss gaps."""
    t0 = time.time()
    tasks = plan_L3(evidence) if variant == "L3" else plan_L4(evidence, gaps)
    paragraphs = []
    for task, evs in tasks:
        if not evs:
            text = writer._call(f"Write one or two sentences for a scientific review on {writer.topic}. "
                                f"{task}\nNo citations, no numbers.\n", writer.wm, 120, "gap")
            paragraphs.append({"text": text, "evidence": [], "first_check": None, "repaired": False, "removed": []})
            continue
        p = writer.grounded_paragraph(section, task, evs)
        if variant == "L5" and p["text"]:
            p = writer.restyle(section, p)
        paragraphs.append(p)
    sec = writer.polish(section, paragraphs)
    return {"variant": variant, "section": section, "paragraphs": paragraphs, **sec,
            "seconds": round(time.time() - t0, 1)}
