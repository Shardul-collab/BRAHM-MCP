"""
Pilot benchmark for GANESH writing (2026-09-11). Local models only.

Stage A - writer model: every candidate writes the SAME paragraph tasks (the
L4 claim plan of the pilot sections), each through write -> verify -> one
repair -> strip. Stage B - variants: L3/L4/L5 with the chosen writer and the
14B flow pass, full pilot sections, plus a blind-read file.

  ../shani/venv/bin/python -m ganesh.writing.bench A --models granite4.2:8b,qwen2.5:7b --out /path
  ../shani/venv/bin/python -m ganesh.writing.bench B --writer X --polish qwen2.5:14b --out /path
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import time
from pathlib import Path

from ganesh.llm_client import call_llm, ensure_local_models
from ganesh.writing.claims import corpus_property_coverage
from ganesh.writing.evidence import document_packets
from ganesh.writing.grounding import check_text, summarise
from ganesh.writing.style import paragraph_style, repeated_openings, synthesis_metrics
from ganesh.writing.variants import Writer, plan_L4, write_section

ROOT = Path(__file__).resolve().parents[4]
DB = ROOT / "agents" / "shani" / "database" / "research_workflow.db"
SECTIONS = ["Synthesis Methods", "Properties & Results"]
TOPIC = "the growth, polymorphism and properties of In2Se3 thin films"
SUBJECT = {"In2Se3"}


def _ctx(conn, workflow_id):
    props = [p.strip() for p in (conn.execute(
        "SELECT properties FROM WorkflowResearchConfig WHERE workflow_id=?", (workflow_id,)).fetchone()[0] or "").split(",") if p.strip()]
    gaps = corpus_property_coverage(conn, workflow_id, props)
    # same packets as the real G1 builds for these sections (D13)
    evidence = {s: v for s, v in document_packets(conn, workflow_id, SECTIONS).items()}
    return gaps, evidence


def _para_record(p, seconds=None):
    st = paragraph_style(p["text"]) if p["text"] else {}
    ev = {e.eid: e for e in p["evidence"]}
    checks = check_text(p["text"], ev, SUBJECT) if p["text"] else []
    return {
        "first": p.get("first_check"), "after_repair": p.get("after_repair"),
        "removed": len(p.get("removed", [])), "sentences": p.get("sentences_before_strip"),
        "papers_cited": len({ev[e].paper_id for c in checks for e in c.cited if e in ev}),
        "style": st, "text": p["text"], "seconds": seconds,
    }


def stage_a(models, out: Path, workflow_id=1, max_tasks=18, repair=None):
    ensure_local_models(models + ([repair] if repair else []))
    conn = sqlite3.connect(DB)
    gaps, evidence = _ctx(conn, workflow_id)
    tasks = []
    per_section = max(1, max_tasks // len(SECTIONS))
    for s in SECTIONS:
        tasks += [(s, t, e) for t, e in plan_L4(evidence[s], gaps if s == "Properties & Results" else None)
                  if e][:per_section]
    results = {}
    for m in models:
        w = Writer(call_llm, m, None, TOPIC, SUBJECT, repair_model=repair)
        recs = []
        for s, t, e in tasks:
            t0 = time.time()
            p = w.grounded_paragraph(s, t, e)
            recs.append(_para_record(p, round(time.time() - t0, 1)))
            print(f"[A] {m} | {s[:12]} | first {p['first_check']['grounding_rate']:.2f} "
                  f"after {p['after_repair']['grounding_rate']:.2f} removed {len(p['removed'])} "
                  f"| {recs[-1]['seconds']}s", flush=True)
        results[m] = {"paragraphs": recs, "calls": w.log, "summary": _aggregate(recs)}
        (out / "stage_a.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return results


def _aggregate(recs):
    def tot(key, sub):
        return sum((r[key] or {}).get(sub, 0) for r in recs if r[key])
    ch_f, ch_a = tot("first", "checkable"), tot("after_repair", "checkable")
    return {
        "paragraphs": len(recs),
        "first_pass_grounding": round(tot("first", "supported") / ch_f, 3) if ch_f else None,
        "after_repair_grounding": round(tot("after_repair", "supported") / ch_a, 3) if ch_a else None,
        "sentences_removed_share": round(sum(r["removed"] for r in recs) / max(1, sum(r["sentences"] or 0 for r in recs)), 3),
        "supported_claims_kept": tot("after_repair", "supported"),
        "mean_papers_cited": round(statistics.mean(r["papers_cited"] for r in recs), 2) if recs else 0,
        "mean_words": round(statistics.mean(r["style"].get("words", 0) for r in recs), 1) if recs else 0,
        "mean_sentence_len_cv": round(statistics.mean(r["style"].get("sentence_len_cv", 0) for r in recs), 3) if recs else 0,
        "mean_mtld": round(statistics.mean(r["style"].get("mtld", 0) for r in recs), 1) if recs else 0,
        "cliches_per_paragraph": round(statistics.mean(len(r["style"].get("cliches", [])) for r in recs), 2) if recs else 0,
        "mean_seconds_per_paragraph": round(statistics.mean(r["seconds"] for r in recs if r["seconds"]), 1) if recs else 0,
    }


def stage_b(writer_model, polish_model, out: Path, variants=("L3", "L4", "L5"), workflow_id=1, repair=None):
    ensure_local_models([writer_model] + ([polish_model] if polish_model else []) + ([repair] if repair else []))
    conn = sqlite3.connect(DB)
    gaps, evidence = _ctx(conn, workflow_id)
    results = {}
    for v in variants:
        for s in SECTIONS:
            w = Writer(call_llm, writer_model, polish_model, TOPIC, SUBJECT, repair_model=repair)
            r = write_section(w, v, s, evidence[s], gaps if s == "Properties & Results" else None)
            ev = {e.eid: e for p in r["paragraphs"] for e in p["evidence"]}
            paras = [x for x in r["text"].split("\n\n") if x.strip()]
            para_checks = [check_text(x, ev, SUBJECT) for x in paras]
            results[f"{v}|{s}"] = {
                "seconds": r["seconds"], "polish": r["polish"], "polish_detail": r.get("polish_detail"),
                "section_check": summarise(r["checks"]),
                "synthesis": synthesis_metrics(para_checks, ev),
                "repeated_openings": repeated_openings(paras),
                "style": [paragraph_style(x) for x in paras],
                "paragraph_records": [_para_record(p) for p in r["paragraphs"]],
                "calls": w.log, "text": r["text"],
            }
            print(f"[B] {v} | {s} | {r['seconds']}s | polish: {r['polish']} | "
                  f"{summarise(r['checks'])}", flush=True)
            (out / "stage_b.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    _blind(results, out, variants)
    return results


def _blind(results, out: Path, variants):
    rng = random.Random(20260911)
    key, lines = {}, ["# Blind read — which version of each section reads best?\n",
                      "Rate each version 1-5 for accuracy of synthesis, flow, and whether it reads like a "
                      "real review. Citations are shown as [E#].\n"]
    for s in SECTIONS:
        order = list(variants)
        rng.shuffle(order)
        lines.append(f"\n## {s}\n")
        for label, v in zip("ABC", order):
            key[f"{s}|{label}"] = v
            lines.append(f"\n### Version {label}\n\n{results[f'{v}|{s}']['text']}\n")
    (out / "blind_read.md").write_text("\n".join(lines))
    (out / "blind_key.json").write_text(json.dumps(key, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["A", "B"])
    ap.add_argument("--models", default="")
    ap.add_argument("--writer", default="")
    ap.add_argument("--polish", default="qwen2.5:14b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tasks", type=int, default=18)
    ap.add_argument("--variants", default="L3,L4,L5")
    ap.add_argument("--repair", default=None, help="model for the verifier-driven repair (default: the writer)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.stage == "A":
        stage_a([m for m in a.models.split(",") if m], out, max_tasks=a.max_tasks, repair=a.repair)
    else:
        stage_b(a.writer, a.polish or None, out, repair=a.repair,
                variants=tuple(v for v in a.variants.split(",") if v))
