"""
ganesh/tools/load_context.py
=============================
G1 — Load research context using FAISS vector search.
Builds a per-section evidence map so G3 prompts stay small.

Context source: Chitragupta /context/load (filtered, ready papers only).
FAISS vector routing runs on top for per-section evidence selection.
"""
from __future__ import annotations
import json
import re
import sys
import os
from datetime import datetime
from pathlib import Path

SHANI_ROOT = Path("/mnt/d/brahm/agents/shani")
if str(SHANI_ROOT) not in sys.path:
    sys.path.insert(0, str(SHANI_ROOT))

CHITRAGUPTA_BASE = "http://localhost:8003"
CHITRAGUPTA_KEY  = os.getenv("API_KEY", "chitragupta_api_2026Uzp7N9dRpYguBAEiljqFpn075xIpGdEI")

# ── Section-aware evidence routing ──────────────────────────────────────
# Maps section names to the ResearchKnowledge categories that are actually
# relevant to that section. Sections not listed here keep the old broad,
# unfiltered behavior (Executive Summary, Introduction, Conclusion).
#
# Rationale: without this, every section pulled the same LIMIT-30 pool,
# dominated by high-volume generic categories (characterization, material),
# which is why specific numbers (mobility, growth temp) rarely surfaced.
SECTION_CATEGORY_MAP = {
    "Methodology": {
        "synthesis_method", "growth_temperature", "growth_duration",
        "chamber_pressure", "gas_flow", "annealing_condition",
        "computational_method",
    },
    "Results": {
        "field_effect_mobility", "on_off_ratio", "threshold_voltage",
        "subthreshold_swing", "contact_resistance", "photoresponsivity",
        "electrical_property", "optical_property", "defect_type",
        "doping_parameter",
    },
    "Discussion": {
        "field_effect_mobility", "on_off_ratio", "threshold_voltage",
        "subthreshold_swing", "contact_resistance", "photoresponsivity",
        "electrical_property", "optical_property", "defect_type",
        "doping_parameter",
    },
}

# 'characterization' is a catch-all category mixing structural (XRD/TEM/SEM),
# optical (Raman/PL), and electrical (TLM/EIS) techniques. Confirmed via
# direct sampling of ResearchKnowledge.value on 2026-07-02 — not resolved by
# category alone, so we sub-classify by keyword on the `value` field only
# (not `sentence`, to avoid false positives from unrelated body text).
SECTION_CHARACTERIZATION_SUBTYPES = {
    "Methodology": {"structural"},
    "Results":     {"optical", "electrical"},
    "Discussion":  {"optical", "electrical"},
}

_STRUCTURAL_PATTERNS  = [r"\\bxrd\\b", r"x-ray diffraction",
                          r"\\btem\\b", r"transmission electron micro",
                          r"\\bafm\\b", r"atomic force micro",
                          r"\\bsem\\b", r"scanning electron micro",
                          r"\\bbet\\b"]
_OPTICAL_PATTERNS     = [r"\\bpl\\b", r"\\braman\\b", r"photoluminescence",
                          r"uv-?vis", r"\\bel\\b", r"electroluminescence",
                          r"\\babsorption\\b"]
_ELECTRICAL_PATTERNS  = [r"\\btlm\\b", r"transmission line measure",
                          r"\\beis\\b", r"electrochemical impedance",
                          r"i[-–]v\\b", r"\\bhall\\b"]


def _classify_characterization(value: str) -> set:
    """Sub-classify a 'characterization' row's value text by technique family.
    Returns a set since values often list multiple techniques
    (e.g. 'Raman spectroscopy, XPS')."""
    v = (value or "").lower()
    subtypes = set()
    if any(re.search(p, v) for p in _STRUCTURAL_PATTERNS):
        subtypes.add("structural")
    if any(re.search(p, v) for p in _OPTICAL_PATTERNS):
        subtypes.add("optical")
    if any(re.search(p, v) for p in _ELECTRICAL_PATTERNS):
        subtypes.add("electrical")
    if not subtypes:
        subtypes.add("other")
    return subtypes


def _chit_headers() -> dict:
    return {"X-API-Key": CHITRAGUPTA_KEY}


def _fetch_context_from_chitragupta(source_ids: list, document_type: str) -> dict | None:
    """Call Chitragupta /context/load. Returns context package or None on failure."""
    try:
        import httpx
        resp = httpx.post(
            f"{CHITRAGUPTA_BASE}/v1/context/load",
            headers=_chit_headers(),
            json={
                "workflow_ids":     source_ids,
                "document_type":    document_type,
                "max_per_category": 100,
            },
            timeout=30.0,
        )
        if resp.status_code == 200:
            return resp.json()
        print(f"[G1] Chitragupta /context/load returned {resp.status_code} — falling back to direct DB")
        return None
    except Exception as exc:
        print(f"[G1] Chitragupta unreachable: {exc} — falling back to direct DB")
        return None


def _fetch_summary_from_chitragupta(source_ids: list) -> dict | None:
    """Call Chitragupta /context/knowledge_summary. Returns dict or None."""
    try:
        import httpx
        ids_str = ",".join(str(i) for i in source_ids)
        resp = httpx.get(
            f"{CHITRAGUPTA_BASE}/v1/context/knowledge_summary",
            headers=_chit_headers(),
            params={"workflow_ids": ids_str},
            timeout=15.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Convert by_category list → dict[category, list[value]]
            summary: dict = {}
            for row in data.get("by_category", []):
                summary[row["category"]] = []
            return summary
        return None
    except Exception:
        return None


def load_context(repo, document_id: int, config: dict) -> dict:
    source_ids    = json.loads(config.get("source_ids") or "[]")
    source_type   = config.get("source_type", "shani")
    document_type = config.get("document_type", "literature_review")

    print(f"[G1] Loading context for document_id={document_id}, "
          f"source_type={source_type}, source_ids={source_ids}")

    # ── Get workflow material context (still from SHANI config) ───────────────
    material_context = _get_material_context(repo, source_ids)

    # ── Get section names for this document type ──────────────────────────────
    from ganesh.document_types.literature_review import LITERATURE_REVIEW_SECTIONS
    from ganesh.document_types.dft_report        import DFT_REPORT_SECTIONS
    from ganesh.document_types.research_report   import RESEARCH_REPORT_SECTIONS
    from ganesh.document_types.technical_summary import TECHNICAL_SUMMARY_SECTIONS
    from ganesh.document_types.manuscript_draft  import MANUSCRIPT_DRAFT_SECTIONS
    SECTION_MAP = {
        "literature_review": LITERATURE_REVIEW_SECTIONS,
        "dft_report":        DFT_REPORT_SECTIONS,
        "research_report":   RESEARCH_REPORT_SECTIONS,
        "technical_summary": TECHNICAL_SUMMARY_SECTIONS,
        "manuscript_draft":  MANUSCRIPT_DRAFT_SECTIONS,
    }
    sections      = SECTION_MAP.get(document_type, LITERATURE_REVIEW_SECTIONS)
    section_names = [s["section_name"] for s in sections]

    # ── G1: Fetch curated context from Chitragupta ────────────────────────────
    chit_package = _fetch_context_from_chitragupta(source_ids, document_type)

    if chit_package:
        print(f"[G1] Chitragupta context: {chit_package['total_papers']} papers, "
              f"{chit_package['total_knowledge']} knowledge rows")
        # Build valid_paper_ids from top_papers returned by Chitragupta
        valid_paper_ids = set(p["id"] for p in chit_package.get("top_papers", []))
        # Flat knowledge pool from Chitragupta (all categories)
        chit_knowledge: list = []
        for entries in chit_package.get("knowledge", {}).values():
            for e in entries:
                chit_knowledge.append(e)
    else:
        # Fallback: query SHANI DB directly
        print("[G1] Using direct SHANI DB fallback for paper list")
        if source_ids:
            placeholders = ",".join("?" * len(source_ids))
            valid_paper_ids = set(
                r["id"] for r in repo.fetch_all(
                    f"SELECT id FROM Paper WHERE workflow_id IN ({placeholders}) "
                    f"AND status IN ('extracted','knowledge_ready','completed')",
                    tuple(source_ids),
                )
            )
        else:
            valid_paper_ids = set()
        chit_knowledge = []

    print(f"[G1] {len(valid_paper_ids)} valid papers in target workflows")

    # ── FAISS vector search per section ───────────────────────────────────────
    section_evidence_map = {}
    total_knowledge = 0

    try:
        from services.vector_db_service import VectorDBService
        vs = VectorDBService(str(SHANI_ROOT / "database" / "vector_index.faiss"))
        vector_available = vs.index.ntotal > 0
    except Exception as exc:
        print(f"[G1] Vector search unavailable: {exc} — using fallback")
        vector_available = False

    for section_name in section_names:
        if vector_available and valid_paper_ids:
            query      = f"{material_context} {section_name}"
            raw_results = vs.search(query, top_k=50, return_scores=True)
            filtered    = [
                (pid, score) for pid, score in raw_results
                if pid in valid_paper_ids
            ][:8]
            relevant_paper_ids = [pid for pid, _ in filtered]
        else:
            relevant_paper_ids = list(valid_paper_ids)[:8]

        # Fetch knowledge rows for these papers from SHANI DB.
        # Category-aware: sections with a defined map only pull knowledge from
        # categories relevant to that section, instead of an unfiltered
        # LIMIT 30 that gets swamped by generic categories (characterization,
        # material). Unmapped sections keep the original broad behavior.
        mapped_categories    = SECTION_CATEGORY_MAP.get(section_name)
        wanted_char_subtypes = SECTION_CHARACTERIZATION_SUBTYPES.get(section_name, set())

        if relevant_paper_ids:
            placeholders = ",".join("?" * len(relevant_paper_ids))

            if mapped_categories is not None:
                query_categories = set(mapped_categories)
                if wanted_char_subtypes:
                    query_categories.add("characterization")
                query_categories = list(query_categories)
                cat_placeholders = ",".join("?" * len(query_categories))
                knowledge_rows = repo.fetch_all(
                    f"SELECT category, value, sentence as context FROM ResearchKnowledge "
                    f"WHERE paper_id IN ({placeholders}) "
                    f"AND category IN ({cat_placeholders}) LIMIT 30",
                    tuple(relevant_paper_ids) + tuple(query_categories),
                )
            else:
                knowledge_rows = repo.fetch_all(
                    f"SELECT category, value, sentence as context FROM ResearchKnowledge "
                    f"WHERE paper_id IN ({placeholders}) LIMIT 30",
                    tuple(relevant_paper_ids),
                )

            evidence = []
            for r in knowledge_rows:
                row = dict(r)
                if row["category"] == "characterization" and wanted_char_subtypes:
                    subtypes = _classify_characterization(row.get("value", ""))
                    if not (subtypes & wanted_char_subtypes):
                        continue  # wrong characterization sub-type for this section
                evidence.append(row)

            # Fetch synthesized findings (S5_5) for the same papers - higher-quality
            # narrative evidence, kept separate from raw ResearchKnowledge fragments.
            finding_rows = repo.fetch_all(
                f"SELECT finding_text as value, material, property_name, "
                f"property_value, source_sentence as context FROM ResearchFinding "
                f"WHERE paper_id IN ({placeholders}) LIMIT 20",
                tuple(relevant_paper_ids),
            )
            for r in finding_rows:
                row = dict(r)
                row["category"] = "finding"
                evidence.append(row)
        elif chit_knowledge:
            # Use Chitragupta flat pool as section fallback
            evidence = chit_knowledge[:30]
        else:
            evidence = []

        section_evidence_map[section_name] = evidence
        total_knowledge += len(evidence)
        print(f"[G1]   {section_name}: {len(evidence)} knowledge rows "
              f"from {len(relevant_paper_ids)} papers")

    # ── Knowledge summary — prefer Chitragupta, fall back to direct DB ────────
    knowledge_summary = (
        _fetch_summary_from_chitragupta(source_ids)
        or _build_knowledge_summary(repo, source_ids)
    )

    context_bundle = {
        "document_type":        document_type,
        "source_workflow_ids":  source_ids,
        "material_context":     material_context,
        "section_evidence_map": section_evidence_map,
        "knowledge_summary":    knowledge_summary,
        "dft_results":          [],
        "total_papers":         len(valid_paper_ids),
        "total_knowledge_rows": total_knowledge,
    }

    # ── Persist to GaneshContext ──────────────────────────────────────────────
    now = datetime.utcnow().isoformat()
    with repo.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO GaneshContext
                (document_id, context_type, context_ref, context_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                document_id,
                source_type,
                json.dumps(source_ids),
                json.dumps(context_bundle),
                now,
            ),
        )
        cursor.execute(
            "UPDATE GaneshDocument SET status='planning', updated_at=? WHERE id=?",
            (now, document_id),
        )

    print(f"[G1] Context loaded: {len(valid_paper_ids)} papers, "
          f"{total_knowledge} knowledge rows (section-indexed)")

    return {
        "status":          "success",
        "context_bundle":  context_bundle,
        "paper_count":     len(valid_paper_ids),
        "knowledge_count": total_knowledge,
    }


def _get_material_context(repo, workflow_ids: list) -> str:
    """Get material/focus keywords from SHANI workflow configs."""
    if not workflow_ids:
        return "materials science"
    placeholders = ",".join("?" * len(workflow_ids))
    rows = repo.fetch_all(
        f"SELECT material, focus FROM WorkflowResearchConfig "
        f"WHERE workflow_id IN ({placeholders})",
        tuple(workflow_ids),
    )
    parts = []
    for r in rows:
        if r["material"]: parts.append(r["material"])
        if r["focus"]:    parts.append(r["focus"][:100])
    return " ".join(parts)[:300] if parts else "materials science"


def _build_knowledge_summary(repo, workflow_ids: list) -> dict:
    """Fallback: top 10 values per category directly from SHANI DB."""
    if not workflow_ids:
        return {}
    placeholders = ",".join("?" * len(workflow_ids))
    rows = repo.fetch_all(
        f"""
        SELECT rk.category, rk.value, COUNT(*) as cnt
        FROM ResearchKnowledge rk
        JOIN Paper p ON p.id = rk.paper_id
        WHERE p.workflow_id IN ({placeholders})
        GROUP BY rk.category, rk.value
        ORDER BY rk.category, cnt DESC
        """,
        tuple(workflow_ids),
    )
    summary: dict = {}
    for r in rows:
        cat = r["category"]
        summary.setdefault(cat, [])
        if len(summary[cat]) < 10:
            summary[cat].append(r["value"])
    return summary
