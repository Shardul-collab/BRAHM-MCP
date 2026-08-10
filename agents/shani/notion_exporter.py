"""
SHANI -> Notion Exporter (v2 — relevance-filtered, 300 papers)
"""

import sys
import time
import re
import logging
from pathlib import Path

CHITRAGUPTA_PATH = Path("/mnt/d/chitragupta")
sys.path.insert(0, str(CHITRAGUPTA_PATH))

from dotenv import load_dotenv
load_dotenv(CHITRAGUPTA_PATH / ".env")

from notion.notion_client import create_database, create_page, NotionAPIError
from notion.schema_manager import (
    create_schema, schema_to_notion_properties,
    update_notion_id, SchemaAlreadyExistsError, load_schema
)
from config.settings import NOTION_PAGE_ID

sys.path.insert(0, str(Path(__file__).parent))
from repositories.repository import Repository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shani.notion_exporter")

DB_NAME        = "ZnSe Research Knowledge Base"
RATE_LIMIT_SEC = 0.38
ABSTRACT_LIMIT = 800
PREVIEW_LIMIT  = 600
TOP_N          = 20   # top papers per workflow theme

SCHEMA_FIELDS = [
    {"name": "Title",          "type": "title"},
    {"name": "Year",           "type": "number", "format": "number"},
    {"name": "DOI",            "type": "url"},
    {"name": "PDF URL",        "type": "url"},
    {"name": "Source",         "type": "select",
     "options": ["arxiv", "openalex", "semantic_scholar", "local", "other"]},
    {"name": "Workflow Theme", "type": "select", "options": [
        "ZnSe Fundamentals", "ZnO Fundamentals", "ZnSe vs ZnO Comparative",
        "ZnSeO Alloy Formation", "Thermodynamics & Phase Stability",
        "Oxygen Incorporation", "Defect Chemistry", "Bandgap Engineering",
        "Optical Properties", "Charge Transport", "Thin Film Synthesis",
        "Post-Deposition Treatments", "Characterization Techniques",
        "Applications", "Challenges & Research Gaps",
    ]},
    {"name": "Paper Status",         "type": "select",
     "options": ["extracted", "knowledge_ready", "completed"]},
    {"name": "Relevance Score",      "type": "number", "format": "number"},
    {"name": "Sections Found",       "type": "rich_text"},
    {"name": "Abstract",             "type": "rich_text"},
    {"name": "Synthesis Methods",    "type": "rich_text"},
    {"name": "Characterization",     "type": "rich_text"},
    {"name": "Key Properties",       "type": "rich_text"},
    {"name": "Defects & Mechanisms", "type": "rich_text"},
    {"name": "Applications",         "type": "rich_text"},
    {"name": "Content Preview",      "type": "rich_text"},
]

SECTION_MAP = {
    "abstract": "abstract", "abstract:": "abstract", "abstract.": "abstract",
    "synthesis": "synthesis", "methodology": "synthesis", "methods": "synthesis",
    "experimental": "synthesis", "materials": "synthesis", "method": "synthesis",
    "synthetic": "synthesis",
    "characterization": "characterization", "characterization.": "characterization",
    "characterization,": "characterization", "technique": "characterization",
    "structural": "characterization", "optical": "characterization",
    "spectroscopy": "characterization",
    "results": "results", "result": "results", "property": "results",
    "discussion": "discussion", "discussions": "discussion",
    "discussion.": "discussion", "discussion:": "discussion",
    "applications": "applications", "applications.": "applications",
    "application": "applications", "application.": "applications",
    "conclusion": "conclusion", "conclusions": "conclusion",
    "conclusion.": "conclusion", "conclusion:": "conclusion", "summary": "conclusion",
    "introduction": "introduction", "introduction.": "introduction",
    "introduction:": "introduction", "review": "introduction",
}

SYNTHESIS_KW        = ["deposition","sputtering","mbe","mocvd","cvd","spray pyrolysis",
                        "sol-gel","pld","evaporation","substrate","annealing","growth rate",
                        "thickness","precursor","flow rate","rf power","oxygen partial"]
CHARACTERIZATION_KW = ["xrd","x-ray","sem","tem","afm","edx","eds","xps","ftir",
                        "raman","uv-vis","photoluminescence","pl ","hall effect",
                        "ellipsometry","sims","rheed","diffraction"]
PROPERTIES_KW       = ["bandgap","band gap","ev","carrier","mobility","conductivity",
                        "resistivity","transmittance","refractive index","absorption",
                        "lattice","grain size","crystallite","urbach","optical gap"]
DEFECT_KW           = ["defect","vacancy","interstitial","trap","deep level","dlts",
                        "recombination","substitution","antisite","doping","donor",
                        "acceptor","compensat","native defect","oxygen vacancy",
                        "selenium vacancy","passivation"]
APPLICATION_KW      = ["led","laser","solar cell","photovoltaic","gas sensor","detector",
                        "photodetector","optoelectronic","transistor","diode",
                        "electroluminescence","sensor","photocatalyst","photonic"]

# Material relevance keywords — weighted
MATERIAL_KW_HIGH = ["znse", "znse1", "znse₁", "znseo", "zinc selenide"]
MATERIAL_KW_MED  = ["zno", "zinc oxide", "ii-vi", "ii vi", "chalcogenide",
                     "selenide", "semiconductor thin film"]


# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_relevance(paper: dict, rk_count: int, content_count: int) -> float:
    """
    Score = material_score*0.40 + knowledge_density*0.40 + content_completeness*0.20
    All components normalized to 0-1 range.
    Returns float 0.0 - 1.0
    """
    # 1. Material relevance (0-1)
    haystack = (
        (paper.get("title") or "") + " " +
        (paper.get("abstract") or "")
    ).lower()

    mat_score = 0.0
    for kw in MATERIAL_KW_HIGH:
        if kw in haystack:
            mat_score = 1.0
            break
    if mat_score == 0.0:
        for kw in MATERIAL_KW_MED:
            if kw in haystack:
                mat_score = 0.5
                break

    # 2. Knowledge density (0-1), normalized: 10+ rows = max
    kd_score = min(rk_count / 10.0, 1.0)

    # 3. Content completeness (0-1), normalized: 6+ sections = max
    cc_score = min(content_count / 6.0, 1.0)

    return round(mat_score * 0.40 + kd_score * 0.40 + cc_score * 0.20, 4)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str, limit: int = 0) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "..."
    return text

def _extract_sentences(text: str, keywords: list, max_chars: int = 800) -> str:
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    matched, total = [], 0
    for s in sentences:
        if any(kw in s.lower() for kw in keywords):
            c = _clean(s)
            if c and len(c) > 20:
                matched.append(c)
                total += len(c)
                if total >= max_chars:
                    break
    return " ".join(matched)[:max_chars]

def _extract_year(text: str):
    m = re.findall(r"\b(19[89]\d|20[012]\d)\b", text or "")
    return int(m[0]) if m else None

def _map_theme(name: str) -> str:
    n = name.lower()
    if "fundamental" in n and "zno" in n: return "ZnO Fundamentals"
    if "fundamental" in n:                return "ZnSe Fundamentals"
    if "comparative" in n or " vs " in n: return "ZnSe vs ZnO Comparative"
    if "alloy" in n or "formation" in n:  return "ZnSeO Alloy Formation"
    if "thermodynamic" in n or "phase" in n: return "Thermodynamics & Phase Stability"
    if "oxygen incorpor" in n:            return "Oxygen Incorporation"
    if "defect" in n:                     return "Defect Chemistry"
    if "bandgap" in n or "band gap" in n: return "Bandgap Engineering"
    if "optical" in n:                    return "Optical Properties"
    if "transport" in n or "carrier" in n: return "Charge Transport"
    if "synthesis" in n or "thin film" in n: return "Thin Film Synthesis"
    if "anneal" in n or "doping" in n or "post" in n: return "Post-Deposition Treatments"
    if "characterization" in n:           return "Characterization Techniques"
    if "application" in n:                return "Applications"
    if "challenge" in n or "gap" in n:    return "Challenges & Research Gaps"
    return "ZnSe Fundamentals"

def _rtext(v: str) -> dict:
    return {"rich_text": [{"type": "text", "text": {"content": v[:2000]}}]}

def _select(v: str) -> dict:
    return {"select": {"name": v[:100]}}

def _url(v: str) -> dict:
    return {"url": v.strip()[:2000]} if v and v.strip().startswith("http") else {"url": None}

def _number(v) -> dict:
    return {"number": v}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING + RANKING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_rank_papers(repo: Repository) -> list:
    """
    Load all extracted papers, score each, return top TOP_N per workflow.
    """
    # Load all extracted papers with workflow name
    rows = repo.fetch_all("""
        SELECT p.id, p.title, p.source, p.pdf_url, p.abstract,
               p.doi, p.status, p.created_at, p.workflow_id,
               w.name as workflow_name
        FROM Paper p
        JOIN Workflow w ON p.workflow_id = w.id
        WHERE p.status = 'extracted'
          AND EXISTS (SELECT 1 FROM PaperContent pc WHERE pc.paper_id = p.id)
        ORDER BY p.workflow_id ASC, p.id ASC
    """)

    # Load ResearchKnowledge counts per paper
    rk_rows = repo.fetch_all(
        "SELECT paper_id, COUNT(*) as cnt FROM ResearchKnowledge GROUP BY paper_id"
    )
    rk_map = {r["paper_id"]: r["cnt"] for r in rk_rows}

    # Load PaperContent section counts per paper
    pc_rows = repo.fetch_all(
        "SELECT paper_id, COUNT(*) as cnt FROM PaperContent GROUP BY paper_id"
    )
    pc_map = {r["paper_id"]: r["cnt"] for r in pc_rows}

    log.info("Scoring %d candidate papers...", len(rows))

    # Score all papers
    scored = []
    for row in rows:
        p = dict(row)
        rk_count  = rk_map.get(p["id"], 0)
        pc_count  = pc_map.get(p["id"], 0)
        p["relevance_score"] = compute_relevance(p, rk_count, pc_count)
        scored.append(p)

    # Group by workflow_id, take top TOP_N per group by score
    from collections import defaultdict
    by_workflow = defaultdict(list)
    for p in scored:
        by_workflow[p["workflow_id"]].append(p)

    selected = []
    for wf_id, papers in by_workflow.items():
        top = sorted(papers, key=lambda x: x["relevance_score"], reverse=True)[:TOP_N]
        selected.extend(top)
        log.info(
            "Workflow %d: %d candidates → top %d selected (score range: %.3f – %.3f)",
            wf_id, len(papers), len(top),
            top[-1]["relevance_score"] if top else 0,
            top[0]["relevance_score"] if top else 0,
        )

    log.info("Total selected: %d papers across %d workflows",
             len(selected), len(by_workflow))

    # Load PaperContent sections for selected papers only
    for p in selected:
        content_rows = repo.fetch_all(
            "SELECT section_name, content FROM PaperContent WHERE paper_id = ?",
            (p["id"],)
        )
        sections = {}
        for cr in content_rows:
            canonical = SECTION_MAP.get(cr["section_name"].lower().strip())
            if canonical:
                sections.setdefault(canonical, [])
                sections[canonical].append(cr["content"] or "")
        p["sections"] = sections
        p["raw_section_names"] = [cr["section_name"] for cr in content_rows]

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# ROW BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_row(paper: dict) -> dict:
    sections = paper.get("sections", {})

    def _get(*keys):
        parts = []
        for k in keys:
            parts.extend(sections.get(k, []))
        return " ".join(parts)

    abstract_raw    = paper.get("abstract") or _get("abstract")
    abstract        = _clean(abstract_raw, ABSTRACT_LIMIT)
    year            = _extract_year(paper.get("created_at","")) or _extract_year(abstract_raw)
    synthesis       = _extract_sentences(_get("synthesis","results"), SYNTHESIS_KW)
    characterization= _extract_sentences(_get("characterization","methodology","results"), CHARACTERIZATION_KW)
    key_properties  = _extract_sentences(_get("results","discussion","conclusion"), PROPERTIES_KW)
    defects         = _extract_sentences(_get("results","discussion","methodology"), DEFECT_KW, 600)
    applications    = _extract_sentences(_get("applications","conclusion","introduction"), APPLICATION_KW, 600)
    preview_src     = _get("results") or _get("discussion") or _get("conclusion") or abstract_raw or ""
    preview         = _clean(preview_src, PREVIEW_LIMIT)
    sections_str    = ", ".join(sorted(set(paper.get("raw_section_names", [])))[:20])
    theme           = _map_theme(paper.get("workflow_name",""))
    doi_raw         = paper.get("doi") or ""
    doi_url         = f"https://doi.org/{doi_raw.strip()}" if doi_raw.strip() else None
    score           = paper.get("relevance_score", 0.0)

    props = {
        "Title":               {"title": [{"type": "text", "text": {"content": paper["title"][:500]}}]},
        "Workflow Theme":      _select(theme),
        "Paper Status":        _select(paper.get("status","extracted")),
        "Source":              _select(paper.get("source","other")),
        "PDF URL":             _url(paper.get("pdf_url") or ""),
        "DOI":                 _url(doi_url or ""),
        "Relevance Score":     _number(round(score, 4)),
        "Sections Found":      _rtext(sections_str),
        "Abstract":            _rtext(abstract),
        "Synthesis Methods":   _rtext(synthesis),
        "Characterization":    _rtext(characterization),
        "Key Properties":      _rtext(key_properties),
        "Defects & Mechanisms":_rtext(defects),
        "Applications":        _rtext(applications),
        "Content Preview":     _rtext(preview),
    }
    if year:
        props["Year"] = _number(year)
    return props


# ─────────────────────────────────────────────────────────────────────────────
# NOTION SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_database() -> str:
    try:
        schema = create_schema(DB_NAME, SCHEMA_FIELDS)
        log.info("Schema created locally.")
    except SchemaAlreadyExistsError:
        log.info("Schema exists — reusing.")
        schema = load_schema(DB_NAME)

    existing_id = schema.get("notion_database_id","").strip()
    if existing_id:
        log.info("Reusing existing Notion database | id=%s", existing_id)
        return existing_id

    notion_props = schema_to_notion_properties(schema)
    result = create_database(NOTION_PAGE_ID, DB_NAME, notion_props)
    db_id  = result["id"]
    update_notion_id(DB_NAME, db_id)
    log.info("Notion database created | id=%s", db_id)
    return db_id


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_export():
    log.info("=" * 60)
    log.info("SHANI -> Notion Exporter v2 (top %d per workflow)", TOP_N)
    log.info("=" * 60)

    repo = Repository()
    try:
        papers = load_and_rank_papers(repo)
    finally:
        repo.close()

    if not papers:
        log.warning("No papers selected. Exiting.")
        return

    log.info("Selected %d papers for export.", len(papers))

    db_id = setup_database()
    log.info("Target DB: https://notion.so/%s", db_id.replace("-",""))
    log.info("Estimated time: ~%.0f minutes", (len(papers) * RATE_LIMIT_SEC) / 60)

    success, failed, skipped = 0, 0, 0

    for i, paper in enumerate(papers, 1):
        try:
            props = build_row(paper)
            create_page(db_id, props)
            success += 1
            if i % 10 == 0 or i == len(papers):
                log.info("Progress: %d/%d | OK=%d ERR=%d SKIP=%d",
                         i, len(papers), success, failed, skipped)
        except NotionAPIError as e:
            log.error("Notion error paper %d '%s': %s",
                      paper["id"], paper["title"][:50], e)
            failed += 1
            if e.notion_status == 429:
                log.warning("Rate limited — sleeping 5s")
                time.sleep(5)
        except Exception as e:
            log.error("Error paper %d: %s", paper["id"], e)
            skipped += 1
        time.sleep(RATE_LIMIT_SEC)

    log.info("=" * 60)
    log.info("Done. Pushed=%d  Failed=%d  Skipped=%d / %d total",
             success, failed, skipped, len(papers))
    log.info("Notion: https://notion.so/%s", db_id.replace("-",""))
    log.info("=" * 60)


if __name__ == "__main__":
    run_export()
