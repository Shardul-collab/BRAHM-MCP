# ============================================================
# paper_ingestor.py
#
# ROLE IN PIPELINE:
#   Central scoring, ranking, deduplication, and insertion
#   authority for all papers retrieved by S2 (search_papers).
#
# WHAT CHANGED vs previous version:
#   This module was previously a thin wrapper around
#   paper_repo.create_paper(). It now contains a full
#   content-aware scoring engine with four domain-specific
#   scorers and a weighted final score.
#
#   ingest_search_results() now:
#     1. Scores every candidate with compute_final_score()
#     2. Sorts by score (descending)
#     3. Deduplicates by fuzzy title match
#     4. Inserts into Paper table in ranked order up to `limit`
#     5. Returns list of inserted paper IDs
#
#   Public API is BACKWARD COMPATIBLE:
#     ingest_search_results(repo, workflow_id, papers)
#   New optional parameters:
#     config=None   — WorkflowResearchConfig dict for scoring
#     limit=500     — max papers to insert
#   If config is None, scoring is skipped and all papers are
#   inserted in arrival order (safe fallback).
#
# SCORER OVERVIEW:
#   compute_material_score()        weight=0.40
#   compute_objective_score()       weight=0.25
#   compute_synthesis_score()       weight=0.20
#   compute_characterization_score()weight=0.10
#   compute_application_score()     weight=0.05
#
# ALL SCORING IS:
#   - Purely deterministic (string + pattern matching)
#   - No ML, no external calls, no probabilistic methods
#   - No new dependencies
# ============================================================

import difflib
import re

import repositories.paper_repo as paper_repo


# ============================================================
# MATERIAL KNOWLEDGE TABLES
#
# Material families group chemically related compounds.
# Keys are lowercase, no spaces, no hyphens — normalised form.
# Base elements map element symbols to their common compounds.
# ============================================================

# Family → set of compound keys (normalised)
_MATERIAL_FAMILIES: dict[str, set] = {
    "iii_vi_layered": {
        "in2se3", "in2s3", "in2te3",
        "ga2se3", "ga2s3", "ga2te3",
        "inse", "ins", "inte",
        "gase", "gas", "gate",
        "bi2se3", "bi2s3", "bi2te3",
        "sb2se3", "sb2s3", "sb2te3",
    },
    "ii_vi_chalcogenide": {
        "znse", "zns", "znte", "zno",
        "cdse", "cds", "cdte", "cdo",
        "hgse", "hgs", "hgte",
        "mgs", "mgo", "mgte",
        "beo", "bes",
    },
    "iii_v_nitride": {
        "gan", "aln", "inn", "bnn",
        "gaas", "gaaalasn",
    },
    "iii_v_arsenide": {
        "gaas", "inas", "alas", "alinas",
    },
    "iv_vi_chalcogenide": {
        "pbs", "pbse", "pbte",
        "sns", "snse", "snte", "sns2",
        "ges", "gese", "gete",
    },
    "transition_metal_dichalcogenide": {
        "mos2", "ws2", "mose2", "wse2",
        "mote2", "wte2", "res2", "nbs2",
    },
    "perovskite": {
        "mapbi3", "cspbi3", "fapbi3",
        "mapbbr3", "cspbbr3",
        "batio3", "srtio3",
    },
    "oxide_semiconductor": {
        "zno", "tio2", "sno2", "in2o3",
        "ga2o3", "al2o3", "cuo", "cu2o",
        "fe2o3", "fe3o4", "wo3", "v2o5",
    },
    "silicon_carbide": {
        "sic", "4hsic", "6hsic",
    },
}

# Base element → set of compound keys (normalised)
_BASE_ELEMENTS: dict[str, set] = {
    "zn": {"znse", "zns", "znte", "zno"},
    "cd": {"cdse", "cds", "cdte", "cdo"},
    "ga": {"gan", "gaas", "gap", "ga2o3"},
    "in": {"inp", "inas", "in2o3", "inse", "inn", "in2se3", "in2s3", "in2te3"},
    "sn": {"sno2", "sns", "sns2", "snse"},
    "pb": {"pbs", "pbse", "pbte"},
    "mo": {"mos2", "mose2", "mote2"},
    "w":  {"ws2", "wse2", "wte2"},
    "ti": {"tio2", "tin", "tic"},
    "si": {"sic", "si"},
}

# Structural keywords that cross material boundaries
_STRUCTURE_TERMS = {
    "thin film", "nanoparticle", "nanowire", "nanorod",
    "nanosheet", "quantum dot", "bulk crystal", "single crystal",
    "polycrystalline", "epitaxial", "heterostructure",
    "nanostructure", "nanotube", "core-shell",
}


# ============================================================
# SYNTHESIS KNOWLEDGE TABLES
# ============================================================

# Canonical class → set of method keywords (lowercase)
_SYNTHESIS_CLASSES: dict[str, set] = {
    "pvd": {
        "sputtering", "magnetron sputtering", "rf sputtering",
        "dc sputtering", "reactive sputtering",
        "evaporation", "thermal evaporation", "e-beam evaporation",
        "electron beam evaporation", "pulsed laser deposition",
        "pld", "laser ablation",
    },
    "cvd": {
        "chemical vapor deposition", "cvd",
        "mocvd", "metalorganic cvd", "metal organic cvd",
        "pecvd", "lpcvd", "atmospheric pressure cvd",
        "atomic layer deposition", "ald",
        "mbe", "molecular beam epitaxy",
        "hvpe", "halide vapor phase epitaxy",
    },
    "solution": {
        "sol-gel", "sol gel", "spray pyrolysis",
        "spin coating", "dip coating",
        "chemical bath deposition", "cbd",
        "silar", "successive ionic layer adsorption",
        "electrodeposition", "electrochemical deposition",
        "hydrothermal", "solvothermal",
        "co-precipitation", "coprecipitation",
        "microwave synthesis", "sonochemical",
        "hot injection", "colloidal synthesis",
    },
    "solid_state": {
        "sintering", "solid state reaction",
        "ball milling", "high energy milling",
        "calcination", "solid-state",
    },
    "epitaxial": {
        "mbe", "molecular beam epitaxy",
        "movpe", "mocvd",
        "hvpe", "epitaxial growth",
        "homoepitaxy", "heteroepitaxy",
    },
}

# Flat lookup: keyword → canonical class name
_METHOD_TO_CLASS: dict[str, str] = {}
for _cls, _kws in _SYNTHESIS_CLASSES.items():
    for _kw in _kws:
        _METHOD_TO_CLASS[_kw] = _cls


# ============================================================
# CHARACTERIZATION TECHNIQUE TABLE
# ============================================================

# Canonical technique name → list of matching keywords
_CHAR_TECHNIQUES: dict[str, list] = {
    "XRD":          ["xrd", "x-ray diffraction", "x ray diffraction"],
    "SEM":          ["sem", "scanning electron microscop"],
    "TEM":          ["tem", "transmission electron microscop", "hrtem"],
    "EDX":          ["edx", "eds", "energy dispersive", "energy-dispersive"],
    "XPS":          ["xps", "x-ray photoelectron", "x ray photoelectron"],
    "FTIR":         ["ftir", "infrared spectroscop", "ir spectroscop"],
    "Raman":        ["raman"],
    "UV-Vis":       ["uv-vis", "uv–vis", "uv vis", "optical absorbance",
                     "optical transmittance"],
    "PL":           ["photoluminescence", " pl ", "pl spectroscop"],
    "AFM":          ["afm", "atomic force microscop"],
    "Hall":         ["hall measurement", "hall effect", "hall coefficient"],
    "EIS":          ["eis", "electrochemical impedance"],
    "CV":           ["cyclic voltammetry"],
    "I-V":          ["i-v", "current-voltage", "iv characteristic"],
    "TGA":          ["tga", "thermogravimetric"],
    "BET":          ["bet", "brunauer"],
    "Cathodoluminescence": ["cathodoluminescence", " cl "],
    "DLTS":         ["dlts", "deep level transient"],
}

# Flat lookup: keyword → canonical technique name
_KEYWORD_TO_TECHNIQUE: dict[str, str] = {}
for _tech, _kws in _CHAR_TECHNIQUES.items():
    for _kw in _kws:
        _KEYWORD_TO_TECHNIQUE[_kw] = _tech


# ============================================================
# APPLICATION TERMS TABLE
# ============================================================

# Canonical application → matching keywords
_APPLICATION_TERMS: dict[str, list] = {
    "photodetector":   ["photodetector", "photodetection", "photoconductor",
                        "uv detector", "visible detector"],
    "solar_cell":      ["solar cell", "photovoltaic", "pv device"],
    "led":             ["led", "light-emitting diode", "electroluminescence"],
    "laser":           ["laser", "lasing", "laser diode"],
    "gas_sensor":      ["gas sensor", "gas sensing", "chemical sensor"],
    "transistor":      ["transistor", "fet", "mosfet", "field effect"],
    "photocatalysis":  ["photocatalysis", "photocatalytic", "dye degradation",
                        "water splitting"],
    "battery":         ["battery", "lithium", "li-ion", "sodium-ion",
                        "energy storage"],
    "thermoelectric":  ["thermoelectric", "seebeck"],
    "scintillator":    ["scintillator", "scintillation"],
    "piezoelectric":   ["piezoelectric", "piezo"],
}

# Flat lookup: keyword → canonical application
_KEYWORD_TO_APP: dict[str, str] = {}
for _app, _kws in _APPLICATION_TERMS.items():
    for _kw in _kws:
        _KEYWORD_TO_APP[_kw] = _app


# ============================================================
# SCORING FINAL WEIGHTS
# ============================================================

_WEIGHTS = {
    "material":         0.40,
    "objective":        0.25,
    "synthesis":        0.20,
    "characterization": 0.10,
    "application":      0.05,
}

MIN_SCORE = 0.12  # Papers below this threshold are dropped before insertion

MIN_SCORE = 0.12  # Papers below this threshold are dropped before insertion


# ============================================================
# HELPERS
# ============================================================

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation/spaces for key matching."""
    return re.sub(r"[\s\-_/]", "", text.lower())


_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def _paper_text(paper: dict) -> tuple[str, str, str]:
    """
    Returns (title_lower, abstract_lower, combined_lower).
    Handles both 'summary' (search pool) and 'abstract' (DB row).
    Unicode subscript digits (e.g. MoS₂) are normalised to ASCII
    so material matching works regardless of source formatting.
    """
    title    = (paper.get("title")   or "").lower().translate(_SUBSCRIPT_MAP)
    abstract = (paper.get("summary") or paper.get("abstract") or "").lower().translate(_SUBSCRIPT_MAP)
    return title, abstract, title + " " + abstract


def _keyword_hits(text: str, keywords: list[str]) -> int:
    """Count how many keywords appear in text."""
    return sum(1 for kw in keywords if kw in text)


# ============================================================
# SCORER 1 — MATERIAL SIMILARITY
#
# Hierarchy (returns 0.0–1.0):
#   1.0  exact material string found in title or abstract
#   0.8  exact in title only (title hit is stronger signal)
#   0.6  different compound from same chemical family
#   0.3  same base element (e.g. Zn-based: ZnO, ZnS, ZnSe…)
#   0.15 same structural form (thin film, nanowire…)
#   0.0  no match
# ============================================================

def compute_material_score(paper: dict, config: dict) -> float:
    # ----------------------------------------------------------------
    # config["material"] may be a comma-separated list of aliases, e.g.
    # "In2Se3, alpha-In2Se3, indium selenide"
    # We score each alias independently and return the highest hit.
    # ----------------------------------------------------------------

    material_raw = (config.get("material") or "").strip()
    if not material_raw:
        return 0.0

    # Parse all aliases
    aliases = [a.strip() for a in material_raw.split(",") if a.strip()]
    if not aliases:
        return 0.0

    title, abstract, combined = _paper_text(paper)

    best = 0.0

    for material in aliases:

        mat_lower = material.lower()
        mat_key   = _normalise(material)

        # --- Level 1: exact match ---
        if mat_lower in title:
            return 1.0          # can't do better — short-circuit
        if mat_lower in abstract:
            best = max(best, 0.8)
            continue

        # --- Level 2: same chemical family ---
        family_hit = False
        for family, members in _MATERIAL_FAMILIES.items():
            if mat_key in members:
                family_hit = True
                for sibling in members:
                    if sibling != mat_key and len(sibling) >= 3:
                        if sibling in combined.replace(" ", "").replace("-", ""):
                            best = max(best, 0.6)
                break

        if family_hit:
            continue

        # --- Level 3: same base element ---
        element_hit = False
        for element, compounds in _BASE_ELEMENTS.items():
            if mat_key in compounds:
                element_hit = True
                if f" {element}" in combined or combined.startswith(element):
                    best = max(best, 0.3)
                break

        if element_hit:
            continue

        # --- Level 4: same structural form ---
        structure = (config.get("structure") or "").lower()
        if structure:
            for stype in _STRUCTURE_TERMS:
                if stype in structure and stype in combined:
                    best = max(best, 0.15)
                    break

    return best


# ============================================================
# SCORER 2 — SYNTHESIS METHOD MATCHING
#
# Scoring (returns 0.0–1.0):
#   1.0  exact method keyword found in paper text
#   0.6  different method from same synthesis class (PVD, CVD…)
#   0.0  no match
# ============================================================

def compute_synthesis_score(paper: dict, config: dict) -> float:

    method_config = (config.get("method") or "").lower()
    if not method_config:
        return 0.0

    _, _, combined = _paper_text(paper)

    config_methods  = [m.strip() for m in method_config.split(",") if m.strip()]
    config_classes  = set()

    for cm in config_methods:
        # --- exact keyword hit ---
        for keyword in _METHOD_TO_CLASS:
            if keyword in cm or keyword in combined:
                if keyword in combined and keyword in cm:
                    return 1.0  # both config and paper mention same term
                if keyword in cm:
                    config_classes.add(_METHOD_TO_CLASS[keyword])

        # --- any config keyword appears verbatim in paper ---
        if cm in combined:
            return 1.0

    # --- same synthesis class ---
    paper_classes = set()
    for keyword, cls in _METHOD_TO_CLASS.items():
        if keyword in combined:
            paper_classes.add(cls)

    if config_classes & paper_classes:  # non-empty intersection
        return 0.6

    return 0.0


# ============================================================
# SCORER 3 — RESEARCH OBJECTIVE (focus + properties)
#
# Uses token overlap between config focus/properties and
# paper title + abstract. Returns 0.0–1.0.
#
# Scoring:
#   Each focus/property keyword found in title  → +0.15 (cap 3)
#   Each focus/property keyword found in abstract → +0.08 (cap 5)
#   Raw sum normalised to [0, 1].
# ============================================================

def compute_objective_score(paper: dict, config: dict) -> float:

    focus_raw      = config.get("focus")      or ""
    properties_raw = config.get("properties") or ""

    objective_terms = []

    for field in (focus_raw, properties_raw):
        for term in field.split(","):
            t = term.strip().lower()
            if t and len(t) > 2:
                objective_terms.append(t)

    if not objective_terms:
        return 0.0

    title, abstract, _ = _paper_text(paper)

    raw = 0.0
    title_hits    = 0
    abstract_hits = 0

    for term in objective_terms:
        if term in title and title_hits < 3:
            raw += 0.15
            title_hits += 1
        elif term in abstract and abstract_hits < 5:
            raw += 0.08
            abstract_hits += 1

    # cap at 1.0
    return min(raw, 1.0)


# ============================================================
# SCORER 4 — CHARACTERIZATION TECHNIQUE MATCHING
#
# Returns 0.0–1.0.
# Each config technique that also appears in the paper → +score.
# Bonus if paper uses ≥3 relevant techniques (thorough study).
# ============================================================

def compute_characterization_score(paper: dict, config: dict) -> float:

    char_config = (config.get("characterization") or "").lower()
    if not char_config:
        # If config has no characterization, still reward papers
        # that use multiple techniques (quality signal).
        _, _, combined = _paper_text(paper)
        hits = sum(
            1 for kw in _KEYWORD_TO_TECHNIQUE if kw in combined
        )
        return min(hits * 0.1, 0.4)

    _, _, combined = _paper_text(paper)

    config_techs = {t.strip().lower() for t in char_config.split(",")}
    score        = 0.0
    matched      = 0

    for keyword, canonical in _KEYWORD_TO_TECHNIQUE.items():
        # Config requests this technique AND paper mentions it
        if keyword in combined:
            canon_lower = canonical.lower()
            for ct in config_techs:
                if ct in canon_lower or canon_lower in ct or ct == keyword:
                    score  += 0.25
                    matched += 1
                    break

    # Bonus: paper uses ≥3 distinct techniques (comprehensive study)
    total_in_paper = len({
        v for k, v in _KEYWORD_TO_TECHNIQUE.items() if k in combined
    })
    if total_in_paper >= 3:
        score += 0.2

    return min(score, 1.0)


# ============================================================
# SCORER 5 — APPLICATION MATCHING
#
# Compares config focus/properties against known application
# vocabularies. Returns 0.0–1.0.
# ============================================================

def compute_application_score(paper: dict, config: dict) -> float:

    focus_raw      = (config.get("focus")      or "").lower()
    properties_raw = (config.get("properties") or "").lower()
    config_text    = focus_raw + " " + properties_raw

    _, _, combined = _paper_text(paper)

    # Collect which applications the config is targeting
    config_apps = set()
    for keyword, app in _KEYWORD_TO_APP.items():
        if keyword in config_text:
            config_apps.add(app)

    if not config_apps:
        return 0.0

    # Check which of those appear in the paper
    paper_apps = set()
    for keyword, app in _KEYWORD_TO_APP.items():
        if keyword in combined:
            paper_apps.add(app)

    overlap = config_apps & paper_apps
    if not overlap:
        return 0.0

    return min(len(overlap) * 0.4, 1.0)


# ============================================================
# FINAL SCORE COMBINER
#
# S = 0.40 * material
#   + 0.25 * objective
#   + 0.20 * synthesis
#   + 0.10 * characterization
#   + 0.05 * application
#
# Returns (total: float, breakdown: dict) for logging.
# ============================================================

def compute_final_score(paper: dict, config: dict) -> tuple[float, dict]:

    breakdown = {
        "material":         compute_material_score(paper, config),
        "objective":        compute_objective_score(paper, config),
        "synthesis":        compute_synthesis_score(paper, config),
        "characterization": compute_characterization_score(paper, config),
        "application":      compute_application_score(paper, config),
    }

    total = sum(
        breakdown[k] * _WEIGHTS[k]
        for k in _WEIGHTS
    )

    return round(total, 4), breakdown


# ============================================================
# DUPLICATE DETECTION
# (unchanged from search_papers.py — moved here)
# ============================================================

def _is_duplicate(title: str, seen_titles: set) -> bool:
    for t in seen_titles:
        if difflib.SequenceMatcher(None, title.lower(), t.lower()).ratio() > 0.9:
            return True
    return False


# ============================================================
# MAIN ENTRY POINT
#
# ingest_search_results(repo, workflow_id, papers,
#                       config=None, limit=500)
#
# Papers is a list of dicts with at minimum 'title' and
# optionally 'source', 'pdf_url', 'summary'/'abstract',
# 'doi', 'year'.
#
# When config is provided, papers are scored, sorted by
# final score, deduplicated, and inserted in ranked order.
#
# When config is None (safe fallback), papers are inserted
# in arrival order without scoring.
#
# Returns list[int] of inserted paper IDs.
# ============================================================

def ingest_search_results(
    repo,
    workflow_id: int,
    papers: list,
    config: dict = None,
    limit: int = 500
) -> list:
    """
    Score, rank, deduplicate, and insert papers into the DB.

    Args:
        repo:        Repository instance
        workflow_id: parent workflow ID
        papers:      raw candidate list from search_papers
        config:      WorkflowResearchConfig dict (for scoring)
        limit:       max papers to insert

    Returns:
        list[int]: inserted paper IDs in score-descending order
    """

    if not papers:
        return []

    # --------------------------------------------------
    # SCORE every candidate (skip if no config)
    # --------------------------------------------------
    scored: list[tuple[dict, float, dict]] = []

    if config:
        for p in papers:
            if not p.get("title"):
                continue
            total, breakdown = compute_final_score(p, config)
            scored.append((p, total, breakdown))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = [(p, s, b) for p, s, b in scored if s >= MIN_SCORE]

        # Log score distribution
        high  = sum(1 for _, s, _ in scored if s >= 0.4)
        mid   = sum(1 for _, s, _ in scored if 0.15 <= s < 0.4)
        low   = sum(1 for _, s, _ in scored if s < 0.15)
        print(
            f"[Ingestor] Score distribution: "
            f"{high} high (≥0.4) | {mid} mid | {low} low"
        )
        if scored:
            top = scored[0]
            print(
                f"[Ingestor] Top paper: '{top[0].get('title', '')[:60]}' "
                f"score={top[1]} breakdown={top[2]}"
            )
    else:
        # No config — use arrival order, score=0 for all
        print("[Ingestor] No config supplied — inserting in arrival order.")
        for p in papers:
            if p.get("title"):
                scored.append((p, 0.0, {}))

    # --------------------------------------------------
    # DEDUPLICATE + INSERT in score-descending order
    # --------------------------------------------------
    seen_titles  = set()
    inserted_ids = []

    for p, score, breakdown in scored:

        title = (p.get("title") or "").strip()
        if not title:
            continue

        if _is_duplicate(title, seen_titles):
            continue

        seen_titles.add(title)

        # Skip if already in DB for this workflow
        existing = repo.fetch_one(
            "SELECT id FROM Paper WHERE workflow_id = ? AND title = ?",
            (workflow_id, title)
        )
        if existing:
            continue

        abstract = (p.get("summary") or p.get("abstract") or "").strip() or None

        paper_id = paper_repo.create_paper(
            repo=repo,
            workflow_id=workflow_id,
            title=title,
            source=p.get("source", "unknown"),
            pdf_url=p.get("pdf_url"),
            status="pending",
            abstract=abstract,
            year=p.get("year")
        )

        if not paper_id:
            continue

        # Store DOI if available
        doi = p.get("doi")
        if doi:
            with repo.transaction() as cursor:
                cursor.execute(
                    "UPDATE Paper SET doi = ? WHERE id = ?",
                    (doi, paper_id)
                )

        inserted_ids.append(paper_id)

        if len(inserted_ids) >= limit:
            break

    print(
        f"[Ingestor] Inserted {len(inserted_ids)} papers "
        f"(dedup removed {len(seen_titles) - len(inserted_ids)} duplicates)"
    )

    return inserted_ids
