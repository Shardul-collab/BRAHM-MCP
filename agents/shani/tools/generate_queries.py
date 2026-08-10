# ============================================================
# generate_queries.py  — S1
#
# CHANGES vs previous version:
#
# 1. INTENT-SPECIFIC QUERY EXPANSION
#    Added _build_intent_queries() which, when the material is
#    a recognised II-VI / wide-bandgap semiconductor (e.g. ZnSe,
#    ZnO, CdS, GaN), generates a fixed bank of phenomenon-targeted
#    queries covering:
#      - Se/S/O vacancy defects
#      - doping concentration effects
#      - annealing temperature vs optical/electrical properties
#      - carrier lifetime and recombination
#      - UV photocurrent / photodetector optimisation
#    These queries are appended to the config-driven set.
#
# 2. QUERY COUNT
#    Hard cap raised from 7 to 12 so the intent queries reach S2.
#
# 3. SCORING UPDATE
#    Scoring now also rewards "defect", "doping", "annealing",
#    "carrier" and "photocurrent" tokens — the new intent signals.
#
# Everything else (config parsing, deduplication, stopwords) is
# unchanged for full pipeline compatibility.
# ============================================================


# ============================================================
# INTENT QUERY BANK
#
# Key: lowercase material alias (partial match ok)
# Value: list of intent-targeted query strings
#
# Add new material entries here as the pipeline is reused for
# other systems.  Queries should target PHENOMENA, not just
# the material name — this is what differentiates intent-aware
# retrieval from generic keyword search.
# ============================================================

_INTENT_QUERIES: dict = {
    "in2se3": [
        "alpha-In2Se3 ferroelectric polarization switching",
        "In2Se3 FeFET field effect transistor nonvolatile memory",
        "In2Se3 phase transition alpha beta CVD MBE",
        "In2Se3 quintuple layer van der Waals ferroelectricity",
        "In2Se3 piezoresponse force microscopy PFM SHG",
        "alpha-In2Se3 neuromorphic synaptic device",
        "In2Se3 out-of-plane in-plane polarization dipole locking",
    ],
    "znse": [
        "ZnSe selenium vacancy point defect concentration",
        "ZnSe doping Al Cl In concentration carrier density",
        "ZnSe annealing temperature optical electrical properties",
        "ZnSe carrier lifetime recombination photoluminescence",
        "ZnSe UV photocurrent photodetector responsivity",
        "ZnSe bandgap stoichiometry Se-rich Zn-rich",
        "ZnSe trap state density defect passivation",
    ],
    "zno": [
        "ZnO oxygen vacancy defect density concentration",
        "ZnO doping Al Ga In concentration conductivity",
        "ZnO annealing atmosphere temperature resistivity",
        "ZnO carrier lifetime UV emission recombination",
        "ZnO UV photodetector photocurrent responsivity",
    ],
    "cds": [
        "CdS sulfur vacancy defect photoluminescence",
        "CdS doping In Cl annealing temperature",
        "CdS carrier lifetime trap density recombination",
    ],
    "gan": [
        "GaN nitrogen vacancy defect concentration",
        "GaN Mg Si doping concentration carrier density",
        "GaN annealing temperature HVPE MOCVD defect",
        "GaN carrier lifetime recombination deep UV",
    ],
}

# Tokens that boost query score beyond config fields
_INTENT_SCORE_TOKENS = {
    "defect", "vacancy", "doping", "dopant", "annealing",
    "carrier", "lifetime", "recombination", "photocurrent",
    "responsivity", "trap", "stoichiometry", "passivation",
    "bandgap", "photoluminescence",
}

STOPWORDS = {"of", "and", "in", "on", "for", "with", "can", "the", "a", "an"}


# ============================================================
# INTENT QUERY LOOKUP
# ============================================================

def _build_intent_queries(material: str) -> list:
    """
    Returns intent-specific queries for recognised semiconductors.
    Matches by substring — so "n-type znse thin film" still returns
    ZnSe-specific queries.  Returns [] for unknown materials.
    """
    mat_lower = material.lower().replace("-", "").replace(" ", "")
    for key, queries in _INTENT_QUERIES.items():
        if key in mat_lower:
            return list(queries)
    return []


# ============================================================
# MAIN TOOL — S1
# ============================================================

def generate_queries(repo, workflow_id, execution_attempt_id=None, **kwargs):

    # --------------------------------------------------
    # FETCH WORKFLOW TITLE (fallback)
    # --------------------------------------------------

    workflow = repo.fetch_one(
        "SELECT name FROM Workflow WHERE id = ?",
        (workflow_id,)
    )

    topic = workflow["name"] if workflow and workflow["name"] else "research topic"

    # --------------------------------------------------
    # FETCH CONFIG
    # --------------------------------------------------

    config = repo.fetch_one(
        """
        SELECT
            material,
            structure,
            focus,
            method,
            properties,
            characterization
        FROM WorkflowResearchConfig
        WHERE workflow_id = ?
        """,
        (workflow_id,)
    )

    # --------------------------------------------------
    # BASIC FALLBACK (HARD SAFETY)
    # --------------------------------------------------

    if not config or not config["material"]:

        material = topic

        queries = [
            f"{material}",
            f"{material} research",
            f"{material} review"
        ]

        return {
            "status": "success",
            "data":   queries,
            "error":  None
        }

    # --------------------------------------------------
    # PRIMARY
    # --------------------------------------------------

    material = config["material"].strip()
    # Parse aliases — use first one as the primary query token.
    # The full comma-separated string is NOT a valid search query.
    _aliases = [a.strip() for a in material.split(",") if a.strip()]
    primary  = _aliases[0].lower() if _aliases else material.lower()

    # --------------------------------------------------
    # HELPER
    # --------------------------------------------------

    def parse_keywords(value):
        if not value or value == "ALL":
            return []
        return [
            v.strip().lower()
            for v in value.split(",")
            if v.strip().lower() not in STOPWORDS
        ]

    # --------------------------------------------------
    # INTENT EXTRACTION
    # --------------------------------------------------

    focus          = parse_keywords(config["focus"])[:3]
    structure      = parse_keywords(config["structure"])
    method         = parse_keywords(config["method"])
    properties     = parse_keywords(config["properties"])
    characterization = parse_keywords(config["characterization"])

    # --------------------------------------------------
    # DETERMINISTIC CONFIG-DRIVEN QUERY BUILDER
    # (unchanged logic from previous version)
    # --------------------------------------------------

    queries = set()

    def safe_add(q):
        q = q.strip().lower()
        tokens = q.split()
        if len(tokens) != len(set(tokens)):
            return
        if q:
            queries.add(q)

    # 1. base
    safe_add(primary)

    # 2. focus-driven
    for f in focus:
        safe_add(f"{primary} {f}")

    # 3. structure
    for s in structure:
        safe_add(f"{primary} {s}")

    # 4. method
    for m in method:
        safe_add(f"{primary} {m}")

    # 5. characterization
    for c in characterization:
        safe_add(f"{primary} {c}")

    # 6. focus + method
    for f in focus:
        for m in method:
            safe_add(f"{primary} {f} {m}")

    # 7. focus + characterization
    for f in focus:
        for c in characterization:
            safe_add(f"{primary} {f} {c}")

    queries = list(queries)

    # --------------------------------------------------
    # SCORING
    # CHANGE: intent score tokens added alongside config fields.
    # --------------------------------------------------

    def score_query(q):
        score = 0

        for f in focus:
            if f in q:
                score += 3

        for m in method:
            if m in q:
                score += 2

        for c in characterization:
            if c in q:
                score += 2

        # NEW: reward intent-signal tokens
        for tok in _INTENT_SCORE_TOKENS:
            if tok in q:
                score += 2

        score -= 0.1 * len(q.split())

        return score

    queries.sort(key=score_query, reverse=True)

    # --------------------------------------------------
    # NEW: APPEND INTENT-SPECIFIC QUERIES
    #
    # These are phenomenon-targeted queries that go beyond
    # config vocabulary.  They are appended after scoring so
    # they always appear in the final list regardless of how
    # the config is filled in.
    # --------------------------------------------------

    intent_queries = _build_intent_queries(material)

    if intent_queries:
        print(
            f"[S1] Intent queries for '{primary}': {len(intent_queries)}"
        )
    else:
        print(
            f"[S1] No intent query bank found for '{primary}' "
            f"— using config-only queries."
        )

    # --------------------------------------------------
    # FINAL MERGE + LIMIT
    #
    # CHANGE: cap raised to 12 (was 7) to accommodate intent
    # queries without displacing config-driven ones.
    # --------------------------------------------------

    # Deduplicate: intent queries are case-mixed; normalise for check
    existing_lower = {q.lower() for q in queries}
    for iq in intent_queries:
        if iq.lower() not in existing_lower:
            queries.append(iq)
            existing_lower.add(iq.lower())

    final_queries = queries[:12]

    if len(final_queries) < 3:
        final_queries = list(set(final_queries + [
            primary,
            f"{primary} research",
            f"{primary} review"
        ]))

    print(f"[S1] Final queries ({len(final_queries)}):")
    for q in final_queries:
        print(f"  · {q}")

    return {
        "status": "success",
        "data":   final_queries,
        "error":  None
    }
