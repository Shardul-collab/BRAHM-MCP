import re
import spacy

import repositories.paper_repo as paper_repo
import repositories.research_knowledge_repo as rk_repo
import repositories.failure_repo as failure_repo

from services.llm_service import LLMService, OllamaClient, GeminiClient, CerebrasClient, GroqClient


# ============================================================
# EXTRACT LIGHTWEIGHT KNOWLEDGE — S2_75
#
# Pipeline position: between S2 (search_papers) and
#                    S2_5 (resolve_pdf)
#
# Purpose:
#   Extract structured research knowledge from paper title
#   and abstract before any PDF is downloaded. Papers that
#   never get a PDF (paywalled, dead links) still contribute
#   knowledge to S6 section generation.
#
# Input:
#   Paper rows with abstract IS NOT NULL and no existing
#   ResearchKnowledge entries (idempotent — safe to re-run).
#
# Output:
#   ResearchKnowledge rows with:
#     source_type = 'abstract'
#     confidence  = 'low' (rule-based) or 'medium' (LLM)
#
# Design constraints:
#   - Lightweight: short prompts, low token budget per call
#   - No rejection: all papers with abstracts are processed
#   - No overwrite: skips papers that already have knowledge
#   - Deterministic: rule extraction runs first, LLM second
#   - Follows standard tool interface: tool(repo, workflow_id)
# ============================================================

# NLP model — loaded once at module level
nlp = spacy.load("en_core_web_sm")

# Maximum words from abstract sent to LLM.
# Title + first 120 words of abstract ≈ 150-200 tokens.
# This keeps each LLM call well within Mistral 7B limits.
MAX_ABSTRACT_WORDS = 120

# Max value length — real entity names are short
MAX_VALUE_LENGTH = 60


# ============================================================
# RULE-BASED EXTRACTION DICTIONARIES
#
# Keyed by category. Each entry maps a keyword/phrase that
# must appear in the title or abstract (case-insensitive) to
# a canonical value to store in ResearchKnowledge.
#
# These cover the most common patterns for materials science
# and nanomaterials research. They are intentionally broad —
# false positives at this stage are acceptable because S5
# (full PDF extraction) will produce higher-confidence entries
# for the same papers later.
# ============================================================

RULE_PATTERNS = {

    "synthesis_method": {
        "hydrothermal":              "hydrothermal synthesis",
        "solvothermal":              "solvothermal synthesis",
        "co-precipitation":          "co-precipitation",
        "coprecipitation":           "co-precipitation",
        "sol-gel":                   "sol-gel",
        "chemical vapor deposition": "CVD",
        "cvd":                       "CVD",
        "pvd":                       "PVD",
        "physical vapor deposition": "PVD",
        "electrodeposition":         "electrodeposition",
        "electrochemical deposition":"electrodeposition",
        "atomic layer deposition":   "ALD",
        "ald":                       "ALD",
        "spray pyrolysis":           "spray pyrolysis",
        "magnetron sputtering":      "magnetron sputtering",
        "ball milling":              "ball milling",
        "calcination":               "calcination",
        "annealing":                 "annealing",
        "sintering":                 "sintering",
        "green synthesis":           "green synthesis",
        "microwave":                 "microwave synthesis",
        "sonochemical":              "sonochemical synthesis",
        "electrospinning":           "electrospinning",
    },

    "characterization": {
        "xrd":                        "XRD",
        "x-ray diffraction":          "XRD",
        "sem":                        "SEM",
        "scanning electron microscop":"SEM",
        "tem":                        "TEM",
        "transmission electron microscop": "TEM",
        "hrtem":                      "HRTEM",
        "ftir":                       "FTIR",
        "infrared spectroscop":       "FTIR",
        "raman":                      "Raman spectroscopy",
        "xps":                        "XPS",
        "x-ray photoelectron":        "XPS",
        "bet":                        "BET",
        "brunauer":                   "BET",
        "uv-vis":                     "UV-Vis",
        "uv–vis":                     "UV-Vis",
        "edx":                        "EDX",
        "energy dispersive":          "EDX",
        "thermogravimetric":          "TGA",
        "tga":                        "TGA",
        "cyclic voltammetry":         "CV",
        " cv ":                       "CV",
        "electrochemical impedance":  "EIS",
        "eis":                        "EIS",
        "photoluminescence":          "PL spectroscopy",
        " pl ":                       "PL spectroscopy",
        "afm":                        "AFM",
        "atomic force microscop":     "AFM",
    },

    "application": {
        "supercapacitor":           "supercapacitor",
        "energy storage":           "energy storage",
        "photocatalysis":           "photocatalysis",
        "photocatalytic":           "photocatalysis",
        "gas sensor":               "gas sensing",
        "gas sensing":              "gas sensing",
        "biosensor":                "biosensor",
        "antibacterial":            "antibacterial activity",
        "antimicrobial":            "antimicrobial activity",
        "photovoltaic":             "photovoltaic",
        "solar cell":               "solar cell",
        "lithium":                  "lithium-ion battery",
        "li-ion":                   "lithium-ion battery",
        "sodium-ion":               "sodium-ion battery",
        "electrocatalysis":         "electrocatalysis",
        "water splitting":          "water splitting",
        "hydrogen evolution":       "hydrogen evolution",
        "oxygen evolution":         "oxygen evolution",
        "dye degradation":          "dye degradation",
        "wastewater":               "wastewater treatment",
        "drug delivery":            "drug delivery",
        "luminescence":             "luminescence",
        "led":                      "LED",
        "transistor":               "transistor",
        "dielectric":               "dielectric material",
        "piezoelectric":            "piezoelectric",
        "ferroelectric":            "ferroelectric",
        "magnetic":                 "magnetic material",
    },
}

# Material formula patterns — matches chemical-formula-like
# uppercase tokens: ZnO, TiO2, Fe3O4, WSe2, MoS2, etc.
MATERIAL_FORMULA_PATTERN = re.compile(
    r"\b([A-Z][a-z]?\d*){1,4}(?:/[A-Z][a-z]?\d*)*\b"
)

# Noise patterns — sentences containing these are skipped
NOISE_PATTERNS = [
    "creative commons", "copyright", "doi:", "http",
    "all rights reserved", "correspondence", "received:",
    "accepted:", "journal of", "elsevier", "springer",
    "©", "this article", "this paper"
]


# ============================================================
# HELPERS
# ============================================================

def is_noise(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in NOISE_PATTERNS)


def is_valid_value(value: str) -> bool:
    v = value.strip()
    return bool(v) and 2 <= len(v) <= MAX_VALUE_LENGTH


def truncate_abstract(abstract: str) -> str:
    """
    Return at most MAX_ABSTRACT_WORDS words from the abstract.
    Prevents token overflow on the LLM call.
    """
    words = abstract.split()
    if len(words) <= MAX_ABSTRACT_WORDS:
        return abstract
    return " ".join(words[:MAX_ABSTRACT_WORDS]) + "..."


# ============================================================
# RULE-BASED EXTRACTION
#
# Scans title + abstract for known patterns.
# Returns list of dicts with category, value, sentence,
# confidence='low'.
#
# Each pattern is checked once per text. The first match
# wins — multiple matches of the same canonical value are
# deduplicated before storage.
# ============================================================

def extract_by_rules(title: str, abstract: str) -> list:

    combined = f"{title}. {abstract}".lower()
    results  = []
    seen     = set()   # (category, value) dedup within this paper

    # Synthesis, characterization, application
    for category, patterns in RULE_PATTERNS.items():
        for keyword, canonical_value in patterns.items():
            if keyword in combined:
                key = (category, canonical_value)
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "category":   category,
                        "value":      canonical_value,
                        "sentence":   None,
                        "confidence": "low"
                    })

    # Material formulas from title (higher signal than abstract)
    title_tokens = MATERIAL_FORMULA_PATTERN.findall(title)
    for token in title_tokens:
        if is_valid_value(token):
            key = ("material", token)
            if key not in seen:
                seen.add(key)
                results.append({
                    "category":   "material",
                    "value":      token,
                    "sentence":   None,
                    "confidence": "low"
                })

    return results


# ============================================================
# LLM EXTRACTION
#
# One LLM call per paper. Prompt is deliberately short:
# title + truncated abstract → JSON array of entities.
# max_tokens=200 keeps the call lightweight.
#
# Returns list of dicts with category, value, sentence,
# confidence='medium' (LLM-confirmed).
#
# Only called if rule extraction produced fewer than 2 entries
# for a paper — avoids redundant LLM calls on well-covered
# papers.
# ============================================================

def extract_by_llm(
    title: str,
    abstract: str,
    service: LLMService,
    existing_values: set
) -> list:

    short_abstract = truncate_abstract(abstract)

    prompt = (
        "Extract scientific entities from the title and abstract below.\n"
        "Return a JSON array only. Each item must have:\n"
        '  "category": one of material|synthesis_method|'
        'characterization|application|computational_method\n'
        '  "value": the specific entity name (1-5 words, no sentences)\n\n'
        f'Title: "{title}"\n'
        f'Abstract: "{short_abstract}"\n\n'
        "JSON array:"
    )

    try:
        items = service.extract(prompt, stage="S2_75")
    except Exception as e:
        print(f"[S2_75] LLM extraction failed: {e}")
        return []

    results = []
    seen    = set()

    for item in items:
        category = item.get("category", "")
        value    = item.get("value", "")

        if not is_valid_value(value):
            continue

        # Skip if already captured by rules
        if value.lower() in existing_values:
            continue

        key = (category, value)
        if key in seen:
            continue

        seen.add(key)
        results.append({
            "category":   category,
            "value":      value,
            "sentence":   short_abstract[:200],
            "confidence": "medium"
        })

    return results


# ============================================================
# MAIN TOOL — S2_75
# ============================================================

def extract_lightweight_knowledge(
    repo, workflow_id, execution_attempt_id=None, **kwargs
):
    """
    Standard tool interface: tool(repo, workflow_id, **kwargs)

    Processes all papers that have:
    - abstract IS NOT NULL
    - no existing ResearchKnowledge entries

    For each qualifying paper:
    1. Run rule-based extraction (no LLM, confidence='low')
    2. If rule coverage < 2 entries: run one LLM call
       (confidence='medium')
    3. Store all results via create_lightweight_knowledge()
    4. Log failures to FailureLog without stopping pipeline

    Returns standard result dict.
    """

    # --------------------------------------------------
    # FETCH PAPERS WITH ABSTRACTS
    # --------------------------------------------------
    papers = repo.fetch_all(
        """
        SELECT id, title, abstract
        FROM Paper
        WHERE workflow_id = ?
          AND abstract IS NOT NULL
          AND length(trim(abstract)) > 50
        ORDER BY id ASC
        """,
        (workflow_id,)
    )

    if not papers:
        print("[S2_75] No papers with abstracts found. Skipping.")
        return {"status": "success", "data": [], "error": None}

    print(f"[S2_75] Found {len(papers)} papers with abstracts")

    # --------------------------------------------------
    # LLM INIT — single instance for all papers
    # --------------------------------------------------
    llm     = GroqClient(timeout=120)
    service = LLMService(llm)

    # --------------------------------------------------
    # COUNTERS FOR LOGGING
    # --------------------------------------------------
    papers_processed  = 0
    papers_skipped    = 0
    knowledge_created = 0
    failures          = 0

    # --------------------------------------------------
    # PROCESS EACH PAPER
    # --------------------------------------------------
    for paper in papers:

        paper_id = paper["id"]
        title    = paper["title"] or ""
        abstract = paper["abstract"] or ""

        # IDEMPOTENCY CHECK — skip if knowledge already exists
        if rk_repo.paper_has_knowledge(repo, paper_id):
            papers_skipped += 1
            continue

        if is_noise(title) or is_noise(abstract):
            papers_skipped += 1
            continue

        try:

            # STEP 1: Rule-based extraction
            rule_results = extract_by_rules(title, abstract)

            # STEP 2: LLM extraction if rules found < 2 entities
            llm_results = []

            if len(rule_results) < 2:
                existing_values = {
                    r["value"].lower() for r in rule_results
                }
                llm_results = extract_by_llm(
                    title, abstract, service, existing_values
                )

            all_results = rule_results + llm_results

            if not all_results:
                # No knowledge found — paper had abstract but
                # nothing extractable. Not a failure.
                papers_processed += 1
                continue

            # STEP 3: Persist all extracted knowledge
            # Uses create_lightweight_knowledge() which always
            # sets source_type='abstract'. Does NOT overwrite
            # any existing entries — INSERT only.
            with repo.transaction() as cursor:
                for item in all_results:
                    cursor.execute(
                        """
                        INSERT INTO ResearchKnowledge (
                            paper_id,
                            category,
                            value,
                            section_source,
                            sentence,
                            source_type,
                            confidence,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (
                            paper_id,
                            item["category"],
                            item["value"],
                            "abstract",
                            item.get("sentence"),
                            "abstract",
                            item["confidence"]
                        )
                    )
                    knowledge_created += 1

            papers_processed += 1

        except Exception as e:

            error_msg = str(e)
            print(f"[S2_75] ⚠️ Failed paper {paper_id}: {error_msg}")

            failure_repo.log_failure(
                repo,
                workflow_id,
                "LIGHTWEIGHT_EXTRACTION_ERROR",
                error_msg,
                paper_id=paper_id
            )

            failures += 1
            # Continue — one paper failure does not stop stage

    # --------------------------------------------------
    # LOGGING
    # --------------------------------------------------
    print(
        f"\n[S2_75] Complete — "
        f"processed: {papers_processed} | "
        f"skipped: {papers_skipped} | "
        f"knowledge entries created: {knowledge_created} | "
        f"failures: {failures}"
    )

    return {
        "status": "success",
        "data": {
            "papers_processed":  papers_processed,
            "papers_skipped":    papers_skipped,
            "knowledge_created": knowledge_created,
            "failures":          failures
        },
        "error": None
    }
