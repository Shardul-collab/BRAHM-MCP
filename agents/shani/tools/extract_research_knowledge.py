# ============================================================
# extract_research_knowledge.py  — S5  (REWRITE)
#
# Section-aware, paper-type-aware extraction.
#
# PRIMARY PATH (paper has PaperContent rows):
#   1. Classify paper as REVIEW or EXPERIMENTAL
#   2. Select sections based on type
#   3. Concatenate selected section text
#   4. Run sentence-level rules (formula / keyword / regex) on full text
#   5. Chunk text (~3500 chars, 200 char overlap) → LLM extraction per chunk
#
# FALLBACK PATH (no PaperContent, raw_text present):
#   Segment raw_text by header patterns → pseudo-sections →
#   same classify / select / chunk flow
#
# DEAD PAPER (no sections, no raw_text):
#   Skip immediately, mark knowledge_ready.
# ============================================================

import re
import json
from collections import defaultdict
import spacy

from tools.text_cleaner import clean_scientific_text
from tools.normalise_paper_content import run_normalisation
from tools.relation_extractor import extract_relations
from tools.keyword_match import keyword_in

import repositories.paper_repo as paper_repo
import repositories.paper_content_repo as pc_repo
from services.llm_service import ensure_model_available, LLMService, OllamaClient, GeminiClient, CerebrasClient, GroqClient
from services.vector_db_service import VectorDBService


# ============================================================
# PIPELINE STATUS CONTRACT
# ============================================================

class _PipelineStatus:
    S5_INPUT  = "extracted"
    S5_OUTPUT = "knowledge_ready"
    S5_FAILED = "knowledge_failed"

PIPELINE_STATUS = _PipelineStatus()


# ============================================================
# NLP + VECTOR DB
# ============================================================

nlp = spacy.load("en_core_web_sm")
vector_service = VectorDBService()


# ============================================================
# PAPER TYPE CLASSIFICATION
# ============================================================

# If ANY section name matches one of these patterns → REVIEW paper.
# Roman-numeral prefixes (i_, ii_, iii_ …) are the strongest signal.
_REVIEW_PATTERNS = [
    re.compile(r"^i_"),
    re.compile(r"^ii_"),
    re.compile(r"^iii_"),
    re.compile(r"^iv_"),
    re.compile(r"^v_"),
    re.compile(r"^vi_"),
    re.compile(r"^vii_"),
    re.compile(r"^viii_"),
    re.compile(r"summary_and_outlook"),
    re.compile(r"future_direction"),
    re.compile(r"perspective"),
    re.compile(r"outlook"),
]

def classify_paper(section_names: list) -> str:
    """Return 'REVIEW' or 'EXPERIMENTAL'."""
    for name in section_names:
        n = name.lower().strip()
        for pat in _REVIEW_PATTERNS:
            if pat.search(n):
                return "REVIEW"
    return "EXPERIMENTAL"


# ============================================================
# SECTION SELECTION
# ============================================================

# Sections to ALWAYS skip regardless of paper type.
_SKIP_ALWAYS = {
    "references", "reference", "viii_references",
    "acknowledgements", "acknowledgments", "acknowledgement", "acknowledgment",
    "vi_acknowledgements",
    "author_contributions", "author_information", "corresponding_author",
    "data_availability", "code_availability",
    "additional_information", "supporting_information",
    "competing_interests",
    "figure_captions",
    "notes",
    "front_matter",
}

# For EXPERIMENTAL papers, also skip these sections.
_SKIP_EXPERIMENTAL = {
    "introduction", "i_introduction",
}

# Front matter is judged by content in normalisation (tools/front_matter.py,
# decision D1 2026-09-11) and labelled "front_matter"; that label is skipped
# here. "preamble" is no longer judged at all at this stage: it used to be
# skipped by name (which dropped 84% of paper 20), then by size (<=4,000 chars
# and <30% of the paper), which is the rule that dropped the abstracts of
# papers 13, 15 and 17 along with their author blocks.


def select_sections(sections: dict, paper_type: str) -> dict:
    """
    sections: {section_name: text}
    Returns filtered dict with only sections we want to extract from.
    """
    skip = set(_SKIP_ALWAYS)
    if paper_type == "EXPERIMENTAL":
        skip |= _SKIP_EXPERIMENTAL
    return {name: text for name, text in sections.items()
            if text and text.strip() and name.lower().strip() not in skip}


# ============================================================
# RAW TEXT FALLBACK SEGMENTATION
# ============================================================

# Header patterns that commonly appear in paper raw_text.
_HEADER_RE = re.compile(
    r"^(?:"
    r"abstract|introduction|background|"
    r"method(?:s|ology)?|experimental|materials?\s+and\s+methods?|"
    r"result(?:s)?(?:\s+and\s+discussion)?|"
    r"discussion|conclusion(?:s)?|summary|outlook"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE
)

def segment_raw_text(raw_text: str) -> dict:
    """
    Attempt to split raw_text into pseudo-sections by header patterns.
    Returns {section_name: text} dict.
    Falls back to {"body": raw_text} if no headers found.
    """
    lines = raw_text.split("\n")
    sections = {}
    current_name = "body"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if _HEADER_RE.match(stripped):
            if current_lines:
                sections[current_name] = "\n".join(current_lines).strip()
            current_name = stripped.lower().replace(" ", "_")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_name] = "\n".join(current_lines).strip()

    if not sections or (len(sections) == 1 and "body" in sections):
        return {"body": raw_text}

    return sections


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE    = 3500   # chars
CHUNK_OVERLAP = 200    # chars

def chunk_text(text: str) -> list:
    """
    Split text into overlapping chunks of ~CHUNK_SIZE chars.
    Tries to break at sentence boundaries ('. ').
    Returns list of strings.
    """
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start  = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at a sentence boundary within the last 300 chars of window
        boundary = text.rfind(". ", start + CHUNK_SIZE - 300, end)
        if boundary != -1:
            end = boundary + 1  # include the period

        chunks.append(text[start:end].strip())
        start = end - CHUNK_OVERLAP

    return [c for c in chunks if c.strip()]


# ============================================================
# NOISE FILTERING
# ============================================================

NOISE_PATTERNS = [
    "creative commons", "copyright", "all rights reserved",
    "doi:", "http://", "https://", "www.",
    "correspondence:", "received:", "accepted:", "published:",
    "journal of", "elsevier", "springer", "wiley", "mdpi",
    "this article", "this paper", "this work", "this study",
    "the authors", "author contributions", "funding:",
    "conflict of interest", "supplementary",
    "©", "™", "®"
]

MAX_VALUE_LENGTH = 120

VALUE_BLOCKLIST = {
    "usa", "uk", "eu", "uae", "us", "un",
    "url", "doi", "isbn", "issn", "pdf", "html", "xml", "api",
    "lc", "rc", "dc", "ac", "ii", "iii", "iv", "vi", "vii",
    "et", "al", "fig", "eq", "ref", "sec", "vol", "no",
    "mdpi", "ieee", "aip", "acs", "rsc", "iop",
    "a", "b", "c", "n", "p", "x", "y", "z",
}

MIN_PAPER_SCORE = 0.03


def is_noise_sentence(sentence: str) -> bool:
    s = sentence.lower()
    if any(p in s for p in NOISE_PATTERNS):
        return True
    if re.search(r'\d[A-Z][a-z]', sentence) and re.search(
        r'Department|Institute|University|College|Laboratory|Lab', sentence
    ):
        return True
    if re.search(r'[A-Z][a-z]+\d,\s+[A-Z][a-z]+\d', sentence):
        return True
    words = sentence.split()
    if len(words) < 20 and sentence.count(',') >= 3:
        alpha_words = [w for w in words if w.isalpha() and len(w) > 1]
        digit_words = [w for w in words if any(c.isdigit() for c in w)]
        if len(digit_words) >= 2 and len(alpha_words) <= len(words) * 0.7:
            return True
    if len(sentence.strip()) < 40 and re.search(r'\d', sentence) and re.search(r'[,\)\(]', sentence):
        return True
    return False


# 2026-09-11 (decision D11): for these categories a value is a name, not a
# number, and requiring it to share a word with the workflow query dropped
# 8 of 10 application values ('photodetectors', 'solar cells', 'ferroelectric
# semiconductor field effect transistors') plus 'HAADF-STEM', 'excess Se' and
# 'photoluminescence spectroscopy' on the replay. They still go through the
# length/blocklist checks here and the placeholder check in is_valid_knowledge.
RELAXED_VALUE_CATEGORIES = {"application", "defect_type", "characterization"}


def is_valid_value(value: str, query_terms: set, category: str = None) -> bool:
    if not value or not value.strip():
        return False
    v = value.strip()
    if len(v) > MAX_VALUE_LENGTH or len(v) < 3:
        return False
    if v.lower() in VALUE_BLOCKLIST:
        return False
    if category in RELAXED_VALUE_CATEGORIES:
        return True
    if re.fullmatch(r"[A-Z]", v):
        return False
    if re.fullmatch(r"[A-Z]{3,5}", v):
        return True
    if re.fullmatch(r"[A-Za-z]{1,3}_[A-Za-z]{1,3}", v):
        return True
    value_words = {w.lower() for w in re.findall(r"\b\w+\b", v)}
    if value_words & query_terms:
        return True
    if re.search(r"(?:[A-Z][a-z]?\d*){2,}", v) and re.search(r"[a-z]|\d", v):
        return True
    if re.search(r"\d", v):
        return True
    return False


# ============================================================
# CATEGORY-AWARE KNOWLEDGE VALIDATOR
# ============================================================

_OPTICAL_UNITS = re.compile(
    r'\d\s*(?:eV|nm|cm[\u207b\-]1|A\s*W[\u207b\-]1|AW[\u207b\-]1|meV|\u03bcm)'
)
_ELECTRICAL_UNITS = re.compile(
    r'\d\s*(?:ns|\u03bcs|ps|ms|cm[\u207b\-]?[23]|\u03a9|ohm|A\s*/\s*W|A/W|Jones|'
    r'cm2|V\b|nA|\u03bcA|mA\b|S\s*cm|S/cm|m\*)'
)
_JONES_RE    = re.compile(r'\bJones\b')

# 2026-09-11 (decision D3): categories for quantities the 20-category schema
# could not hold. Measured on a replay of the last full S5 run: MBE fluxes and
# flux ratios arrived as doping_parameter and failed its unit check, paper 20's
# dielectric constants (17, 6.29) failed optical_property's unit list, and 13
# photodetector figures of merit were emitted under invented categories and
# dropped. Each new category still requires a number plus a unit or a word that
# says what the number is.
_FLUX_RE = re.compile(
    r'cm\s*(?:\^|\*\*)?\s*[-\u2212\u207b]\s*2|atoms?\s*/|/\s*cm2|\bBEP\b|beam[- ]equivalent|'
    r'\bTorr\b|\bmbar\b|\bPa\b|\u00c5\s*/\s*(?:s|min)|\bA\s*/\s*s\b|nm\s*/\s*(?:s|min|h)|'
    r'\bML\s*/\s*(?:s|min)|\u03bcm\s*/\s*h|um\s*/\s*h|\bratio\b|\b[A-Z][a-z]?\s*:\s*[A-Z][a-z]?\b|\bflux\b|\brate\b',
    re.IGNORECASE)
_FERRO_RE = re.compile(
    r'pm\s*/\s*V|[\u03bcu\u00b5]C\s*/?\s*cm|kV\s*/\s*cm|MV\s*/\s*cm|\d\s*V\b|\u00b0C|\d\s*K\b|'
    r'\bd33\b|\bd_?33\b|polari[sz]ation|coercive|curie', re.IGNORECASE)
_PD_RE = re.compile(
    r'\d\s*(?:[pnu\u03bc\u00b5m]?A\b|Jones|[mu\u03bc\u00b5n]?s\b|%|dB)|detectivity|dark current|'
    r'photocurrent|rise time|decay time|response time|quantum efficiency|\bEQE\b', re.IGNORECASE)
_DIELECTRIC_WORD_RE = re.compile(r'dielectric|permittivity|refractive|extinction|\u03b5', re.IGNORECASE)
_ANNEAL_TEMP = re.compile(r'\d+\s*(?:\u00b0C|K\b)')
_DOPING_UNITS = re.compile(
    r'\d\s*(?:at\.?%|wt\.?%|mol%|cm[\u207b\-]3|\u00d710)'
)
_SUPERSCRIPT_MAP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻', '0123456789-')

_GARBLED_MATERIAL_VALUES = {
    'coco', 'osos', 'osco', 'fesm', 'coos',
    'tialsi', 'tin1', 'in2se320', 'snte37',
    '2d materials', 'bi2x3',
    'cucrs258', 'cuinp2s616',  # confirmed: valid formula + appended citation number
}

# Matches a chemical-formula SHAPE: 2-6 repeats of (Capital letter, optional
# lowercase letter, optional 1-2 digit subscript). e.g. CuCrS2, In2Se3, MoS2.
# Shape alone is not enough — see _looks_like_material below.
_ELEMENT_TOKEN_RE = re.compile(r'^(?:[A-Z][a-z]?\d{0,2}){2,6}$')

# ── Positive material validation ─────────────────────────────────────────────
#
# Added 2026-09-09 after auditing the live In2Se3 corpus. The checks below this
# point were all DENYLIST-shaped: a set of known-bad strings plus two narrow
# regexes. That approach failed badly in practice.
#
# Paper 6 of workflow 2 contributed 605 distinct "material" values — 58% of the
# entire knowledge corpus and 87% of the material axis. Every one of them was
# an author surname carrying an affiliation index, harvested from a CERN LHCb
# particle-physics paper that had been downloaded under an indium-selenide
# title: 'Bediaga1', 'Miranda1', 'Rodrigues1', 'Gomes1', 'LHCb-PAPER-2014-049'.
#
# None were caught, because the existing shape rule was
# `[A-Za-z]{2,4}\d{1,3}` with len<=7: 'Gomes1' has five letters (fails {2,4})
# and 'Bediaga1' has seven. A denylist cannot anticipate an author list.
#
# The fix is to validate POSITIVELY: a material must decompose into real
# periodic-table symbols. 'Gomes1' does not ('G' is not an element; Ga, Gd and
# Ge are). Author names die as a class rather than one string at a time.

_ELEMENTS = {
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S',
    'Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga',
    'Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd',
    'Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm',
    'Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os',
    'Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa',
    'U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg',
    'Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og',
}

# Greek/phase prefixes and decorations that legitimately attach to a formula:
# alpha-In2Se3, β-In2Se3, 2H-MoS2, 1T'-MoS2.
_PHASE_PREFIX_RE = re.compile(
    r"^(?:[α-ω]|alpha|beta|gamma|delta|epsilon|kappa|"
    r"\d+[HTRhtr]['′]?)[-‐-―\s]+",
    re.IGNORECASE,
)
_FORMULA_TOKEN_RE = re.compile(r'([A-Z][a-z]?)(\d{0,3})')


def _strip_phase_prefix(v: str) -> str:
    prev = None
    while prev != v:
        prev = v
        v = _PHASE_PREFIX_RE.sub('', v).strip()
    return v


# Descriptive words that legitimately trail or wrap a formula in extracted
# text. Stripping them lets 'WZ-In2Se3 thin films' validate as In2Se3 rather
# than being thrown away with the author names.
_MATERIAL_DESCRIPTORS = (
    'thin films', 'thin film', 'films', 'film', 'layers', 'layer',
    'substrate', 'substrates', 'nanosheets', 'nanosheet', 'crystals',
    'crystal', 'phase', 'phases', 'type', 'based', 'polymorph-pure',
    'polymorph', 'c-plane', 'a-plane', 'monolayer', 'bulk', 'wz', 'zb',
)
# PDF text extraction frequently emits private-use glyphs for Greek letters
# ( is gamma in several embedded fonts). Map the ones seen in this
# corpus so 'γ-InSe' does not arrive as an unparseable character.
_PUA_GREEK = {'': 'α', '': 'β', '': 'γ', '': 'δ'}

# Subscripts this large are not chemistry — they are appended citation
# numbers. Real subscripts in this corpus top out well below 10 (In2Se3,
# CuInP2S6, Mn2In2Se5). See the note in _looks_like_material about the
# single-digit case, which shape alone cannot resolve.
_MAX_PLAUSIBLE_SUBSCRIPT = 9


def _tokens_are_elements(candidate: str) -> bool:
    """True when candidate decomposes cleanly into real element symbols."""
    if not candidate:
        return False
    pos, tokens = 0, []
    for m in _FORMULA_TOKEN_RE.finditer(candidate):
        if m.start() != pos:      # a gap means an unparseable character
            return False
        sub = m.group(2)
        if sub and int(sub) > _MAX_PLAUSIBLE_SUBSCRIPT:
            return False          # 'GaSe59' = GaSe + citation 59
        tokens.append(m.group(1))
        pos = m.end()
    if pos != len(candidate) or not tokens:
        return False
    return all(t in _ELEMENTS for t in tokens)


def _strip_descriptors(v: str) -> str:
    """Remove parentheticals, quotes and descriptive words around a formula."""
    for pua, greek in _PUA_GREEK.items():
        v = v.replace(pua, greek)
    v = re.sub(r'\([^)]*\)', ' ', v)          # '(WZ type)', '(00l)'
    v = re.sub(r'\[[^\]]*\]', ' ', v)         # '[identity matrix]'
    v = v.replace('’', "'").replace('‘', "'")
    v = re.sub(r"'", ' ', v)
    for word in sorted(_MATERIAL_DESCRIPTORS, key=len, reverse=True):
        v = re.sub(rf'(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])', ' ',
                   v, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', v).strip(" -‐–—/,")


def _looks_like_material(v: str) -> bool:
    """
    Positive test: does this value look like a material rather than a name,
    a citation key, or a sentence fragment?

    Accepts:
      - multi-element formulas: In2Se3, MoS2, CuInP2S6, Bi2Te3
      - phase-prefixed formulas: alpha-In2Se3, beta-In2Se3, 2H-MoS2
      - bare element symbols WITHOUT a subscript: Si, Ge, In, Se
      - simple compound forms joined by / or - where each side qualifies:
        In2Se3/Si, MoS2-WS2

    Rejects:
      - a bare element symbol WITH a subscript: 'Se3', 'Te3'. These are not
        standalone materials in this corpus — they are the tail of a split
        formula. 'Se3' appears 10 times and is In2Se3 with 'In2' lost; the
        one paper titled "Molecular beam epitaxy synthesis of In2Se3 films"
        has its material recorded as 'Se3'.
      - anything whose tokens are not real elements: Gomes1, Bediaga1,
        Universe3, Cousins30.
    """
    v = v.strip()
    if not v:
        return False

    # Strip descriptive wrapping first, then phase prefixes, so that
    # "WZ' type α-In2Se3 film" reduces to "In2Se3" rather than being
    # discarded alongside the author names.
    core = _strip_phase_prefix(_strip_descriptors(v))
    core = _strip_phase_prefix(core)
    if not core:
        return False

    # Split on / and - to handle heterostructures and hyphenated pairs, only
    # after phase prefixes are gone so 'alpha-In2Se3' is not split.
    parts = [p for p in re.split(r'[/‐-―-]', core) if p.strip()]
    if not parts:
        return False

    for part in parts:
        part = part.strip()
        if part in _ELEMENTS:              # bare element, no subscript: fine
            continue
        if not _tokens_are_elements(part):
            return False
        # single element carrying a subscript => split-formula tail
        m = list(_FORMULA_TOKEN_RE.finditer(part))
        if len(m) == 1 and m[0].group(2):
            return False
    return True


# KNOWN LIMIT, recorded rather than hidden: a SINGLE trailing digit cannot be
# told from a real subscript by shape alone. 'SnTe6' is SnTe + citation 6, but
# 'CuInP2S6' is a real formula ending in 6 — both parse identically. The signal
# that resolves it is corpus-level, not per-value: if 'GaSe' also appears in the
# corpus, then 'GaSe59' is almost certainly GaSe + a citation. That belongs in a
# normalisation pass over the whole knowledge table, not in this per-value gate,
# and is not implemented here. Values like 'SnTe6' therefore still get through.


def _has_citation_contaminated_tail(v: str) -> bool:
    """
    Detect 'valid formula + appended 3-digit citation number' contamination
    (e.g. 'CuCrS2' + '58' -> 'CuCrS258', 'CuInP2S6' + '16' -> 'CuInP2S616').

    Deliberately restricted to EXACTLY a 3-digit trailing run. Real chemical
    subscripts in this corpus are essentially always 1-2 digits (In2Se3,
    MoS2, TiO2, ...), so requiring 3 digits avoids false-flagging those while
    still catching the citation-number contamination pattern generically
    (not just the two known instances above).
    """
    if len(v) <= 3:
        return False
    tail = v[-3:]
    if not tail.isdigit():
        return False
    candidate = v[:-3]
    return bool(_ELEMENT_TOKEN_RE.fullmatch(candidate))


_GENERIC_MATERIAL = {
    'layered material', 'thin film', 'nanoparticle',
    'nanowire', 'quantum dot', 'heterostructure', 'substrate',
    'not specified', 'various', 'unknown',
}
_TECHNIQUE_LABELS = {
    'raman intensity', 'pl spectra', 'tr spectra', 'pl and absorption',
    'xps spectra', 'absorption spectra', 'reflectance spectra',
    'time-resolved spectra', 'second harmonic generation',
    'photoluminescence', 'raman spectroscopy',
}


# The S5 prompt tells the model to OMIT anything not stated in the passage.
# It often complies by emitting a placeholder instead - "not specified [FET
# carrier mobility]", "14.56 [not specified unit, likely Torr or Pa]",
# "CVD [inferred from the context ...]". Those passed every gate: the
# "not specified" checks below existed only for material and
# annealing_condition, and is_valid_value() counts the bracket qualifier as
# value content, so "[photoresponsivity]" matched a query term. Measured
# 2026-09-11: 21 stored rows (3.6% of the corpus), 10 of them paper 17's
# numbers with units the model guessed. Enforce the omit rule for every
# category, on the whole value including its qualifier.
_PLACEHOLDER_RE = re.compile(
    r"\bnot\s+(?:explicitly\s+)?(?:specified|reported|stated|mentioned|given|"
    r"available|provided)\b|\binferred\b",
    re.IGNORECASE,
)


def is_valid_knowledge(category: str, value: str, sentence: str) -> bool:
    if not value or not value.strip():
        return False
    v   = value.strip()
    vl  = v.lower()
    v_n = v.translate(_SUPERSCRIPT_MAP)

    if _PLACEHOLDER_RE.search(v):
        return False

    if category == "material":
        if len(v) > 40:
            return False
        if vl in _GARBLED_MATERIAL_VALUES or vl in _GENERIC_MATERIAL:
            return False
        if re.fullmatch(r'[A-Za-z]{2,4}\d{1,3}', v) and len(v) <= 7 and not re.search(r'[a-z][A-Z]', v):
            return False
        if _has_citation_contaminated_tail(v):
            return False
        if "not specified" in vl or "not reported" in vl:
            return False
        # Positive gate (2026-09-09). Everything above is a denylist and was
        # comprehensively defeated by an author list — see _looks_like_material.
        # This must stay LAST so the cheap known-bad checks still short-circuit.
        if not _looks_like_material(v_n):
            return False
        return True

    if category == "optical_property":
        if vl in _TECHNIQUE_LABELS or any(vl.startswith(t) for t in _TECHNIQUE_LABELS):
            return False
        if _OPTICAL_UNITS.search(v_n):
            return True
        # dimensionless: dielectric constant, permittivity, refractive index
        return bool(re.search(r'\d', v_n) and _DIELECTRIC_WORD_RE.search(v_n))

    if category == "growth_flux":
        return bool(re.search(r'\d', v_n) and _FLUX_RE.search(v_n)) and len(v) <= 120

    if category == "ferroelectric_property":
        return bool(re.search(r'\d', v_n) and _FERRO_RE.search(v_n))

    if category == "photodetector_metric":
        return bool(re.search(r'\d', v_n) and _PD_RE.search(v_n))

    if category == "electrical_property":
        if vl in _TECHNIQUE_LABELS:
            return False
        if not _ELECTRICAL_UNITS.search(v_n) and not _JONES_RE.search(v_n):
            return False
        return True

    if category == "annealing_condition":
        if not _ANNEAL_TEMP.search(v_n):
            return False
        if len(v) > 80:
            return False
        if "not specified" in vl or "might be" in vl or "possibly" in vl:
            return False
        return True

    if category == "doping_parameter":
        if not _DOPING_UNITS.search(v_n):
            return False
        return True

    if category == "defect_type":
        if len(v) > 60:
            return False
        if re.search(r'[.]{2,}|equation|table|figure', vl):
            return False
        return True

    return True


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(query: str) -> set:
    doc = nlp(query.lower())
    return {
        token.lemma_
        for token in doc
        if token.pos_ in {"NOUN", "PROPN", "VERB"}
        and not token.is_stop
        and len(token.lemma_) > 2
    }


# ============================================================
# PAPER RELEVANCE SCORING
# ============================================================

# Publishers typeset formulae with Unicode sub/superscript digits: "In₂Se₃",
# not "In2Se3". Every scorer here matches exact tokens against query terms that
# S1 generates in ASCII, so a title written that way scores ZERO on the corpus's
# own subject. Observed 2026-09-10: "Fabrication of ᵞ-In₂Se₃-Based Photodetector"
# tokenised to 'in₂se₃', missed the 'in2se3' query term, scored 0.023 against a
# 0.03 threshold and was skipped - one of the most on-topic papers in the set.
_DIGIT_FOLD = {}
for _i, (_sub, _sup) in enumerate(zip("₀₁₂₃₄₅₆₇₈₉", "⁰¹²³⁴⁵⁶⁷⁸⁹")):
    _DIGIT_FOLD[ord(_sub)] = str(_i)
    _DIGIT_FOLD[ord(_sup)] = str(_i)


def fold_digits(text: str) -> str:
    """Map Unicode sub/superscript digits to ASCII so formulae tokenise."""
    return (text or "").translate(_DIGIT_FOLD)


# ── Term specificity ────────────────────────────────────────────────────────
# The query is built by flattening every WorkflowResearchConfig field into one
# string, which throws away the one thing the config actually knows: which term
# is the SUBJECT and which are generic technique words. Every term then weighed
# the same, so 'beam' + 'epitaxy' outscored 'in2se3'.
#
# Measured on workflow 1 (2026-09-10): the gate admitted a BaBiO3-on-SrTiO3
# paper at 0.035 and rejected two In2Se3 papers at 0.029, against a 0.03
# threshold - separating those decisions by 0.006, with the sign backwards.
#
# Weights come from the field a term arrived in, not from a hand-written list,
# so they follow whatever the user configured for the workflow. They are chosen
# so the total weight stays close to the term count and MIN_PAPER_SCORE keeps
# its meaning.
FIELD_WEIGHTS = {
    "material":         3.0,   # the subject of the review - decisive
    "structure":        1.5,
    "focus":            1.5,   # what the review is asking about
    "properties":       1.5,
    "method":           0.5,   # MBE/CVD: shared by most papers in any corpus
    "characterization": 0.5,   # XRD/Raman/TEM: likewise
}
GENERIC_TERM_WEIGHT = 0.5      # the hardcoded defect/carrier/bandgap tail
DEFAULT_TERM_WEIGHT = 1.0


def build_term_weights(config, generic_text=""):
    """
    Map each query term to a weight based on which config field produced it.
    Highest weight wins when a term appears in several fields.
    """
    weights = {}

    def add(text, weight):
        if not text:
            return
        for term in extract_keywords(text):
            if weights.get(term, 0) < weight:
                weights[term] = weight

    for field, weight in FIELD_WEIGHTS.items():
        try:
            value = config[field] if config is not None else None
        except (KeyError, IndexError):
            value = None          # sqlite3.Row raises when the column wasn't selected
        add(value, weight)
    add(generic_text, GENERIC_TERM_WEIGHT)
    return weights


def _weight_of(term, term_weights):
    return term_weights.get(term, DEFAULT_TERM_WEIGHT) if term_weights else 1.0


def _weighted_hits(words, query_terms, term_weights):
    return sum(_weight_of(w, term_weights) for w in words if w in query_terms)


def jaccard_title(title, query_terms, term_weights=None):
    words = {w.lower() for w in re.findall(r"\b\w+\b", fold_digits(title))}
    if not words:
        return 0
    return _weighted_hits(words, query_terms, term_weights) / len(words | query_terms)


def abstract_density(text, query_terms, term_weights=None):
    words = re.findall(r"\b\w+\b", fold_digits(text).lower())[:1000]
    if not words:
        return 0
    return _weighted_hits(words, query_terms, term_weights) / len(words)


def title_overlap(title, query_terms, term_weights=None):
    words = {w.lower() for w in re.findall(r"\b\w+\b", fold_digits(title))}
    if not query_terms:
        return 0
    total = sum(_weight_of(t, term_weights) for t in query_terms)
    if not total:
        return 0
    return _weighted_hits(words, query_terms, term_weights) / total


# ── Subject-element affinity ────────────────────────────────────────────────
# Weighting terms by config field fixed the BaBiO3 case but broke two others:
# "MBE of Mn2In2Se5 van der Waals Layers" and "Mixed polytype/polymorph
# formation ... in InSe" both dropped below threshold, because neither
# 'mn2in2se5' nor 'inse' is a query term - only the exact string 'in2se3' is.
# Both papers are squarely on-topic for an In2Se3 polymorphism review: one is
# an In-Se compound, the other the parent binary.
#
# Exact-token matching cannot see that. Element sets can: parse formula-shaped
# tokens out of the title and compare their elements to the subject material's.
# Case matters for parsing ('InSe' is In+Se, 'inse' is ambiguous), so this runs
# on the ORIGINAL-CASE title, before the lowercasing the other scorers do.
SUBJECT_COMPOUND_AFFINITY = 1.0    # title names a compound of ALL subject elements
SHARED_ELEMENT_AFFINITY   = 0.4    # ...of some of them
AFFINITY_WEIGHT           = 0.05   # a strong nudge, not a veto


def subject_elements(material: str) -> set:
    """Element symbols in the workflow's subject material. 'In2Se3' -> {In, Se}."""
    if not material:
        return set()
    core = _strip_phase_prefix(material.strip())
    return {m.group(1) for m in _FORMULA_TOKEN_RE.finditer(core)
            if m.group(1) in _ELEMENTS}


def material_affinity(title: str, subj_elements: set) -> float:
    """
    Does the title name a material built from the subject's elements?
    Returns the best match found: full, partial, or none.
    """
    if not subj_elements or not title:
        return 0.0
    best = 0.0
    for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", fold_digits(title)):
        if not _looks_like_material(token):
            continue
        els = {m.group(1) for m in _FORMULA_TOKEN_RE.finditer(token)
               if m.group(1) in _ELEMENTS}
        if not els:
            continue
        if subj_elements <= els:
            return SUBJECT_COMPOUND_AFFINITY
        if subj_elements & els:
            best = max(best, SHARED_ELEMENT_AFFINITY)
    return best


def compute_score(title, text_sample, query_terms, term_weights=None,
                  subj_elements=None):
    """Relevance score from the title and a sample of the paper's own text."""
    return min(
        0.4 * jaccard_title(title, query_terms, term_weights)
        + 0.35 * abstract_density(text_sample, query_terms, term_weights)
        + 0.25 * title_overlap(title, query_terms, term_weights)
        + AFFINITY_WEIGHT * material_affinity(title, subj_elements or set()),
        1,
    )


# ============================================================
# CHARACTERIZATION KEYWORD RULES
# ============================================================

CHAR_KEYWORDS = {
    "xrd": "XRD", "x-ray diffraction": "XRD",
    "gixrd": "GIXRD", "grazing incidence x-ray": "GIXRD",
    "saxs": "SAXS", "waxs": "WAXS",
    "rheed": "RHEED", "leed": "LEED", "xrf": "XRF",
    "sem": "SEM", "scanning electron": "SEM",
    "fesem": "FE-SEM", "field emission sem": "FE-SEM",
    "tem": "TEM", "transmission electron": "TEM",
    "hrtem": "HRTEM", "high-resolution tem": "HRTEM",
    "stem": "STEM", "haadf": "HAADF-STEM",
    "eels": "EELS", "electron energy loss": "EELS",
    "ebsd": "EBSD", "electron backscatter": "EBSD",
    "fib": "FIB", "focused ion beam": "FIB",
    "afm": "AFM", "atomic force": "AFM",
    "stm": "STM", "scanning tunneling": "STM",
    "kpfm": "KPFM", "kelvin probe": "KPFM",
    "pfm": "PFM", "piezoresponse force": "PFM",
    "xps": "XPS", "x-ray photoelectron": "XPS", "esca": "XPS",
    "ups": "UPS", "ultraviolet photoelectron": "UPS",
    "aes": "AES", "auger electron": "AES",
    "arpes": "ARPES", "angle-resolved photoemission": "ARPES",
    "exafs": "EXAFS", "extended x-ray absorption": "EXAFS",
    "xanes": "XANES", "near-edge x-ray": "XANES",
    "ftir": "FTIR", "infrared spectroscop": "FTIR",
    "atr": "ATR-FTIR", "attenuated total reflectance": "ATR-FTIR",
    "raman": "Raman spectroscopy",
    "photoluminescence": "PL spectroscopy",
    "cathodoluminescence": "CL",
    "electroluminescence": "EL",
    "time-resolved photoluminescence": "TRPL",
    "time resolved photoluminescence": "TRPL",
    "trpl": "TRPL", "tcspc": "TCSPC",
    "shg": "SHG", "second harmonic generation": "SHG",
    "uv-vis": "UV-Vis", "uv–vis": "UV-Vis",
    "uv-vis-nir": "UV-Vis-NIR",
    "ellipsomet": "ellipsometry",
    "reflectance spectroscop": "reflectance spectroscopy",
    "eis": "EIS", "electrochemical impedance": "EIS",
    "cyclic voltammetry": "CV", " cv ": "CV",
    "lsv": "LSV", "linear sweep voltammetry": "LSV",
    "tga": "TGA", "thermogravimetric": "TGA",
    "dsc": "DSC", "differential scanning calorimetry": "DSC",
    "dta": "DTA", "differential thermal analysis": "DTA",
    "bet": "BET", "brunauer": "BET",
    "dls": "DLS", "dynamic light scattering": "DLS",
    "sims": "SIMS", "secondary ion mass": "SIMS",
    "rbs": "RBS", "rutherford backscattering": "RBS",
    "pixe": "PIXE",
    "inductively coupled plasma": "ICP",
    "gdms": "GDMS", "gdoes": "GDOES", "libs": "LIBS",
    "dlts": "DLTS", "deep level transient": "DLTS",
    "dlos": "DLOS", "picts": "PICTS",
    "hall effect": "Hall effect",
    "four-point probe": "four-point probe",
    "four point probe": "four-point probe",
    "photoconductivity": "photoconductivity measurement",
    "edx": "EDX", "energy dispersive": "EDX",
    "eds": "EDS", "edax": "EDX", "wds": "WDS",
    "profilometr": "profilometry",
    "absorption spectroscop": "absorption spectroscopy",
}

SYNTH_KEYWORDS = {
    "molecular beam epitaxy":       "MBE",
    "chemical vapor deposition":    "CVD",
    "chemical vapour deposition":   "CVD",
    "metalorganic chemical vapor":  "MOCVD",
    "metalorganic cvd":             "MOCVD",
    "metal-organic cvd":            "MOCVD",
    "plasma-enhanced cvd":          "PECVD",
    "plasma enhanced cvd":          "PECVD",
    "low-pressure cvd":             "LPCVD",
    "atmospheric pressure cvd":     "APCVD",
    "atomic layer deposition":      "ALD",
    "plasma-enhanced ald":          "PEALD",
    "pulsed laser deposition":      "PLD",
    "physical vapor deposition":    "PVD",
    "physical vapour deposition":   "PVD",
    "rf sputtering":                "RF sputtering",
    "dc sputtering":                "DC sputtering",
    "magnetron sputtering":         "magnetron sputtering",
    "reactive sputtering":          "reactive sputtering",
    "thermal evaporation":          "thermal evaporation",
    "electron beam evaporation":    "e-beam evaporation",
    "e-beam evaporation":           "e-beam evaporation",
    "metal organic vapor phase":    "MOVPE",
    "metalorganic vapor phase":     "MOVPE",
    "halide vapor phase":           "HVPE",
    "spin coating":                 "spin coating",
    "sol-gel":                      "sol-gel",
    "spray pyrolysis":              "spray pyrolysis",
    "chemical bath deposition":     "CBD",
    "successive ionic layer":       "SILAR",
    "silar":                        "SILAR",
    "electrodeposition":            "electrodeposition",
    "electroplating":               "electroplating",
    "hydrothermal":                 "hydrothermal synthesis",
    "solvothermal":                 "solvothermal synthesis",
    "close space sublimation":      "CSS",
    "close-space sublimation":      "CSS",
}

CHAR_WHITELIST = {
    "SEM": "SEM", "FESEM": "FE-SEM", "HRSEM": "HR-SEM", "ESEM": "ESEM",
    "TEM": "TEM", "HRTEM": "HRTEM", "STEM": "STEM", "HAADF": "HAADF-STEM", "ETEM": "ETEM",
    "AFM": "AFM", "STM": "STM", "MFM": "MFM", "KPFM": "KPFM", "SKPM": "KPFM",
    "EFM": "EFM", "CAFM": "C-AFM", "PFM": "PFM",
    "FIB": "FIB", "FIBSEM": "FIB-SEM",
    "XRD": "XRD", "HRXRD": "HRXRD", "GIXRD": "GIXRD", "GIXD": "GIXRD",
    "SAXS": "SAXS", "WAXS": "WAXS", "GISAXS": "GISAXS", "GIWAXS": "GIWAXS",
    "RHEED": "RHEED", "LEED": "LEED", "SAED": "SAED", "CBED": "CBED",
    "EBSD": "EBSD", "XRF": "XRF", "SXRD": "SXRD", "XRR": "XRR",
    "XPS": "XPS", "ESCA": "XPS", "HAXPES": "HAXPES",
    "UPS": "UPS", "AES": "AES", "ARPES": "ARPES", "ARUPS": "ARUPS",
    "EXAFS": "EXAFS", "XANES": "XANES", "NEXAFS": "NEXAFS", "XAS": "XAS",
    "EDS": "EDS", "EDX": "EDX", "EDAX": "EDX", "WDS": "WDS", "EPMA": "EPMA",
    "EELS": "EELS", "HREELS": "HREELS", "EFTEM": "EFTEM",
    "FTIR": "FTIR", "ATR": "ATR-FTIR", "DRIFTS": "DRIFTS",
    "PL": "PL spectroscopy", "TRPL": "TRPL", "TCSPC": "TCSPC",
    "CL": "CL", "EL": "EL", "SHG": "SHG", "THG": "THG",
    "UVVIS": "UV-Vis", "UVVISNIR": "UV-Vis-NIR",
    "CV": "CV", "EIS": "EIS", "LSV": "LSV", "GEIS": "EIS", "PEIS": "EIS",
    "TGA": "TGA", "DSC": "DSC", "DTA": "DTA", "TMA": "TMA", "DTGA": "TGA",
    "BET": "BET", "DLS": "DLS",
    "SIMS": "SIMS", "TOFSIMS": "ToF-SIMS", "RBS": "RBS",
    "ERDA": "ERDA", "NRA": "NRA", "PIXE": "PIXE",
    "NMR": "NMR", "EPR": "EPR", "ESR": "ESR",
    "DLTS": "DLTS", "DLOS": "DLOS", "ICTS": "ICTS", "PICTS": "PICTS", "TSC": "TSC",
    "ICP": "ICP", "ICPMS": "ICP-MS", "ICPAES": "ICP-AES",
    "GDMS": "GDMS", "GDOES": "GDOES", "LIBS": "LIBS",
    "CLSM": "CLSM", "LEEM": "LEEM", "PEEM": "PEEM",
    "HALL": "Hall effect",
}

SYNTH_WHITELIST = {
    "CVD": "CVD", "MOCVD": "MOCVD", "MOVPE": "MOVPE",
    "PECVD": "PECVD", "LPCVD": "LPCVD", "APCVD": "APCVD",
    "RTCVD": "RTCVD", "HWCVD": "HWCVD", "ALCVD": "ALD",
    "OMCVD": "OMCVD", "OMVPE": "OMVPE",
    "VPE": "VPE", "HVPE": "HVPE", "OVPE": "OVPE",
    "MBE": "MBE", "MOMBE": "MOMBE", "GSMBE": "GSMBE",
    "CBE": "CBE", "PAMBE": "PA-MBE", "RFMBE": "RF-MBE",
    "ALD": "ALD", "PEALD": "PEALD", "SALD": "SALD", "TALD": "TALD",
    "PVD": "PVD", "PLD": "PLD", "IBD": "IBD", "IBAD": "IBAD",
    "CBD": "CBD", "SILAR": "SILAR", "CSS": "CSS", "IBS": "IBS",
}

SYNTH_SUFFIXES = {"CVD", "MBE", "PVD", "ALD", "VPE", "PLD"}


def classify_acronym(token: str):
    t = token.upper().replace("-", "").replace("_", "")
    if t in CHAR_WHITELIST:
        return "characterization", CHAR_WHITELIST[t]
    if t in SYNTH_WHITELIST:
        return "synthesis_method", SYNTH_WHITELIST[t]
    for suffix in SYNTH_SUFFIXES:
        if t.endswith(suffix) and len(t) > len(suffix):
            return "synthesis_method", token
    return None, None


# ============================================================
# DEFECT KEYWORD RULES
# ============================================================

DEFECT_KEYWORDS = {
    "selenium vacancy": "Se vacancy (V_Se)",
    "se vacancy":       "Se vacancy (V_Se)",
    "v_se":             "Se vacancy (V_Se)",
    "vse":              "Se vacancy (V_Se)",
    "s vacancy":        "S vacancy (V_S)",
    "sulfur vacancy":   "S vacancy (V_S)",
    "zinc vacancy":     "Zn vacancy (V_Zn)",
    "zn vacancy":       "Zn vacancy (V_Zn)",
    "v_zn":             "Zn vacancy (V_Zn)",
    "zinc interstitial":"Zn interstitial (Zn_i)",
    "zn interstitial":  "Zn interstitial (Zn_i)",
    "zni":              "Zn interstitial (Zn_i)",
    "antisite":         "antisite defect",
    "se_zn":            "Se antisite (Se_Zn)",
    "zn_se":            "Zn antisite (Zn_Se)",
    "deep level":       "deep-level defect",
    "deep trap":        "deep-level trap",
    "shallow donor":    "shallow donor",
    "shallow acceptor": "shallow acceptor",
    "donor level":      "donor level",
    "acceptor level":   "acceptor level",
    "native defect":    "native point defect",
    "point defect":     "point defect",
    "stacking fault":   "stacking fault",
    "dislocation":      "dislocation",
    "grain boundary":   "grain boundary defect",
}


# ============================================================
# REGEX PARAMETER PATTERNS
# ============================================================

def _fmt_doping(m):
    elem = (m.group("elem") or "").strip()
    val  = (m.group("val")  or "").strip()
    unit = (m.group("unit") or "").strip()
    return f"{elem} {val} {unit}".strip() if elem else f"{val} {unit}".strip()

def _fmt_temperature(m):
    val  = (m.group("val")  or "").strip()
    unit = (m.group("unit") or "°C").strip()
    return f"{val} {unit}"

def _fmt_bandgap(m):
    return f"{(m.group('val') or '').strip()} eV"

def _fmt_carrier_lifetime(m):
    val  = (m.group("val")  or "").strip()
    unit = (m.group("unit") or "").strip()
    return f"{val} {unit}"

def _fmt_carrier_density(m):
    val  = (m.group("val") or "").strip()
    exp  = (m.group("exp") or "").strip()
    unit = (m.group("unit") or "cm⁻³").strip()
    return f"{val}×10^{exp} {unit}" if exp else f"{val} {unit}"

def _fmt_photocurrent(m):
    val  = (m.group("val")  or "").strip()
    unit = (m.group("unit") or "").strip()
    return f"{val} {unit}"


PARAM_PATTERNS = [
    (
        re.compile(
            r"(?P<elem>[A-Z][a-z]?)"
            r"[\s\-]?(?:doped|doping|dopant|content|concentration)?"
            r"\s*(?:of\s*)?(?P<val>\d+(?:\.\d+)?(?:\s*[×x]\s*10\^?\d+)?)"
            r"\s*(?P<unit>at\.?%|mol\.?%|wt\.?%|%)",
            re.IGNORECASE
        ),
        _fmt_doping,
        "doping_parameter"
    ),
    (
        re.compile(
            r"(?P<elem>[A-Z][a-z]?)?\s*"
            r"(?P<val>\d+(?:\.\d+)?)\s*[×xX]\s*10\^?(?P<exp>\d+)"
            r"\s*(?P<unit>cm[⁻\-]?3|cm\^?\-?3)",
            re.IGNORECASE
        ),
        _fmt_carrier_density,
        "doping_parameter"
    ),
    (
        re.compile(
            r"(?:anneal(?:ing|ed)?|temper(?:ing|ed)|heat(?:ed|ing)|"
            r"sintered?|calcined?|post.?deposition)"
            r"[\s\w]{0,25}?"
            r"(?P<val>\d{2,4})\s*(?P<unit>°C|°F)",
            re.IGNORECASE
        ),
        _fmt_temperature,
        "annealing_condition"
    ),
    (
        re.compile(
            r"(?:anneal(?:ing|ed)?|temper(?:ing|ed)|heat(?:ed|ing)|"
            r"sintered?|calcined?|post.?deposition)"
            r"[\s\w]{0,25}?"
            r"(?P<val>\d{2,4})\s*(?P<unit>[Kk]elvin|K\b)",
            re.IGNORECASE
        ),
        _fmt_temperature,
        "annealing_condition"
    ),
    (
        re.compile(
            r"(?:band.?gap|E_?g|optical.?gap|energy.?gap)"
            r"[\s\w]{0,15}?"
            r"(?P<val>\d+\.\d+)\s*eV",
            re.IGNORECASE
        ),
        _fmt_bandgap,
        "optical_property"
    ),
    (
        re.compile(
            r"(?:PL|photoluminescence|emission|luminescence)"
            r"[\s\w]{0,20}?"
            r"(?P<val>\d{3,4})\s*(?P<unit>nm)",
            re.IGNORECASE
        ),
        lambda m: f"PL {m.group('val')} nm",
        "optical_property"
    ),
    (
        re.compile(
            r"(?:carrier\s+)?(?:lifetime|τ|tau|recombination\s+time)"
            r"[\s\w=~]{0,12}?"
            r"(?P<val>\d+(?:\.\d+)?)\s*"
            r"(?P<unit>ns|μs|us|ps|ms)\b",
            re.IGNORECASE
        ),
        _fmt_carrier_lifetime,
        "electrical_property"
    ),
    (
        re.compile(
            r"(?:resistivity|conductivity)"
            r"[\s\w=~]{0,12}?"
            r"(?P<val>\d+(?:\.\d+)?(?:\s*[×x]\s*10[⁻\-]?\d+)?)"
            r"\s*(?P<unit>Ω[·.]?cm|S[/.]cm|S\s*cm[⁻\-]1)",
        ),
        lambda m: f"{m.group('val')} {m.group('unit')}",
        "electrical_property"
    ),
    (
        re.compile(
            r"(?:photocurrent|dark\s+current|photo\s+response|responsivity)"
            r"[\s\w=~]{0,15}?"
            r"(?P<val>\d+(?:\.\d+)?(?:\s*[×x]\s*10[⁻\-]?\d+)?)"
            r"\s*(?P<unit>μA|mA|nA|A/W|mA/W)",
            re.IGNORECASE
        ),
        _fmt_photocurrent,
        "electrical_property"
    ),
]


# ============================================================
# SENTENCE-LEVEL RULE EXTRACTION
#
# Runs formula / keyword / regex rules against every sentence
# in the selected text.  Called once per paper on the full
# concatenated selected text — not per-chunk.
# ============================================================

def run_sentence_rules(full_text: str, query_terms: set) -> list:
    """
    Returns list of knowledge dicts extracted by deterministic rules.
    No LLM involved.
    """
    doc      = nlp(full_text[:500_000])   # spacy hard cap
    results  = []

    for sent in doc.sents:
        sentence = sent.text.strip()
        if len(sentence) < 40 or is_noise_sentence(sentence):
            continue

        s_lower = sentence.lower()

        # ── Token gate: acronyms + chemical formulas ──────────
        seen_tokens: set = set()
        for tok_m in re.finditer(r"\b([A-Z][A-Z0-9a-z\-]{1,})\b", sentence):
            tok = tok_m.group(1)
            if tok in seen_tokens:
                continue
            seen_tokens.add(tok)
            is_all_caps = bool(re.fullmatch(r"[A-Z][A-Z0-9\-]+", tok))
            if is_all_caps:
                cat, label = classify_acronym(tok)
                if cat is not None:
                    results.append({
                        "category":       cat,
                        "value":          label,
                        "sentence":       sentence,
                        "section_source": "rule",
                        "equation_id":    None,
                        "source_type":    "rule",
                        "confidence":     "medium",
                    })
            else:
                if re.search(r"[A-Z][a-z]", tok) and re.search(r"\d", tok):
                    if is_valid_value(tok, query_terms):
                        results.append({
                            "category":       "material",
                            "value":          tok,
                            "sentence":       sentence,
                            "section_source": "pattern",
                            "equation_id":    None,
                            "source_type":    "pattern",
                            "confidence":     "medium",
                        })
                elif re.fullmatch(r"(?:[A-Z][a-z]\d*){2,}", tok):
                    if is_valid_value(tok, query_terms):
                        results.append({
                            "category":       "material",
                            "value":          tok,
                            "sentence":       sentence,
                            "section_source": "pattern",
                            "equation_id":    None,
                            "source_type":    "pattern",
                            "confidence":     "medium",
                        })

        # ── Characterisation keywords (substring) ─────────────
        for kw, label in CHAR_KEYWORDS.items():
            if keyword_in(kw, s_lower):
                results.append({
                    "category":       "characterization",
                    "value":          label,
                    "sentence":       sentence,
                    "section_source": "rule",
                    "equation_id":    None,
                    "source_type":    "rule",
                    "confidence":     "medium",
                })
                break

        # ── Synthesis keywords (substring) ────────────────────
        for kw, label in SYNTH_KEYWORDS.items():
            if keyword_in(kw, s_lower):
                results.append({
                    "category":       "synthesis_method",
                    "value":          label,
                    "sentence":       sentence,
                    "section_source": "rule",
                    "equation_id":    None,
                    "source_type":    "rule",
                    "confidence":     "medium",
                })
                break

        # ── Defect keywords ───────────────────────────────────
        for kw, label in DEFECT_KEYWORDS.items():
            if keyword_in(kw, s_lower):
                results.append({
                    "category":       "defect_type",
                    "value":          label,
                    "sentence":       sentence,
                    "section_source": "rule",
                    "equation_id":    None,
                    "source_type":    "rule",
                    "confidence":     "medium",
                })
                break

        # ── Regex parameter patterns ──────────────────────────
        seen_params: set = set()
        for pattern, fmt_fn, category in PARAM_PATTERNS:
            for m in pattern.finditer(sentence):
                try:
                    value = fmt_fn(m).strip()
                except Exception:
                    continue
                if not value:
                    continue
                key = (category, value.lower())
                if key in seen_params:
                    continue
                seen_params.add(key)
                results.append({
                    "category":       category,
                    "value":          value,
                    "sentence":       sentence,
                    "section_source": "regex",
                    "equation_id":    None,
                    "source_type":    "pattern",
                    "confidence":     "medium",
                })

    return results


# ============================================================
# LLM CHUNK EXTRACTION
# ============================================================

LLM_EXTRACTION_PROMPT = """\
You are a scientific information extraction engine.

CRITICAL RULE: Extract ONLY information that is explicitly and directly stated in the passage \
below. Do NOT infer, generalise, or add any knowledge from outside the text. If a value is \
not clearly present in the passage, omit it entirely.

Return a JSON array. Each item must have exactly two keys:
  "category": one of
    material | synthesis_method | characterization | application |
    computational_method | defect_type | doping_parameter |
    annealing_condition | optical_property | electrical_property |
    growth_temperature | chamber_pressure | gas_flow | growth_duration |
    field_effect_mobility | on_off_ratio | threshold_voltage |
    subthreshold_swing | contact_resistance | photoresponsivity |
    growth_flux | ferroelectric_property | photodetector_metric
  "value": the extracted value as stated in the passage

QUALIFIER RULE — mandatory for ALL numeric values:
  Every numeric value MUST include its experimental context in square brackets.
  Format: "NUMBER UNIT [context]"
  Examples:
    "550 °C [CVD substrate temperature]"
    "0.3 Torr [chamber pressure]"
    "50 sccm [Ar carrier gas flow]"
    "30 min [growth duration]"
    "2.7 eV [optical bandgap]"
    "23 cm²/V·s [field-effect mobility]"
    "10^6 [FET on/off ratio]"
  Non-numeric values do NOT need brackets (e.g. "CVD", "XRD", "Se vacancy").

CATEGORY RULES:
- growth_temperature: substrate or zone temperature DURING deposition/growth only.
- chamber_pressure: process chamber pressure during growth (Torr, mbar, Pa).
- gas_flow: carrier gas, precursor gas, or flow rates during growth (sccm, slm).
- growth_duration: deposition or growth time (min, hours, seconds).
- annealing_condition: post-deposition annealing/sintering/calcination ONLY. NOT growth temp.
- optical_property: bandgap (eV), PL peak (nm/eV), absorption edge, dielectric constant /
  permittivity and refractive index (these may be dimensionless). NOT XPS photon energy.
- electrical_property: carrier lifetime, resistivity, carrier density ONLY.
- field_effect_mobility: FET carrier mobility in cm²/V·s.
- on_off_ratio: FET Ion/Ioff ratio (dimensionless).
- threshold_voltage: FET threshold voltage in V.
- subthreshold_swing: SS in mV/dec.
- contact_resistance: Rc in Ω·μm or kΩ·μm.
- photoresponsivity: R in A/W or mA/W.
- synthesis_method: technique name only (CVD, MBE, ALD etc). NOT parameters.
- growth_flux: source/beam flux, beam-equivalent pressure (BEP) of a source, flux ratio
  (e.g. Se:In), growth or deposition rate.
- ferroelectric_property: polarization (uC/cm2), piezoelectric coefficient d33 (pm/V),
  coercive field (kV/cm) or coercive voltage (V), Curie temperature.
- photodetector_metric: dark current, photocurrent, detectivity (Jones), external quantum
  efficiency, rise/decay (response) time. Responsivity goes in photoresponsivity.

OMIT if:
- The value is not explicitly stated in the passage.
- The value is generic, unknown, estimated, or described as "not specified".
- You are unsure of the category.

Passage:
\"\"\"
{passage}
\"\"\"

JSON array:"""


# ── Evidence for LLM items ───────────────────────────────────────────────────
# LLM items used to store chunk[:300] as their `sentence` — the first 300 chars
# of a 3,500-char chunk, whatever the value was. Measured 2026-09-11: of 254
# live LLM rows mapped back to their chunks, the stored `sentence` contained the
# value for only 59 (23%), while 225 (89%) of the values were in the chunk.
# GANESH reads that column as the context for each claim, so a review would
# cite text that does not say what it claims.
# Locate the sentence that actually carries the value; fall back to the old
# behaviour only when it cannot be found.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_EVIDENCE_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
EVIDENCE_MAX_CHARS = 500


def _has_number(n: str, text: str) -> bool:
    # '5' must not match inside '15' or '0.5'
    return re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", text) is not None


def evidence_sentence(chunk: str, value: str) -> str:
    core = re.sub(r"\s*\[[^\]]*\]\s*", " ", value or "").strip()
    nums = _EVIDENCE_NUM_RE.findall(core)
    sentences = [x.strip() for x in _SENT_SPLIT_RE.split(chunk or "") if x.strip()]
    if nums:
        hit = next((x for x in sentences if all(_has_number(n, x) for n in nums)), None)
        if hit is None and len(sentences) > 1:
            # a value can straddle a sentence boundary the splitter invented
            pairs = (a + " " + b for a, b in zip(sentences, sentences[1:]))
            hit = next((x for x in pairs if all(_has_number(n, x) for n in nums)), None)
    else:
        needle = core.lower()
        hit = next((x for x in sentences if needle and needle in x.lower()), None)
    if hit:
        return hit[:EVIDENCE_MAX_CHARS]
    return (chunk or "")[:300]


def run_llm_chunks(
    text: str,
    query_terms: set,
    service: LLMService,
    section_name: str = "body",
) -> list:
    """
    Chunk text and run LLM extraction on each chunk.
    Returns list of knowledge dicts.
    """
    chunks  = chunk_text(text)
    results = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        prompt = LLM_EXTRACTION_PROMPT.replace("{passage}", chunk[:4000])

        try:
            items = service.extract(prompt, stage="S5")
        except Exception as e:
            print(f"[S5][LLM] Chunk {i+1}/{len(chunks)} SKIPPED — error: {type(e).__name__}: {e}")
            print(f"[S5][LLM] Chunk preview: {chunk[:120]!r}")
            continue

        for item in items:
            val = item.get("value", "")
            if not is_valid_value(val, query_terms, item.get("category")):
                continue
            results.append({
                "category":       item.get("category", "material"),
                "value":          val,
                "sentence":       evidence_sentence(chunk, val),
                "section_source": section_name,
                "equation_id":    None,
                "source_type":    "llm",
                "confidence":     "high",
            })

    return results


# ============================================================
# AGGREGATION  (unchanged from previous S5)
# ============================================================

_UNIT_ALIASES: dict = {
    "°c": "°C", "deg c": "°C", "degrees c": "°C", "celsius": "°C",
    "°f": "°F", "k": "K", "kelvin": "K",
    "ev": "eV",
    "at.%": "at.%", "at%": "at.%",
    "mol.%": "mol.%", "mol%": "mol.%",
    "wt.%": "wt.%", "wt%": "wt.%",
    "cm-3": "cm⁻³", "cm^-3": "cm⁻³", "cm⁻3": "cm⁻³",
    "nm": "nm", "um": "μm", "μm": "μm",
    "ns": "ns", "us": "μs", "μs": "μs", "ps": "ps", "ms": "ms",
    "ω·cm": "Ω·cm", "ohm·cm": "Ω·cm",
    "s/cm": "S/cm",
    "μa": "μA", "ma": "mA", "na": "nA",
    "a/w": "A/W", "ma/w": "mA/W",
}

_PARAM_CATEGORIES = {
    "doping_parameter", "annealing_condition",
    "optical_property", "electrical_property", "defect_type",
    "growth_flux", "ferroelectric_property", "photodetector_metric",
}

_NUM_RE  = re.compile(r"(\d+(?:\.\d+)?)")
_UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s*([^\d\s].*)")


def _parse_numeric(value: str):
    nm = _NUM_RE.search(value)
    if not nm:
        return None, None
    num = float(nm.group(1))
    um  = _UNIT_RE.search(value)
    raw_unit  = um.group(1).strip() if um else ""
    canonical = _UNIT_ALIASES.get(raw_unit.lower(), raw_unit)
    return num, canonical


def _mode(values: list) -> str:
    if not values:
        return ""
    from collections import Counter
    return Counter(values).most_common(1)[0][0]


def _detect_trend(nums: list, category: str, unit: str) -> str:
    if len(nums) < 4:
        return ""
    ups   = sum(1 for a, b in zip(nums, nums[1:]) if b > a)
    dns   = sum(1 for a, b in zip(nums, nums[1:]) if b < a)
    total = ups + dns
    if total == 0:
        return ""
    _AXIS = {
        "annealing_condition": f"annealing temperature ({unit})",
        "doping_parameter":    f"doping concentration ({unit})",
        "optical_property":    f"optical property value ({unit})",
        "electrical_property": f"electrical property value ({unit})",
        "defect_type":         f"defect density ({unit})",
    }
    axis = _AXIS.get(category, f"value ({unit})")
    if ups / total > 0.60:
        return f"increasing {axis} trend across papers"
    if dns / total > 0.60:
        return f"decreasing {axis} trend across papers"
    return f"no consistent trend in {axis}"


def aggregate_parameters(results: list) -> dict:
    by_cat: dict = defaultdict(list)
    for item in results:
        cat = item.get("category", "")
        if cat in _PARAM_CATEGORIES:
            by_cat[cat].append(item)

    aggregation: dict = {}

    for cat, items in by_cat.items():
        all_values = [item["value"] for item in items if item.get("value")]
        paper_ids  = {item.get("paper_id") for item in items if item.get("paper_id")}

        parsed         = [_parse_numeric(v) for v in all_values]
        num_unit_pairs = [(n, u) for n, u in parsed if n is not None]

        if num_unit_pairs:
            unit_counts: dict = defaultdict(int)
            for _, u in num_unit_pairs:
                unit_counts[u] += 1
            dominant_unit    = max(unit_counts, key=unit_counts.get)
            same_unit_nums   = sorted(n for n, u in num_unit_pairs if u == dominant_unit)
            min_val  = f"{same_unit_nums[0]} {dominant_unit}"  if same_unit_nums else ""
            max_val  = f"{same_unit_nums[-1]} {dominant_unit}" if same_unit_nums else ""
            rng      = f"{same_unit_nums[0]}–{same_unit_nums[-1]} {dominant_unit}" \
                       if len(same_unit_nums) > 1 else min_val
            trend    = _detect_trend(same_unit_nums, cat, dominant_unit)
        else:
            dominant_unit = same_unit_nums = []
            min_val = max_val = rng = trend = ""

        aggregation[cat] = {
            "count":       len(all_values),
            "papers":      len(paper_ids),
            "values":      all_values,
            "numeric":     same_unit_nums if isinstance(same_unit_nums, list) else [],
            "unit":        dominant_unit,
            "min":         min_val,
            "max":         max_val,
            "range":       rng,
            "most_common": _mode(all_values),
            "trend":       trend,
        }

        print(
            f"[S5][AGG] {cat}: {len(all_values)} values "
            f"from {len(paper_ids)} papers | "
            f"range={rng or 'n/a'} | most_common={_mode(all_values)[:40]}"
        )

    return aggregation


# ============================================================
# EQUATION-CONTEXT KNOWLEDGE EXTRACTION  (preserved from previous S5)
# ============================================================

def extract_equation_knowledge(
    repo,
    paper_id: int,
    equations: list,
    query_terms: set,
    service: LLMService
) -> list:
    knowledge = []

    for eq in equations:
        context = " ".join(filter(None, [
            eq.get("context_before", ""),
            eq.get("context_after",  "")
        ])).strip()

        if not context or len(context.strip()) < 40 or is_noise_sentence(context):
            continue

        eq_db_id       = eq.get("db_id")
        eq_section_src = eq.get("section_source") or "body"
        ctx_lower      = context.lower()

        for kw, label in CHAR_KEYWORDS.items():
            if keyword_in(kw, ctx_lower):
                knowledge.append({
                    "category":       "characterization",
                    "value":          label,
                    "sentence":       context[:200],
                    "section_source": eq_section_src,
                    "equation_id":    eq_db_id,
                    "source_type":    "rule",
                    "confidence":     "medium",
                })
                break

        prompt = (
            "Extract ONLY the scientific property or material explicitly described by this "
            "equation context. Do NOT infer or add knowledge from outside the text.\n"
            "Return JSON array. Each item: "
            '{"category":"material|application|synthesis_method|characterization|'
            'defect_type|doping_parameter|annealing_condition|optical_property|'
            'electrical_property|growth_temperature|chamber_pressure|gas_flow|'
            'growth_duration|field_effect_mobility|on_off_ratio|threshold_voltage|'
            'subthreshold_swing|contact_resistance|photoresponsivity|'
            'growth_flux|ferroelectric_property|photodetector_metric",'
            '"value":"short name or value with unit [context]"}\n'
            "Numeric values MUST include context in brackets e.g. \"2.7 eV [optical bandgap]\".\n\n"
            f'Context: "{context[:250]}"\n\nJSON array:'
        )

        try:
            items = service.extract(prompt, stage="S5_eq")
            for item in items:
                val = item.get("value", "")
                if is_valid_value(val, query_terms, item.get("category")):
                    knowledge.append({
                        "category":       item["category"],
                        "value":          val,
                        "sentence":       context[:200],
                        "section_source": eq_section_src,
                        "equation_id":    eq_db_id,
                        "source_type":    "llm",
                        "confidence":     "medium",
                    })
        except Exception:
            pass

    return knowledge


# ============================================================
# MAIN TOOL — S5
# ============================================================

def extract_research_knowledge(repo, workflow_id, execution_attempt_id=None, **kwargs):

    # ── Fetch workflow config ─────────────────────────────────
    config = repo.fetch_one(
        """
        SELECT material, structure, focus, method, properties, characterization
        FROM WorkflowResearchConfig
        WHERE workflow_id = ?
        """,
        (workflow_id,)
    )

    query_parts = []
    if config:
        query_parts.extend([
            config["material"]         or "",
            config["structure"]        or "",
            config["focus"]            or "",
            config["method"]           or "",
            config["properties"]       or "",
            config["characterization"] or ""
        ])
    query_parts.append(
        "defect vacancy doping annealing carrier lifetime "
        "bandgap photocurrent recombination trap"
    )

    generic_terms_text = query_parts[-1]

    query        = " ".join(q for q in query_parts if q).strip() or "materials science research"
    query_terms  = extract_keywords(query)
    term_weights = build_term_weights(config, generic_terms_text)
    subj_elements = subject_elements(config["material"] if config else "")
    print(f"[S5] Query terms: {sorted(query_terms)}")
    print("[S5] Term weights: " + ", ".join(
        f"{t}={term_weights[t]:g}" for t in sorted(term_weights, key=lambda t: (-term_weights[t], t))))

    # ── Fetch papers ──────────────────────────────────────────
    papers = repo.fetch_all(
        """
        SELECT id, title, raw_text, doi, year
        FROM Paper
        WHERE workflow_id = ?
          AND status = ?
        """,
        (workflow_id, PIPELINE_STATUS.S5_INPUT)
    )
    print(f"[S5] Papers at status='{PIPELINE_STATUS.S5_INPUT}': {len(papers)}")

    # Model comes from DEFAULT_LOCAL_MODEL / SHANI_LOCAL_MODEL (see the
    # benchmark table in services/llm_service.py). Verified before the loop
    # so a missing model fails once, loudly, instead of 404-ing every chunk
    # while the stage still reports success.
    ensure_model_available()
    llm     = OllamaClient()
    service = LLMService(llm)

    run_normalisation(workflow_id)

    all_results = []
    processed   = 0
    skipped     = 0

    for paper in papers:
        paper    = dict(paper)
        paper_id = paper["id"]
        title    = paper["title"] or ""
        raw_text = paper["raw_text"] or ""

        # ── 1. Get sections ───────────────────────────────────
        sections = pc_repo.get_paper_content(repo, paper_id)  # {name: text} or None

        if not sections:
            # Fallback: segment raw_text
            if raw_text.strip():
                raw_clean = clean_scientific_text(raw_text)
                sections  = segment_raw_text(raw_clean)
                print(f"[S5] FALLBACK raw_text segmentation for paper {paper_id}: "
                      f"{len(sections)} pseudo-sections")
            else:
                # Dead paper — nothing to work with
                print(f"[S5] SKIP (no data): {title[:70]}")
                paper_repo.update_paper_status(repo, paper_id, PIPELINE_STATUS.S5_OUTPUT)
                skipped += 1
                continue

        # ── 2. Relevance score ────────────────────────────────
        # Use abstract section if available, else first 1000 chars of raw_text
        # abstract_density is 35% of the score, and it needs actual body text.
        # This used to read abstract -> preamble -> raw_text[:1000]; when S4
        # emitted neither an abstract nor a preamble section - 7 of 9 papers in
        # workflow 1 - it silently scored the paper on raw_text's first 1000
        # chars, which for a PDF is the title block, authors and affiliations.
        # Sampling every section instead measures the paper, not its cover page.
        text_sample = sections.get("abstract", "") or " ".join(
            (body or "")[:1500] for body in sections.values()
        ) or raw_text[:5000]
        score = compute_score(title, text_sample, query_terms, term_weights,
                              subj_elements)

        if score < MIN_PAPER_SCORE:
            hits = sorted(
                {w.lower() for w in re.findall(r"\b\w+\b", fold_digits(title))} & query_terms)
            print(f"[S5] SKIP (score={score:.3f}, title terms={hits}): {title[:70]}")
            paper_repo.update_paper_status(repo, paper_id, PIPELINE_STATUS.S5_OUTPUT)
            skipped += 1
            continue

        # ── 3. Classify + select sections ─────────────────────
        paper_type       = classify_paper(list(sections.keys()))
        selected         = select_sections(sections, paper_type)

        print(f"[S5] {paper_type} (score={score:.3f}) | "
              f"sections: {list(selected.keys())} | {title[:60]}")

        if not selected:
            print(f"[S5] No usable sections after filter: {title[:70]}")
            paper_repo.update_paper_status(repo, paper_id, PIPELINE_STATUS.S5_OUTPUT)
            skipped += 1
            continue

        # ── 4. Build full extraction text ─────────────────────
        # Concatenated for sentence-level rules; per-section for LLM.
        full_text = "\n\n".join(
            f"[{name}]\n{text}"
            for name, text in selected.items()
            if text.strip()
        )

        knowledge: list = []

        # ── 5. Sentence-level deterministic rules ─────────────
        rule_items = run_sentence_rules(full_text, query_terms)
        knowledge.extend(rule_items)
        print(f"[S5] Rule items: {len(rule_items)}")

        # ── 6. LLM extraction — per section ───────────────────
        llm_total = 0
        for sec_name, sec_text in selected.items():
            if not sec_text.strip():
                continue
            items = run_llm_chunks(sec_text, query_terms, service, section_name=sec_name)
            knowledge.extend(items)
            llm_total += len(items)
        print(f"[S5] LLM items: {llm_total}")

        # ── 7. Equation-context extraction ───────────────────
        equations = pc_repo.get_equations_for_paper(repo, paper_id)
        if equations:
            for eq in equations:
                eq["db_id"] = eq["id"]
            eq_knowledge = extract_equation_knowledge(
                repo, paper_id, equations, query_terms, service
            )
            knowledge.extend(eq_knowledge)
            print(f"[S5] Equation-context knowledge: {len(eq_knowledge)}")

        # ── 8. Category-aware filter ──────────────────────────
        before    = len(knowledge)
        knowledge = [
            item for item in knowledge
            if is_valid_knowledge(
                item["category"], item["value"], item.get("sentence", "")
            )
        ]
        after = len(knowledge)
        if before != after:
            print(f"[S5][FILTER] Rejected {before - after} invalid items for paper {paper_id}")

        if not knowledge:
            print(f"[S5] No valid knowledge extracted: {title[:70]}")
            paper_repo.update_paper_status(repo, paper_id, PIPELINE_STATUS.S5_OUTPUT)
            processed += 1
            continue

        # ── 9. Deduplicate within paper ───────────────────────
        seen_kv: set = set()
        deduped: list = []
        for item in knowledge:
            key = (item["category"], item["value"].lower().strip())
            if key not in seen_kv:
                seen_kv.add(key)
                deduped.append(item)
        knowledge = deduped
        print(f"[S5] After dedup: {len(knowledge)} items")

        # ── 10. Relations ─────────────────────────────────────
        relations = extract_relations(raw_text, knowledge)

        # ── 11. Insert to DB + vector index ───────────────────
        embedding_batch: list = []

        with repo.transaction() as cursor:
            # Replace, don't append (D5/D8, 2026-09-11). Re-running S5 on a
            # paper used to add a second copy of its rows and relations, so
            # every reset needed a hand-written DELETE - and relations were
            # always forgotten: 397 of 1,792 (22%) came from runs whose
            # knowledge had been deleted. The abstract-path rows (S2_75) are
            # a different stage and are left alone.
            cursor.execute(
                "DELETE FROM ResearchKnowledge WHERE paper_id = ? "
                "AND source_type IN ('llm', 'rule', 'pattern')", (paper_id,))
            cursor.execute("DELETE FROM ResearchRelation WHERE paper_id = ?", (paper_id,))
            for r in relations:
                try:
                    cursor.execute(
                        """
                        INSERT INTO ResearchRelation
                        (paper_id, subject, relation, object, created_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (paper_id, r["subject"], r["relation"], r["object"])
                    )
                except Exception:
                    pass

            for item in knowledge:
                cursor.execute(
                    """
                    INSERT INTO ResearchKnowledge (
                        paper_id, category, value, sentence,
                        section_source, equation_id,
                        source_type, confidence,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        paper_id,
                        item["category"],
                        item["value"],
                        item.get("sentence"),
                        item.get("section_source"),
                        item.get("equation_id"),
                        item.get("source_type"),
                        item.get("confidence"),
                    )
                )
                knowledge_id = cursor.lastrowid
                text_embed   = f"{item['category']} : {item['value']}"
                if item.get("sentence"):
                    text_embed += f" | {item['sentence'][:200]}"

                embedding_batch.append({
                    "knowledge_id": knowledge_id,
                    "paper_id":     paper_id,
                    "workflow_id":  workflow_id,
                    "category":     item["category"],
                    "value":        item["value"],
                    "sentence":     (item.get("sentence") or "")[:300],
                    "doi":          paper.get("doi", ""),
                    "title":        paper.get("title", ""),
                    "year":         paper.get("year"),
                    "text":         text_embed,
                })

        try:
            vector_service.add_knowledge_records(embedding_batch)
        except Exception as ve:
            print(f"[S5][VECTOR ERROR] {ve}")

        paper_repo.update_paper_status(repo, paper_id, PIPELINE_STATUS.S5_OUTPUT)

        for item in knowledge:
            item["paper_id"] = paper_id
        all_results.extend(knowledge)
        processed += 1
        print(f"[S5] Done paper {paper_id}: {len(knowledge)} knowledge items")

    print(f"\n[S5] Complete: {processed} processed | {skipped} skipped")

    # ── Reconcile the vector index with the DB (D4, 2026-09-11) ───
    try:
        import sqlite3
        from tools.normalise_paper_content import DB_PATH
        from tools.vector_index_maintenance import rebuild_vector_index
        with sqlite3.connect(DB_PATH) as _conn:
            rebuild_vector_index(_conn, vector_service)
    except Exception as ve:
        print(f"[S5][VECTOR REBUILD ERROR] {ve}")

    # ── Aggregation ───────────────────────────────────────────
    aggregation = {}
    try:
        aggregation = aggregate_parameters(all_results)
        print(f"[S5][AGG] Keys: {list(aggregation.keys())}")
    except Exception as agg_err:
        print(f"[S5][AGG] Failed (non-fatal): {agg_err}")

    return {
        "status": "success",
        "data": {
            "knowledge":   all_results,
            "aggregation": aggregation,
        },
        "error": None,
    }