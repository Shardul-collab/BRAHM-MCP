# tools/s4_vision/s4c_semantic_structuring.py
# ============================================================
# S4C — SEMANTIC STRUCTURING AGENT
#
# Responsibilities:
#   1. Consume all plain_text + title regions from S4B
#   2. Reconstruct section hierarchy (title → section body)
#   3. Map equations to their enclosing section
#   4. Build sections dict + latex_sections dict
#      (same format as current S4 extract_sections_by_blocks)
#   5. Prepare semantic chunks for S5
#
# Output schema matches current S4's output exactly so
# pc_repo.create_paper_content() calls remain unchanged.
# ============================================================

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .s4a_document_vision import Region, DocumentVisionResult, TEXT_LABELS


# ── Known section heading keywords (from current S4) ─────────
SECTION_KEYWORDS = [
    "abstract", "introduction", "background",
    "related work", "literature review",
    "methods", "methodology", "experimental", "materials",
    "synthesis", "fabrication", "characterization",
    "results", "discussion", "results and discussion",
    "conclusion", "conclusions", "summary",
    "acknowledgement", "acknowledgements", "acknowledgment",
    "references", "bibliography",
    "supplementary", "appendix",
    "theory", "theoretical",
    "computational", "simulation",
    "electrochemical", "optical", "structural",
    "morphological", "thermal", "mechanical",
]

# Compiled pattern for fast heading detection
_HEADING_PATTERN = re.compile(
    r"^(?:\d+\.?\s+)?(" +
    "|".join(re.escape(k) for k in SECTION_KEYWORDS) +
    r")",
    re.IGNORECASE,
)

# Maximum characters in a title region to be treated as a heading
MAX_HEADING_LEN = 120


# ────────────────────────────────────────────────────────────
# SECTION DETECTION
# ────────────────────────────────────────────────────────────

def _is_section_heading(region: Region) -> bool:
    """
    Return True if the region looks like a section heading.
    Criteria:
      1. YOLO label is "title", OR
      2. label is "plain_text" AND text matches a known keyword
         AND text is short enough to be a heading.
    """
    text = (region.text or "").strip()
    if not text:
        return False

    if region.label == "title":
        # Title regions from YOLO are already headings
        return len(text) <= MAX_HEADING_LEN

    if region.label == "plain_text":
        if len(text) > MAX_HEADING_LEN:
            return False
        return bool(_HEADING_PATTERN.match(text))

    return False


def _normalize_section_name(heading: str) -> str:
    """
    Convert a raw heading string to a canonical section key.
    e.g. "2. Results and Discussion" → "results_and_discussion"
    """
    # Strip leading numbering
    text = re.sub(r"^\d+\.?\s*", "", heading.strip())
    text = text.lower().strip()
    # Collapse whitespace/special chars to underscore
    text = re.sub(r"[\s\-/&]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "section"


# ────────────────────────────────────────────────────────────
# MAIN SECTION RECONSTRUCTION
# ────────────────────────────────────────────────────────────

def _build_sections_from_regions(
    regions: List[Region],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Walk through regions in reading order and reconstruct
    section text.

    Returns
    -------
    sections : dict[section_name → text_content]
    latex_sections : dict[section_name → latex_text]
        Currently latex_text == text_content (LaTeX rendering
        is handled by S4B pix2tex for equations, not for prose).
    """
    sections:       Dict[str, str] = {}
    latex_sections: Dict[str, str] = {}

    current_section = "preamble"
    current_texts:  List[str] = []

    def _flush():
        nonlocal current_section, current_texts
        body = " ".join(current_texts).strip()
        if body:
            if current_section in sections:
                sections[current_section] += " " + body
            else:
                sections[current_section] = body
            latex_sections[current_section] = sections[current_section]
        current_texts = []

    for region in regions:
        if region.label not in TEXT_LABELS:
            continue
        if region.is_garbage:
            continue

        text = (region.text or "").strip()
        if not text:
            continue

        if _is_section_heading(region):
            _flush()
            current_section = _normalize_section_name(text)
        else:
            current_texts.append(text)

    _flush()   # flush last section

    return sections, latex_sections


# ────────────────────────────────────────────────────────────
# EQUATION → SECTION MAPPING
# ────────────────────────────────────────────────────────────

def _assign_equation_sections(
    equations: List[Dict],
    sections: Dict[str, str],
) -> List[Dict]:
    """
    For each equation, find which section it most likely belongs to
    by searching for the equation's context in section bodies.
    Mirrors the mapping logic in current S4's main loop.
    """
    for eq in equations:
        context = (
            eq.get("context_before", "") + " " + eq.get("context_after", "")
        ).lower()

        for sec_name in sections:
            if sec_name.lower()[:6] in context:
                eq["section_source"] = sec_name
                break

    return equations


# ────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ────────────────────────────────────────────────────────────

def run_semantic_structuring(
    vision_result: DocumentVisionResult,
) -> Dict:
    """
    S4C entry point.

    Builds the final structured output from the vision result.

    Returns
    -------
    A dict with:
        "sections"        : dict[section_name → text]
        "latex_sections"  : dict[section_name → text]
        "raw_text"        : str (all text joined)
        "figures"         : list[dict]  (from S4B)
        "tables"          : list[dict]  (from S4B)
        "equations"       : list[dict]  (from S4B, section mapped)
        "used_method"     : str ("vision" | "vision_fallback")

    The caller (extract_paper_content.py) writes these to the
    DB using the existing repo functions — no schema changes.
    """
    regions   = vision_result.regions
    figures   = getattr(vision_result, "extracted_figures",   [])
    tables    = getattr(vision_result, "extracted_tables",    [])
    equations = getattr(vision_result, "extracted_equations", [])

    print(f"[S4C] Semantic structuring — {len(regions)} regions")

    # ── Build sections ────────────────────────────────────────
    sections, latex_sections = _build_sections_from_regions(regions)

    # ── Fallback: if no sections found ───────────────────────
    if not sections:
        print("[S4C] No sections found — emitting raw_text only")
        all_text = " ".join(
            (r.text or "").strip()
            for r in regions
            if r.label in TEXT_LABELS and not r.is_garbage and r.text
        )
        sections       = {"raw_text": all_text[:5000]} if all_text else {}
        latex_sections = dict(sections)

    # ── Map equations to sections ─────────────────────────────
    equations = _assign_equation_sections(equations, sections)

    # ── Build flat raw_text ───────────────────────────────────
    raw_text = " ".join(v for v in sections.values() if v).strip()

    method = (
        "vision_fallback"
        if vision_result.fallback_used
        else "vision"
    )

    print(
        f"[S4C] Done — "
        f"sections={list(sections.keys())} "
        f"method={method}"
    )

    return {
        "sections":       sections,
        "latex_sections": latex_sections,
        "raw_text":       raw_text,
        "figures":        figures,
        "tables":         tables,
        "equations":      equations,
        "used_method":    method,
    }
