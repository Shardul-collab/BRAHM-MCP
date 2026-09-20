"""
Tests for S5's paper relevance gate.

The bug these pin, measured on workflow 1 (In2Se3 review) on 2026-09-10:
the gate ADMITTED a BaBiO3-on-SrTiO3 paper at 0.035 and REJECTED two In2Se3
papers at 0.029, against a MIN_PAPER_SCORE of 0.03 — separating those
decisions by 0.006, with the sign backwards.

Three causes, one test class each:
  * Unicode subscripts. Publishers typeset "In₂Se₃"; S1 generates "in2se3".
    The title tokenised to 'in₂se₃' and matched nothing.
  * Flat term weights. The query is every WorkflowResearchConfig field
    concatenated, so 'beam' + 'epitaxy' (method) outweighed 'in2se3'
    (material) two to one.
  * Exact-token matching. 'mn2in2se5' and 'inse' are not query terms, so an
    In-Se compound and the parent binary scored the same as an unrelated oxide.
"""

import os
import sys

TOOLS_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TOOLS_PARENT not in sys.path:
    sys.path.insert(0, TOOLS_PARENT)

from tools.extract_research_knowledge import (      # noqa: E402
    MIN_PAPER_SCORE, build_term_weights, compute_score, fold_digits,
    material_affinity, subject_elements,
)

# The real workflow-1 configuration.
CONFIG = {
    "material": "In2Se3",
    "structure": None,
    "focus": ("structural polymorphism and phase identification across "
              "alpha, beta, beta-prime, gamma and wurtzite phases"),
    "method": "molecular beam epitaxy, chemical vapour deposition",
    "properties": "ferroelectricity, photoresponsivity, bandgap",
    "characterization": "XRD, Raman, TEM, XPS, AFM",
}
GENERIC = ("defect vacancy doping annealing carrier lifetime bandgap "
           "photocurrent recombination trap")

WEIGHTS = build_term_weights(CONFIG, GENERIC)
TERMS = set(WEIGHTS)
ELEMENTS = subject_elements(CONFIG["material"])


def score(title, body=""):
    return compute_score(title, body, TERMS, WEIGHTS, ELEMENTS)


# ─── Unicode subscripts ───────────────────────────────────────────────────────

def test_subscript_digits_fold_to_ascii():
    assert fold_digits("In₂Se₃") == "In2Se3"
    assert fold_digits("Bi₂Te₃ and MoS⁴") == "Bi2Te3 and MoS4"


def test_subscripted_title_scores_like_the_ascii_one():
    unicode_title = "Fabrication of ᵞ-In₂Se₃-Based Photodetector"
    ascii_title = "Fabrication of g-In2Se3-Based Photodetector"
    assert score(unicode_title) == score(ascii_title)


# ─── Term weighting by config field ───────────────────────────────────────────

def test_material_outweighs_technique_terms():
    """'in2se3' comes from `material`; 'beam'/'epitaxy' from `method`."""
    assert WEIGHTS["in2se3"] > WEIGHTS["beam"]
    assert WEIGHTS["in2se3"] > WEIGHTS["epitaxy"]
    assert WEIGHTS["beta"] > WEIGHTS["xrd"]        # focus beats characterization


def test_unknown_terms_do_not_crash_scoring():
    assert score("A paper about nothing in particular") >= 0


def test_missing_config_is_survivable():
    assert build_term_weights(None, GENERIC)       # generic terms still weighted
    assert subject_elements("") == set()
    assert subject_elements(None) == set()


# ─── Subject-element affinity ─────────────────────────────────────────────────

def test_subject_elements_parsed_from_formula():
    assert subject_elements("In2Se3") == {"In", "Se"}
    assert subject_elements("alpha-In2Se3") == {"In", "Se"}


def test_compound_of_subject_elements_is_full_affinity():
    assert material_affinity("MBE of Mn2In2Se5 van der Waals Layers", ELEMENTS) == 1.0
    assert material_affinity("Polytype formation in InSe films", ELEMENTS) == 1.0


def test_unrelated_oxide_has_no_affinity():
    assert material_affinity(
        "Epitaxial growth of BaBiO3 thin films on SrTiO3(001) and MgO",
        ELEMENTS) == 0.0


def test_partial_element_overlap_scores_between():
    partial = material_affinity("Twin-free Bi2Se3 layers", ELEMENTS)
    assert 0 < partial < 1.0


# ─── The regression, end to end ───────────────────────────────────────────────

def test_on_topic_papers_pass_and_off_topic_paper_does_not():
    """
    The exact set that exposed the bug. Every In-Se paper must clear the
    threshold; the BaBiO3 paper must not.
    """
    on_topic = [
        "Fabrication of ᵞ-In₂Se₃-Based Photodetector Using RF Magnetron Sputtering",
        "Thickness-dependent Dielectric Constant of Few-layer In2Se3 Nano-flakes",
        "Molecular Beam Epitaxy of Mn2In2Se5 van der Waals Layers",
        "Mixed polytype/polymorph formation and its effects in InSe",
        "Wafer-scale fabrication of fast two-dimensional beta-In2Se3 photodetectors",
    ]
    off_topic = "Epitaxial growth of BaBiO3 thin films on SrTiO3(001) and MgO by molecular beam epitaxy"

    for title in on_topic:
        assert score(title) >= MIN_PAPER_SCORE, f"wrongly skipped: {title}"
    assert score(off_topic) < MIN_PAPER_SCORE


def test_on_topic_margin_is_not_marginal():
    """
    The old gate separated pass from skip by 0.006. Decisions that close are
    noise, not judgement — require a real gap.
    """
    worst_on_topic = min(
        score("Molecular Beam Epitaxy of Mn2In2Se5 van der Waals Layers"),
        score("Mixed polytype/polymorph formation and its effects in InSe"),
    )
    off_topic = score(
        "Epitaxial growth of BaBiO3 thin films on SrTiO3(001) and MgO by molecular beam epitaxy")
    assert worst_on_topic > 2 * off_topic


# ─── Section selection: front matter by content, preamble never by name/size ──
#
# "preamble" was skipped by name (paper 20: 84% of the paper never extracted),
# then by size, which dropped the abstracts of papers 13, 15 and 17 with their
# author blocks. Since D1 (2026-09-11) normalisation labels front matter by
# content as "front_matter"; that label is what gets skipped.

from tools.extract_research_knowledge import select_sections   # noqa: E402


def test_front_matter_is_skipped():
    sections = {"front_matter": "Ryan Trice1, Mingyu Yu2", "methods": "x" * 20000}
    assert set(select_sections(sections, "EXPERIMENTAL")) == {"methods"}


def test_small_preamble_is_extracted_from():
    """A short preamble holding an abstract is content, whatever its size."""
    sections = {"preamble": "Abstract—Metal chalcogenide In2Se3 thin films were prepared.",
                "methods": "x" * 20000}
    assert set(select_sections(sections, "EXPERIMENTAL")) == {"preamble", "methods"}


def test_large_preamble_is_extracted_from():
    sections = {"preamble": "x" * 14237, "methods": "y" * 2686}
    assert set(select_sections(sections, "EXPERIMENTAL")) == {"preamble", "methods"}


def test_preamble_alone_is_never_dropped():
    """Dropping it would leave S5 with nothing at all for this paper."""
    assert set(select_sections({"preamble": "short"}, "EXPERIMENTAL")) == {"preamble"}


def test_ordinary_skips_still_apply():
    sections = {"references": "r", "acknowledgements": "a",
                "introduction": "i", "results": "res"}
    assert set(select_sections(sections, "EXPERIMENTAL")) == {"results"}
    assert set(select_sections(sections, "REVIEW")) == {"introduction", "results"}
