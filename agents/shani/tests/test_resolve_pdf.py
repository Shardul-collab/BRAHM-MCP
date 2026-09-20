"""
Tests for S2_5 PDF resolution.

These exist because of the most expensive bug found in SHANI so far.

lookup_arxiv() ran a free-text arXiv search on the paper title, took the
first result unconditionally, and never compared it to what was asked for.
For workflow 2 it returned arXiv 1411.4413 — a CERN LHCb particle-physics
paper — as the match for two different indium-selenide papers, and that
unverified guess was ranked at priority 2 while each paper's own known-good
URL sat at priority 5. download_papers tries candidates in ascending
priority, so both papers downloaded the LHCb PDF and never touched their
correct link. 6.pdf and 13.pdf are byte-identical on disk.

Cost, measured on the live corpus:
  - 635 knowledge rows from paper 6 alone: 58% of the whole corpus and
    87% of the material axis, every value an LHCb author surname
  - 152,332 of S5's 472,807 input characters spent on the wrong paper

The similarity numbers below are from the real titles: the LHCb title
scores 0.27 and 0.29 against the two it was returned for, while genuine
formatting variants score 0.93-1.00. The 0.75 threshold sits in a very
wide gap, so these tests are not brittle.
"""

import pytest

from tools.resolve_pdf import titles_match, _normalise_title


LHCB = ("Observation of the rare $B^0_s\\to\\mu^+\\mu^-$ decay from the "
        "combined analysis of CMS and LHCb data")

# The two real titles that arXiv returned the LHCb paper for.
POISONED = [
    "Formation of Beta-Indium Selenide Layers Grown via Selenium "
    "Passivation of InP(111)B Substrate",
    "Molecular Beam Epitaxy of Twin-Free Bi2Se3 and Sb2Te3 on In2Se3/InP(111)",
]


@pytest.mark.parametrize("requested", POISONED)
def test_lhcb_paper_is_rejected(requested):
    """The exact regression. Neither title may ever match the LHCb paper."""
    assert not titles_match(requested, LHCB)


@pytest.mark.parametrize("requested,returned", [
    # subscript markup: publisher titles carry <sub>, arXiv does not
    ("Molecular Beam Epitaxy of Mn<sub>2</sub>In<sub>2</sub>Se<sub>5</sub>",
     "Molecular Beam Epitaxy of Mn2In2Se5"),
    # unicode vs ascii, case, and arXiv's wrapped-line whitespace
    ("Growth of Nanometer-Thick γ-InSe on Si(111) 7 × 7 by Molecular Beam Epitaxy",
     "Growth of nanometer-thick gamma-InSe on Si(111) 7x7 by\n  molecular beam epitaxy"),
    ("Hybrid ferroelectric tunnel junctions: State-of-the-art, challenges and opportunities",
     "Hybrid ferroelectric tunnel junctions: state of the art, challenges and\n  opportunities"),
])
def test_genuine_variants_still_match(requested, returned):
    """Formatting differences must not cost us a real match."""
    assert titles_match(requested, returned)


def test_empty_titles_never_match():
    assert not titles_match("", LHCB)
    assert not titles_match("Some Real Title", "")
    assert not titles_match(None, None)


def test_normalisation_strips_markup_and_punctuation():
    assert _normalise_title("Mn<sub>2</sub>In<sub>2</sub>Se<sub>5</sub>") == "mn2in2se5"
    assert _normalise_title("State-of-the-art!") == "state of the art"


def test_threshold_gap_is_wide():
    """
    Guards the threshold itself. If a future change narrows the gap between
    true and false matches, this fails before a wrong paper gets downloaded.
    """
    import difflib
    worst_true = min(
        difflib.SequenceMatcher(None, _normalise_title(a), _normalise_title(b)).ratio()
        for a, b in [
            ("Molecular Beam Epitaxy of Mn<sub>2</sub>In<sub>2</sub>Se<sub>5</sub>",
             "Molecular Beam Epitaxy of Mn2In2Se5"),
            ("Growth of Nanometer-Thick γ-InSe on Si(111) 7 × 7 by Molecular Beam Epitaxy",
             "Growth of nanometer-thick gamma-InSe on Si(111) 7x7 by molecular beam epitaxy"),
        ]
    )
    best_false = max(
        difflib.SequenceMatcher(None, _normalise_title(t), _normalise_title(LHCB)).ratio()
        for t in POISONED
    )
    assert best_false < 0.50 < worst_true, (
        f"true matches bottom out at {worst_true:.2f}, false matches top out "
        f"at {best_false:.2f} — the threshold no longer separates them"
    )
