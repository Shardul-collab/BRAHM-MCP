"""A row cannot vouch for a material its evidence sentence never mentions.

From the 50-claim TRL-5 audit (2026-09-19). One of the two UNSUPPORTED claims
read "α-In2Se3 on B substrates previously displayed anti-phase domains due to
substrate height variations [E236]". Paper 6 says that about **Bi2Se3**. The
verifier passed it because Evidence.formula_text included the row's own `value`,
which is literally "α-In2Se3" — BRAHM's label for the row, not a sentence the
paper wrote — and, failing that, the paper's title+abstract formulas, which say
the paper is an In2Se3 paper somewhere.

Measured on document 2: dropping `value` alone flags 3 sentences, dropping the
paper-level formulas too flags 4, and the 4th is that claim. Sentence-only flags
19 and is too strict.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ganesh.writing.grounding import (  # noqa: E402
    Evidence, check_sentence, is_formula, is_roman_numeral)


def _ev(eid, value, sentence, title="", paper_formulas=""):
    return Evidence(eid, 6, "material", value, sentence,
                    title=title, paper_formulas=paper_formulas)


def test_a_rows_own_value_does_not_vouch_for_the_material():
    """E236 exactly: value says α-In2Se3, the sentence is about Bi2Se3."""
    ev = {"E236": _ev("E236", "α-In2Se3",
                      "B substrates have previously shown anti-phase domains due to "
                      "variations in substrate height and Bi2Se3 grown on flat InP(111)B "
                      "has shown twin boundaries.")}
    c = check_sentence("α-In2Se3 on B substrates previously displayed anti-phase "
                       "domains driven by substrate height variations [E236].", ev)
    assert c.status == "unsupported"
    assert "In2Se3" in c.missing


def test_a_paper_level_mention_does_not_license_a_sentence_level_claim():
    """Paper 6 IS an In2Se3 paper, which is why dropping `value` alone was not
    enough — the abstract's formulas still covered it."""
    ev = {"E236": _ev("E236", "α-In2Se3",
                      "Bi2Se3 grown on flat InP(111)B has shown twin boundaries.",
                      title="Molecular Beam Epitaxy of Twin-Free Bi2Se3",
                      paper_formulas="In2Se3 Bi2Se3")}
    assert check_sentence("α-In2Se3 showed anti-phase domains [E236].", ev).status == "unsupported"


def test_the_sentence_itself_still_vouches_for_its_own_material():
    ev = {"E1": _ev("E1", "twin boundaries",
                    "Bi2Se3 grown on flat InP(111)B has shown twin boundaries.")}
    assert check_sentence("Bi2Se3 showed twin boundaries [E1].", ev).status == "supported"


def test_the_paper_title_still_vouches():
    """A title states what the reported work is, so it stays in scope."""
    ev = {"E1": _ev("E1", "1.28 eV", "The band gap was 1.28 eV.",
                    title="Epitaxial growth of β-InSe thin films")}
    assert check_sentence("β-InSe films show a band gap of 1.28 eV [E1].", ev).status == "supported"


@pytest.mark.parametrize("tok", ["II", "IV", "VI", "IX", "XII", "III"])
def test_roman_numerals_are_not_chemical_formulas(tok):
    """Every Roman numeral character is also an element symbol, so "Phase II"
    parsed as iodine-iodine and "Phase IV" as iodine-vanadium — a false
    unsupported the moment the subject check was tightened."""
    assert is_roman_numeral(tok)
    assert not is_formula(tok)


@pytest.mark.parametrize("tok", ["In2Se3", "Bi2Se3", "Al2O3", "MoS2", "SnO2", "VC"])
def test_real_formulas_still_parse(tok):
    """VC is vanadium carbide and is NOT a valid Roman numeral, so the exclusion
    must not swallow it."""
    assert is_formula(tok)


def test_phase_numbering_no_longer_breaks_a_sentence():
    ev = {"E1": _ev("E1", "300 °C", "The peak temperature was 300 °C.")}
    c = check_sentence("The peak temperature for Phase II is 300 °C [E1].", ev)
    assert c.status == "supported", c.missing
