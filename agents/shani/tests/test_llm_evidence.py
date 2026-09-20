"""
S5 LLM items must carry the sentence that evidences them.

Until 2026-09-11 every LLM item stored chunk[:300] — the first 300 characters of
a ~3,500-char chunk — as its `sentence`. Of 254 live LLM rows mapped back to
their chunks, only 59 (23%) had a `sentence` containing their value, though 225
(89%) of the values were in the chunk. GANESH reads that column as each claim's
context.
"""
import os
import sys

SHANI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SHANI not in sys.path:
    sys.path.insert(0, SHANI)

from tools.extract_research_knowledge import evidence_sentence, is_valid_knowledge  # noqa: E402

CHUNK = (
    "To understand the growth mechanism for BBO thin films on both STO and MgO "
    "bulk substrates, a series of samples was prepared. The films were grown at "
    "700°C under a background pressure of 3E-6 Torr. After growth the samples were "
    "annealed at 800°C for 30 minutes in oxygen. The RMS roughness was 0.52 nm for "
    "the film grown at 700°C and 1.5 nm at 650°C. Bulk In2Se3 has a dielectric "
    "constant of 17, falling to 6.29 in the thinnest flakes."
)


def test_numeric_value_gets_its_own_sentence():
    s = evidence_sentence(CHUNK, "800°C for 30 minutes [post-deposition annealing]")
    assert s.startswith("After growth") and "800" in s and "30" in s


def test_qualifier_numbers_are_ignored():
    # the bracket holds context, not evidence; '700' in it must not steer the match
    s = evidence_sentence(CHUNK, "0.52 nm [RMS roughness at 700°C]")
    assert "0.52 nm" in s


def test_number_does_not_match_inside_a_larger_number():
    s = evidence_sentence(CHUNK, "1.5 nm [roughness]")
    assert "1.5 nm at 650" in s
    assert evidence_sentence("Grown at 15 K. Measured at 5 K.", "5 K") == "Measured at 5 K."


def test_dimensionless_value():
    s = evidence_sentence(CHUNK, "6.29 [dielectric constant of ultra-thin In2Se3]")
    assert "6.29" in s and "dielectric constant" in s


def test_text_value():
    s = evidence_sentence("Samples were grown by MBE. Films were imaged by AFM.", "AFM")
    assert s == "Films were imaged by AFM."


def test_unlocatable_value_falls_back_to_chunk_head():
    assert evidence_sentence(CHUNK, "42 GPa [bulk modulus]") == CHUNK[:300]


# Real values stored in workflow 1 before 2026-09-11.
PLACEHOLDERS = [
    ("field_effect_mobility", "not specified [FET carrier mobility]"),
    ("photoresponsivity", "not explicitly stated [photoresponsivity]"),
    ("chamber_pressure", "14.56 [not specified unit, likely Torr or Pa]"),
    ("growth_duration", "MnSe layer duration [not explicitly stated]"),
    ("synthesis_method", "CVD [inferred from the context of growth and RHEED monitoring]"),
    ("growth_temperature", "deposition temperature [not explicitly stated]"),
]


def test_placeholders_are_rejected_in_every_category():
    for cat, val in PLACEHOLDERS:
        assert not is_valid_knowledge(cat, val, ""), (cat, val)


def test_real_values_still_pass():
    for cat, val in [("growth_temperature", "690-750 °C [substrate temperature during deposition]"),
                     ("gas_flow", "30-200 sccm [Ar carrier gas flow]"),
                     ("synthesis_method", "van der Waals epitaxy"),
                     ("chamber_pressure", "5 × 10^-10 Torr [base pressure of MBE chamber]")]:
        assert is_valid_knowledge(cat, val, ""), (cat, val)
