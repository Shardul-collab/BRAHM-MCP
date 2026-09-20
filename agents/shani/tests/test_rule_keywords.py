"""
Tests for the rule-based extractors' keyword and formula matching (S2_75, S5).

Two defects, both measured on the live In2Se3 corpus on 2026-09-11:

  * Substring keyword matching. `keyword in text` let "bet" match "between",
    "tem" match "temperature", "sem" match "semiconductor" and "led" match
    "controlled". Every one of the 9 full-text papers carried a BET row; none
    used BET. 28 of 92 abstract-path rows had no whole-word support.

  * re.findall on a pattern with a capturing group inside a repetition returns
    the LAST repetition, not the match: 'In2Se3' -> 'Se3', 'InSe' -> 'Se',
    'GaAs' -> 'As'. That is where the corpus's split-formula materials came
    from.
"""

import os
import sys

SHANI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SHANI not in sys.path:
    sys.path.insert(0, SHANI)

from tools.keyword_match import keyword_in                     # noqa: E402
from tools.extract_lightweight_knowledge import extract_by_rules, is_noise  # noqa: E402


class TestKeywordIn:
    def test_short_acronyms_do_not_match_inside_words(self):
        for kw, text in [("bet", "the spacing between layers"),
                         ("bet", "beta-in2se3"),
                         ("tem", "substrate temperature of 300 c"),
                         ("tem", "the mbe system"),
                         ("sem", "a layered semiconductor"),
                         ("led", "precisely controlled thickness"),
                         ("ups", "hydroxyl groups"),
                         ("eds", "hundreds of nanometres"),
                         ("tga", "the substrate was outgassed"),
                         ("aes", "purchased from alfa aesar"),
                         ("vse", "the inverse relation")]:
            assert not keyword_in(kw, text), (kw, text)

    def test_short_acronyms_match_as_words(self):
        for kw, text in [("tem", "cross-sectional tem images"),
                         ("tem", "haadf-stem and tem."),
                         ("tem", "tems of both films"),
                         ("sem", "sem/eds mapping"),
                         ("xrd", "(xrd) patterns"),
                         ("bet", "bet surface area")]:
            assert keyword_in(kw, text), (kw, text)

    def test_long_stems_keep_prefix_behaviour(self):
        assert keyword_in("infrared spectroscop", "fourier-transform infrared spectroscopy")
        assert keyword_in("photoluminescen", "photoluminescence spectra")
        assert keyword_in("transmission electron", "transmission electron microscopy")

    def test_padded_and_punctuated_keywords_keep_substring_behaviour(self):
        assert keyword_in(" cv ", "measured by cv curves")
        assert keyword_in("v_se", "defect v_se forms")


class TestS2_75Rules:
    # Real title + abstract fragments from workflow 1.
    def _values(self, title, abstract, category):
        return {r["value"] for r in extract_by_rules(title, abstract) if r["category"] == category}

    def test_title_formulas_are_captured_whole(self):
        mats = self._values("Epitaxial growth of γ-InSe and α, β, and γ-In2Se3 on ε-GaSe", "", "material")
        assert {"InSe", "In2Se3", "GaSe"} <= mats
        assert "Se" not in mats and "Se3" not in mats

    def test_quaternary_and_iii_v_formulas(self):
        mats = self._values("Molecular Beam Epitaxy of Mn2In2Se5 van der Waals Layers", "", "material")
        assert "Mn2In2Se5" in mats and "Se5" not in mats
        mats = self._values("InSe films grown by molecular beam epitaxy on GaAs(111)B", "", "material")
        assert "GaAs" in mats and "As" not in mats

    def test_no_keyword_hits_from_ordinary_words(self):
        abstract = ("The band gap decreases between the monolayer and bilayer; the "
                    "growth temperature was precisely controlled and the film is a "
                    "semiconductor, as revealed by transport.")
        found = {r["value"] for r in extract_by_rules("Growth of a layered film", abstract)}
        assert not found & {"BET", "TEM", "SEM", "LED"}, found


class TestS2_75PaperGate:
    def test_authors_describing_their_own_work_is_not_noise(self):
        # paper 16's real abstract opening; it was skipped outright
        abstract = ("Ferroelectric materials have received great attention in the field of "
                    "data storage. In this article, we realized the molecular beam epitaxial "
                    "(MBE) growth of beta-In2Se3 films on bilayer graphene substrates.")
        assert not is_noise(abstract)
        assert not is_noise("In this paper, we use aberration-corrected STEM.")

    def test_boilerplate_is_still_noise(self):
        assert is_noise("(c) 2023 Elsevier B.V. All rights reserved.")


class TestS2_75Evidence:
    """D7 (2026-09-11): abstract rows carry evidence; the LLM sees the whole abstract."""

    ABSTRACT = ("In this article, we realized the molecular beam epitaxial (MBE) growth of β–In2Se3 films "
                "on bilayer graphene substrates. Combining in situ scanning tunneling microscopy (STM) and "
                "ARPES measurements, we found that the four-monolayer β–In2Se3 is a semiconductor. "
                "The band gap of In2Se3 film decreases after potassium doping on its surface. "
                "Raman spectroscopy confirmed the beta phase.")

    def test_rule_rows_store_the_matching_sentence(self):
        rows = extract_by_rules("Surface Reconstruction in Epitaxial β–In2Se3 Thin Films", self.ABSTRACT)
        raman = [r for r in rows if r["category"] == "characterization" and "raman" in r["value"].lower()]
        assert raman and raman[0]["sentence"] == "Raman spectroscopy confirmed the beta phase."
        mats = [r for r in rows if r["category"] == "material"]
        assert mats and all(r["sentence"] for r in mats)

    def test_llm_sees_the_full_abstract_and_values_get_evidence(self):
        from tools.extract_lightweight_knowledge import extract_by_llm

        class Stub:
            prompts = []
            def extract(self, prompt, stage="S5"):
                self.prompts.append(prompt)
                return [{"category": "optical_property", "value": "2.1 eV [band gap before doping]"},
                        {"category": "application", "value": "2D ferroelectric devices"},
                        {"category": "growth_duration", "value": "not specified [growth time]"}]

        long_abstract = self.ABSTRACT + " " + "The films were uniform over the wafer. " * 40 + \
            "A band gap of 2.1 eV was measured before doping."
        stub = Stub()
        out = extract_by_llm("Title", long_abstract, stub, set())
        assert "2.1 eV was measured before doping" in stub.prompts[0]      # no 120-word cut
        vals = {r["value"]: r for r in out}
        assert "not specified [growth time]" not in vals
        assert vals["2.1 eV [band gap before doping]"]["sentence"] == "A band gap of 2.1 eV was measured before doping."
        assert "2D ferroelectric devices" in vals


def test_title_tokens_go_through_the_material_validator():
    rows = extract_by_rules("Fabrication of In2Se3 Photodetector Using RF Magnetron Sputtering", "")
    mats = {r["value"] for r in rows if r["category"] == "material"}
    assert "In2Se3" in mats and "RF" not in mats
