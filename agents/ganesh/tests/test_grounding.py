import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ganesh.writing.grounding import Evidence, check_text, summarise, strip_failing, claims_in  # noqa: E402

EV = {
    "E1": Evidence("E1", 13, "growth_temperature", "280 °C [substrate temperature]",
                   "The substrate was held at 280 °C during Mn2In2Se5 growth."),
    "E2": Evidence("E2", 20, "optical_property", "17 [bulk In2Se3 dielectric constant]",
                   "Bulk In2Se3 has a dielectric constant of 17."),
}


def statuses(text, **kw):
    return [c.status for c in check_text(text, EV, **kw)]


def test_supported_number_and_formula():
    assert statuses("Mn2In2Se5 layers were grown at 280 °C [E1].") == ["supported"]


def test_invented_number_is_caught():
    c = check_text("Growth proceeded at 300 °C [E1].", EV)[0]
    assert c.status == "unsupported" and c.missing == ["300"]


def test_number_must_come_from_the_cited_item_not_any_item():
    assert statuses("The dielectric constant is 17 [E1].") == ["unsupported"]


def test_uncited_claim_and_bad_citation():
    assert statuses("The dielectric constant of bulk In2Se3 is 17.") == ["uncited"]
    assert statuses("Films were grown at 280 °C [E9].") == ["bad_citation"]


def test_subject_formula_can_be_allowed_and_narrative_passes():
    assert statuses("In2Se3 has several polymorphs.", always_allowed={"In2Se3"}) == ["narrative"]
    assert statuses("These results point in the same direction.") == ["narrative"]


def test_formula_digits_are_not_numbers():
    nums, forms = claims_in("β-In2Se3 on Al2O3 [E1]")
    assert nums == [] and set(forms) == {"In2Se3", "Al2O3"}


def test_summary_and_strip():
    text = "Mn2In2Se5 grew at 280 °C [E1]. It melts at 900 °C [E1]. Bulk In2Se3 has a dielectric constant of 17 [E2]."
    checks = check_text(text, EV)
    s = summarise(checks)
    assert s["supported"] == 2 and s["unsupported"] == 1 and abs(s["grounding_rate"] - 2 / 3) < 1e-9
    assert "900" not in strip_failing(text, checks)


def test_latex_formulas_from_gemma_are_normalised():
    from ganesh.writing.grounding import normalise_output
    raw = r"$\text{In}_2\text{Se}_3$ layers were formed at 505 $^\circ$C and In$_2$Se$_3$ on $\text{Al}_{2}\text{O}_{3}$."
    out = normalise_output(raw)
    assert "In2Se3 layers" in out and "on Al2O3" in out and "505 °C" in out.replace("  ", " "), out


# 2026-09-11 pm: digits inside a formula in the cited evidence are not values
def test_formula_digits_in_evidence_do_not_ground_a_bare_number():
    # E2 contains "In2Se3" but no value 3 or 2
    assert statuses("The films were 3 nm thick [E2].") == ["unsupported"]
    assert statuses("Two phases, 2 in total, coexist [E2].") == ["unsupported"]


def test_formula_must_match_a_whole_formula_in_evidence():
    # "In2Se" is a substring of "In2Se3" but not a formula the evidence states
    assert statuses("In2Se films were grown at 280 °C [E1].") == ["unsupported"]
    assert statuses("In2Se3 has a dielectric constant of 17 [E2].") == ["supported"]


def test_abstract_check_masks_formula_digits():
    from ganesh.writing.assemble import grounded_abstract
    secs = [{"section_name": "S", "content": "In2Se3 films were grown on mica [E2]."}]
    kept, dropped = grounded_abstract(lambda p, max_tokens=500: "In2Se3 films 3 nm thick were grown. In2Se3 films were grown on mica.", "T", secs)
    assert dropped == ["In2Se3 films 3 nm thick were grown."]


def test_subject_formula_is_checked_in_cited_sentences():
    ev = {"E9": Evidence("E9", 14, "optical_property", "1.28 eV [band gap for β-InSe]",
                         "A band gap of 1.28 eV was found for β-InSe.", title="Polytypes in InSe films")}
    c = check_text("In2Se3 films have a band gap of 1.28 eV [E9].", ev, {"In2Se3"})[0]
    assert c.status == "unsupported" and c.missing == ["In2Se3"]
    # uncited framing may still name the subject
    assert check_text("In2Se3 is a layered semiconductor.", ev, {"In2Se3"})[0].status == "narrative"
    # the paper title can carry the material a row's sentence does not repeat
    ev2 = {"E1": Evidence("E1", 9, "growth_temperature", "300 °C", "The substrate was held at 300 °C.",
                          title="Polymorph selection of MBE-grown In2Se3")}
    assert check_text("In2Se3 was grown at 300 °C [E1].", ev2, {"In2Se3"})[0].status == "supported"


def test_normalise_greek_latex():
    from ganesh.writing.grounding import normalise_output
    assert normalise_output(r"γ and/or \kappa-In2Se3, \tau_{spv} = 14 ms") == "γ and/or κ-In2Se3, τspv = 14 ms"
