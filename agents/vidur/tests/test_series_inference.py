"""Series/sample inference from a set of filenames.

There is no fixed convention; structure comes from what varies across the set.
"""
import pytest
from series import group, tokenise


def _named(stems):
    return {g["series"]: [m["sample"] for m in g["members"]] for g in group(stems)}


def test_one_varying_token_becomes_the_sample_name():
    assert _named(["ZnSe_S1_UV", "ZnSe_S2_UV", "ZnSe_S3_UV"]) == {
        "ZnSe_UV": ["S1", "S2", "S3"]}


def test_two_series_in_one_folder_stay_separate():
    out = _named(["ZnSe_S1_UV", "ZnSe_S2_UV", "SnO2_S1_XRD", "SnO2_S2_XRD"])
    assert sorted(out.values()) == [["S1", "S2"], ["S1", "S2"]]
    assert len(out) == 2


def test_unrelated_files_do_not_become_a_series():
    """Grouping on token count alone put four unrelated names in one series."""
    stems = ["532nm_191216_Si_200mu", "SMC_1_Initial_RT",
             "XRD_918_16_10", "m_54313_om2th_10"]
    out = group(stems)
    assert len(out) == 4
    assert all(not g["inferred"] for g in out)


@pytest.mark.parametrize("formula", ["In2Se3", "SnO2", "TiO2", "Bi2Se3"])
def test_chemical_formulas_survive_tokenisation(formula):
    """An earlier version split trailing digits to turn "sample1" into
    ("sample", "1"), which also turned SnO2 into ("SnO", "2"). There is no rule
    that separates a sample index from a formula, so tokens stay whole."""
    assert formula in tokenise(f"{formula}_450C_S1")


def test_multiple_varying_tokens_are_joined():
    out = _named(["In2Se3_450C_S1", "In2Se3_450C_S2", "In2Se3_500C_S1"])
    assert list(out) == ["In2Se3"]
    assert sorted(out["In2Se3"]) == ["450C_S1", "450C_S2", "500C_S1"]


def test_names_written_without_separators_are_segmented():
    """Shardul's photodetector files carry their structure in the letter/digit
    alternation: 16znse03iv150 is 16 | znse | 03 | iv | 150. Without splitting
    those, every such file was a series of one -- measured on the real folder,
    2026-09-19."""
    assert tokenise("16znse03iv150") == ["16", "znse", "03", "iv", "150"]
    out = _named(["16znse03iv150", "16znse03rt150"])
    assert list(out.values()) == [["iv", "rt"]]


def test_segmentation_only_applies_when_there_were_no_separators():
    """A punctuated name keeps its tokens whole -- that is what stops SnO2
    becoming ("SnO", "2")."""
    assert tokenise("SnO2_S1_XRD") == ["SnO2", "S1", "XRD"]
    assert tokenise("In2Se3_450C_S1") == ["In2Se3", "450C", "S1"]
