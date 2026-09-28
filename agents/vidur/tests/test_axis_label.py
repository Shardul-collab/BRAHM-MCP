"""The x-axis label a file declares, and what VIDUR does when it declares none.

2026-09-27 regression. 11 of the 37 RRD-A patterns (Claude outputs/5. RRD-A)
were refused with "axis is q as declared by the file" although none holds q.
Three causes, each reproduced below with the header lines copied from those
files (data rows shortened):

1. x_label() took the first word of the last text line before the data --
   titles, instrument metadata and beamline comments became "axis labels";
2. it treated a line as data when its first word parsed as a number, so the
   real header "2 Theta,Intensity" and the comment "#      7983" ended the
   header search;
3. _parse_ascii() set axis_is_two_theta=False for any label it did not
   recognise, and every consumer reads False as "the file declares q".
"""
import numpy as np
import pytest

from extractor import extract
from parsers import xrd
from parsers._ascii import header_label
import line_profile as lp
import prep


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _parsed(path):
    return xrd.parse(extract(path))


# Header shapes that are 2theta, or carry no column header at all.
TWO_THETA_CASES = {
    # RRD-A 003: a title line, no column header.
    "title.txt": "Industrial TiO2 - Su\n2.0      1578\n2.05     1513\n2.1      1378\n",
    # RRD-A 007: Bruker EVA export, quoted metadata line, decimal-comma data.
    "bruker_eva.txt": "'Id:\tZnFe2O4\tComment:\t\tOperator:\tAdministrator\tAnode:\tCu\n"
                      "10\t1,296070088\t\t\n10,0157538\t1,33703782\t\t\n10,0315\t1,3929\t\t\n",
    # RRD-A 015: section title above a real header whose first word is "2".
    "multiblock.csv": "﻿Eu11Zn4Sn2As12,,,,\nPXRD,,,Resistivity vs T,\n"
                      "2 Theta,Intensity,,Temperature (K),Resistivity (mOhm m)\n"
                      "20,117,,326,5160.21\n20.01,114,,349.15,4606.52\n20.02,122,,373,4061.98\n",
    # RRD-A 019: Spanish label, semicolon separator, decimal comma.
    "spanish.csv": "2Teta (°);Intensity (a.u)\n15,01;48\n15,03;58\n15,05;35\n",
    # RRD-A 026/028: ALBA MSPD .xye; motor names in comments, then a point count.
    "alba.xye": "# /beamlines/bl04/DATA/21LMO_f00169.xye : mythproc.py merge\n"
                "# Wave = 0.8258  CalPath /beamlines/bl04/Si13keVnose2_f00002.cal\n"
                "# icurr 249.9177  mocoIn 6.6333E-06  mocoOut 0.8258\n"
                "# dif_tth 13.9499  energyc 11000.0154\n"
                "#      9942\n"
                "1.5496 120.0 11.0\n1.5556 118.0 10.9\n1.5616 121.0 11.0\n",
    # Recognised 2theta spellings.
    "theta_symbol.xy": "2θ I\n20 1\n21 2\n22 3\n",
    "two_theta_words.csv": "Two Theta,Counts\n20,1\n21,2\n22,3\n",
    "motor.xy": "dif_tth intensity\n20 1\n21 2\n22 3\n",
    # The 09-26 reproduction: a free-text comment is not a column header.
    "freetext.xy": "# columns copied verbatim from the instrument export\n20 1\n21 2\n22 3\n",
}


@pytest.mark.parametrize("name", sorted(TWO_THETA_CASES))
def test_a_file_that_does_not_declare_q_is_not_reported_as_q(tmp_path, name):
    d = _parsed(_write(tmp_path, name, TWO_THETA_CASES[name]))
    assert d["metadata"]["axis_is_two_theta"] is True, d["metadata"]
    assert d["axis_name"] == "2Theta"


def test_the_header_is_read_from_the_line_above_the_data(tmp_path):
    def hl(name):
        return header_label(_write(tmp_path, name, TWO_THETA_CASES[name]))
    assert hl("multiblock.csv") == ("2 Theta", True)
    assert hl("spanish.csv")[0].startswith("2Teta") and hl("spanish.csv")[1]
    assert hl("two_theta_words.csv") == ("Two Theta", True)
    assert hl("title.txt") == ("Industrial", False)     # a title, not a header
    assert hl("alba.xye") == (None, False)               # "#      9942"


@pytest.mark.parametrize("text", [
    "Q,I(Q)\n0.5 100\n1.0 200\n1.5 150\n",
    "# q I\n0.5 100\n1.0 200\n1.5 150\n",
    "q (1/A)\tI\n0.5\t100\n1.0\t200\n1.5\t150\n",
    # Spaces inside the labels, so the header does not split into two fields:
    # the q spelling is still honoured rather than defaulting to 2theta.
    "q (1/A)   I (a.u.)\n0.5 100\n1.0 200\n1.5 150\n",
])
def test_a_declared_q_axis_is_still_q(tmp_path, text):
    d = _parsed(_write(tmp_path, "q.xy", text))
    assert d["metadata"]["axis_is_two_theta"] is False
    assert d["axis_name"] == "q"


# RRD-A 013: a real header whose x label is neither 2theta nor q.
UNDECLARED = ("Fe2TiO5_x\tFe2TiO5_y\n10\t0\n10.02\t10.14\n10.04\t-22.45\n")


def test_an_unrecognised_label_is_undeclared_not_q(tmp_path):
    d = _parsed(_write(tmp_path, "xrd.txt", UNDECLARED))
    assert d["metadata"]["axis_is_two_theta"] is None
    assert d["metadata"]["axis_label_in_file"] == "Fe2TiO5_x"
    assert d["axis_name"] == "Fe2TiO5_x"


def test_line_profile_asks_about_an_undeclared_axis_instead_of_calling_it_q(tmp_path):
    p = _write(tmp_path, "xrd.txt", UNDECLARED)
    with pytest.raises(lp.AxisUndeclared) as exc:
        lp.load_xrd(p)
    assert "q as declared" not in str(exc.value)
    rep = lp.run(p, {})
    assert rep["status"] == "needs_input"
    assert "axis_is_two_theta" in rep["questions"]


def test_the_caller_can_state_an_undeclared_axis_is_two_theta(tmp_path):
    x, y, md = lp.load_xrd(_write(tmp_path, "xrd.txt", UNDECLARED), axis_is_two_theta=True)
    assert md["axis_is_two_theta"] is True
    assert md["axis_is_two_theta_source"] == "caller"
    assert len(x) == 3


def test_the_caller_cannot_overrule_a_declared_q_axis(tmp_path):
    p = _write(tmp_path, "q.xy", "Q,I(Q)\n0.5 100\n1.0 200\n1.5 150\n")
    with pytest.raises(ValueError, match="q as declared"):
        lp.load_xrd(p, axis_is_two_theta=True)


def test_prep_does_not_call_an_undeclared_axis_q():
    cols, notes = prep.derive("XRD", np.array([20.0, 21.0]), np.array([100.0, 200.0]),
                              {"wavelength_a": 1.5406, "axis_is_two_theta": None})
    assert "d_A" not in cols and "q_invA" not in cols
    assert not any("q as declared" in n for n in notes)
    assert any("neither 2theta nor q" in n for n in notes)


# 2026-09-27 (Shardul): no column header -> 2theta automatically, but recorded as
# an assumption and warned about wherever results are reported.

def _headerless_pattern(tmp_path):
    x = np.arange(20.0, 60.0, 0.02)
    y = 50.0 + sum(h * np.exp(-0.5 * ((x - c) / 0.08) ** 2)
                   for c, h in ((28.0, 3000.0), (47.0, 1500.0)))
    y = np.random.default_rng(1).poisson(y).astype(float)
    p = tmp_path / "ZnSe_S1_XRD.xy"
    p.write_text("\n".join(f"{a:.3f} {b:.0f}" for a, b in zip(x, y)))
    return str(p)


def test_no_header_is_two_theta_but_marked_assumed(tmp_path):
    md = _parsed(_headerless_pattern(tmp_path))["metadata"]
    assert md["axis_is_two_theta"] is True
    assert md["axis_assumed"] == "2theta"


def test_a_declared_two_theta_is_not_marked_assumed(tmp_path):
    md = _parsed(_write(tmp_path, "h.xy", "2Theta I\n20 1\n21 2\n22 3\n"))["metadata"]
    assert md["axis_is_two_theta"] is True
    assert "axis_assumed" not in md


def test_line_profile_report_warns_the_axis_was_assumed(tmp_path):
    rep = lp.run(_headerless_pattern(tmp_path), {"wavelength_a": 1.5406, "ka2_ratio": 0})
    assert rep["status"] == "ok"
    assert rep["axis_assumed"] == "2theta"
    assert rep["caveats"][0].startswith("x axis ASSUMED to be 2theta")


def test_batch_notes_the_assumed_axis(tmp_path):
    import batch
    _headerless_pattern(tmp_path)
    # A headerless file scores "uncertain" in detection, so batch needs the
    # technique stated; the axis assumption must still be reported.
    out = batch.process(tmp_path, {"params": {"wavelength_a": 1.5406},
                                   "techniques": {"ZnSe_S1_XRD.xy": "XRD"}})
    assert len(out["written"]) == 1, out
    assert any("assumed 2theta" in n["note"] for n in out["notes"]), out
