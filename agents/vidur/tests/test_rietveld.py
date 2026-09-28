"""VIDUR Rietveld (FullProf) -- strict CIF identity, the PCR round trip, and one
real refinement.

Decisions these encode (Shardul, 2026-09-27): FullProf is the engine; a CIF
whose identity does not match the phase named is REJECTED, never refined;
RRD-A 006, 024 (Cu CIF) and 027 (precursor CIFs) are the real cases.
"""
from pathlib import Path

import numpy as np
import pytest

import rietveld as rv

REPO = Path(__file__).resolve().parents[3]
RRDA = REPO / "Claude outputs/5. RRD-A/RRD_A"
LOGS = REPO / "agents/shani/logs/vidur_rietveld_2026-09-27"


def _rrda(rel):
    p = RRDA / rel
    if not p.is_file():
        pytest.skip(f"{rel} not present")
    return str(p)


PNMA_CIF = """data_test
_chemical_formula_sum 'Fe Li O4 P'
_cell_length_a 10.33
_cell_length_b 6.00
_cell_length_c 4.69
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_cell_formula_units_Z 4
_symmetry_space_group_name_H-M 'P n m a'
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
-x+1/2,-y,z+1/2
-x,y+1/2,-z
x+1/2,-y+1/2,-z+1/2
-x,-y,-z
x+1/2,y,-z+1/2
x,-y+1/2,z
-x+1/2,y+1/2,z+1/2
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_B_iso_or_equiv
Li1 Li+1 0 0 0 1 1.0
Fe1 Fe+2 0.2822 0.25 0.9747 1 0.5
P1 P+5 0.0948 0.25 0.4182 1 0.4
O1 O-2 0.0966 0.25 0.7428 1 0.6
O2 O-2 0.4571 0.25 0.2060 1 0.6
O3 O-2 0.1657 0.0465 0.2848 1 0.6
"""


def _cif(tmp_path, text, name="t.cif"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_formula_parser():
    assert rv.parse_formula("LiFePO4") == {"Li": 1, "Fe": 1, "P": 1, "O": 4}
    assert rv.parse_formula("Fe Li O4 P") == {"Fe": 1, "Li": 1, "O": 4, "P": 1}
    assert rv.parse_formula("Ca5(PO4)3(OH)") == {"Ca": 5, "P": 3, "O": 13, "H": 1}
    assert rv.same_composition(rv.parse_formula("NiFe2O4"), rv.parse_formula("Fe2NiO4"))


def test_site_multiplicities_come_from_the_symmetry_operators(tmp_path):
    c = rv.read_cif(_cif(tmp_path, PNMA_CIF))
    mult = {a["label"]: a["mult"] for a in c.atoms}
    assert mult == {"Li1": 4, "Fe1": 4, "P1": 4, "O1": 4, "O2": 4, "O3": 8}
    assert c.cell[0] == pytest.approx(10.33) and c.z == 4


def test_a_consistent_cif_passes(tmp_path):
    v = rv.validate_phase(rv.Phase("LiFePO4", _cif(tmp_path, PNMA_CIF), "LiFePO4"))
    assert v["ok"], v["rejections"]


def test_a_cif_for_another_compound_is_rejected(tmp_path):
    v = rv.validate_phase(rv.Phase("FePO4", _cif(tmp_path, PNMA_CIF), "FePO4"))
    assert not v["ok"] and "elements differ" in v["rejections"][0]


def test_a_cif_whose_sites_contradict_its_formula_is_rejected(tmp_path):
    bad = PNMA_CIF.replace("Li1 Li+1 0 0 0 1 1.0\n", "")
    v = rv.validate_phase(rv.Phase("LiFePO4", _cif(tmp_path, bad), "LiFePO4"))
    assert not v["ok"] and any("atom sites give" in r for r in v["rejections"])


def test_a_source_id_mismatch_is_rejected(tmp_path):
    v = rv.validate_phase(rv.Phase("LiFePO4", _cif(tmp_path, PNMA_CIF, "1101111.cif"),
                                   "LiFePO4", source_id="1234567"))
    assert not v["ok"] and "source id mismatch" in v["rejections"][0]


def test_non_physical_biso_is_a_warning_and_is_reported(tmp_path):
    t = PNMA_CIF.replace("Li1 Li+1 0 0 0 1 1.0", "Li1 Li+1 0 0 0 1 -3.1")
    v = rv.validate_phase(rv.Phase("LiFePO4", _cif(tmp_path, t), "LiFePO4"))
    assert v["ok"] and any("Li1: Biso -3.1" in w for w in v["warnings"])


# ── the RRD-A decisions (Shardul, 2026-09-27) ─────────────────────────────────

def test_rrda_006_is_rejected_on_provenance():
    v = rv.validate_phase(rv.Phase(
        "NiFe2O4", _rrda("CACR-RRDA-006_F/Fe2NiO4_mp-22684_computed.cif"), "NiFe2O4",
        source_id="mp-24941"))
    assert not v["ok"] and any("mp-22684" in r for r in v["rejections"])


def test_rrda_024_cu_cif_is_rejected_for_al():
    v = rv.validate_phase(rv.Phase("Al", _rrda("CACR-RRDA-024_F/9013014.cif"), "Al"))
    assert not v["ok"] and "elements differ" in v["rejections"][0]


@pytest.mark.parametrize("cif", ["2225444.cif", "9007690.cif"])
def test_rrda_027_precursor_cifs_are_rejected_for_maricite(cif):
    v = rv.validate_phase(rv.Phase("NaMnPO4", _rrda(f"CACR-RRDA-027_F/{cif}"), "NaMnPO4"))
    assert not v["ok"]


def test_rrda_026_models_are_accepted():
    for cif, f in (("1101111.cif", "LiFePO4"), ("1525576.cif", "FePO4")):
        v = rv.validate_phase(rv.Phase(f, _rrda(f"CACR-RRDA-026_F/{cif}"), f,
                                       source_id=cif[:-4]))
        assert v["ok"], v["rejections"]


def test_refine_refuses_to_run_with_a_rejected_phase(tmp_path):
    x = np.arange(10, 20, 0.01)
    d = tmp_path / "p.xye"
    d.write_text("\n".join(f"{a} 100 10" for a in x))
    r = rv.refine(str(d), [rv.Phase("FePO4", _cif(tmp_path, PNMA_CIF), "FePO4")],
                  rv.Instrument(wavelength=1.5406), str(tmp_path / "w"))
    assert r["status"] == "rejected" and not (tmp_path / "w" / "vidur.pcr").exists()


def _scripted_refine(tmp_path, monkeypatch, chi2_by_run):
    """refine() with FullProf replaced by a script of chi2 values, one per run."""
    runs = iter(chi2_by_run)
    wrote = []

    def fake_run(wd, code, datafile, timeout=300):
        (Path(wd) / f"{code}.sum").write_text("x")
        wrote.append((Path(wd) / f"{code}.pcr").read_text())
        return "END"

    def fake_sum(path):
        c = next(runs)
        return {"chi2": c, "Rwp": c, "Rp": c, "Rexp": 1.0, "phases": []}

    monkeypatch.setattr(rv, "run_fp2k", fake_run)
    monkeypatch.setattr(rv, "parse_sum", fake_sum)
    monkeypatch.setattr(rv, "read_state", lambda pcr, st: st)
    x = np.arange(10, 20, 0.01)
    d = tmp_path / "p.xye"
    d.write_text("\n".join(f"{a} 100 10" for a in x))
    r = rv.refine(str(d), [rv.Phase("LiFePO4", _cif(tmp_path, PNMA_CIF), "LiFePO4")],
                  rv.Instrument(wavelength=1.5406), str(tmp_path / "w"))
    return r, wrote


def test_a_step_that_raises_chi2_is_not_accepted(tmp_path, monkeypatch):
    # steps 1-2 improve; step 3 (background) raises chi2 twice; 4-6 improve;
    # one extra run re-computes the accepted model at the end
    r, wrote = _scripted_refine(tmp_path, monkeypatch, [10, 9, 12, 11, 8, 7, 6, 6])
    s3 = r["steps"][2]
    assert s3["outcome"] == "chi2_increased" and s3["groups_dropped"] == ["background"]
    assert s3["chi2"] == 9                                   # the model kept is step 2's
    assert "background" not in r["steps"][3]["free"]        # not freed again
    assert r["status"] == "ok" and r["groups_dropped"] == ["background"]
    assert " 0.30  0.30" in wrote[3]                         # the retry was damped


def test_a_diverged_step_is_retried_damped_and_can_recover(tmp_path, monkeypatch):
    r, _ = _scripted_refine(tmp_path, monkeypatch, [10, 9, float("nan"), 8, 7, 6, 5])
    assert r["steps"][2]["outcome"] == "accepted_after_damped_retry"
    assert r["steps"][2]["first_attempt"]["outcome"] == "diverged"
    assert r["status"] == "ok" and r["groups_dropped"] == []


def test_the_mcp_tool_returns_the_rejection(tmp_path):
    import asyncio
    import json
    import sys
    sys.path.insert(0, str(REPO))
    vidur_mod = pytest.importorskip("brahm.agents.vidur", reason="needs the mcp package")
    x = np.arange(10, 20, 0.01)
    d = tmp_path / "p.xye"
    d.write_text("\n".join(f"{a} 100 10" for a in x))
    out = asyncio.run(vidur_mod.vidur_rietveld({
        "data_files": [str(d)], "work_dir": str(tmp_path / "w"), "wavelength_a": 1.5406,
        "phases": [{"name": "FePO4", "cif": _cif(tmp_path, PNMA_CIF), "formula": "FePO4"}]}))
    text = json.dumps(out)
    assert "rejected" in text and "elements differ" in text
    assert '"refinement_status": "rejected"' in text


# ── one real refinement (FullProfAPP LiFePO4/FePO4 case, frame 20) ───────────

def test_fullprof_refines_the_lfp_fp_frame(tmp_path):
    try:
        rv.fullprof_exe()
    except FileNotFoundError:
        pytest.skip("FullProf not installed")
    src = LOGS / "iu5063sup3"
    if not src.is_dir():
        pytest.skip("iu5063sup3 not extracted")
    irf = rv.read_irf(str(src / "IRF_NOTOS_July2022_MR.irf"))
    phases = [rv.Phase("LiFePO4", str(LOGS / "cif/1101111.cif"), "LiFePO4", "1101111"),
              rv.Phase("FePO4", str(LOGS / "cif/1525576.cif"), "FePO4", "1525576"),
              rv.Phase("Al", str(LOGS / "cif/4313206.cif"), "Al", "4313206", method="lebail")]
    r = rv.refine(str(src / "LiFePO4_14LFPKa/14LFPka_f00020.xye"), phases,
                  rv.Instrument(irf["wavelength"], str(src / "IRF_NOTOS_July2022_MR.irf"),
                                irf["cthm"], ilo=3), str(tmp_path))
    assert r["status"] == "ok"
    chi2 = [s["chi2"] for s in r["steps"]]
    assert chi2[-1] < chi2[0]                       # the staged workflow converges
    assert r["steps"][-1]["chi2"] < 4.6             # inside the paper's sequence range
    fp = r["result"]["phases"][1]
    assert fp["cell"][0] == pytest.approx(9.81, abs=0.02)      # FePO4 a
    assert sum(p.get("weight_pct", 0) for p in r["result"]["phases"]) == pytest.approx(100, abs=0.1)
