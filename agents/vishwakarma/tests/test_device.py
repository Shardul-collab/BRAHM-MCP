"""
Tests for vishwakarma.device: parameter provenance, BoltzTraP2 band transport
on a real Quantum ESPRESSO run, and DEVSIM simulations checked against
closed-form physics.

The transport fixture is the data-file-schema.xml of a real NSCF run
(primitive Si, PBE, 16x16x16, 16 bands; job 1ef22e16 writing into the
outdir of SCF job 5fd2b783, 2026-10-15). Expected values below are textbook
silicon numbers fixed before the code was run, not values read back from it.

DEVSIM tests take ~5 s (diode) and ~1-2 min (MOSFET). Set VK_SKIP_SLOW=1 to
skip the MOSFET.
"""

import importlib.util
import math
import os
import shutil
from pathlib import Path

import pytest

from conftest import FIXTURES

from vishwakarma.device.params import MaterialParams, Param
from vishwakarma.device import analytic, transport

HAS_BTP = importlib.util.find_spec("BoltzTraP2") is not None


def _has_devsim() -> bool:
    os.environ.setdefault("DEVSIM_MATH_LIBS", "liblapack.so.3:libblas.so.3")
    try:
        import devsim  # noqa: F401
        return True
    except Exception:
        return False


HAS_DEVSIM = _has_devsim()

LIT = {"source": "literature", "ref": "test"}


def si_textbook() -> MaterialParams:
    m = MaterialParams(name="Si-test")
    for k, v in (("band_gap_eV", 1.12), ("electron_affinity_eV", 4.05), ("eps_r", 11.8),
                 ("Nc_cm3", 2.8e19), ("Nv_cm3", 1.04e19), ("mu_n_cm2Vs", 1500),
                 ("mu_p_cm2Vs", 450)):
        m.set(k, v, **LIT)
    m.set("srh_lifetime_n_s", 1e-6, "assumed")
    m.set("srh_lifetime_p_s", 1e-6, "assumed")
    return m


@pytest.fixture
def si_save_dir(tmp_path):
    d = tmp_path / "si.save"
    d.mkdir()
    shutil.copy(FIXTURES / "si_nscf16_data-file-schema.xml", d / "data-file-schema.xml")
    return d


# ─── provenance rules ──────────────────────────────────────────────────────

def test_literature_value_requires_ref():
    with pytest.raises(ValueError):
        Param(1.12, "eV", "literature")


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        Param(1.12, "eV", "vibes")


def test_validate_catches_density_given_in_m3():
    m = si_textbook()
    m.set("Nc_cm3", 2.8e25, "user")            # 2.8e19 cm^-3 written in m^-3
    assert any("Nc_cm3" in p for p in m.validate())


def test_missing_parameters_reported():
    m = MaterialParams(name="x")
    m.set("band_gap_eV", 1.0, "user")
    assert "eps_r" in m.missing()
    assert m.validate()


def test_roundtrip_dict():
    m = si_textbook()
    m2 = MaterialParams.from_dict(m.to_dict())
    assert m2.provenance_table() == m.provenance_table()
    assert m2.assumed_keys() == ["srh_lifetime_n_s", "srh_lifetime_p_s"]


def test_ni_from_nc_nv_eg():
    # sqrt(2.8e19 * 1.04e19) * exp(-1.12 / (2 kT/q)) at 300 K
    m = si_textbook()
    expected = math.sqrt(2.8e19 * 1.04e19) * math.exp(-1.12 / (2 * 0.0258520))
    assert m.ni() == pytest.approx(expected, rel=1e-4)
    assert 5e9 < m.ni() < 1.5e10


# ─── band transport (BoltzTraP2) on the real Si run ────────────────────────

@pytest.mark.skipif(not HAS_BTP, reason="BoltzTraP2 not installed")
def test_si_band_transport_matches_textbook(si_save_dir):
    r = transport.band_transport(si_save_dir, check_convergence=True)
    assert r["run"]["calculation"] == "nscf"
    # PBE gap of Si is ~0.5-0.65 eV (experiment 1.12) -- the known underestimate
    assert 0.45 < r["band_gap_dft_eV"] < 0.70
    # Nc: 2.8e19 (Sze) / 2.86e19 (Green 1990); allow 10 %
    assert r["electrons"]["Neff_cm3"] == pytest.approx(2.8e19, rel=0.10)
    # electron conductivity mass 3/(1/ml + 2/mt) = 0.26 for ml=0.92, mt=0.19
    assert 0.24 < r["electrons"]["m_cond"] < 0.30
    # cubic crystal: conductivity tensor isotropic
    mp = r["electrons"]["m_cond_principal"]
    assert max(mp) / min(mp) < 1.01
    assert r["convergence"]["relative_change"]["m_cond_n"] < 0.05


def test_tau_mobility_roundtrip():
    tau = transport.tau_from_mobility(1500, 0.26)
    assert 1e-13 < tau < 5e-13                      # ~0.2 ps for Si electrons
    assert transport.mobility_from_tau(tau, 0.26) == pytest.approx(1500)


def test_locate_save_dir_reroots_old_absolute_outdir(tmp_path, si_save_dir):
    work = tmp_path / "jobs"
    scf, nscf = work / "scf-1", work / "nscf-2"
    (scf / "out").mkdir(parents=True)
    shutil.copytree(si_save_dir, scf / "out" / "si_dev.save")
    nscf.mkdir()
    (nscf / "input.in").write_text(
        "&CONTROL\n  calculation = 'nscf'\n  prefix = 'si_dev'\n"
        "  outdir = '/mnt/d/brahm/agents/vishwakarma/jobs/scf-1/out'\n/\n")
    found = transport.locate_save_dir(job_id="nscf-2", workdir=str(work))
    assert found == scf / "out" / "si_dev.save"


# ─── DEVSIM vs closed-form physics ─────────────────────────────────────────

@pytest.mark.skipif(not HAS_DEVSIM, reason="DEVSIM or BLAS/LAPACK unavailable")
def test_diode_matches_shockley():
    from vishwakarma.device import tcad
    res = tcad.pn_diode_1d(si_textbook())
    assert res["success"], res.get("error")
    chk = res["analytic_check"]
    assert chk["pass"] is True, chk["summary"]
    assert chk["n_window_points"] >= 3
    # reverse current: negative, and no larger than the generation upper bound
    rev = chk["reverse_info"]
    assert rev["J_sim"] < 0 and abs(rev["J_sim"]) <= abs(rev["J_generation_bound"]) * 1.2


@pytest.mark.skipif(not HAS_DEVSIM or os.environ.get("VK_SKIP_SLOW"),
                    reason="DEVSIM unavailable or VK_SKIP_SLOW set")
def test_nmosfet_matches_charge_sheet():
    from vishwakarma.device import tcad
    res = tcad.nmosfet_2d(si_textbook(), gate_workfunction_eV=4.05)
    assert res["success"], res.get("error")
    chk = res["analytic_check"]
    assert chk["pass"] is True, chk["summary"]
    # constant-mobility long channel: gm should give back the input mobility
    assert chk["mu_eff_extracted_cm2Vs"] == pytest.approx(1500, rel=0.10)
    # Id must increase monotonically with Vg
    ids = [p["Id_A_per_um"] for p in res["transfer_vd_lin"]]
    assert all(b >= a for a, b in zip(ids, ids[1:]))


def test_charge_sheet_threshold_above_classical():
    # max-gm extrapolation sits a few kT/q above the 2*phiF definition
    m = si_textbook()
    vgs = [round(-0.3 + 0.05 * i, 3) for i in range(45)]
    cs = analytic.charge_sheet_transfer(m, 1e17, 1e-6, 3.9, 4.05, 2e-4, 0.05, vgs)
    vt_cs, _, _ = analytic.extract_vt_maxgm([(r["Vg"], r["Id_A_per_um"]) for r in cs], 0.05)
    vt_cl = analytic.mos_threshold(m, 1e17, 1e-6, 3.9, 4.05)["Vt"]
    assert 0.02 < vt_cs - vt_cl < 0.25
