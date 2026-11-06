"""
Tests for vishwakarma.plotting.

The fixture is the data-file-schema.xml of a real NSCF run (primitive Si, PBE,
16x16x16, 16 bands; job 1ef22e16, 2026-10-15), the same file test_device.py
uses. Expected numbers are physics fixed beforehand: Si has 8 valence
electrons, a PBE indirect gap of about 0.6 eV with the valence maximum at Gamma
and the conduction minimum along Gamma-X, and a 16^3 mesh has 4096 points.
"""

import shutil

import numpy as np
import pytest

from conftest import FIXTURES

from vishwakarma import plotting
from vishwakarma.plotting import bandgrid, figures

pytest.importorskip("spglib")
pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def save_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("si_save")
    shutil.copy(FIXTURES / "si_nscf16_data-file-schema.xml", d / "data-file-schema.xml")
    return d


@pytest.fixture(scope="module")
def grid(save_dir):
    return plotting.load_band_grid(save_dir)


def test_unfold_fills_whole_mesh(grid):
    assert grid.mesh == (16, 16, 16)
    assert grid.energies.shape == (16, 16, 16, 16)
    assert not np.isnan(grid.energies).any()
    assert grid.n_sym_ops == 48                      # diamond point group (Oh) incl. fractional translations


def test_si_band_edges(grid):
    s = grid.gap_summary()
    assert grid.n_occ == 4 and grid.nelec == 8
    assert 0.45 < s["gap_eV"] < 0.70                 # PBE Si indirect gap ~0.6 eV
    assert s["vbm_k_crystal"] == [0.0, 0.0, 0.0]     # valence maximum at Gamma
    assert s["direct_gap_min_eV"] > 2.0              # direct gap at Gamma is much larger
    assert not s["metallic"]


def test_dos_integrates_to_valence_electrons(grid):
    x, dos, ref, _ = figures.compute_dos(grid, sigma_eV=0.05, window_eV=(-14.0, 10.0))
    s = grid.gap_summary()
    n = np.trapezoid(dos[x < s["vbm_eV"] - ref + 1e-9], x[x < s["vbm_eV"] - ref + 1e-9])
    assert n == pytest.approx(8.0, abs=0.05)


def test_dos_gap_is_not_eaten_by_small_broadening(grid):
    """The reason this module exists: a narrow gaussian leaves the real 0.6 eV gap visible."""
    x, dos, ref, _ = figures.compute_dos(grid, sigma_eV=0.02)
    s = grid.gap_summary()
    mid = 0.5 * (s["vbm_eV"] + s["cbm_eV"]) - ref
    assert dos[np.argmin(abs(x - mid))] < 1e-3
    zero = x[dos < 1e-3]
    zero = zero[(zero > s["vbm_eV"] - ref - 0.1) & (zero < s["cbm_eV"] - ref + 0.1)]
    assert zero.max() - zero.min() > 0.45


def test_first_zone_mask_is_the_wigner_seitz_cell(grid):
    # Gamma is inside; the point at a full reciprocal vector is not
    pts = np.array([[0.0, 0.0, 0.0], 1.0 * grid.recip_ang[0], 0.4 * grid.recip_ang[0]])
    inside = figures._in_first_bz(pts, grid.recip_ang)
    assert inside.tolist() == [True, False, True]


def test_band_surfaces_ordering_and_edges(grid):
    KX, KY, surf, bands = figures.band_surfaces(grid, n=81)
    vb, cb = surf
    assert bands == (3, 4)
    assert np.nanmax(vb) == pytest.approx(grid.gap_summary()["vbm_eV"], abs=0.02)   # Gamma is on the plane
    assert np.nanmin(cb) == pytest.approx(grid.gap_summary()["cbm_eV"], abs=0.06)   # X valleys lie on kz=0
    assert np.nanmin(cb) > np.nanmax(vb)


def test_plot_job_writes_files(save_dir, tmp_path):
    res = plotting.plot_job(save_dir=str(save_dir), kind="all", out_dir=str(tmp_path))
    for key in ("dos_png", "bands3d_png", "bands3d_html"):
        f = tmp_path / res["files"][key].split("/")[-1]
        assert f.is_file() and f.stat().st_size > 5_000, key
    assert res["band_edges"]["gap_eV"] == pytest.approx(0.58, abs=0.03)
    assert res["dos"]["electrons_below_gap"] == pytest.approx(8.0, abs=0.05)


def test_unknown_kind_and_non_mesh_runs_are_rejected(save_dir, tmp_path):
    with pytest.raises(ValueError):
        plotting.plot_job(save_dir=str(save_dir), kind="nonsense")
    # a bands-path style XML (explicit k list, no monkhorst_pack) must be refused, not mis-plotted
    txt = (FIXTURES / "si_nscf16_data-file-schema.xml").read_text()
    import re
    broken = re.sub(r"<monkhorst_pack[^>]*>[^<]*</monkhorst_pack>", "", txt)
    assert broken != txt
    d = tmp_path / "path"
    d.mkdir()
    (d / "data-file-schema.xml").write_text(broken)
    with pytest.raises(plotting.BandGridError):
        plotting.load_band_grid(d)


# ---------------------------------------------------------------------------
# k-path runs (2D band structure). Fixture: data-file-schema.xml of the 'bands'
# NSCF of primitive Si (PBE, 121 k-points along L-G-X-W-K-G, 12 bands; job
# 0566bacc, 2026-11-06), produced through the normal band_structure workflow.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def path_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("si_path")
    shutil.copy(FIXTURES / "si_bands_path_data-file-schema.xml", d / "data-file-schema.xml")
    return d


@pytest.fixture(scope="module")
def bpath(path_dir):
    from vishwakarma.plotting.bandpath import load_band_path
    return load_band_path(path_dir)


def test_path_nodes_are_found_and_labelled(bpath):
    assert bpath.labels == ["L", "Γ", "X", "W", "K", "Γ"]
    assert len(bpath.nodes) == 6
    assert len(bpath.x) == 121 and bpath.energies.shape == (121, 12)
    assert np.all(np.diff(bpath.x) >= 0)


def test_path_band_physics_of_si(bpath):
    e = bpath.edges()
    assert 0.45 < e["gap_eV"] < 0.70
    assert bpath.labels[1] == "Γ" and e["vbm_index"] in (bpath.nodes[1], bpath.nodes[5])   # VBM at Gamma
    ic = e["cbm_index"]
    assert bpath.nodes[1] < ic < bpath.nodes[2]            # CBM on Gamma-X, not at X or Gamma
    frac = (bpath.x[ic] - bpath.x[bpath.nodes[1]]) / (bpath.x[bpath.nodes[2]] - bpath.x[bpath.nodes[1]])
    assert 0.78 < frac < 0.92                              # ~0.85 of the way to X
    assert e["direct_gap_min_eV"] > 2.0


def test_path_and_mesh_vbm_agree(bpath, grid):
    """Two independent runs (8x8x8 SCF + path, and 16^3 NSCF) must place the VBM at the same energy."""
    assert bpath.edges()["vbm_eV"] == pytest.approx(grid.gap_summary()["vbm_eV"], abs=0.01)


def test_plot_job_bands_png(path_dir, tmp_path):
    res = plotting.plot_job(save_dir=str(path_dir), kind="all", out_dir=str(tmp_path))
    assert res["run_type"] == "k-path"
    f = tmp_path / res["files"]["bands2d_png"].split("/")[-1]
    assert f.is_file() and f.stat().st_size > 20_000
    assert res["band_edges"]["gap_eV"] == pytest.approx(0.57, abs=0.03)


def test_kind_must_match_run_type(path_dir, save_dir, tmp_path):
    for k in ("dos", "bands3d"):
        with pytest.raises(plotting.BandGridError):
            plotting.plot_job(save_dir=str(path_dir), kind=k, out_dir=str(tmp_path))
    with pytest.raises(plotting.BandGridError):
        plotting.plot_job(save_dir=str(save_dir), kind="bands", out_dir=str(tmp_path))


def test_custom_labels_and_wrong_count(path_dir):
    from vishwakarma.plotting.bandpath import load_band_path
    bp = load_band_path(path_dir, labels=list("ABCDEF"))
    assert bp.labels == list("ABCDEF")
    with pytest.raises(plotting.BandGridError):
        load_band_path(path_dir, labels=["A", "B"])


# ---------------------------------------------------------------------------
# Charge density. Fixture: charge-density.hdf5 of the primitive-Si SCF (job
# cd93c035, 2026-11-06) with the k-path NSCF xml of the same structure.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dens_dir(tmp_path_factory):
    pytest.importorskip("h5py")
    d = tmp_path_factory.mktemp("si_density")
    shutil.copy(FIXTURES / "si_bands_path_data-file-schema.xml", d / "data-file-schema.xml")
    shutil.copy(FIXTURES / "si_charge-density.hdf5", d / "charge-density.hdf5")
    return d


@pytest.fixture(scope="module")
def dens(dens_dir):
    from vishwakarma.plotting.density import load_density
    return load_density(dens_dir)


def test_density_integrates_to_electron_count(dens):
    assert dens.nelec == 8
    assert dens.rho.mean() * dens.volume_ang3 == pytest.approx(8.0, abs=1e-6)
    assert dens.rho.min() > 0                              # valence density is positive everywhere in Si
    assert dens.checks["electrons_from_rho_G0"] == pytest.approx(8.0, abs=1e-6)
    assert dens.checks["max_abs_imag_over_max_real"] < 1e-10


def test_density_profile_along_the_bond(dens):
    """Structure of the Si-Si bond line, as measured on this fixture (ultrasoft rrkjus PP,
    whose augmentation charge sits on the nuclei): maximum at the atom, a dip toward the
    bond, a broad covalent plateau at the bond centre, and an empty interstitial region."""
    from scipy.interpolate import RegularGridInterpolator
    n = dens.rho.shape
    f = RegularGridInterpolator([np.arange(k + 1) / k for k in n], np.pad(dens.rho, ((0, 1),) * 3, mode="wrap"))
    a0, a1 = (np.array(c) for _, c in dens.atoms)

    def at(t):
        fr = np.linalg.solve(dens.cell_ang.T, a0 + t * (a1 - a0)) % 1.0
        return float(f(fr[None, :])[0])

    bond = at(0.5)
    assert at(0.0) == pytest.approx(dens.rho.max(), rel=0.02)        # peak on the nucleus
    assert at(0.2) < 0.6 * bond                                      # dip between core and bond
    assert at(0.4) > 0.9 * bond and at(0.6) > 0.9 * bond             # plateau around the bond centre
    assert dens.rho.min() < 0.1 * bond                               # empty interstitial region


def test_plot_job_density_writes_files(dens_dir, tmp_path):
    res = plotting.plot_job(save_dir=str(dens_dir), kind="density", out_dir=str(tmp_path))
    assert (tmp_path / res["files"]["density_png"].split("/")[-1]).stat().st_size > 20_000
    html = tmp_path / res["files"]["density_html"].split("/")[-1]
    assert html.stat().st_size > 100_000                   # plotly.js is embedded: works offline
    d = res["density"]
    assert 0.05 < d["cell_fraction_above_isovalue"] < 0.15  # default level encloses ~10 % of the cell


def test_density_isovalue_out_of_range_is_rejected(dens_dir, tmp_path):
    with pytest.raises(plotting.BandGridError):
        plotting.plot_job(save_dir=str(dens_dir), kind="density", isovalue=50.0, out_dir=str(tmp_path))


def test_density_refuses_missing_file_and_wrong_electron_count(dens_dir, tmp_path):
    from vishwakarma.plotting.density import load_density
    empty = tmp_path / "empty"
    empty.mkdir()
    shutil.copy(FIXTURES / "si_bands_path_data-file-schema.xml", empty / "data-file-schema.xml")
    with pytest.raises(plotting.BandGridError):
        load_density(empty)
    # a density from a different structure must not be silently plotted with this xml's electron count
    xml = (dens_dir / "data-file-schema.xml").read_text()
    bad = tmp_path / "bad"
    bad.mkdir()
    shutil.copy(dens_dir / "charge-density.hdf5", bad / "charge-density.hdf5")
    assert 'nelec' in xml or 'NELEC' in xml.upper()
    import re
    (bad / "data-file-schema.xml").write_text(re.sub(r"(<nelec>)\s*[0-9.Ee+-]+", r"\g<1>10.0", xml, count=1))
    with pytest.raises(plotting.BandGridError):
        load_density(bad)
