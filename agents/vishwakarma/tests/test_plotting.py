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
