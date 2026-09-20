"""Derived columns. The rule: raw is never altered, every step adds a column,
and anything needing a constant the file does not carry is SKIPPED with a
reason rather than defaulted silently.
"""
import numpy as np
import pytest
import prep


def test_raw_is_always_present_and_unmodified():
    y = np.array([1.0, -2.0, 3.0])
    cols, _ = prep.derive("XRD", np.array([10.0, 20.0, 30.0]), y, {})
    assert list(cols["raw"]) == [1.0, -2.0, 3.0]


def test_xrd_without_wavelength_skips_d_and_q_and_says_so():
    cols, notes = prep.derive("XRD", np.array([10.0, 20.0]), np.array([1.0, 2.0]), {})
    assert "d_A" not in cols and "q_invA" not in cols
    assert any("wavelength" in n for n in notes)
    assert any("not assumed" in n for n in notes)


def test_xrd_with_wavelength_matches_bragg():
    two_theta = np.array([15.01])
    cols, _ = prep.derive("XRD", two_theta, np.array([9.0]), {"wavelength_a": 1.5406})
    theta = np.radians(15.01 / 2)
    assert cols["d_A"][0] == pytest.approx(1.5406 / (2 * np.sin(theta)))
    assert cols["q_invA"][0] == pytest.approx(4 * np.pi * np.sin(theta) / 1.5406)


def test_uvvis_without_thickness_does_not_substitute_absorbance_for_alpha():
    """The literature warns specifically against this substitution."""
    cols, notes = prep.derive("UV-Vis", np.array([400.0, 500.0]),
                              np.array([1.0, 0.5]), {})
    assert "tauc_direct" not in cols and "alpha_invcm" not in cols
    assert any("thickness" in n for n in notes)


def test_uvvis_photon_energy():
    cols, _ = prep.derive("UV-Vis", np.array([400.0]), np.array([1.0]), {})
    assert cols["photon_energy_eV"][0] == pytest.approx(3.0996, abs=1e-3)


def test_normalisations_do_not_change_the_number_of_points():
    x = np.linspace(100, 200, 64)
    y = np.random.default_rng(0).normal(size=64)
    cols, _ = prep.derive("Raman", x, y, {})
    for name, values in cols.items():
        assert len(values) == 64, name


def test_reference_band_normalisation_skips_with_a_reason_when_out_of_range():
    x = np.linspace(580, 1378, 100)          # no Si band at 520.7 in this window
    cols, notes = prep.derive("Raman", x, np.ones(100), {"reference_band_cm1": 520.7})
    assert not any(k.startswith("norm_ref") for k in cols)
    assert any("reference-band" in n for n in notes)


def test_transmittance_to_absorbance():
    assert prep.transmittance_to_absorbance(np.array([100.0]))[0] == pytest.approx(0.0)
    assert prep.transmittance_to_absorbance(np.array([10.0]))[0] == pytest.approx(1.0)


def test_a_q_axis_never_gets_bragg_applied_to_it():
    """ixdat's fixture set has .xy files whose x column is q, not 2theta.
    Labelling those degrees, or running Bragg on them, is the same fabrication
    as inventing an axis. Found 2026-09-19 on real fixtures."""
    cols, notes = prep.derive("XRD", np.array([0.5, 1.0]), np.array([100.0, 200.0]),
                              {"wavelength_a": 1.5406, "axis_is_two_theta": False})
    assert "d_A" not in cols and "q_invA" not in cols
    assert any("q as declared by the file" in n for n in notes)
