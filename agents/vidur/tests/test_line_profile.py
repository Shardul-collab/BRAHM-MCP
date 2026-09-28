"""line_profile.py against patterns whose answer is known.

The synthetic patterns are built here, independently of line_profile.py: a
sample Voigt convolved with an instrument Voigt is again a Voigt whose
Lorentzian widths add and whose Gaussian widths add in quadrature, so the
measured profile can be written down exactly and the analysis must recover the
sample parameters that went in. Poisson noise is added with a fixed seed.

The tolerances are statements about the method, not tuned to pass: each one is
the stated relative error, and the fits are well above noise.
"""
import math

import numpy as np
import pytest
from scipy.special import voigt_profile

import line_profile as lp

LAM1, LAM2, R = 1.540598, 1.544426, 0.5
FWHM_PER_SIGMA = 2 * math.sqrt(2 * math.log(2))
# instrument: Caglioti U,V,W (deg^2) for Gaussian FWHM, X,Y (deg) for Lorentzian
U, V, W, X, Y = 0.004, -0.002, 0.0025, 0.01, 0.02
ZNSE = [27.22, 45.20, 53.57, 65.88, 72.61, 83.47]       # zinc-blende ZnSe, Cu Ka
LAB6 = [21.36, 30.38, 37.44, 43.51, 48.96, 53.99, 63.22, 67.55, 71.75, 75.85, 79.84]


def inst_widths(tt):
    th = math.radians(tt / 2)
    t = math.tan(th)
    return math.sqrt(U * t * t + V * t + W), X * t + Y / math.cos(th)


def pattern(positions, sample_fn, *, ka2=True, height=4000.0, bkg=60.0, seed=0,
            x=None, instrument=True):
    """sample_fn(tt) -> (fwhm_gauss_deg, fwhm_lorentz_deg) of the sample."""
    x = np.arange(15, 90, 0.01) if x is None else x
    y = np.full_like(x, bkg)
    for tt in positions:
        sg, sl = sample_fn(tt)
        ig, il = inst_widths(tt) if instrument else (0.0, 0.0)
        fg = math.sqrt(sg ** 2 + ig ** 2)
        fl = sl + il
        s, g = fg / FWHM_PER_SIGMA, fl / 2
        area = height / voigt_profile(0.0, s, g)
        y += area * voigt_profile(x - tt, s, g)
        if ka2:
            tt2 = 2 * math.degrees(math.asin(LAM2 / LAM1 * math.sin(math.radians(tt / 2))))
            y += R * area * voigt_profile(x - tt2, s, g)
    rng = np.random.default_rng(seed)
    return x, rng.poisson(y).astype(float)


def rad(ka2=True):
    return lp.Radiation(lam1=LAM1, lam2=LAM2 if ka2 else None,
                        ka2_ratio=R if ka2 else 0.0)


def at(report, tt):
    """The fitted reflection nearest a known position (skips Ka2 partners and
    rejected noise, which are also listed in report['peaks'])."""
    cand = [p for p in report["peaks"]
            if p["status"] in ("ok", "suspect", "unresolved")]
    p = min(cand, key=lambda q: abs(q["two_theta"] - tt))
    assert abs(p["two_theta"] - tt) < 0.1, (tt, p["two_theta"])
    return p


def standard_instrument(seed=1):
    x, y = pattern(LAB6, lambda tt: (0.0, 0.0), seed=seed, height=20000)
    pk, _ = lp.find_peaks(x, y)
    fitted, _, _ = lp.fit_peaks(x, y, pk, rad(), counts=True)
    return lp.instrument_from_peaks(fitted)


# ── known-answer tests ──────────────────────────────────────────────────────

def test_voigt_components_round_trip():
    """Olivero-Longbothum and the integral breadth reduce to the pure cases."""
    assert lp.voigt_fwhm(0.0, 1.0) == pytest.approx(1.0, rel=1e-3)
    assert lp.voigt_fwhm(1.0, 0.0) == pytest.approx(1.0, rel=1e-3)
    assert lp.voigt_integral_breadth(0.0, 2.0) == pytest.approx(2.0)
    assert lp.voigt_integral_breadth(2.0, 0.0) == pytest.approx(2.0)


def test_instrument_function_recovered_from_a_standard():
    inst = standard_instrument()
    for tt in (25.0, 45.0, 70.0):
        g_true, l_true = inst_widths(tt)
        g, l, flag = inst.at(tt)
        assert flag is None
        assert g == pytest.approx(g_true, rel=0.05)
        assert l == pytest.approx(l_true, rel=0.10)


def test_single_line_voigt_recovers_size_and_strain():
    """Sample: D = 30 nm (Lorentzian), eps = 2e-3 (Gaussian)."""
    D, eps = 30.0, 2e-3

    def sample(tt):
        th = math.radians(tt / 2)
        beta_l = (LAM1 / 10) / (D * math.cos(th))          # rad, integral breadth
        beta_g = 4 * eps * math.tan(th)
        fl = math.degrees(beta_l) * 2 / math.pi              # -> FWHM, deg
        fg = math.degrees(beta_g) / math.sqrt(2 * math.pi) * FWHM_PER_SIGMA
        return fg, fl

    x, y = pattern(ZNSE, sample)
    inst = standard_instrument()
    report, _ = lp.analyse_pattern(x, y, rad(), inst)
    for tt in ZNSE:
        p = at(report, tt)
        assert p["status"] == "ok"
        assert p["D_voigt_nm"] == pytest.approx(D, rel=0.15), p["two_theta"]
        assert p["strain_voigt"] == pytest.approx(eps, rel=0.15), p["two_theta"]


def test_williamson_hall_recovers_size_and_strain_when_its_model_holds():
    """WH assumes breadths add linearly. Build the sample that way (all
    Lorentzian, FWHM = K lam/(D cos) + 4 eps tan) and it must come back."""
    D, eps, K = 25.0, 1.5e-3, 0.9

    def sample(tt):
        th = math.radians(tt / 2)
        b = K * (LAM1 / 10) / (D * math.cos(th)) + 4 * eps * math.tan(th)
        return 0.0, math.degrees(b)

    x, y = pattern(ZNSE, sample, seed=3)
    report, _ = lp.analyse_pattern(x, y, rad(), standard_instrument())
    wh = report["williamson_hall"]
    assert wh["status"] == "ok" and wh["n"] == len(ZNSE)
    assert wh["D_nm"] == pytest.approx(D, rel=0.10)
    assert wh["strain"] == pytest.approx(eps, rel=0.15)
    assert wh["r2"] > 0.95


def test_size_strain_plot_recovers_lorentz_size_and_gauss_strain():
    """The SSP's own model: Lorentzian size, Gaussian strain, integral breadths.
    The first version used the literature form '(K/D)(d^2 beta cos)' with no
    lambda -- dimensionally inconsistent -- and returned D = 234 nm here."""
    D, eps = 30.0, 2e-3

    def sample(tt):
        th = math.radians(tt / 2)
        bl = (LAM1 / 10) / (D * math.cos(th))
        bg = 4 * eps * math.tan(th)
        return (math.degrees(bg) / math.sqrt(2 * math.pi) * FWHM_PER_SIGMA,
                math.degrees(bl) * 2 / math.pi)

    x, y = pattern(ZNSE, sample, seed=100)
    report, _ = lp.analyse_pattern(x, y, rad(), standard_instrument(seed=200))
    ssp = report["size_strain_plot"]
    assert ssp["status"] == "ok"
    assert ssp["D_nm"] == pytest.approx(D, rel=0.10)
    assert ssp["strain"] == pytest.approx(eps, rel=0.20)


def test_scherrer_on_pure_size_broadening():
    """No strain, Lorentzian size broadening with FWHM = K lam / (D cos)."""
    D, K = 40.0, 0.9

    def sample(tt):
        th = math.radians(tt / 2)
        return 0.0, math.degrees(K * (LAM1 / 10) / (D * math.cos(th)))

    x, y = pattern(ZNSE, sample, seed=4)
    report, _ = lp.analyse_pattern(x, y, rad(), standard_instrument())
    for tt in ZNSE:
        p = at(report, tt)
        assert p["D_scherrer_nm"] == pytest.approx(D, rel=0.08), tt


def test_ignoring_ka2_is_measurably_wrong():
    """Where the doublet is not resolved, reading it as one peak inflates the
    breadth. Measured 2026-09-25 on this synthetic set: FWHM x1.52 and D off by
    18 % at 53.6 deg, D = 40 nm. This is why the ratio is asked for rather than
    defaulted to 0."""
    D, tt = 40.0, 53.57

    def sample(t):
        th = math.radians(t / 2)
        return 0.0, math.degrees((LAM1 / 10) / (D * math.cos(th)) * 2 / math.pi)

    x, y = pattern([tt], sample, seed=5)
    inst = standard_instrument()
    right = at(lp.analyse_pattern(x, y, rad(True), inst)[0], tt)
    wrong = at(lp.analyse_pattern(x, y, rad(False), inst)[0], tt)
    assert right["D_voigt_nm"] == pytest.approx(D, rel=0.10)
    assert wrong["meas_fwhm_deg"] > 1.3 * right["meas_fwhm_deg"]
    assert abs(wrong["D_voigt_nm"] - D) / D > 0.10


# ── honesty tests ───────────────────────────────────────────────────────────

def test_no_instrument_is_flagged_as_a_lower_bound():
    x, y = pattern(ZNSE, lambda tt: (0.05, 0.1))
    report, _ = lp.analyse_pattern(x, y, rad(), lp.Instrument(kind="none"))
    assert report["instrument_corrected"] is False
    assert "lower" in report["caveats"][0]


def test_instrument_only_pattern_is_not_given_a_size():
    """A sample no broader than the instrument has no measurable size; the
    subtraction must say so, not return a huge D."""
    x, y = pattern(ZNSE, lambda tt: (0.0, 0.0), seed=7, height=20000)
    inst = standard_instrument()
    report, _ = lp.analyse_pattern(x, y, rad(), inst)
    for tt in ZNSE:
        p = at(report, tt)
        if p["status"] == "ok":
            # whatever survives the subtraction is noise-level: huge D, not a size
            assert p["D_scherrer_nm"] is None or p["D_scherrer_nm"] > 300
        else:
            assert p["status"] == "unresolved"


def test_williamson_hall_refuses_two_points():
    x, y = pattern(ZNSE[:2], lambda tt: (0.05, 0.1))
    report, _ = lp.analyse_pattern(x, y, rad(), standard_instrument())
    assert report["williamson_hall"]["status"] == "not_computed"
    assert report["size_strain_plot"]["status"] == "not_computed"


def test_raw_column_is_the_input_unchanged():
    x, y = pattern(ZNSE, lambda tt: (0.05, 0.1))
    _, cols = lp.analyse_pattern(x, y, rad(), lp.Instrument(kind="none"))
    assert np.array_equal(cols["raw"], y)
    fitted = np.isfinite(cols["model"])
    assert np.allclose(cols["residual"][fitted], y[fitted] - cols["model"][fitted])


def test_missing_wavelength_and_ka2_become_questions():
    r, q = lp.resolve_radiation({}, {})
    assert r is None
    assert set(q) == {"wavelength_a", "ka2_ratio"}
    r, q = lp.resolve_radiation({}, {"wavelength_a": 1.5406, "ka2_ratio": 0})
    assert q == {} and r.doublet is False


def test_file_values_outrank_the_caller():
    r, q = lp.resolve_radiation(
        {"wavelength_a": 1.540598, "wavelength_ka2_a": 1.544426, "ka2_ratio": 0.5},
        {"wavelength_a": 1.0, "ka2_ratio": 0.0})
    assert r.lam1 == 1.540598 and r.ka2_ratio == 0.5
    assert r.source["wavelength_a"] == "file"


def test_pure_noise_does_not_produce_peaks():
    """A fixed 5-sigma prominence cut accepted ~30-40 'peaks' in 7,500 points of
    pure Poisson noise (measured 2026-09-25). With the false-alarm limit, 20
    noise patterns together may yield at most a couple."""
    found = 0
    for seed in range(20):
        y = np.random.default_rng(seed).poisson(60.0, 7500).astype(float)
        pk, info = lp.find_peaks(np.arange(15, 90, 0.01), y)
        found += len(pk)
    assert found <= 2
