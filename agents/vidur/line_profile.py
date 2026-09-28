"""XRD line-profile analysis: crystallite size, microstrain, dislocation density.

Every formula here is written out, with its source and its limits, in
XRD-LINE-PROFILE-METHODS.md (same directory). Read that first; section numbers
in comments below (§n) refer to it.

This is the interpretation layer VIDUR-DESIGN.md keeps out of prep.py. The same
rules apply:
  * the raw intensity is never altered -- fit, background and residual are
    extra columns beside it;
  * no constant is assumed -- wavelength, Ka2 and the instrument function come
    from the file or the caller, and what is missing is returned as a question;
  * every number carries the model that produced it and its standard error.

Angles: user-facing values are degrees 2theta; breadths inside the formulas are
radians of 2theta; theta is the Bragg angle (half of 2theta).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks as _sp_find_peaks, peak_widths
from scipy.special import voigt_profile, erfcx
from scipy.stats import linregress

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))   # Gaussian FWHM / sigma
SQRT_2PI = math.sqrt(2.0 * math.pi)

DEFAULTS = {
    "min_snr": 5.0,          # §2  floor on the prominence threshold, noise sigmas
    "false_alarm": 0.01,     # §2  chance per pattern that pure noise passes
    "min_points_fwhm": 5,    # §2  points above half height needed to fit a breadth
    "window_fwhm": 4.0,      # §3.3 fit window half-width, in estimated FWHMs
    "scherrer_k": 0.9,       # §5.1 FWHM Scherrer constant (convention, not universal)
    "ssp_k": 1.0,            # §6.2 K=1 -> volume-weighted size; 0.75 = literature sphere convention
    "min_wh_peaks": 3,       # §6  fewer points than this and nothing tests the line
}


# ── §3.1 profile helpers ─────────────────────────────────────────────────────

def voigt_fwhm(fwhm_l, fwhm_g):
    """Olivero & Longbothum (1977), ~0.02 % accurate."""
    return 0.5346 * fwhm_l + math.sqrt(0.2166 * fwhm_l ** 2 + fwhm_g ** 2)


def voigt_integral_breadth(beta_l, beta_g):
    """beta = beta_G * exp(-k^2) / erfc(k), k = beta_L / (sqrt(pi) beta_G).
    erfcx(k) = exp(k^2) erfc(k), which stays finite for large k."""
    if beta_g <= 0:
        return beta_l
    if beta_l <= 0:
        return beta_g
    k = beta_l / (math.sqrt(math.pi) * beta_g)
    return beta_g / erfcx(k)


def ka2_position(two_theta1_deg, lam1, lam2):
    """§3.4 Kalpha2 position fixed by Bragg's law from the Kalpha1 position."""
    s = (lam2 / lam1) * np.sin(np.radians(np.asarray(two_theta1_deg) / 2.0))
    return 2.0 * np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))


def noise_sigma(y):
    """§2 robust noise: 1.4826 * MAD(diff(y)) / sqrt(2)."""
    d = np.diff(np.asarray(y, dtype=float))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0))


# ── §2 peak finding ──────────────────────────────────────────────────────────

def is_counts(y, meta=None):
    """Raw detector counts (Poisson) or not. The file's own unit wins; otherwise
    non-negative integers are taken as counts."""
    unit = str((meta or {}).get("intensity_unit") or "").lower()
    if unit:
        return unit in ("counts", "count", "cts")
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    return bool(y.size and np.all(y >= 0) and np.allclose(y, np.round(y)))


_NOISE_K_CACHE: dict = {}


def noise_prominence_limit(n, alpha=0.01, trials=400):
    """§2 The prominence (in noise sigmas) that pure noise of n points exceeds
    with probability alpha somewhere in the pattern.

    Measured 2026-09-25: a prominence is roughly a local max minus a local min,
    so pure Poisson noise over 7,500 points gives ~30-40 'peaks' above 5 sigma
    and a largest one of ~7 sigma. A fixed 5-sigma cut is therefore a false-
    alarm generator; this limit replaces it by a stated false-alarm rate per
    pattern (default 1 %). Deterministic: fixed seed, cached by n."""
    key = (int(n), float(alpha))
    if key not in _NOISE_K_CACHE:
        rng = np.random.default_rng(12345)
        mx = np.empty(trials)
        for t in range(trials):
            z = rng.standard_normal(int(n))
            _, pr = _sp_find_peaks(z, prominence=0)
            mx[t] = pr["prominences"].max() if pr["prominences"].size else 0.0
        _NOISE_K_CACHE[key] = float(np.quantile(mx, 1.0 - alpha))
    return _NOISE_K_CACHE[key]


def find_peaks(x, y, min_snr=DEFAULTS["min_snr"], counts=None,
               false_alarm=DEFAULTS["false_alarm"]):
    """§2 Peaks whose prominence is >= k noise sigmas, k = the larger of
    min_snr and the noise_prominence_limit for this pattern length.

    Two noise models, and the output says which ran:
      counts  -> Poisson: sigma = sqrt(local baseline) for each peak. Counting
                 noise grows with intensity, so one global sigma over-detects in
                 low-count regions and under-detects on a high background
                 (measured 2026-09-25 on m54313_om2th_10.xrdml: global MAD sigma
                 2.1 counts against a 40-count background -> 1-point spikes
                 accepted at 'SNR 10').
      other   -> one robust global sigma, 1.4826 * MAD(diff y) / sqrt 2.
    Returns (peaks, noise_info)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    counts = is_counts(y) if counts is None else counts
    sig = noise_sigma(y)
    if not np.isfinite(sig) or sig <= 0:
        # Noise-free (synthetic or pre-smoothed) data: fall back to a fraction of
        # the range so that the threshold is still stated, not absent.
        sig = 1e-3 * (np.nanmax(y) - np.nanmin(y))
    k_thr = max(min_snr, noise_prominence_limit(y.size, false_alarm))
    floor = k_thr * (min(sig, 1.0) if counts else sig)
    idx, props = _sp_find_peaks(y, prominence=floor)
    info = {"model": "poisson" if counts else "global_mad", "global_sigma": sig,
            "threshold_sigmas": k_thr, "false_alarm_per_pattern": false_alarm,
            "min_snr_floor": min_snr}
    if idx.size == 0:
        return [], info
    w, _, lip, rip = peak_widths(y, idx, rel_height=0.5)
    step = np.median(np.diff(x))
    out = []
    for i, k in enumerate(idx):
        prom = float(props["prominences"][i])
        # Poisson sigma from the local background level, taken as the median
        # of a window much wider than the peak. (The prominence base -- a local
        # minimum -- sits ~3 sigma low in noise and understated sigma by ~20 %.)
        half = int(max(50, 5 * w[i]))
        base = float(np.median(y[max(0, k - half):k + half + 1]))
        s_loc = math.sqrt(max(base, 1.0)) if counts else sig
        if prom < k_thr * s_loc:
            continue
        left = np.interp(lip[i], np.arange(x.size), x)
        right = np.interp(rip[i], np.arange(x.size), x)
        out.append({
            "index": int(k),
            "two_theta": float(x[k]),
            "prominence": prom,
            "snr": prom / s_loc,
            "fwhm_est": float(abs(right - left)) or float(abs(step)),
            "points_fwhm": int(round(w[i])),
        })
    return out, info


# ── §3 profile fitting ───────────────────────────────────────────────────────

@dataclass
class Radiation:
    lam1: float                       # Angstrom
    lam2: float | None = None
    ka2_ratio: float = 0.0            # area(Ka2)/area(Ka1); 0 = Ka1 only
    source: dict = field(default_factory=dict)

    @property
    def doublet(self):
        return bool(self.ka2_ratio) and self.lam2 is not None


def _cluster_model(n, rad: Radiation):
    """n Voigt peaks (+ Ka2 partners) on a shared linear background.
    Parameter vector: [x0, area, sigma, gamma] * n + [b0, b1]."""
    def f(x, *p):
        b0, b1 = p[-2], p[-1]
        xm = f.xmid
        y = b0 + b1 * (x - xm)
        for j in range(n):
            x0, a, s, g = p[4 * j:4 * j + 4]
            y = y + a * voigt_profile(x - x0, s, g)
            if rad.doublet:
                x2 = ka2_position(x0, rad.lam1, rad.lam2)
                y = y + a * rad.ka2_ratio * voigt_profile(x - x2, s, g)
        return y
    f.xmid = 0.0
    return f


def _clusters(peaks, window_fwhm, ka2_extra):
    """§3.3 group peaks whose fit windows overlap."""
    spans = []
    for p in sorted(peaks, key=lambda q: q["two_theta"]):
        half = window_fwhm * p["fwhm_est"]
        lo = p["two_theta"] - half
        hi = p["two_theta"] + half + ka2_extra(p["two_theta"])
        if spans and lo <= spans[-1]["hi"]:
            spans[-1]["hi"] = max(spans[-1]["hi"], hi)
            spans[-1]["peaks"].append(p)
        else:
            spans.append({"lo": lo, "hi": hi, "peaks": [p]})
    return spans


def fit_peaks(x, y, peaks, rad: Radiation, window_fwhm=DEFAULTS["window_fwhm"],
              min_points_fwhm=DEFAULTS["min_points_fwhm"], noise=None, counts=False):
    """Fit every peak. Returns (list of per-peak results, model array,
    background array). Model/background are NaN outside fitted windows."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    model = np.full_like(y, np.nan)
    bkg = np.full_like(y, np.nan)
    results = []

    fittable, skipped = [], []
    # §3.4 with a doublet, the Kalpha2 line of a strong reflection is found as a
    # peak of its own. It is part of the Kalpha1 model, not a second reflection:
    # fitting it as one would give it its own Kalpha2 and split the breadth.
    peaks, partners = _drop_ka2_partners(peaks, rad)
    for q, parent in partners:
        skipped.append({**_peak_stub(q), "status": "ka2_partner",
                        "reason": f"Kalpha2 of the reflection at {parent:.3f} deg; "
                                  "fitted as part of it"})
    for p in peaks:
        if p["points_fwhm"] < min_points_fwhm:
            skipped.append({**_peak_stub(p), "status": "not_fitted",
                            "reason": (f"{p['points_fwhm']} points above half height "
                                       f"(< {min_points_fwhm}); breadth would be set "
                                       "by the step size, not the sample")})
        else:
            fittable.append(p)
    results.extend(skipped)

    def ka2_extra(tt):
        if not rad.doublet:
            return 0.0
        return float(ka2_position(tt, rad.lam1, rad.lam2) - tt)

    # Reflections that cannot be fitted (too few points: typically a single-
    # crystal substrate line) are still in the data. Inside another peak's window
    # they would be absorbed into its background or its tails, so their points
    # are left out of that fit -- masked, not deleted; the ranges are reported.
    unfit = [q for q in skipped if q["status"] == "not_fitted"]
    for cl in _clusters(fittable, window_fwhm, ka2_extra):
        sel = (x >= cl["lo"]) & (x <= cl["hi"])
        masked = []
        weakest = min(p["prominence"] for p in cl["peaks"])
        for q in unfit:
            # Only a feature stronger than the weakest peak being fitted can pull
            # the fit; weaker ones (thickness fringes, noise) stay in the data.
            if q["prominence"] <= weakest:
                continue
            half = 3.0 * q["fwhm_est"]
            if not (cl["lo"] - half <= q["two_theta"] <= cl["hi"] + half):
                continue
            if any(abs(q["two_theta"] - p["two_theta"]) < p["fwhm_est"] for p in cl["peaks"]):
                continue           # inside a fitted peak's own core: same feature
            m = (x >= q["two_theta"] - half) & (x <= q["two_theta"] + half)
            if rad.doublet:        # its Kalpha2 line too
                x2 = float(ka2_position(q["two_theta"], rad.lam1, rad.lam2))
                m |= (x >= x2 - half) & (x <= x2 + half)
            if np.any(sel & m):
                sel &= ~m
                masked.append([round(q["two_theta"] - half, 4), round(q["two_theta"] + half, 4)])
        xs, ys = x[sel], y[sel]
        n = len(cl["peaks"])
        if xs.size < 4 * n + 2 + 3:
            for p in cl["peaks"]:
                results.append({**_peak_stub(p), "status": "not_fitted",
                                "reason": "too few points in the fit window"})
            continue
        f = _cluster_model(n, rad)
        f.xmid = float(xs.mean())
        p0, lo, hi = [], [], []
        edge = max(3, xs.size // 20)
        b_left, b_right = np.median(ys[:edge]), np.median(ys[-edge:])
        for p in cl["peaks"]:
            w = p["fwhm_est"]
            height = max(p["prominence"], 1e-12)
            # area of a peak of this height and width, split between the two
            # components half and half as a neutral start
            s0, g0 = 0.5 * w / FWHM_PER_SIGMA * 1.4, 0.25 * w
            a0 = height / max(voigt_profile(0.0, s0, g0), 1e-12)
            if rad.doublet:
                a0 /= (1.0 + rad.ka2_ratio)
            p0 += [p["two_theta"], a0, s0, g0]
            lo += [p["two_theta"] - w, 0.0, 1e-7, 1e-7]
            hi += [p["two_theta"] + w, np.inf, 5 * w, 5 * w]
        slope0 = (b_right - b_left) / (xs[-1] - xs[0]) if xs[-1] != xs[0] else 0.0
        p0 += [min(b_left, b_right), slope0]
        lo += [-np.inf, -np.inf]
        hi += [np.inf, np.inf]
        try:
            popt, pcov = curve_fit(f, xs, ys, p0=p0, bounds=(lo, hi),
                                   method="trf", maxfev=20000)
        except Exception as exc:  # scipy raises RuntimeError / ValueError
            for p in cl["peaks"]:
                results.append({**_peak_stub(p), "status": "not_fitted",
                                "reason": f"fit did not converge: {exc}"})
            continue
        fit = f(xs, *popt)
        model[sel] = fit
        b0, b1 = popt[-2], popt[-1]
        bkg[sel] = b0 + b1 * (xs - f.xmid)
        resid = ys - fit
        rms = float(np.sqrt(np.mean(resid ** 2)))
        # Reduced chi^2 with Poisson variance (= model). ~1 means the fit is at
        # counting noise. Only meaningful when the intensity is raw counts.
        dof = max(xs.size - len(popt), 1)
        chi2 = float(np.sum(resid ** 2 / np.maximum(fit, 1.0)) / dof)
        perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if np.all(np.isfinite(pcov)) \
            else np.full(len(popt), np.nan)
        for j, p in enumerate(cl["peaks"]):
            x0, a, s, g = popt[4 * j:4 * j + 4]
            ex0, ea, es, eg = perr[4 * j:4 * j + 4]
            cov4 = pcov[4 * j:4 * j + 4, 4 * j:4 * j + 4]
            res = _peak_stub(p)
            res.update({
                "status": "ok",
                "two_theta": float(x0), "two_theta_err": float(ex0),
                "area": float(a), "area_err": float(ea),
                "sigma_deg": float(s), "sigma_err": float(es),
                "gamma_deg": float(g), "gamma_err": float(eg),
                "height": float(a * voigt_profile(0.0, s, g)),
                "fit_window": [float(xs[0]), float(xs[-1])],
                "masked_ranges": masked or None,
                "peaks_in_window": n,
                "residual_rms": rms,
                "chi2_red_poisson": chi2,
                "_cov_sg": cov4[2:4, 2:4].tolist(),
            })
            # A 'peak' whose fitted area is not distinguishable from zero is
            # noise that crossed the prominence threshold. Judged by the fit's own
            # uncertainty (area / its standard error < 3), not a retuned threshold.
            if counts:
                from scipy.stats import chi2 as _chi2
                pval = float(_chi2.sf(chi2 * dof, dof))
                res["chi2_pvalue"] = pval
                if pval < 1e-3:
                    res.setdefault("flags", []).append(
                        f"fit not within counting noise (chi2_red {chi2:.1f}, p < 1e-3): "
                        "asymmetry or an unmodelled component; breadth less reliable")
            if not np.isfinite(ea) or a < 3.0 * ea:
                res["status"] = "not_significant"
                res["reason"] = (f"fitted area {a:.3g} +/- {ea:.3g}: below 3 standard "
                                 "errors, indistinguishable from noise")
            elif not (lo[4 * j] < x0 < hi[4 * j]):
                res["status"] = "suspect"
                res["reason"] = "centre ran to its bound"
            results.append(res)
    results.sort(key=lambda r: r["two_theta"])
    return results, model, bkg


def _drop_ka2_partners(peaks, rad):
    if not rad.doublet:
        return list(peaks), []
    keep, partners, taken = [], [], set()
    for p in sorted(peaks, key=lambda q: -q["prominence"]):
        if id(p) in taken:
            continue
        keep.append(p)
        x2 = float(ka2_position(p["two_theta"], rad.lam1, rad.lam2))
        for q in peaks:
            if q is p or id(q) in taken or q["prominence"] >= p["prominence"]:
                continue
            if abs(q["two_theta"] - x2) <= max(p["fwhm_est"], q["fwhm_est"]):
                taken.add(id(q))
                partners.append((q, p["two_theta"]))
    keep = [p for p in keep if id(p) not in taken]
    return keep, partners


def _peak_stub(p):
    return {"two_theta": p["two_theta"], "snr": round(p["snr"], 1),
            "prominence": p["prominence"],
            "fwhm_est": p["fwhm_est"], "points_fwhm": p["points_fwhm"]}


# ── §4 instrument function ───────────────────────────────────────────────────

@dataclass
class Instrument:
    """FWHM_G(2theta) and FWHM_L(2theta) of the instrument, degrees."""
    kind: str                                 # "standard" | "stated" | "none"
    U: float = 0.0; V: float = 0.0; W: float = 0.0     # Caglioti, Gaussian
    X: float = 0.0; Y: float = 0.0                      # TCH, Lorentzian
    fwhm_g_const: float | None = None
    fwhm_l_const: float | None = None
    range_deg: tuple | None = None
    n_peaks: int = 0
    notes: list = field(default_factory=list)

    def at(self, two_theta_deg):
        if self.kind == "none":
            return 0.0, 0.0, None
        if self.kind == "stated":
            return self.fwhm_g_const or 0.0, self.fwhm_l_const or 0.0, None
        th = math.radians(two_theta_deg / 2.0)
        t = math.tan(th)
        g2 = self.U * t * t + self.V * t + self.W
        l = self.X * t + self.Y / math.cos(th)
        flag = None
        if g2 < 0 or l < 0:
            flag = "instrument function negative here (fit of the standard does not hold)"
        if self.range_deg and not (self.range_deg[0] <= two_theta_deg <= self.range_deg[1]):
            flag = (f"extrapolated: standard spans {self.range_deg[0]:.1f}-"
                    f"{self.range_deg[1]:.1f} deg 2theta")
        return math.sqrt(max(g2, 0.0)), max(l, 0.0), flag

    def describe(self):
        d = {"kind": self.kind, "notes": self.notes}
        if self.kind == "standard":
            d.update({"caglioti_UVW": [self.U, self.V, self.W],
                      "lorentz_XY": [self.X, self.Y],
                      "range_deg": self.range_deg, "n_peaks": self.n_peaks})
        elif self.kind == "stated":
            d.update({"fwhm_gauss_deg": self.fwhm_g_const,
                      "fwhm_lorentz_deg": self.fwhm_l_const})
        return d


def instrument_from_peaks(fitted):
    """§4.1 fit Caglioti (Gaussian) and TCH X,Y (Lorentzian) to the widths of a
    standard's peaks."""
    ok = [p for p in fitted if p.get("status") == "ok"]
    notes = []
    if not ok:
        raise ValueError("no peak of the standard could be fitted")
    tt = np.array([p["two_theta"] for p in ok])
    th = np.radians(tt / 2.0)
    t = np.tan(th)
    fg = np.array([p["sigma_deg"] for p in ok]) * FWHM_PER_SIGMA
    fl = np.array([p["gamma_deg"] for p in ok]) * 2.0
    inst = Instrument(kind="standard", range_deg=(float(tt.min()), float(tt.max())),
                      n_peaks=len(ok))
    if len(ok) >= 3:
        A = np.column_stack([t * t, t, np.ones_like(t)])
        inst.U, inst.V, inst.W = map(float, np.linalg.lstsq(A, fg ** 2, rcond=None)[0])
    else:
        inst.W = float(np.mean(fg ** 2))
        notes.append(f"only {len(ok)} standard peak(s): Gaussian width taken as "
                     "constant (Caglioti needs >= 3)")
    if len(ok) >= 2:
        A = np.column_stack([t, 1.0 / np.cos(th)])
        inst.X, inst.Y = map(float, np.linalg.lstsq(A, fl, rcond=None)[0])
    else:
        inst.Y = float(fl[0] * np.cos(th[0]))
        notes.append("only 1 standard peak: Lorentzian width scaled as 1/cos(theta)")
    inst.notes = notes
    return inst


# ── §4 deconvolution, §5 single-line, §7 dislocation ─────────────────────────

def _propagate(fun, x, cov):
    """First-order error propagation with a numerical gradient."""
    x = np.asarray(x, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(cov)):
        return float("nan")
    f0 = fun(*x)
    if f0 is None or not np.isfinite(f0):
        return float("nan")
    grad = np.zeros_like(x)
    for i in range(x.size):
        h = 1e-6 * max(abs(x[i]), 1e-9)
        xp = x.copy(); xp[i] += h
        fp = fun(*xp)
        if fp is None or not np.isfinite(fp):
            return float("nan")
        grad[i] = (fp - f0) / h
    return float(math.sqrt(max(grad @ cov @ grad, 0.0)))


def analyse_peak(pk, rad: Radiation, inst: Instrument, K=DEFAULTS["scherrer_k"],
                 burgers_nm=None):
    """Add sample breadths, Scherrer and Voigt size/strain and delta to one
    fitted peak (in place). Returns pk."""
    if pk.get("status") not in ("ok", "suspect"):
        return pk
    tt = pk["two_theta"]
    th = math.radians(tt / 2.0)
    lam_nm = rad.lam1 / 10.0
    pk["d_A"] = rad.lam1 / (2.0 * math.sin(th))

    fg_i, fl_i, iflag = inst.at(tt)
    pk["inst_fwhm_gauss_deg"] = fg_i
    pk["inst_fwhm_lorentz_deg"] = fl_i
    if iflag:
        pk.setdefault("flags", []).append(iflag)

    s, g = pk["sigma_deg"], pk["gamma_deg"]
    cov = pk.pop("_cov_sg")

    def comps(s_, g_):
        fg_h, fl_h = s_ * FWHM_PER_SIGMA, 2.0 * g_
        return fg_h ** 2 - fg_i ** 2, fl_h - fl_i      # (FWHM_G^2, FWHM_L) sample

    fg2, fl = comps(s, g)
    pk["meas_fwhm_deg"] = voigt_fwhm(2 * g, s * FWHM_PER_SIGMA)
    pk["meas_fwhm_err"] = _propagate(lambda a, b: voigt_fwhm(2 * b, a * FWHM_PER_SIGMA),
                                     [s, g], cov)
    pk["meas_beta_deg"] = math.degrees(voigt_integral_breadth(
        math.radians(math.pi * g), math.radians(s * SQRT_2PI)))

    # A component whose fitted width is within 2 standard errors of zero is not
    # determined by the data (typically gamma or sigma sitting on its bound).
    # Dividing by it gives a size of 10^7 nm -- measured on XRD-918-16_10.xrdml
    # before this check existed. Treat it as not resolved.
    es, eg = (math.sqrt(max(cov[0][0], 0.0)), math.sqrt(max(cov[1][1], 0.0))) \
        if np.all(np.isfinite(cov)) else (float("inf"), float("inf"))
    g_det, l_det = s > 2.0 * es, g > 2.0 * eg
    for name, det in (("gaussian", g_det), ("lorentzian", l_det)):
        if not det:
            pk.setdefault("flags", []).append(
                f"measured {name} width not distinguishable from 0 (< 2 s.e.)")
    # Deconvolution of the total breadth uses the fitted components whether or
    # not each is individually determined: sigma and gamma are strongly
    # correlated, and their combination (the FWHM) can be well determined when
    # neither is. Only the component-wise method (§5.2) needs each one.
    g_ok, l_ok = fg2 > 0, fl > 0
    pk["gauss_resolved"], pk["lorentz_resolved"] = bool(g_ok), bool(l_ok)
    if not g_ok and not l_ok:
        pk["status"] = "unresolved"
        pk["reason"] = ("measured breadth is not larger than the instrument's: "
                        "sample broadening not resolved, no size reported")
        return pk
    zeroed = [n for n, ok in (("gaussian", g_ok), ("lorentzian", l_ok)) if not ok]
    if zeroed:
        pk.setdefault("flags", []).append(
            f"{zeroed[0]} sample component <= instrument; taken as 0")

    def sample_fwhm(s_, g_):
        a, b = comps(s_, g_)
        return voigt_fwhm(max(b, 0.0), math.sqrt(max(a, 0.0)))

    def sample_beta_rad(s_, g_):
        a, b = comps(s_, g_)
        fg = math.sqrt(max(a, 0.0))
        bg = math.radians(fg / FWHM_PER_SIGMA * SQRT_2PI)
        bl = math.radians(math.pi * max(b, 0.0) / 2.0)
        return voigt_integral_breadth(bl, bg)

    fw = sample_fwhm(s, g)
    pk["sample_fwhm_deg"] = fw
    pk["sample_fwhm_err"] = _propagate(sample_fwhm, [s, g], cov)
    pk["sample_beta_deg"] = math.degrees(sample_beta_rad(s, g))
    if not fw > 2.0 * pk["sample_fwhm_err"]:
        pk["status"] = "unresolved"
        pk["reason"] = (f"sample FWHM {fw:.4g} +/- {pk['sample_fwhm_err']:.2g} deg is not "
                        "distinguishable from zero after the instrument subtraction")
        return pk

    cos_t, tan_t = math.cos(th), math.tan(th)

    # §5.1 Scherrer, FWHM, K as given
    def D_fwhm(s_, g_):
        b = math.radians(sample_fwhm(s_, g_))
        return K * lam_nm / (b * cos_t) if b > 0 else float("nan")
    pk["D_scherrer_nm"] = D_fwhm(s, g)
    pk["D_scherrer_err"] = _propagate(D_fwhm, [s, g], cov)

    # §5.1 Scherrer, integral breadth, K = 1 -> volume-weighted column length
    def D_ib(s_, g_):
        b = sample_beta_rad(s_, g_)
        return lam_nm / (b * cos_t) if b > 0 else float("nan")
    pk["D_intbreadth_nm"] = D_ib(s, g)
    pk["D_intbreadth_err"] = _propagate(D_ib, [s, g], cov)

    # §5.2 single-line Voigt: size from Lorentzian, strain from Gaussian
    def D_v(s_, g_):
        _, b = comps(s_, g_)
        bl = math.radians(math.pi * b / 2.0)
        return lam_nm / (bl * cos_t) if b > 0 else float("nan")

    def eps_v(s_, g_):
        a, _ = comps(s_, g_)
        if a <= 0:
            return float("nan")
        bg = math.radians(math.sqrt(a) / FWHM_PER_SIGMA * SQRT_2PI)
        return bg / (4.0 * tan_t)
    pk["D_voigt_nm"] = D_v(s, g) if (l_ok and l_det) else None
    pk["D_voigt_err"] = _propagate(D_v, [s, g], cov) if (l_ok and l_det) else None
    pk["strain_voigt"] = eps_v(s, g) if (g_ok and g_det) else None
    pk["strain_voigt_err"] = _propagate(eps_v, [s, g], cov) if (g_ok and g_det) else None

    # §7.1 delta = 1/D^2, per D
    pk["delta_scherrer_m2"] = _delta(pk["D_scherrer_nm"])
    # §7.2 only with a stated Burgers vector
    if burgers_nm and pk.get("strain_voigt") and pk.get("D_voigt_nm"):
        pk["rho_ws_m2"] = (2 * math.sqrt(3) * pk["strain_voigt"]
                           / (pk["D_voigt_nm"] * 1e-9 * burgers_nm * 1e-9))
    return pk


def _delta(D_nm):
    if D_nm is None or not np.isfinite(D_nm) or D_nm <= 0:
        return None
    return 1.0 / (D_nm * 1e-9) ** 2


# ── §6 multi-line ────────────────────────────────────────────────────────────

def _usable(peaks):
    return [p for p in peaks if p.get("sample_fwhm_deg") and p.get("sample_beta_deg")]


def williamson_hall(peaks, rad: Radiation, K=DEFAULTS["scherrer_k"],
                    min_n=DEFAULTS["min_wh_peaks"], breadth="fwhm"):
    """§6.1 beta cos(theta) = K lam / D + 4 eps sin(theta).
    breadth="fwhm": sample FWHM with K (the thin-film literature convention).
    breadth="integral": sample integral breadth with K = 1, as Williamson & Hall
    (1953) wrote it; D is then a volume-weighted apparent size."""
    use = _usable(peaks)
    if breadth == "integral":
        K = 1.0
    out = {"model": "UDM", "breadth": breadth, "K": K, "n": len(use),
           "peaks_2theta": [round(p["two_theta"], 4) for p in use]}
    if len(use) < min_n:
        out["status"] = "not_computed"
        out["reason"] = (f"{len(use)} usable reflection(s); need >= {min_n} so that "
                         "the straight line is tested rather than fitted exactly")
        return out
    key = "sample_fwhm_deg" if breadth == "fwhm" else "sample_beta_deg"
    th = np.radians([p["two_theta"] / 2.0 for p in use])
    b = np.radians([p[key] for p in use])
    xv, yv = 4.0 * np.sin(th), b * np.cos(th)
    r = linregress(xv, yv)
    lam_nm = rad.lam1 / 10.0
    D = K * lam_nm / r.intercept if r.intercept > 0 else None
    out.update({
        "status": "ok",
        "strain": float(r.slope), "strain_err": float(r.stderr),
        "intercept": float(r.intercept), "intercept_err": float(r.intercept_stderr),
        "r2": float(r.rvalue ** 2),
        "D_nm": D,
        "D_err": (K * lam_nm * r.intercept_stderr / r.intercept ** 2) if D else None,
        "delta_m2": _delta(D),
        "points": [{"x_4sin": float(a), "y_bcos": float(c)} for a, c in zip(xv, yv)],
    })
    if r.intercept <= 0:
        out["flags"] = ["intercept <= 0: size not defined by this fit"]
    if r.slope < 0:
        out.setdefault("flags", []).append(
            "negative slope: the uniform-strain model does not describe these "
            "reflections (often anisotropic broadening); reported, not clamped")
    return out


def size_strain_plot(peaks, rad: Radiation, K=DEFAULTS["ssp_k"],
                     min_n=DEFAULTS["min_wh_peaks"]):
    """§6.2 in reciprocal units, with the sample integral breadth:
        (beta*/d*)^2 = (K/D) (beta*/d*^2) + (2e)^2
    beta* = beta cos(theta) / lambda, d* = 1/d. e is the same apparent strain as
    in Williamson-Hall (beta_D = 4 e tan theta), so the two are comparable. The
    literature's '(eps/2)^2' uses eps = 4e (Stokes-Wilson eta); reported too."""
    use = _usable(peaks)
    out = {"model": "SSP", "breadth": "integral", "K": K, "n": len(use),
           "peaks_2theta": [round(p["two_theta"], 4) for p in use]}
    if len(use) < min_n:
        out["status"] = "not_computed"
        out["reason"] = f"{len(use)} usable reflection(s); need >= {min_n}"
        return out
    lam_nm = rad.lam1 / 10.0
    th = np.radians([p["two_theta"] / 2.0 for p in use])
    b = np.radians([p["sample_beta_deg"] for p in use])
    dstar = 10.0 / np.array([p["d_A"] for p in use])          # 1/nm
    bstar = b * np.cos(th) / lam_nm                            # 1/nm
    xv, yv = bstar / dstar ** 2, (bstar / dstar) ** 2
    r = linregress(xv, yv)
    D = K / r.slope if r.slope > 0 else None
    e = math.sqrt(r.intercept) / 2.0 if r.intercept > 0 else None
    out.update({
        "status": "ok", "slope": float(r.slope), "slope_err": float(r.stderr),
        "intercept": float(r.intercept), "intercept_err": float(r.intercept_stderr),
        "r2": float(r.rvalue ** 2), "D_nm": D,
        "D_err": (K * r.stderr / r.slope ** 2) if D else None,
        "strain": e,
        "strain_err": (r.intercept_stderr / (4.0 * e)) if e else None,
        "strain_eta_literature": 4.0 * e if e else None,
        "delta_m2": _delta(D),
    })
    if r.intercept <= 0:
        out["flags"] = ["intercept <= 0: strain not defined; model inconsistent "
                        "with these reflections (reported, not clamped)"]
    return out


# ── driver ───────────────────────────────────────────────────────────────────

def resolve_radiation(meta: dict, params: dict):
    """Radiation from the file first, the caller second. Returns (Radiation or
    None, questions)."""
    q = {}
    src = {}
    lam1 = meta.get("wavelength_a")
    src["wavelength_a"] = "file" if lam1 else None
    if not lam1 and params.get("wavelength_a"):
        lam1, src["wavelength_a"] = float(params["wavelength_a"]), "caller"
    if not lam1:
        q["wavelength_a"] = ("Kalpha1 wavelength in Angstrom. The file does not "
                             "state it; Cu Ka1 = 1.5406 is not assumed.")
    ratio = meta.get("ka2_ratio")
    lam2 = meta.get("wavelength_ka2_a")
    if ratio is not None:
        src["ka2_ratio"] = "file"
    elif params.get("ka2_ratio") is not None:
        ratio, src["ka2_ratio"] = float(params["ka2_ratio"]), "caller"
    else:
        q["ka2_ratio"] = ("Is the beam Kalpha1-only (monochromated) -> 0, or "
                          "Kalpha1+Kalpha2 -> the area ratio (PANalytical writes 0.5; "
                          "NIST 2017 measured 0.52 for Cu). The file does not say.")
    if lam2 is None and params.get("wavelength_ka2_a"):
        lam2 = float(params["wavelength_ka2_a"])
        src["wavelength_ka2_a"] = "caller"
    elif lam2 is not None:
        src["wavelength_ka2_a"] = "file"
    if ratio and not lam2:
        q["wavelength_ka2_a"] = "Kalpha2 wavelength in Angstrom (needed for the doublet)."
    if q:
        return None, q
    return Radiation(lam1=float(lam1), lam2=lam2, ka2_ratio=float(ratio or 0.0),
                     source=src), {}


def analyse_pattern(x, y, rad: Radiation, inst: Instrument, params: dict | None = None):
    """Run §2-§7 on one pattern. Returns (report dict, columns dict)."""
    p = {**DEFAULTS, **(params or {})}
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = p.get("two_theta_range")
    if rng:
        sel = (x >= rng[0]) & (x <= rng[1])
    else:
        sel = np.ones_like(x, dtype=bool)
    counts = p.get("counts")
    if counts is None:
        counts = is_counts(y[sel])
    peaks, noise = find_peaks(x[sel], y[sel], p["min_snr"], counts, p["false_alarm"])
    if p.get("peaks"):                   # caller-chosen positions
        want = [float(v) for v in p["peaks"]]
        peaks = [min(peaks, key=lambda q: abs(q["two_theta"] - w)) for w in want
                 if peaks] if peaks else []
        peaks = list({q["index"]: q for q in peaks}.values())
    fitted, model_s, bkg_s = fit_peaks(x[sel], y[sel], peaks, rad, p["window_fwhm"],
                                       p["min_points_fwhm"], counts=counts)
    model = np.full_like(y, np.nan); model[sel] = model_s
    bkg = np.full_like(y, np.nan); bkg[sel] = bkg_s
    for pk in fitted:
        analyse_peak(pk, rad, inst, p["scherrer_k"], p.get("burgers_nm"))
    for pk in fitted:
        pk.pop("_cov_sg", None)

    good = [q for q in fitted if q.get("status") == "ok"]
    if p.get("wh_peaks"):
        want = [float(v) for v in p["wh_peaks"]]
        good = [min(good, key=lambda q: abs(q["two_theta"] - w)) for w in want] if good else []
    wh = williamson_hall(good, rad, p["scherrer_k"], p["min_wh_peaks"], "fwhm")
    wh_ib = williamson_hall(good, rad, 1.0, p["min_wh_peaks"], "integral")
    ssp = size_strain_plot(good, rad, p["ssp_k"], p["min_wh_peaks"])

    ds = [q["D_scherrer_nm"] for q in good if q.get("D_scherrer_nm")]
    report = {
        "method_reference": "agents/vidur/XRD-LINE-PROFILE-METHODS.md",
        "radiation": {"lambda1_A": rad.lam1, "lambda2_A": rad.lam2,
                      "ka2_ratio": rad.ka2_ratio, "doublet_fitted": rad.doublet,
                      "source": rad.source},
        "instrument": inst.describe(),
        "instrument_corrected": inst.kind != "none",
        "constants": {k: p[k] for k in DEFAULTS},
        "noise": noise,
        "peaks_found": len(peaks),
        "peaks_fitted_ok": len(good),
        "peaks": fitted,
        "williamson_hall": wh,
        "williamson_hall_integral": wh_ib,
        "size_strain_plot": ssp,
        "summary": {
            "D_scherrer_mean_nm": float(np.mean(ds)) if ds else None,
            "D_scherrer_sd_nm": float(np.std(ds, ddof=1)) if len(ds) > 1 else None,
            "delta_from_mean_scherrer_m2": _delta(float(np.mean(ds))) if ds else None,
            "D_wh_fwhm_nm": wh.get("D_nm"), "strain_wh_fwhm": wh.get("strain"),
            "D_wh_integral_nm": wh_ib.get("D_nm"), "strain_wh_integral": wh_ib.get("strain"),
            "D_ssp_nm": ssp.get("D_nm"), "strain_ssp": ssp.get("strain"),
        },
        "caveats": [
            "delta = 1/D^2 is a conventional comparison index, not a measured "
            "dislocation density (Gubicza 2022; methods doc s7.3).",
            "standard errors are statistical only; the model and the instrument "
            "function dominate the real uncertainty (s3.5).",
        ],
    }
    if inst.kind == "none":
        report["caveats"].insert(0, "NO instrument correction: sizes are lower "
                                 "bounds and strains upper bounds (s4.1).")
    cols = {"two_theta": x, "raw": y, "model": model, "background": bkg,
            "residual": y - model}
    return report, cols


def instrument_from_params(params: dict, load_pattern=None):
    """Build the Instrument from params['instrument']. load_pattern(path) ->
    (x, y, meta) is used for a standard file."""
    spec = (params or {}).get("instrument") or {}
    if spec.get("standard_file"):
        if load_pattern is None:
            raise ValueError("a standard file needs a loader")
        xs, ys, meta = load_pattern(spec["standard_file"])
        rad, q = resolve_radiation(meta, spec)
        if q:
            raise ValueError(f"standard file: {q}")
        c = is_counts(ys, meta)
        pk, _ = find_peaks(xs, ys, spec.get("min_snr", DEFAULTS["min_snr"]), c)
        fitted, _, _ = fit_peaks(xs, ys, pk, rad, counts=c)
        inst = instrument_from_peaks(fitted)
        inst.notes.append(f"standard: {os.path.basename(spec['standard_file'])}")
        return inst
    if spec.get("fwhm_gauss_deg") is not None or spec.get("fwhm_lorentz_deg") is not None:
        return Instrument(kind="stated",
                          fwhm_g_const=float(spec.get("fwhm_gauss_deg") or 0.0),
                          fwhm_l_const=float(spec.get("fwhm_lorentz_deg") or 0.0),
                          notes=["stated by the caller, constant over 2theta"])
    return Instrument(kind="none", notes=["no standard and no stated widths"])


class AxisUndeclared(ValueError):
    """The file has a column header, but its x label is neither 2theta nor q."""


def load_xrd(path, axis_is_two_theta=None):
    """Parse an XRD file through VIDUR's own extractor/parser. Returns
    (axis, intensity, metadata). Refuses an uncalibrated or q axis, and an
    axis the file leaves undeclared unless the caller states it is 2theta
    (axis_is_two_theta=True)."""
    from extractor import extract
    from parsers import xrd
    parsed = xrd.parse(extract(path))
    md = parsed.get("metadata") or {}
    if not md.get("axis_calibrated", True):
        raise ValueError("axis is channel index, not 2theta: no breadth in angle "
                         "can be measured")
    declared = md.get("axis_is_two_theta", True)
    label = md.get("axis_label_in_file")
    if declared is False:
        raise ValueError(f"axis is q as declared by the file (x label {label!r}), "
                         "not 2theta")
    if declared is None:
        if axis_is_two_theta is not True:
            raise AxisUndeclared(
                f"The file's x label {label!r} does not say whether the axis is "
                "2theta or q. Pass axis_is_two_theta=true if it is 2theta in degrees.")
        md = {**md, "axis_is_two_theta": True, "axis_is_two_theta_source": "caller"}
    return np.asarray(parsed["axis"]), np.asarray(parsed["intensity"]), md


def run(path, params: dict | None = None, out_dir: str | None = None):
    """Full analysis of one file. Returns a JSON-safe report; writes
    <stem>_peaks.csv, <stem>_fit.csv, <stem>_microstructure.json when out_dir
    is given."""
    params = params or {}
    try:
        x, y, meta = load_xrd(path, params.get("axis_is_two_theta"))
    except AxisUndeclared as exc:
        return {"file": path, "status": "needs_input",
                "questions": {"axis_is_two_theta": str(exc)}}
    if "counts" not in params:
        params = {**params, "counts": is_counts(y, meta)}
    rad, q = resolve_radiation(meta, params)
    try:
        inst = instrument_from_params(params, load_xrd)
    except Exception as exc:
        q["instrument"] = f"instrument standard could not be used: {exc}"
        inst = None
    if q:
        return {"file": path, "status": "needs_input", "questions": q}
    report, cols = analyse_pattern(x, y, rad, inst, params)
    report["file"] = path
    report["status"] = "ok"
    if meta.get("axis_assumed"):
        report["axis_assumed"] = meta["axis_assumed"]
        report["caveats"].insert(0, "x axis ASSUMED to be 2theta: the file has no column "
                                    "header declaring it. If it is q or d, every result "
                                    "below is wrong.")
    if inst.kind == "none":
        report["questions"] = {"instrument": (
            "No instrumental broadening given. Pass instrument.standard_file (LaB6 "
            "SRM 660 or Si SRM 640 on the same optics) or instrument.fwhm_gauss_deg/"
            "fwhm_lorentz_deg. Results below are uncorrected.")}
    report = _jsonable(report)
    if out_dir:
        report["written"] = write_outputs(report, cols, path, out_dir)
    return report


PEAK_COLUMNS = [
    "two_theta", "two_theta_err", "d_A", "status", "reason", "height", "area",
    "meas_fwhm_deg", "meas_fwhm_err", "meas_beta_deg", "sigma_deg", "gamma_deg",
    "inst_fwhm_gauss_deg", "inst_fwhm_lorentz_deg",
    "sample_fwhm_deg", "sample_fwhm_err", "sample_beta_deg",
    "D_scherrer_nm", "D_scherrer_err", "D_intbreadth_nm", "D_intbreadth_err",
    "D_voigt_nm", "D_voigt_err", "strain_voigt", "strain_voigt_err",
    "delta_scherrer_m2", "rho_ws_m2", "chi2_red_poisson", "snr", "flags",
]


def write_outputs(report, cols, path, out_dir):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    p_peaks = os.path.join(out_dir, f"{stem}_peaks.csv")
    p_fit = os.path.join(out_dir, f"{stem}_fit.csv")
    p_json = os.path.join(out_dir, f"{stem}_microstructure.json")
    with open(p_peaks, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(PEAK_COLUMNS)
        for pk in report["peaks"]:
            row = []
            for c in PEAK_COLUMNS:
                v = pk.get(c)
                row.append("; ".join(v) if isinstance(v, list) else ("" if v is None else v))
            w.writerow(row)
    with open(p_fit, "w", newline="") as fh:
        w = csv.writer(fh)
        names = list(cols)
        w.writerow(names)
        for i in range(len(cols["raw"])):
            w.writerow(["" if not np.isfinite(cols[n][i]) else f"{cols[n][i]:.8g}"
                        for n in names])
    with open(p_json, "w") as fh:
        json.dump(report, fh, indent=2)
    return [p_peaks, p_fit, p_json]


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
