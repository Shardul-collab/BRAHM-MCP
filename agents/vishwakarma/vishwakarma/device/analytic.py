# vishwakarma/device/analytic.py
#
# Closed-form device physics used as an independent cross-check on every
# DEVSIM result. The numerical simulation and these formulas share only the
# material parameters -- if they disagree beyond the stated tolerance, either
# the mesh/solver setup or the parameter plumbing is wrong, and the result is
# flagged instead of returned as if it were fine.
#
# Tolerances are fixed here, before any comparison is run, and are not to be
# loosened to make a result pass:
#   diode  : J_sim / J_diffusion within [0.80, 1.25] where diffusion current
#            dominates and injection is low (junction voltage corrected for the
#            computed neutral-region series resistance); ideality in [0.97, 1.08]
#   MOSFET : |Vt_sim - Vt_chargesheet| <= 0.10 V, both extracted by max-gm
#            extrapolation (long channel); subthreshold swing within 15 % of
#            the ideal-MOS value

from __future__ import annotations

import math
from .params import MaterialParams, Q_E, EPS0

DIODE_RATIO_BAND = (0.80, 1.25)
DIODE_IDEALITY_BAND = (0.97, 1.08)
MOS_VT_TOL_V = 0.10
MOS_SS_REL_TOL = 0.15


# ─── p-n junction ───────────────────────────────────────────────────────────

def built_in(m: MaterialParams, NA: float, ND: float) -> float:
    return m.Vt * math.log(NA * ND / m.ni() ** 2)


def depletion_width(m: MaterialParams, NA: float, ND: float, V: float) -> float:
    es = m.v("eps_r") * EPS0
    vbi = built_in(m, NA, ND)
    return math.sqrt(2 * es * max(vbi - V, 1e-3) * (NA + ND) / (Q_E * NA * ND))


def _coth(x: float) -> float:
    return 1.0 / math.tanh(x) if x < 20 else 1.0


def diode_J0(m: MaterialParams, NA: float, ND: float, len_p: float, len_n: float,
             V: float = 0.0) -> float:
    """
    Diffusion saturation current with ohmic contacts at finite distance:
    J0 = q ni^2 [ Dn/(NA Ln) coth(Wp/Ln) + Dp/(ND Lp) coth(Wn/Lp) ],
    Wp, Wn = quasi-neutral widths at bias V. Reduces to the long- and
    short-diode limits automatically.
    """
    ni = m.ni()
    W = depletion_width(m, NA, ND, V)
    xp, xn = W * ND / (NA + ND), W * NA / (NA + ND)
    Wp, Wn = max(len_p - xp, 1e-9), max(len_n - xn, 1e-9)
    Dn, Dp = m.diffusivity("n"), m.diffusivity("p")
    Ln, Lp = m.diffusion_length("n"), m.diffusion_length("p")
    return Q_E * ni ** 2 * (Dn / (NA * Ln) * _coth(Wp / Ln) + Dp / (ND * Lp) * _coth(Wn / Lp))


def series_resistance(m: MaterialParams, NA: float, ND: float,
                      len_p: float, len_n: float) -> float:
    """Ohmic resistance of the two quasi-neutral regions, ohm cm^2."""
    W = depletion_width(m, NA, ND, 0.0)
    xp, xn = W * ND / (NA + ND), W * NA / (NA + ND)
    rho_p = 1.0 / (Q_E * m.v("mu_p_cm2Vs") * NA)
    rho_n = 1.0 / (Q_E * m.v("mu_n_cm2Vs") * ND)
    return max(len_p - xp, 0.0) * rho_p + max(len_n - xn, 0.0) * rho_n


def diode_check(m: MaterialParams, NA: float, ND: float, len_p: float, len_n: float,
                iv: list) -> dict:
    """
    Compare simulated forward current with the Shockley diffusion current.

    The comparison uses the junction voltage Vj = V - J*Rs, with Rs the ohmic
    resistance of the neutral regions computed from the same parameters (not
    fitted). Points are compared only where the Shockley model applies: low
    injection on both sides, and the Sah-Noyce-Shockley upper bound on
    depletion-region recombination below 5 % of the diffusion current.
    """
    ni, Vt = m.ni(), m.Vt
    tau_eff = math.sqrt(m.v("srh_lifetime_n_s") * m.v("srh_lifetime_p_s"))
    Rs = series_resistance(m, NA, ND, len_p, len_n)
    rows, window = [], []
    for r in iv:
        V, J = r["V"], r["J_A_cm2"]
        if V <= 0:
            continue
        Vj = V - J * Rs
        Jd = diode_J0(m, NA, ND, len_p, len_n, Vj) * math.expm1(Vj / Vt)
        Jr = Q_E * ni * depletion_width(m, NA, ND, Vj) / (2 * tau_eff) * math.expm1(Vj / (2 * Vt))
        low_inj = (ni ** 2 / NA * math.exp(Vj / Vt) < 0.1 * NA and
                   ni ** 2 / ND * math.exp(Vj / Vt) < 0.1 * ND)
        diff_dom = Jr < 0.05 * Jd
        ratio = J / Jd if Jd > 0 else float("nan")
        rows.append({"V": V, "V_junction": Vj, "J_sim": J, "J_diffusion": Jd,
                     "J_srh_bound": Jr, "ratio": ratio,
                     "in_window": bool(low_inj and diff_dom)})
        if low_inj and diff_dom:
            window.append((Vj, J, ratio))

    out = {"J0_diffusion_A_cm2": diode_J0(m, NA, ND, len_p, len_n, 0.0),
           "V_bi": built_in(m, NA, ND), "R_series_ohm_cm2": Rs, "rows": rows,
           "tolerances": {"ratio": DIODE_RATIO_BAND, "ideality": DIODE_IDEALITY_BAND}}
    rev = [r for r in iv if r["V"] < 0]
    if rev:
        Vr = min(r["V"] for r in rev)
        Jg = Q_E * ni * depletion_width(m, NA, ND, Vr) / (2 * tau_eff)
        out["reverse_info"] = {"V": Vr, "J_sim": next(r["J_A_cm2"] for r in rev if r["V"] == Vr),
                               "J_generation_bound": -Jg,
                               "note": "informational; qniW/2tau is an upper estimate"}
    if len(window) < 2:
        out["pass"] = None
        out["summary"] = ("fewer than two bias points where the Shockley model applies; "
                          "cannot cross-check (add forward points or lengthen lifetimes)")
        return out
    (V1, J1, _), (V2, J2, _) = window[0], window[-1]
    n_ideal = (V2 - V1) / (Vt * math.log(J2 / J1)) if J2 > 0 and J1 > 0 else float("nan")
    ratios = [w[2] for w in window]
    ok_ratio = all(DIODE_RATIO_BAND[0] <= r <= DIODE_RATIO_BAND[1] for r in ratios)
    ok_n = DIODE_IDEALITY_BAND[0] <= n_ideal <= DIODE_IDEALITY_BAND[1]
    out.update({
        "window_Vj": [V1, V2],
        "n_window_points": len(window),
        "ratio_range": [min(ratios), max(ratios)],
        "ideality_factor": n_ideal,
        "pass": bool(ok_ratio and ok_n),
        "summary": (f"J_sim/J_diffusion {min(ratios):.3f}-{max(ratios):.3f} over "
                    f"Vj {V1:.3f}-{V2:.3f} V ({len(window)} pts), ideality {n_ideal:.3f}"),
    })
    return out


# ─── MOS ────────────────────────────────────────────────────────────────────

def max_depletion_width(m: MaterialParams, NA: float) -> float:
    phiF = m.Vt * math.log(NA / m.ni())
    return math.sqrt(2 * m.v("eps_r") * EPS0 * 2 * phiF / (Q_E * NA))


def mos_threshold(m: MaterialParams, NA: float, tox: float, eps_ox: float,
                  gate_wf: float) -> dict:
    """Long-channel n-MOS threshold, no oxide/interface charge."""
    es = m.v("eps_r") * EPS0
    cox = eps_ox * EPS0 / tox
    phiF = m.Vt * math.log(NA / m.ni())
    # phi_s(Fermi) = phi_i + phiF for p-type
    vfb = gate_wf - (m.intrinsic_workfunction() + phiF)
    qd = math.sqrt(2 * es * Q_E * NA * 2 * phiF)
    vt = vfb + 2 * phiF + qd / cox
    cdep = es / max_depletion_width(m, NA)
    ss = math.log(10) * m.Vt * (1 + cdep / cox) * 1e3       # mV/dec
    return {"VFB": vfb, "phiF": phiF, "Cox_F_cm2": cox, "Qdep_C_cm2": qd,
            "Vt": vt, "SS_mV_dec": ss, "Wdmax_cm": max_depletion_width(m, NA)}


def charge_sheet_transfer(m: MaterialParams, NA: float, tox: float, eps_ox: float,
                          gate_wf: float, L: float, vd: float, vgs: list) -> list:
    """
    Long-channel, small-Vd drain current from exact 1D MOS electrostatics
    (Boltzmann, uniform NA): Vg = VFB + psi_s + Qs(psi_s)/Cox, and
    Id = mu_n (1/L) Vd Qi(psi_s) per unit width, with Qi the electron sheet
    charge integrated through the depletion field. No fitting; the same
    extraction is then applied as to the DEVSIM curve, so the comparison is
    between like quantities. Returns [{Vg, Id_A_per_um}].
    """
    import numpy as np
    es = m.v("eps_r") * EPS0
    cox = eps_ox * EPS0 / tox
    Vt, ni = m.Vt, m.ni()
    n0 = ni ** 2 / NA
    vfb = mos_threshold(m, NA, tox, eps_ox, gate_wf)["VFB"]
    LD = math.sqrt(2 * es * Vt / (Q_E * NA))          # E = (Vt/LD)... prefactor below

    def F(psi):
        u = psi / Vt
        return np.sqrt(np.maximum((np.exp(-u) + u - 1) + (n0 / NA) * (np.exp(u) - u - 1), 0))

    def Efield(psi):                                   # V/cm, psi > 0 (depletion side)
        return math.sqrt(2 * Q_E * NA * Vt / es) * F(psi)

    def Qs(psi):
        return es * Efield(psi)

    def Qi(psi_s):
        if psi_s <= 0:
            return 0.0
        psi = np.linspace(psi_s * 1e-6, psi_s, 4000)
        integrand = Q_E * n0 * np.expm1(psi / Vt) / np.maximum(Efield(psi), 1e-30)
        return float(np.trapezoid(integrand, psi)) if hasattr(np, "trapezoid") else \
            float(np.trapz(integrand, psi))

    def vg_of(psi_s):
        return vfb + psi_s + Qs(psi_s) / cox

    out = []
    mu = m.v("mu_n_cm2Vs")
    for vg in vgs:
        lo, hi = 1e-6, 1.6 * 2 * Vt * math.log(NA / ni) + 0.5
        if vg_of(lo) > vg:
            out.append({"Vg": vg, "Id_A_per_um": 0.0})
            continue
        for _ in range(100):                           # bisection on psi_s
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if vg_of(mid) < vg else (lo, mid)
        # Qs(psi)/Cox uses the field form above; Qi integrates n through it.
        Id = mu * Qi(0.5 * (lo + hi)) * vd / L         # A/cm of width
        out.append({"Vg": vg, "Id_A_per_um": Id * 1e-4})
    return out


def extract_vt_maxgm(pts: list, vd: float) -> tuple:
    gm = [((pts[i + 1][1] - pts[i - 1][1]) / (pts[i + 1][0] - pts[i - 1][0]), i)
          for i in range(1, len(pts) - 1)]
    gmax, imax = max(gm)
    vg0, id0 = pts[imax]
    return vg0 - id0 / gmax - vd / 2, gmax, id0


def mosfet_check(m: MaterialParams, NA: float, tox: float, eps_ox: float, gate_wf: float,
                 L: float, vd: float, transfer: list) -> dict:
    """
    Gate: |Vt_sim - Vt_chargesheet| <= 0.10 V, both by max-gm extrapolation on
    the same Vg grid; SS within 15 % of ideal. The classical 2*phiF threshold
    is reported for information only -- max-gm extrapolation sits a few kT/q
    above it by construction (psi_s keeps rising past 2*phiF in inversion).
    """
    th = mos_threshold(m, NA, tox, eps_ox, gate_wf)
    pts = [(r["Vg"], r["Id_A_per_um"]) for r in transfer]
    out = {"analytic": th, "tolerances": {"Vt_V": MOS_VT_TOL_V, "SS_rel": MOS_SS_REL_TOL}}
    if len(pts) < 5:
        out.update({"pass": None, "summary": "too few transfer points"})
        return out

    vt_sim, gmax, id0 = extract_vt_maxgm(pts, vd)
    cs = charge_sheet_transfer(m, NA, tox, eps_ox, gate_wf, L, vd, [p[0] for p in pts])
    vt_cs, _, _ = extract_vt_maxgm([(r["Vg"], r["Id_A_per_um"]) for r in cs], vd)

    sub = [(vg, i) for vg, i in pts if 0 < i < id0 / 100]
    ss_vals = [(b[0] - a[0]) / math.log10(b[1] / a[1]) * 1e3
               for a, b in zip(sub, sub[1:]) if b[1] > a[1]]
    ss_sim = min(ss_vals) if ss_vals else float("nan")
    mu_ext = gmax * 1e4 * L / (th["Cox_F_cm2"] * vd)

    dvt = vt_sim - vt_cs
    ok_vt = abs(dvt) <= MOS_VT_TOL_V
    ok_ss = (not math.isnan(ss_sim)) and abs(ss_sim / th["SS_mV_dec"] - 1) <= MOS_SS_REL_TOL
    out.update({
        "Vt_sim": vt_sim, "Vt_chargesheet_extrapolated": vt_cs, "dVt": dvt,
        "Vt_classical_2phiF": th["Vt"],
        "SS_sim_mV_dec": ss_sim,
        "gm_max_S_per_um": gmax,
        "mu_eff_extracted_cm2Vs": mu_ext,
        "mu_input_cm2Vs": m.v("mu_n_cm2Vs"),
        "chargesheet_transfer": cs,
        "pass": bool(ok_vt and ok_ss),
        "summary": (f"Vt sim {vt_sim:.3f} V vs charge-sheet {vt_cs:.3f} V (dVt {dvt:+.3f}; "
                    f"classical 2phiF {th['Vt']:.3f}); SS sim {ss_sim:.1f} vs ideal "
                    f"{th['SS_mV_dec']:.1f} mV/dec; mu_eff from gm {mu_ext:.0f} vs input "
                    f"{m.v('mu_n_cm2Vs'):.0f} cm2/Vs"),
    })
    return out
