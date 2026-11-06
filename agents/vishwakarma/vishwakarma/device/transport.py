# vishwakarma/device/transport.py
#
# DFT band structure -> transport-relevant band parameters, via BoltzTraP2.
#
# What this computes WITHOUT any scattering assumption (pure band structure):
#   * the DFT gap (on the interpolated mesh; uncorrected -- PBE underestimates)
#   * effective densities of states Nc, Nv, by integrating the interpolated DOS
#     against the Boltzmann factor (no parabolic-band assumption)
#   * conductivity effective masses m_c = n e^2 / (sigma/tau) for electrons
#     and holes, from the conduction / valence manifolds fitted separately
#
# What it CANNOT compute: mobility. BoltzTraP2's constant-relaxation-time
# approximation gives sigma/tau; mobility needs tau, which comes from EPW
# (expensive), the literature (SHANI), or the user. mobility_from_tau() and
# tau_from_mobility() convert between the two once m_c is known, and the
# second is a useful sanity check: a literature mobility that implies a
# momentum relaxation time far outside ~1e-15..1e-12 s points at a unit slip
# or a mismatched band structure.
#
# Electrons and holes are fitted on separate band manifolds so the result does
# not depend on the (wrong) DFT gap: a scissor correction would shift the
# manifolds rigidly and change none of the numbers reported here.

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

from .params import Q_E, M_E, K_B

log = logging.getLogger("vishwakarma.device.transport")

HA_EV = 27.211386245988


# ─── locating the QE data ───────────────────────────────────────────────────

def locate_save_dir(job_id: Optional[str] = None, workdir: Optional[str] = None,
                    save_dir: Optional[str] = None) -> Path:
    """
    Find the directory holding data-file-schema.xml for a Vishwakarma job.

    NSCF/bands steps write into the SCF step's outdir (workflow.py shares it),
    so the job's own folder is not enough: read outdir/prefix from input.in
    first, then fall back to searching the job folder. Old jobs carry WSL-era
    absolute outdirs (/mnt/d/brahm/...); those are re-rooted onto workdir.
    """
    if save_dir:
        p = Path(save_dir)
        if (p / "data-file-schema.xml").is_file():
            return p
        raise FileNotFoundError(f"no data-file-schema.xml in {p}")
    if not (job_id and workdir):
        raise ValueError("pass save_dir, or job_id together with workdir")

    job = Path(workdir) / job_id
    if not job.is_dir():
        raise FileNotFoundError(f"job directory not found: {job}")

    candidates: list[Path] = []
    inp = job / "input.in"
    if inp.is_file():
        text = inp.read_text(errors="replace")
        m_out = re.search(r"outdir\s*=\s*['\"](.+?)['\"]", text)
        m_pre = re.search(r"prefix\s*=\s*['\"](.+?)['\"]", text)
        prefix = m_pre.group(1) if m_pre else "pwscf"
        if m_out:
            out = Path(m_out.group(1))
            if not out.is_absolute():
                out = job / out
            candidates.append(out / f"{prefix}.save")
            # re-root an absolute path from another mount onto workdir
            parts = out.parts
            if "jobs" in parts:
                tail = parts[parts.index("jobs") + 1:]
                candidates.append(Path(workdir).joinpath(*tail) / f"{prefix}.save")
    candidates += sorted(job.glob("**/*.save"))
    for c in candidates:
        if (c / "data-file-schema.xml").is_file():
            return c
    raise FileNotFoundError(
        f"no data-file-schema.xml for job {job_id} (looked in: "
        + ", ".join(str(c) for c in candidates[:6]) + ")")


def xml_run_info(save_dir: Path) -> dict:
    """calculation type, k-point count and band count recorded in the xml."""
    import xml.etree.ElementTree as et
    root = et.parse(Path(save_dir) / "data-file-schema.xml").getroot()
    info = {}
    calc = root.find("./input/control_variables/calculation")
    info["calculation"] = calc.text.strip() if calc is not None and calc.text else None
    nks = root.find("./output/band_structure/nks")
    nbnd = root.find("./output/band_structure/nbnd")
    info["nks"] = int(nks.text) if nks is not None else None
    info["nbnd"] = int(nbnd.text) if nbnd is not None else None
    lsda = root.find("./output/band_structure/lsda")
    nonc = root.find("./output/band_structure/noncolin")
    info["lsda"] = (lsda is not None and lsda.text.strip() == "true")
    info["noncolin"] = (nonc is not None and nonc.text.strip() == "true")
    return info


# ─── the BoltzTraP2 calculation ─────────────────────────────────────────────

def _fit(save_dir: Path, interp_factor: int):
    from BoltzTraP2 import dft, sphere, fite
    data = dft.DFTData(str(save_dir))
    eq = sphere.get_equivalences(data.atoms, data.magmom,
                                 int(interp_factor * len(data.kpoints)))
    coeffs = fite.fitde3D(data, eq)
    eband, vvband, _ = fite.getBTPbands(eq, coeffs, data.get_lattvec(),
                                        curvature=False)
    return data, eband, vvband


def _manifold(data, eband, vvband, sel, edge, sign, T, npts, mu_offset_Ha):
    """Nc/Nv and conductivity mass for one band manifold (CB: sign=+1)."""
    from BoltzTraP2 import bandlib as BL
    eb = eband[sel]
    vv = vvband[sel]                                # (nbands, 3, 3, nk)
    eps, dos, vvdos, _ = BL.BTPDOS(eb, vv, npts=npts)
    de = eps[1] - eps[0]
    kT = K_B * T / Q_E / HA_EV                      # Ha
    vol_cm3 = data.atoms.get_volume() * 1e-24

    x = sign * (eps - edge)                         # >= 0 inside the band
    w = np.where(x >= -0.5 * de, np.exp(-np.clip(x, 0, None) / kT), 0.0)
    neff = data.dosweight * float(np.sum(dos * w)) * de / vol_cm3

    mu = np.array([edge - sign * mu_offset_Ha])     # nondegenerate, in the gap
    Tr = np.array([float(T)])
    N, L0, L1, L2, _ = BL.fermiintegrals(eps, dos, vvdos, mur=mu, Tr=Tr,
                                         dosweight=data.dosweight)
    sigma, _, _, _ = BL.calc_Onsager_coefficients(L0, L1, L2, mu, Tr,
                                                  data.get_volume())
    sig = sigma[0, 0]                               # 3x3, S/(m s) = sigma/tau
    nstates = data.dosweight * eb.shape[0]
    ncell = -N[0, 0] if sign > 0 else nstates + N[0, 0]
    n_m3 = ncell / (vol_cm3 * 1e-6)
    sig_eig = np.linalg.eigvalsh(0.5 * (sig + sig.T))
    m_avg = n_m3 * Q_E ** 2 / (np.trace(sig) / 3.0) / M_E
    m_principal = [n_m3 * Q_E ** 2 / s / M_E for s in sig_eig]
    return {
        "Neff_cm3": neff,
        "m_cond": m_avg,                            # 3 / (sum 1/m_i)  (harmonic mean)
        "m_cond_principal": sorted(m_principal),
        "carriers_cm3_at_probe": ncell / vol_cm3,
        "sigma_over_tau_SI": (sig).tolist(),
    }


def band_transport(save_dir: str | Path, T: float = 300.0, interp_factor: int = 8,
                   npts: int = 6000, mu_offset_eV: float = 0.25,
                   check_convergence: bool = True) -> dict:
    """
    Band-structure transport parameters for a semiconductor DFT run.

    Requires a non-spin-polarised, collinear calculation with conduction bands
    (nbnd > occupied) on a reasonably dense uniform k-mesh -- i.e. the NSCF
    step of dos_workflow, not an SCF on a coarse grid.
    """
    save_dir = Path(save_dir)
    info = xml_run_info(save_dir)
    warnings: list[str] = []
    if info["lsda"] or info["noncolin"]:
        raise ValueError("spin-polarised / noncollinear runs are not supported yet "
                         "(band manifolds would need per-spin handling)")
    if info["calculation"] not in ("nscf", "scf"):
        raise ValueError(f"need a uniform-mesh scf/nscf run, got {info['calculation']!r} "
                         "(a 'bands' run is on a k-path and cannot be interpolated)")
    if info["nks"] is not None and info["nks"] < 60:
        warnings.append(f"only {info['nks']} irreducible k-points -- interpolation "
                        "may be unreliable; use the dense NSCF of dos_workflow")

    data, eband, vvband = _fit(save_dir, interp_factor)
    nval_f = data.nelect / data.dosweight
    nval = int(round(nval_f))
    if abs(nval - nval_f) > 1e-6:
        raise ValueError(f"fractional band filling (nelect={data.nelect}) -- not a "
                         "semiconductor in this calculation")
    if eband.shape[0] - nval < 2:
        raise ValueError(f"only {eband.shape[0] - nval} conduction band(s) present; "
                         "rerun the NSCF with larger nbnd")
    vbm, cbm = float(eband[:nval].max()), float(eband[nval:].min())
    gap = (cbm - vbm) * HA_EV
    grid_gap = (float(data.ebands[nval:].min()) - float(data.ebands[:nval].max())) * HA_EV
    if gap <= 0.02:
        raise ValueError(f"DFT gap {gap:.3f} eV: metallic or near-metallic in this "
                         "functional; drift-diffusion parameters are not meaningful")

    mu_off = mu_offset_eV / HA_EV
    mu_off = min(mu_off, 0.45 * (cbm - vbm))       # keep the probe inside the gap
    elec = _manifold(data, eband, vvband, slice(nval, None), cbm, +1, T, npts, mu_off)
    hole = _manifold(data, eband, vvband, slice(0, nval), vbm, -1, T, npts, mu_off)

    out = {
        "save_dir": str(save_dir),
        "run": info,
        "T_K": T,
        "interp_factor": interp_factor,
        "n_valence_bands": nval,
        "band_gap_dft_eV": gap,
        "band_gap_dft_grid_eV": grid_gap,
        "electrons": elec,
        "holes": hole,
        "warnings": warnings,
        "notes": [
            "band gap is the uncorrected DFT value; supply a corrected gap "
            "(hybrid / GW / DFT+U / literature) for device simulation",
            "no spin-orbit coupling unless the DFT run included it; for heavy "
            "elements this mostly affects the valence band (Nv, m_cond_p)",
        ],
    }

    if check_convergence:
        # Re-run with twice the interpolation density; report relative change.
        _, eb2, vv2 = _fit(save_dir, 2 * interp_factor)
        cbm2, vbm2 = float(eb2[nval:].min()), float(eb2[:nval].max())
        e2 = _manifold(data, eb2, vv2, slice(nval, None), cbm2, +1, T, npts, mu_off)
        h2 = _manifold(data, eb2, vv2, slice(0, nval), vbm2, -1, T, npts, mu_off)
        rel = {
            "Nc": abs(e2["Neff_cm3"] / elec["Neff_cm3"] - 1),
            "Nv": abs(h2["Neff_cm3"] / hole["Neff_cm3"] - 1),
            "m_cond_n": abs(e2["m_cond"] / elec["m_cond"] - 1),
            "m_cond_p": abs(h2["m_cond"] / hole["m_cond"] - 1),
            "gap": abs((cbm2 - vbm2) * HA_EV - gap),
        }
        out["convergence"] = {"compared_interp_factor": 2 * interp_factor,
                              "relative_change": rel}
        worst = max(v for k, v in rel.items() if k != "gap")
        if worst > 0.05:
            warnings.append(f"interpolation not converged: up to {worst:.1%} change "
                            f"on doubling interp_factor -- use a denser NSCF mesh")
    return out


# ─── mobility <-> relaxation time ───────────────────────────────────────────

def mobility_from_tau(tau_s: float, m_cond: float) -> float:
    """mu = e tau / m_c, returned in cm^2/(V s)."""
    return Q_E * tau_s / (m_cond * M_E) * 1e4


def tau_from_mobility(mu_cm2Vs: float, m_cond: float) -> float:
    """Momentum relaxation time implied by a mobility and a conductivity mass."""
    return mu_cm2Vs * 1e-4 * m_cond * M_E / Q_E


def tau_plausibility(tau_s: float) -> Optional[str]:
    if not (1e-16 <= tau_s <= 1e-11):
        return (f"implied momentum relaxation time {tau_s:.2e} s is outside "
                "1e-16..1e-11 s -- check the mobility units or the band structure")
    return None
