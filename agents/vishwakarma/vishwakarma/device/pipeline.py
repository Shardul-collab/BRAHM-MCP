# vishwakarma/device/pipeline.py
#
# DFT job -> MaterialParams. Glue between transport.py (what the band
# structure gives) and params.py (what TCAD needs), with every value tagged.
#
#   from Vishwakarma DFT (+BoltzTraP2):  Nc, Nv, m_cond_n, m_cond_p, DFT gap
#   from literature / SHANI / user:      corrected gap, electron affinity,
#                                        eps_r (or a ph.x epsil run), SRH
#                                        lifetimes, and either mobilities or
#                                        momentum relaxation times
#
# Mobility handling: give tau_momentum_* and mobility is derived as e tau/m_c;
# give mu_* (e.g. a literature Hall mobility) and the implied tau is computed
# and sanity-checked instead. Giving both keeps the mobility and reports the
# disagreement.

from __future__ import annotations

from typing import Optional

from .params import MaterialParams
from . import transport as tr


def material_from_dft(name: str,
                      job_id: Optional[str] = None,
                      workdir: Optional[str] = None,
                      save_dir: Optional[str] = None,
                      T: float = 300.0,
                      overrides: Optional[dict] = None,
                      interp_factor: int = 8,
                      allow_dft_gap: bool = False,
                      check_convergence: bool = True) -> dict:
    sd = tr.locate_save_dir(job_id=job_id, workdir=workdir, save_dir=save_dir)
    rep = tr.band_transport(sd, T=T, interp_factor=interp_factor,
                            check_convergence=check_convergence)
    ref = f"vishwakarma job {job_id}" if job_id else str(sd)
    note = f"BoltzTraP2 interp x{interp_factor}, {rep['run']['nks']} irreducible k"

    m = MaterialParams(name=name, T_K=T)
    m.warnings.extend(rep["warnings"])
    m.set("Nc_cm3", rep["electrons"]["Neff_cm3"], "dft+boltztrap2", ref, note)
    m.set("Nv_cm3", rep["holes"]["Neff_cm3"], "dft+boltztrap2", ref, note)
    m.set("m_cond_n", rep["electrons"]["m_cond"], "dft+boltztrap2", ref, note)
    m.set("m_cond_p", rep["holes"]["m_cond"], "dft+boltztrap2", ref, note)
    m.set("band_gap_dft_eV", rep["band_gap_dft_eV"], "dft", ref,
          "uncorrected; informational unless allow_dft_gap")

    m.merge(overrides)

    if "band_gap_eV" not in m.params and allow_dft_gap:
        m.set("band_gap_eV", rep["band_gap_dft_eV"], "dft", ref,
              "UNCORRECTED DFT gap used for the device")
        m.warnings.append("device uses the uncorrected DFT gap; semi-local "
                          "functionals typically underestimate gaps by 30-50 %, "
                          "which inflates ni by orders of magnitude")
    if "band_gap_eV" in m.params and m.params["band_gap_eV"].source != "dft":
        dg = m.v("band_gap_eV") - rep["band_gap_dft_eV"]
        m.params["band_gap_eV"].note = (m.params["band_gap_eV"].note + "; " if
                                        m.params["band_gap_eV"].note else "") + \
            f"DFT gap {rep['band_gap_dft_eV']:.3f} eV -> scissor {dg:+.3f} eV"

    for c, mu_k, tau_k, mc_k in (("n", "mu_n_cm2Vs", "tau_momentum_n_s", "m_cond_n"),
                                 ("p", "mu_p_cm2Vs", "tau_momentum_p_s", "m_cond_p")):
        mc = m.v(mc_k)
        has_mu, has_tau = mu_k in m.params, tau_k in m.params
        if has_tau and not has_mu:
            m.set(mu_k, tr.mobility_from_tau(m.v(tau_k), mc), "derived",
                  f"e*{tau_k}/{mc_k}", f"tau from {m.params[tau_k].source}")
        elif has_mu:
            tau_implied = tr.tau_from_mobility(m.v(mu_k), mc)
            if not has_tau:
                m.set(tau_k, tau_implied, "derived", f"{mu_k}*{mc_k}/e",
                      "implied by the given mobility and DFT conductivity mass")
            else:
                mu_from_tau = tr.mobility_from_tau(m.v(tau_k), mc)
                m.warnings.append(f"both {mu_k} and {tau_k} given; using {mu_k}="
                                  f"{m.v(mu_k):.0f}; tau would give {mu_from_tau:.0f}")
            w = tr.tau_plausibility(tau_implied)
            if w:
                m.warnings.append(f"{c}: {w}")

    return {"material": m.to_dict(), "transport": rep,
            "ready_for_simulation": not m.validate(),
            "problems": m.validate()}
