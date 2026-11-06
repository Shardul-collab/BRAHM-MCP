"""
End-to-end validation of the device layer on silicon:

    Quantum ESPRESSO (Vishwakarma job)  ->  BoltzTraP2  ->  MaterialParams
                                        ->  DEVSIM diode + MOSFET
                                        ->  analytic cross-checks

Two material sets are simulated:
  A  "dft"         Nc, Nv, conductivity masses from the reference DFT run
                   (run_si_reference_dft.py); gap, affinity, eps_r, mobilities
                   from a literature table; SRH lifetimes assumed (1 us).
  B  "literature"  the same table for everything (control).

Writes validation/device/si_validation.json and SI_VALIDATION.md (+ PNGs if
matplotlib is available).

Run (from agents/vishwakarma):
    .venv/bin/python validation/device/validate_si.py [nscf_job_id]
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
os.environ.setdefault("DEVSIM_MATH_LIBS", "liblapack.so.3:libblas.so.3")

from vishwakarma.device import pipeline, tcad, reference_sets as rs   # noqa: E402
from vishwakarma.device.params import MaterialParams                  # noqa: E402

WORK = os.environ.get("QE_WORKDIR", str(HERE.parents[1] / "jobs"))


def nscf_job_from_reference() -> str:
    d = json.loads((HERE / "si_reference_dft.json").read_text())
    return next(s["job_id"] for s in d["steps"] if s["step"] == "nscf")


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else nscf_job_from_reference()
    out = {"nscf_job_id": job, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    t = time.time()
    A = pipeline.material_from_dft("Si (DFT band edges)", job_id=job, workdir=WORK,
                                   overrides=rs.SI_LITERATURE_OVERRIDES)
    out["transport_seconds"] = round(time.time() - t, 1)
    out["material_dft"] = A
    mA = MaterialParams.from_dict(A["material"])

    mB = MaterialParams(name="Si (literature table)")
    mB.merge(rs.SI_LITERATURE_FULL)
    out["material_literature"] = {"material": mB.to_dict(), "problems": mB.validate()}

    for tag, m in (("dft", mA), ("literature", mB)):
        t = time.time()
        out[f"diode_{tag}"] = tcad.pn_diode_1d(m)
        out[f"diode_{tag}"]["seconds"] = round(time.time() - t, 1)
        t = time.time()
        out[f"mosfet_{tag}"] = tcad.nmosfet_2d(m, gate_workfunction_eV=4.05,
                                               vg_output=[1.0, 2.0])
        out[f"mosfet_{tag}"]["seconds"] = round(time.time() - t, 1)

    (HERE / "si_validation.json").write_text(json.dumps(out, indent=1, default=str))
    write_report(out)
    try:
        plots(out)
    except Exception as exc:                       # matplotlib optional
        print("plots skipped:", exc)
    for k in ("diode_dft", "mosfet_dft", "diode_literature", "mosfet_literature"):
        r = out[k]
        print(k, r.get("success"), (r.get("analytic_check") or {}).get("pass"),
              (r.get("analytic_check") or {}).get("summary", r.get("error")))


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, (int, float)) else str(v)


def write_report(o: dict) -> None:
    A = o["material_dft"]
    tr = A["transport"]
    pa = A["material"]["params"]
    pb = o["material_literature"]["material"]["params"]
    L = []
    L.append("# Device layer validation: silicon\n")
    L.append(f"Generated {o['started']} by `validate_si.py`. NSCF job `{o['nscf_job_id']}`.\n")
    L.append("## 1. DFT band structure -> transport parameters (BoltzTraP2)\n")
    L.append(f"- DFT run: {tr['run']}")
    L.append(f"- DFT gap (PBE, interpolated): {tr['band_gap_dft_eV']:.3f} eV "
             f"(grid {tr['band_gap_dft_grid_eV']:.3f} eV; experiment 1.12 eV)")
    e, h = tr["electrons"], tr["holes"]
    L.append(f"- Nc = {e['Neff_cm3']:.3e} cm^-3, m_cond,n = {e['m_cond']:.3f} m_e "
             f"(principal {', '.join(f'{x:.3f}' for x in e['m_cond_principal'])})")
    L.append(f"- Nv = {h['Neff_cm3']:.3e} cm^-3, m_cond,p = {h['m_cond']:.3f} m_e")
    if "convergence" in tr:
        L.append(f"- interpolation convergence (x{tr['interp_factor']} vs "
                 f"x{tr['convergence']['compared_interp_factor']}): "
                 + ", ".join(f"{k} {v:.2%}" if k != "gap" else f"gap {v*1000:.1f} meV"
                             for k, v in tr["convergence"]["relative_change"].items()))
    L.append(f"- transport step took {o['transport_seconds']} s")
    L.append("")
    L.append("## 2. Material parameters used\n")
    L.append("| parameter | A: DFT band edges | source A | B: literature | source B |")
    L.append("|---|---|---|---|---|")
    for k in sorted(set(pa) | set(pb)):
        a, b = pa.get(k, {}), pb.get(k, {})
        L.append(f"| {k} | {_fmt(a.get('value', '-'))} | {a.get('source', '-')} | "
                 f"{_fmt(b.get('value', '-'))} | {b.get('source', '-')} |")
    dA = A["material"].get("derived", {})
    dB = o["material_literature"]["material"].get("derived", {})
    L.append(f"| ni (derived) | {_fmt(dA.get('ni_cm3'))} | derived | {_fmt(dB.get('ni_cm3'))} | derived |")
    L.append("")
    if A["material"]["warnings"]:
        L.append("Warnings on A: " + "; ".join(A["material"]["warnings"]) + "\n")
    for tag, label in (("dft", "A (DFT band edges)"), ("literature", "B (literature)")):
        d, m = o[f"diode_{tag}"], o[f"mosfet_{tag}"]
        L.append(f"## 3{'a' if tag == 'dft' else 'b'}. Devices, material {label}\n")
        for name, r in (("p-n diode (1D, NA=1e17, ND=1e16, 300 um + 300 um)", d),
                        ("n-MOSFET (2D, L=2 um, tox=10 nm, NA=1e17, n+ gate 4.05 eV)", m)):
            c = r.get("analytic_check") or {}
            status = {True: "PASS", False: "FAIL", None: "NO CHECK"}[c.get("pass")]
            L.append(f"**{name}** -- {status} ({r.get('seconds')} s, "
                     f"{r.get('mesh_nodes')} nodes)  ")
            L.append(f"{c.get('summary', r.get('error'))}  ")
            if r.get("warnings"):
                L.append("warnings: " + "; ".join(r["warnings"]) + "\n")
            else:
                L.append("")
    L.append("## 4. Physics not modelled\n")
    for p in tcad.PHYSICS_LIMITS:
        L.append(f"- {p}")
    L.append("")
    L.append("## 5. Changes made while validating (for review)\n")
    L.append(
        "1. **Diode comparator, series resistance.** First run (before this change): "
        "J_sim/J_Shockley = 0.759 at 0.65 V, outside [0.80, 1.25] -> FAIL. The "
        "neutral regions (300 um each) have Rs = 0.0167 ohm cm^2 computed from the "
        "parameters; at 0.40 A/cm^2 that drops 6.7 mV and predicts a factor "
        "exp(-6.7/25.85) = 0.77, matching 0.759. The comparator now uses the junction "
        "voltage V - J*Rs (Rs computed, not fitted). Tolerance bands unchanged.\n"
        "2. **MOSFET comparator, like-for-like threshold.** First run: Vt by max-gm "
        "extrapolation 0.454 V vs classical 2*phiF threshold 0.338 V (dVt +0.116 V, "
        "tolerance 0.10 V) -> FAIL. These are different quantities: past 2*phiF the "
        "surface potential keeps rising by a few kT/q, so extrapolated Vt sits above "
        "the 2*phiF value by construction. The gate now compares against an exact 1D "
        "charge-sheet model put through the same max-gm extraction (0.449 V, dVt "
        "+0.005 V). Classical Vt is still reported. Tolerance unchanged.\n"
        "3. **Bug fixed: S/D doping expression.** DEVSIM's parser reads "
        "`exp(-(y/yj)^2)` as `exp((-(y/yj))^2)`, so the source/drain doping grew to "
        "~1e63 cm^-3 with depth and the drift-diffusion solve could not converge. "
        "Rewritten as `exp(-1*((y/yj)^2))`; tcad.py now refuses to run if the peak "
        "doping evaluates far from N_sd.\n")
    (HERE / "SI_VALIDATION.md").write_text("\n".join(L))


def plots(o: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for tag, ls in (("dft", "-"), ("literature", "--")):
        d = o[f"diode_{tag}"]
        if d.get("success"):
            iv = [r for r in d["iv"] if r["V"] > 0]
            ax[0].semilogy([r["V"] for r in iv], [r["J_A_cm2"] for r in iv], ls,
                           label=f"DEVSIM, {tag}")
            if tag == "dft":
                rows = d["analytic_check"]["rows"]
                ax[0].semilogy([r["V"] for r in rows], [r["J_diffusion"] for r in rows],
                               "k:", label="Shockley (Vj = V - J Rs)")
        m = o[f"mosfet_{tag}"]
        if m.get("success"):
            tr = m["transfer_vd_lin"]
            ax[1].semilogy([r["Vg"] for r in tr], [max(r["Id_A_per_um"], 1e-18) for r in tr],
                           ls, label=f"DEVSIM, {tag}")
            if tag == "dft":
                cs = m["analytic_check"]["chargesheet_transfer"]
                ax[1].semilogy([r["Vg"] for r in cs], [max(r["Id_A_per_um"], 1e-18) for r in cs],
                               "k:", label="charge-sheet")
                for oc in m["output"]:
                    ax[2].plot([q["Vd"] for q in oc["curve"]],
                               [q["Id_A_per_um"] * 1e6 for q in oc["curve"]],
                               label=f"Vg = {oc['Vg']} V")
    ax[0].set(xlabel="V (V)", ylabel="J (A/cm²)", title="p-n diode, forward")
    ax[1].set(xlabel="Vg (V)", ylabel="Id (A/µm)", title="n-MOSFET, Vd = 50 mV")
    ax[2].set(xlabel="Vd (V)", ylabel="Id (µA/µm)", title="n-MOSFET output, material A")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "si_validation.png", dpi=130)


if __name__ == "__main__":
    main()
