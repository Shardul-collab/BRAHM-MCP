# vishwakarma.device: from DFT to device behavior

This module chains Vishwakarma's DFT output into device-level simulation:

```
QE dense NSCF  ──►  BoltzTraP2  ──►  MaterialParams  ──►  DEVSIM  ──►  analytic check
(unit cell)        (transport.py)     (params.py)        (tcad.py)    (analytic.py)
```

Only the band structure comes from DFT. A plain DFT run can't supply most of what a device needs, so every number carries a source tag and travels with the result.

| parameter | where it comes from |
|---|---|
| Nc, Nv | **DFT + BoltzTraP2**: the interpolated DOS integrated with the Boltzmann factor (no parabolic-band assumption) |
| m_cond,n, m_cond,p | **DFT + BoltzTraP2**: n e² / (σ/τ), with conduction and valence bands fitted separately, so the result doesn't depend on the DFT gap |
| band gap | **literature / user**. The PBE gap is reported but only used if `allow_dft_gap=True`, and then with a warning |
| electron affinity, ε_r | literature / user (ε_r could later come from `ph.x epsil`) |
| μ_n, μ_p | either given directly (the implied τ = μ m_c / e is then sanity-checked), or derived from a momentum relaxation time τ (EPW, literature) as e τ / m_c |
| SRH lifetimes | literature / measurement / user. These depend on the process, so they're tagged `assumed` if nobody knows them |

Source tags are `dft`, `dft+boltztrap2`, `literature` (a ref is required), `user`, `derived` and `assumed`. Any `assumed` value becomes a warning on every simulation result that uses it.

## Devices (v0)

- `pn_diode_1d`: abrupt junction, SRH recombination, J–V. It's checked against Shockley diffusion theory, with finite-length (coth) neutral regions and the junction voltage corrected for computed series resistance.
- `nmosfet_2d`: planar n-MOSFET with a metal gate (work-function difference included) and erfc/Gaussian n+ source and drain. It produces Id–Vg at low Vd and optionally Id–Vd. It's checked against an exact 1D charge-sheet model run through the same max-gm Vt extraction, and against the ideal subthreshold swing.

Tolerances are fixed in `analytic.py` and are not to be loosened to make a result pass. A failed check doesn't throw an error: the result comes back with `analytic_check.pass = false` and a warning attached.

**Physics not included:** field- or doping-dependent mobility, velocity saturation, Fermi–Dirac statistics, band-gap narrowing, quantum confinement, interface or oxide charge, heterojunctions, and ferroelectric polarisation. Treat the output as a long-channel, low-field prototype.

## Use

MCP tools: `vishwakarma_device_material`, then `vishwakarma_device_simulate`.
API: `POST /device/material`, then `POST /device/simulate`.

```python
from vishwakarma.device import pipeline, tcad
from vishwakarma.device.params import MaterialParams

r = pipeline.material_from_dft("ZnSe", job_id="<nscf job>", workdir=JOBS,
        overrides={"band_gap_eV": {"value": 2.7, "source": "literature", "ref": "doi:..."},
                   "electron_affinity_eV": {...}, "eps_r": {...},
                   "tau_momentum_n_s": {...}, "tau_momentum_p_s": {...},
                   "srh_lifetime_n_s": {...}, "srh_lifetime_p_s": {...}})
m = MaterialParams.from_dict(r["material"])
res = tcad.pn_diode_1d(m, NA=1e17, ND=1e17)
res["analytic_check"]["pass"], res["warnings"], res["provenance"]
```

The DFT job has to be a **dense uniform-mesh NSCF with conduction bands**, such as the `nscf` step of `vishwakarma_run_dos` with `nbnd` well above the number of occupied bands. Spin-polarised and noncollinear runs are rejected for now.

## Environment

- `pip install devsim BoltzTraP2`. DEVSIM also needs BLAS/LAPACK at runtime: `DEVSIM_MATH_LIBS` defaults to `liblapack.so.3:libblas.so.3`, which this machine's system libraries provide.
- Each DEVSIM run happens in a fresh subprocess. DEVSIM has global state, isn't thread-safe, and writes to stdout at C level, so running it in-process would corrupt the MCP JSON-RPC stream.

## Validation

`validation/device/run_si_reference_dft.py` runs the reference Si DFT, and `validation/device/validate_si.py` runs the full chain. The results are in `validation/device/SI_VALIDATION.md`.
