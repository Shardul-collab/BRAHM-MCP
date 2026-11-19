# ZnSe phonons and Debye temperature (Vishwakarma / Quantum ESPRESSO)

Raw inputs and outputs from a DFPT phonon calculation on zinc-blende ZnSe,
run through Vishwakarma. Quantum ESPRESSO 7.5 does every calculation;
Vishwakarma prepares, launches, monitors and recovers the jobs.

## Settings (from the input files in this folder)

| Setting | Value |
|---|---|
| Structure | zinc-blende ZnSe, 2-atom fcc primitive cell |
| Lattice constant | a = 5.669 Å (fcc vectors of 2.834 Å components; alat = 7.575 bohr) |
| Functional | PBE |
| Pseudopotentials | PAW, pslibrary 1.0.0: `Zn.pbe-dn-kjpaw_psl.1.0.0.UPF`, `Se.pbe-dn-kjpaw_psl.1.0.0.UPF` |
| Cutoffs | ecutwfc = 50 Ry, ecutrho = 400 Ry |
| k-points | 6 × 6 × 6, shifted (1 1 1) |
| SCF threshold | conv_thr = 1e-8 |
| Phonons | DFPT, 4 × 4 × 4 q-grid, tr2_ph = 1e-14 |
| Interatomic force constants | q2r.x with `zasr = 'crystal'` (acoustic sum rule imposed) |

## Results in this folder

- SCF (`c90751dd…`): converged in 11 iterations, total energy −645.62919810 Ry.
- DFPT (`06f3896c…`): completed (`JOB DONE`), about 2 days 16 hours wall time.
- Dielectric response at Γ, from `q2r.out`: ε∞ ≈ 7.25; Born effective charges
  +2.01 (Zn) and −1.87 (Se) before the sum rule is imposed. LO–TO splitting is
  therefore included.
- `ZnSe.dos`: phonon density of states with partial DOS per species; the
  highest phonon frequency is about 244.5 cm⁻¹.
- `ZnSe.band.freq`: phonon dispersion.

## Debye temperature

Computed by three independent routes (reported by the author, Aug 2026; the
equation-of-state and elastic-constant runs are not in this folder):

| Route | Debye temperature |
|---|---|
| Equation of state | 300.5 K |
| Elastic constants | 289.8 K |
| DFPT phonons | 309.4 K |

All three fall within the reported literature range of 270–317 K.

## The `dyn4` failure

The first `q2r.x` run (`q2r.out`) stopped with `Bad real number` while
reading `dyn4`. That file holds 6 dynamical-matrix blocks, against 24 in the
recomputed `dyn4fix`. The q-point (0.3536, 0, 0) was rerun on its own
(`q4_fix.in` → `q4_fix.out`, completed). `q4_fix2.*` is a second rerun whose
output stops partway. `status.json` was saved while the DFPT job was still
running, so it shows `"status": "running"`.

## Not yet documented

- Whether 5.669 Å is a PBE-relaxed or an experimental lattice constant.
- Comparison of ε∞ and the Born charges with experiment.
- Convergence tests for the cutoffs and the k-mesh.
- The final `q2r.x` and `matdyn.x` inputs used to produce `ZnSe.dos` and
  `ZnSe.band.freq`.
