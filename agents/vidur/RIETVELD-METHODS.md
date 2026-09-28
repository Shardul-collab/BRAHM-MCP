# VIDUR Rietveld refinement — methods and stated defaults

Module: `agents/vidur/rietveld.py`. MCP tool: `vidur_rietveld`. Tests: `tests/test_rietveld.py`.
Added 2026-09-27. If this document and the code disagree, one of them is a bug.

## Decisions (Shardul, 2026-09-27)

- **Engine: FullProf** (`fp2k`, Classic FullProf Suite July 2026, Linux Mint build, installed in
  `~/FullProf_Suite`; override with `FULLPROF`). Chosen because the RRD-A reference refinements
  (002, 026–028) are FullProf, so results stay comparable. GSAS-II is kept for later
  cross-validation; it is not installed.
- **Strict CIF identity.** A CIF whose identity does not match the phase named is rejected and
  the refinement does not run. Never a warning.
- **RRD-A:** 006 rejected until identity is verified (P1 computed cell, MP id mismatch);
  024 Cu CIF rejected (the Al phase has no CIF); 027 rejected for maricite (precursor CIFs only).

## Identity validation (`validate_phase`)

A phase is given as name + CIF + the formula it **is** + optionally the source id the CIF must
carry (COD code or `mp-…` in the file name). Rejections:

1. no complete cell / no space group / no `_chemical_formula_sum`;
2. CIF formula elements or ratios differ from the phase formula (2 % relative tolerance on the
   normalised composition);
3. Rietveld phases: the atom sites, weighted by occupancy × multiplicity (multiplicity computed
   from the CIF's own symmetry operators), do not reproduce the CIF's formula, or Z is
   inconsistent with them (3 % tolerance) — the file contradicts itself;
4. a source id was given and the CIF carries a different one.

Warnings (do not refuse): Biso missing or ≤ 0, partial occupancy, a space-group setting suffix
(`:1`, `:2`, `:H`; FullProf's default setting is used), no symmetry operators to check.

## What goes into FullProf

- One constant-wavelength X-ray pattern per run, `Ins=10` (x, y, σ). Points with σ ≤ 0 are
  excluded regions (listed in the report), never altered or dropped.
- Profile: TCH pseudo-Voigt (`Npr=7`). Instrument: a FullProf IRF (`Res=1`) when given; the
  IRF's `WAVE` and `CTHM` are used. `Ilo=3` (synchrotron x,y,σ polarisation) on request.
- Background: Chebyshev (`Nba=-5`), first 12 of 24 coefficients refined (stated default).
- Phases: Rietveld (`Jbt=0`, atoms from the CIF) or Le Bail (`Jbt=2`, cell and space group only).
- Site occupancy written as `occ × site multiplicity / general multiplicity` (FullProf convention).
- **Model change:** Biso ≤ 0 or missing → 1.0 (the convention of FullProf's own `CIFs_to_PCR`).
  Each substitution is listed under `model_changes`.
- `ATZ = Z × M_w` from the CIF formula (IUPAC atomic weights), so FullProf reports weight
  fractions. Weight fractions are relative to the Rietveld phases; a Le Bail phase reports 0.

## Staged refinement (`refine`)

Default stages follow FullProfAPP (Arcelus et al. 2024): (i) scale, (ii) zero, (iii) background,
(iv) cell, (v) strain, (vi) size. Groups freed at a stage stay free. Sample broadening with an
IRF: **strain = TCH X** (Lorentzian, tanθ), **size = TCH Y** (Lorentzian, 1/cosθ) — the paper
does not say which terms it refines; this is VIDUR's choice.

Guards (all reported in each step's entry):

- **Absent or minor phase** — a Rietveld phase whose scale is < 3σ, or whose weight fraction is
  < 3 wt %, after the previous stage keeps its cell, X and Y fixed; its scale stays free
  (`held_phases`). Measured need: frame 1 of the FullProfAPP LiFePO₄ sequence went NaN when
  FePO₄'s strain was freed while FePO₄ was absent; and in the v2 sequence LiFePO₄ at 0.2 wt %
  refined to TCH X = 9.0 (a broad hump absorbing background), slowing each FullProf run from
  ~3 s to ~2 min.
- **Divergence / χ² rise** — a stage whose χ² is NaN, or rises by more than 2 % over the previous
  accepted stage, is retried once as damped least squares (relaxation 0.3) from the same start.
  If it still fails, the previous model is kept and the stage's new groups are not freed again
  for that pattern (`groups_dropped`). Measured need: the Le Bail Al phase ran away in the
  09-27 v1 sequence (cell 4.049 → 3.686 Å) and every later frame diverged.
- The values FullProf rewrites into the PCR are read back only from accepted stages.

## Sequential refinement (`refine_sequence`)

Pattern *i* starts from pattern *i−1*'s accepted result (the first from the CIFs). A failed
pattern does not seed the next one.

## Stated defaults (not measured optima)

| Name | Value | Where |
|---|---|---|
| `SIG_SCALE` | 3.0 σ | absent-phase test |
| `MIN_WT_PCT` | 3 wt % | minor-phase hold |
| `DAMPED_RELAX` | 0.3 | retry relaxation |
| `CHI2_RISE_TOL` | 2 % | stage acceptance |
| Chebyshev terms refined | 12 | `n_bkg` |
| NCY / Eps | 30 / 0.1 | PCR |
| composition tolerance | 2 % (formula), 3 % (sites) | validation |

## Validation against a published refinement (2026-09-27)

Target: Arcelus et al., *FullProfAPP*, J. Appl. Cryst. 57 (2024), doi:10.1107/S1600576724006885,
LiFePO₄/FePO₄ operando case (ALBA NOTOS, λ = 1.12698 Å). Data, IRF and per-frame FullProf
profiles are the paper's supplement iu5063sup3 (= RRD-A 026). Phases as in the paper:
LiFePO₄ COD 1101111, FePO₄ COD 1525576, Al COD 4313206 (Le Bail); the paper's six stages.
Paper: sequence χ² mean 2.8, min 1.4, max 4.6.

| Frames 3–29 (27 frames) | χ² mean | min | max | frames inside 1.4–4.6 (of 3–30) |
|---|---|---|---|---|
| VIDUR, 12 Chebyshev terms | 4.56 | 2.80 | 7.95 | 16 / 28 |
| VIDUR, 24 Chebyshev terms | 2.82 | 1.63 | 5.44 | 25 / 28 |
| supplement `_prof.cif` | 12.4* | 1.43 | 44.5 | — |

\* the supplement's own χ² for frames 5–13 is 17–45, outside the range the paper states, and
frame 24 is all zeros: the supplement is not the refinement the paper's numbers describe.
Residuals on frame 20 put the 12-term gap at 11.5–20° (broad cell-component humps) and a
background ~10 % low above 20°; 24 terms closes most of it (χ² 2.89 → 1.66; supplement 1.45).
Weight fractions follow LiFePO₄ → FePO₄ on oxidation (frames 3–16) and back (24–29).
Open: frames 26–29 FePO₄ b drifts 5.79 → 5.95 Å (R_Bragg 9 → 19) — misfit or a real
intermediate, not resolved; seeded from it, frame 30 fails (χ² 17.5), while frame 30 refined
from the CIFs gives FePO₄ 99.9 wt %, χ² 1.73. Frames 1–2 are anomalous data (frame 1's
intensity collapses above 2θ ≈ 29°): χ² 32 and 103 here, 79 and 1491 in the supplement.
Step 6 (size, TCH Y) is rejected by the χ²-rise guard on most frames: X and Y are not separable
over 11–59° 2θ here.

## Not done

Atomic coordinates, Biso and occupancies are never refined (the six FullProfAPP stages do not
include them). No preferred orientation, asymmetry, absorption, anisotropic broadening,
multi-pattern, neutron or TOF. No GSAS-II cross-check yet.
