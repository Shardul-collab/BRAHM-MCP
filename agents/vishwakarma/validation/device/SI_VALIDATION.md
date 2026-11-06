# Device layer validation: silicon

Generated 2026-10-15 17:56:12 by `validate_si.py`. NSCF job `1ef22e16-af64-4d20-b84c-b5bf79b33d68`.

## 1. DFT band structure -> transport parameters (BoltzTraP2)

- DFT run: {'calculation': 'nscf', 'nks': 145, 'nbnd': 16, 'lsda': False, 'noncolin': False}
- DFT gap (PBE, interpolated): 0.573 eV (grid 0.580 eV; experiment 1.12 eV)
- Nc = 2.834e+19 cm^-3, m_cond,n = 0.275 m_e (principal 0.275, 0.275, 0.275)
- Nv = 2.260e+19 cm^-3, m_cond,p = 0.339 m_e
- interpolation convergence (x8 vs x16): Nc 3.12%, Nv 1.13%, m_cond_n 0.06%, m_cond_p 1.46%, gap 0.8 meV
- transport step took 18.7 s

## 2. Material parameters used

| parameter | A: DFT band edges | source A | B: literature | source B |
|---|---|---|---|---|
| Nc_cm3 | 2.834e+19 | dft+boltztrap2 | 2.8e+19 | literature |
| Nv_cm3 | 2.26e+19 | dft+boltztrap2 | 1.04e+19 | literature |
| band_gap_dft_eV | 0.5734 | dft | - | - |
| band_gap_eV | 1.12 | literature | 1.12 | literature |
| electron_affinity_eV | 4.05 | literature | 4.05 | literature |
| eps_r | 11.8 | literature | 11.8 | literature |
| m_cond_n | 0.2751 | dft+boltztrap2 | - | - |
| m_cond_p | 0.3394 | dft+boltztrap2 | - | - |
| mu_n_cm2Vs | 1500 | literature | 1500 | literature |
| mu_p_cm2Vs | 450 | literature | 450 | literature |
| srh_lifetime_n_s | 1e-06 | assumed | 1e-06 | assumed |
| srh_lifetime_p_s | 1e-06 | assumed | 1e-06 | assumed |
| tau_momentum_n_s | 2.347e-13 | derived | - | - |
| tau_momentum_p_s | 8.684e-14 | derived | - | - |
| ni (derived) | 9.902e+09 | derived | 6.676e+09 | derived |

## 3a. Devices, material A (DFT band edges)

**p-n diode (1D, NA=1e17, ND=1e16, 300 um + 300 um)** -- PASS (14.0 s, 3465 nodes)  
J_sim/J_diffusion 0.981-1.012 over Vj 0.575-0.643 V (4 pts), ideality 1.012  
warnings: parameters tagged 'assumed' were used: srh_lifetime_n_s, srh_lifetime_p_s

**n-MOSFET (2D, L=2 um, tox=10 nm, NA=1e17, n+ gate 4.05 eV)** -- PASS (223.5 s, 4037 nodes)  
Vt sim 0.454 V vs charge-sheet 0.449 V (dVt +0.005; classical 2phiF 0.338); SS sim 80.0 vs ideal 76.8 mV/dec; mu_eff from gm 1495 vs input 1500 cm2/Vs  
warnings: parameters tagged 'assumed' were used: srh_lifetime_n_s, srh_lifetime_p_s

## 3b. Devices, material B (literature)

**p-n diode (1D, NA=1e17, ND=1e16, 300 um + 300 um)** -- PASS (11.9 s, 3461 nodes)  
J_sim/J_diffusion 0.978-1.018 over Vj 0.575-0.667 V (5 pts), ideality 1.011  
warnings: parameters tagged 'assumed' were used: srh_lifetime_n_s, srh_lifetime_p_s

**n-MOSFET (2D, L=2 um, tox=10 nm, NA=1e17, n+ gate 4.05 eV)** -- PASS (214.6 s, 4037 nodes)  
Vt sim 0.460 V vs charge-sheet 0.455 V (dVt +0.005; classical 2phiF 0.344); SS sim 79.7 vs ideal 76.6 mV/dec; mu_eff from gm 1496 vs input 1500 cm2/Vs  
warnings: parameters tagged 'assumed' were used: srh_lifetime_n_s, srh_lifetime_p_s

## 4. Physics not modelled

- constant low-field mobility (no doping/field dependence, no velocity saturation)
- Boltzmann statistics (invalid for degenerate doping near contacts)
- SRH recombination with mid-gap traps only (no Auger, radiative, or tunnelling)
- no band-gap narrowing, no quantum confinement, no interface/oxide charge
- homojunction only (single semiconductor region)

## 5. Changes made while validating (for review)

1. **Diode comparator, series resistance.** First run (before this change): J_sim/J_Shockley = 0.759 at 0.65 V, outside [0.80, 1.25] -> FAIL. The neutral regions (300 um each) have Rs = 0.0167 ohm cm^2 computed from the parameters; at 0.40 A/cm^2 that drops 6.7 mV and predicts a factor exp(-6.7/25.85) = 0.77, matching 0.759. The comparator now uses the junction voltage V - J*Rs (Rs computed, not fitted). Tolerance bands unchanged.
2. **MOSFET comparator, like-for-like threshold.** First run: Vt by max-gm extrapolation 0.454 V vs classical 2*phiF threshold 0.338 V (dVt +0.116 V, tolerance 0.10 V) -> FAIL. These are different quantities: past 2*phiF the surface potential keeps rising by a few kT/q, so extrapolated Vt sits above the 2*phiF value by construction. The gate now compares against an exact 1D charge-sheet model put through the same max-gm extraction (0.449 V, dVt +0.005 V). Classical Vt is still reported. Tolerance unchanged.
3. **Bug fixed: S/D doping expression.** DEVSIM's parser reads `exp(-(y/yj)^2)` as `exp((-(y/yj))^2)`, so the source/drain doping grew to ~1e63 cm^-3 with depth and the drift-diffusion solve could not converge. Rewritten as `exp(-1*((y/yj)^2))`; tcad.py now refuses to run if the peak doping evaluates far from N_sd.
