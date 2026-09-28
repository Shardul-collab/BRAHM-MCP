# VIDUR — XRD line-profile analysis: formulas and method

Written 2026-09-25. This is the reference `line_profile.py` implements. Every formula
the code uses is here, with where it comes from and when it stops being valid. If
the code and this file disagree, one of them is a bug.

Scope: crystallite size, microstrain and dislocation density from peak broadening
in a θ–2θ (or 2θ–ω) scan. The inputs are measured peak breadths; the outputs are
model-dependent estimates. VIDUR reports them together with the model, the constants,
and the fit quality, so a reader can judge them. It does not pick the "right" one.

---

## 0. Where this sits in VIDUR

VIDUR-DESIGN.md says VIDUR prepares plot-ready data and "is not an interpreter".
Line-profile analysis *is* interpretation, so it lives in its own module and its own
tool (`vidur_xrd_profile`), not in `prep.derive()`. The rules carry over unchanged:

- The raw intensity is never altered. The fit, its background and the residual are
  extra columns next to it.
- No constant is assumed. Wavelength, Kα2 and instrumental broadening come from the
  file or from the caller. What is missing is returned as a question, with evidence.
- Every number is reported with the model that produced it and its uncertainty.

---

## 1. Pipeline

```
raw pattern ─► 2. peak finding ─► 3. profile fitting (Voigt [+Kα2] + linear bkg)
                                         │
       instrument standard or stated ──► 4. instrumental deconvolution
       instrument widths                 │
                                         ▼
                        5. single-line: Scherrer, Voigt size/strain
                        6. multi-line : Williamson–Hall, size–strain plot
                        7. dislocation density
```

All breadths below are in **radians of 2θ**; θ is the Bragg angle (half of 2θ).

---

## 2. Peak finding

Peaks are located with `scipy.signal.find_peaks` on the raw intensity, using
**prominence** (height above the surrounding baseline, not above zero), so a peak on a
sloping or amorphous background is judged by how far it stands out locally.

**Noise σ.** Two models, and the output names the one that ran:

- *Counts* (the file says `unit="counts"`, or every value is a non-negative
  integer): Poisson, σ = √B, where B is the local background, taken as the median of a
  window much wider than the peak. Counting noise grows with intensity, so one global σ
  finds too many peaks in low-count regions and too few on a high background.
- *Other* (normalised or processed data): one robust global σ from the first
  differences, σ = 1.4826 · MAD(Δy) / √2.

**Threshold.** A prominence is roughly a local maximum minus a local minimum, so pure
noise has much larger prominences than its σ suggests. Measured here on pure Poisson
noise: 7,500 points gave 30–40 "peaks" above 5σ, and the largest was about 7σ. A fixed
5σ cut would therefore report noise as peaks. Instead the threshold k is the prominence
that pure Gaussian noise of the same length exceeds, anywhere in the pattern, with
probability `false_alarm` (default 1 %). k is computed by simulation with a fixed seed
and is typically 7–8σ. `min_snr` (default 5) is a floor under k. Both values are
reported. Pass `peaks=[…]` to fit positions you choose instead.

**Too few points.** A peak needs ≥ 5 points above half height before VIDUR fits a
breadth to it. With fewer, the breadth is set by the step size, not the sample (a
single-crystal substrate line in a coarse scan is the usual case). Such peaks are
reported as `not_fitted`, with the reason.

**Significance after fitting.** A candidate whose fitted area is less than 3 standard
errors is reported as `not_significant` (indistinguishable from noise) and is not
analysed further.

---

## 3. Profile fitting

### 3.1 Profile function: Voigt

Each reflection is fitted with a **Voigt** function, the convolution of a Lorentzian
(Cauchy) and a Gaussian:

  V(x) = A · ∫ L(x′; γ) G(x − x′; σ) dx′  (`scipy.special.voigt_profile`)

Why Voigt rather than pseudo-Voigt: the size/strain separation (§5.2) needs the
Lorentzian and Gaussian components separately. A true Voigt gives them directly:

  β_G = σ √(2π)  (Gaussian integral breadth)
  β_L = π γ     (Lorentzian integral breadth)
  FWHM_G = 2σ √(2 ln 2),  FWHM_L = 2γ

and the total FWHM follows Olivero & Longbothum (1977), accurate to ~0.02 %:

  FWHM_V ≈ 0.5346 FWHM_L + √(0.2166 FWHM_L² + FWHM_G²)

Total integral breadth of the Voigt: β = A / I_max (area over peak height).

A pseudo-Voigt fit (η·L + (1−η)·G) would need the de Keijser et al. (1982) empirical
relations to recover the components (φ = FWHM/β):

  β_C/β = 2.0207 − 0.4803 φ − 1.7756 φ²
  β_G/β = 0.6420 + 1.4187 (φ − 2/π)^½ − 2.2043 φ + 1.8706 φ²

Given here for reference only. VIDUR fits the Voigt directly and does not need them.
The coefficients were checked here against the two limits: at φ = 2/π (pure
Lorentzian) they give β_C/β = 0.995 and β_G/β ≈ 0, and at φ = 0.9394 (pure
Gaussian) they give β_C/β ≈ 0 and β_G/β = 1.003.
The Thompson–Cox–Hastings (1987) pseudo-Voigt is the Rietveld-program convention; its
FWHM polynomial (coefficients 2.69269, 2.42843, 4.47163, 0.07842) and η polynomial
(1.36603, −0.47719, 0.11116) are also given in Dinnebier & Scardi (2021).

### 3.2 Background

A **linear background** is fitted *with* the peak, inside the fit window. It is a
fit parameter, not a subtraction: the raw column is untouched and the fitted
background is written out as its own column. Global background subtraction and
smoothing are not used.

### 3.3 Fit window and overlapping peaks

Window = peak centre ± `window_fwhm` × (estimated FWHM), default 4. Peaks whose windows
overlap are fitted **together** in one window (one Voigt per peak, one shared linear
background). Fitting overlapping peaks separately would give each one the other's tail.

A peak that could not be fitted (§2, too few points) but is stronger than a peak being
fitted, and sits in that peak's window, is **masked** out of that fit. Its data are
excluded from that one fit, not deleted, and the masked 2θ ranges are reported per peak.
Without this, a sharp substrate line inside a broad film peak's window is absorbed into
the film peak's fit. Weaker features, such as thickness fringes, are left in.

**Asymmetry is not modelled.** At low angle, axial divergence makes real peaks
asymmetric. A symmetric Voigt then fits them imperfectly, so the breadth is less
reliable, and this shows up in the fit statistic (§3.5).

### 3.4 Kα2

Unmonochromated Cu radiation contains Kα1 and Kα2. The doublet broadens and skews
every peak and, if ignored, is read as sample broadening, most at high angle.

VIDUR models it rather than stripping it (Rachinger stripping changes the data and is
opt-in in VIDUR-DESIGN.md). Each reflection is a pair:

- Kα1 at 2θ₁ (fitted).
- Kα2 at 2θ₂ = 2 arcsin( (λ₂/λ₁) sin θ₁ ): fixed by Bragg's law, not fitted.
- Same widths as Kα1 (the physical widths differ by the ratio λ₂/λ₁, i.e. by < 0.3 %).
- Area ratio I(Kα2)/I(Kα1) = R, fixed.

λ₁, λ₂ and R come from the file when it states them. PANalytical `.xrdml` writes
`<kAlpha1>`, `<kAlpha2>` and `<ratioKAlpha2KAlpha1>`; a ratio of 0 means a Kα1-only
(monochromated) beam, and then no doublet is fitted. For files that do not state them,
the caller supplies `ka2_ratio` (0 for monochromated); VIDUR does not assume a value.

When the doublet is resolved, peak finding also detects the Kα2 line as a peak of its
own. A detected peak within one FWHM of the Kα2 position of a stronger peak is reported
as that peak's `ka2_partner` and is not fitted as a separate reflection.

Why this matters, measured on a synthetic single peak at 53.6° with D = 40 nm (tests):
modelling the doublet recovers D to within 10 %. Fitting the same data as one Kα1 peak
gives a FWHM 1.5× too large and a D about 18 % wrong.

Reference values for Cu: λ(Kα1) = 1.540598 Å, λ(Kα2) = 1.544426 Å, R = 0.5 are the
values PANalytical writes (Hölzer et al. 1997 lineage). Mendenhall et al. (2017, NIST)
measured Kα1 = 0.154059 nm, Kα2 = 0.154438 nm, R = 0.520(2). The difference between
0.5 and 0.52 is small for breadth but not zero; use what the instrument states.

### 3.5 Uncertainties

Parameter standard errors are the square roots of the diagonal of the covariance
matrix from `scipy.optimize.curve_fit` (scaled by reduced χ², i.e.
`absolute_sigma=False`), then propagated to derived quantities to first order. They
are *statistical* only.

Fit quality: χ²_red with Poisson variance (the model itself), χ²_red =
Σ (y − m)²/m / (N − p). This is reported for every peak, and for counts data it is
turned into a p-value. When p < 10⁻³ the peak is flagged: the fit is not within
counting noise, usually because of asymmetry or an unmodelled component. The flag does
not remove the peak.

Seed sweep on synthetic data (20 noise realisations, D = 30 nm, ε = 2×10⁻³): mean
single-line Voigt D = 30.02 nm with SD 0.52 nm, against a median reported s.e. of
0.44 nm. So the reported errors are about right for statistical noise. The dominant errors in line-profile analysis are usually
systematic (the model, the instrument function, anisotropy), and these standard errors
do not include them.

---

## 4. Instrumental broadening

The measured profile h is the convolution of the instrument profile g with the sample
profile f: h = g ⊗ f. For Voigt profiles the convolution is exact in closed form,
because Lorentzians add linearly and Gaussians add in quadrature:

  β_L(f) = β_L(h) − β_L(g)
  β_G(f)² = β_G(h)² − β_G(g)²

(Balzar, *Voigt-function model in diffraction line-broadening analysis*, IUCr
Monographs on Crystallography 10, 1999.)

The popular shortcuts are the two limits of this: β_f = β_h − β_g (both Lorentzian) and
β_f² = β_h² − β_g² (both Gaussian). Which one is used changes the answer. Using the
components avoids choosing.

### 4.1 Getting g

In order of preference:

1. **A standard measured on the same instrument and optics**: NIST SRM 660c LaB₆
   (volume-weighted domain size ≈ 0.8 µm, microstrain refined to zero, so its lines
   are essentially instrumental) or SRM 640 Si. Pass it as `instrument.standard_file`.
   VIDUR fits its peaks with the same Voigt model and fits the angular dependence:

   - Gaussian, Caglioti et al. (1958): FWHM_G² = U tan²θ + V tanθ + W
   - Lorentzian, TCH form: FWHM_L = X tanθ + Y / cosθ

   and evaluates both at each sample peak. The standard must span the sample's 2θ range;
   outside it the fit is extrapolation, and the output says so.

2. **Instrument widths stated by the caller** (`instrument.fwhm_gauss_deg`,
   `instrument.fwhm_lorentz_deg`), constant over 2θ. Coarser, but explicit.

3. **None.** Then nothing is subtracted, and every size is reported as an
   **instrument-uncorrected lower bound** (instrument broadening makes peaks wider,
   so the size comes out too small) and every strain as an upper bound. The output
   carries `instrument_corrected: false` and a question asking for a standard.

If the instrument component is as large as the measured one (β_L(f) ≤ 0 and
β_G(f)² ≤ 0), or the sample FWHM left after subtraction is under 2 standard errors,
the sample broadening is **not resolved**. VIDUR reports that and gives no size, rather
than a very large number that only reflects the subtraction. If only one component is
≤ 0, it is taken as 0, and a flag says which.

The fitted σ and γ are strongly correlated. Their combination (the FWHM) can be well
determined when neither is on its own. So the total-breadth methods (Scherrer,
Williamson–Hall, SSP) use both fitted components as they are. The component-wise
method (§5.2) additionally requires each component to be at least 2 standard errors
from zero. Without that check, a γ sitting on its lower bound gave "sizes" of 10⁷ nm
on a real file.

---

## 5. Single-line methods

λ is the Kα1 wavelength. θ is from the fitted Kα1 position.

### 5.1 Scherrer (1918)

  D = K λ / (β cos θ)

β is the **sample** breadth after §4, in radians. Two variants are reported:

- **FWHM Scherrer**: β = FWHM of the deconvolved sample Voigt, K = `scherrer_k`,
  default **0.9**, the common convention for roughly equiaxed crystallites.
- **Integral-breadth Scherrer**: β = integral breadth of the sample profile, K = 1.
  Here D is the **volume-weighted column length** ⟨L⟩_v (Stokes & Wilson 1942), which
  has a model-free meaning. For spheres of diameter d, ⟨L⟩_v = 3d/4.

K is not universal. Langford & Wilson (1978) tabulate values from about 0.62 to 2.08,
depending on crystallite shape, the breadth used (FWHM or integral), and the size
definition. VIDUR records the K it used; change it with `scherrer_k`.

Validity: Scherrer attributes **all** broadening to size. Any microstrain present makes
D too small. It is meaningful below ~100–200 nm; above that, size broadening is smaller
than typical instrumental broadening.

### 5.2 Single-line Voigt method (de Keijser et al. 1982; Langford 1978)

Size broadening is taken as Lorentzian and microstrain broadening as Gaussian, so one
reflection separates the two:

  ⟨L⟩_v = λ / (β_L(f) cos θ)    (apparent size, volume-weighted)
  ε   = β_G(f) / (4 tan θ)     (apparent microstrain; the "upper-limit" strain,
                     e ≈ 1.25 ε_rms for a Gaussian strain distribution)

The assumption that size is purely Lorentzian and strain purely Gaussian is an
approximation (Balzar 1999). The method is most useful when only one reflection is
available, which is common for highly textured films.

---

## 6. Multi-line methods

These use how breadth changes with angle to separate size (∝ 1/cos θ) from strain
(∝ tan θ). They need **at least 3 fitted reflections**. With 2, a straight line passes
through them exactly and nothing tests the model. With fewer than 3, VIDUR does not
report the result.

### 6.1 Williamson–Hall, uniform deformation model (Williamson & Hall 1953)

Assuming size and strain broadening add linearly:

  β cos θ = K λ / D + 4 ε sin θ

Plot β cos θ against 4 sin θ. Intercept = Kλ/D, slope = ε. VIDUR runs an ordinary
least-squares fit and reports D, ε, their standard errors, R², and the number of
points. It runs two variants:

- `williamson_hall`: β = sample FWHM, K = `scherrer_k` (0.9). This is the thin-film
  literature convention.
- `williamson_hall_integral`: β = sample integral breadth, K = 1. This is how Williamson
  & Hall (1953) wrote it, and D is then a volume-weighted apparent size.

They do not agree, and neither is exact for Voigt peaks, because the linear addition of
size and strain breadths is only strictly true when both are Lorentzian. Measured on
synthetic data with Lorentzian size (D = 30 nm) and Gaussian strain (ε = 2×10⁻³): the
FWHM variant returns D ≈ 57 nm and the integral variant ≈ 37 nm, while the single-line
Voigt method and the SSP recover 30 nm. When the broadening really does add linearly
(a test with all-Lorentzian broadening), the FWHM variant recovers D and ε to within
10–15 %. A **negative slope** is reported as it is. It means the model does not fit
(common with anisotropic broadening), not that the strain is negative in a physical
sense. VIDUR does not clamp it to zero.

The uniform stress (USDM) and energy-density (UDEDM) variants (Mote et al. 2012,
ZnO) additionally need the direction-dependent Young's modulus E_hkl for each
reflection. They are not implemented; anyone who needs them must supply E_hkl.

### 6.2 Size–strain plot (Tagliente & Massaro 2008; Klug & Alexander 1974)

Assume a Lorentzian size profile and a Gaussian strain profile. Use the parabolic Voigt
approximation β² ≈ β·β_S + β_D² for the integral breadths. In reciprocal units,
β* = β cos θ / λ and d* = 1/d, with β_S* = K/D and β_D* = 2e·d*, this gives:

  (β*/d*)² = (K/D) (β*/d*²) + (2e)²

Slope = K/D, intercept = (2e)². Here e is the same apparent strain as in
Williamson–Hall (β_D = 4e tan θ), so the two can be compared directly.

**Two traps in how this is usually quoted.**

1. It is often written "(dβcosθ)² = (K/D)(d²βcosθ) + (ε/2)²", without λ. That form
   is dimensionally inconsistent: the first term has units of length, the others of
   length² or none. The first version of this module used it and returned D = 234 nm
   for a synthetic 30 nm sample. The reciprocal form above recovers 29.3 ± 0.2 nm
   (20-seed sweep).
2. The "ε" in "(ε/2)²" is the Stokes–Wilson apparent strain η = 4e, not the
   Williamson–Hall ε. VIDUR reports e as `strain`, and 4e as `strain_eta_literature`,
   for comparison with papers.

The fit uses the sample **integral breadth**. The default K = 1, so D is a
volume-weighted apparent size, consistent with §5.1 and §5.2. Mote et al. (2012) use
K = 3/4 for spheres; pass `ssp_k=0.75` to match that convention. A negative intercept
means the model is inconsistent with the data, and VIDUR reports it rather than
clamping it. The SSP gives less weight to high-angle reflections, which are the least
precise, and this is its usual advantage over Williamson–Hall.

### 6.3 Caution for thin films

In a symmetric θ–2θ scan only planes parallel to the surface diffract. A textured film
often shows one family (e.g. (111) and (222)), and a Williamson–Hall plot from
different families mixes directions whose size and elastic response can differ.
Orders of the **same** reflection (hkl, 2h2k2l) give the cleanest separation (this is
the basis of Warren–Averbach). VIDUR reports the hkl-agnostic peak list; which peaks
belong in a plot is the analyst's decision, made with `wh_peaks=[…]`.

Peak **shifts** (macro-strain, residual stress) are a different measurement
(sin²ψ) and are not derived from breadth here.

---

## 7. Dislocation density

### 7.1 The thin-film convention

  δ = 1 / D²    (lines m⁻²)

This is Williamson & Smallman's (1956) estimate for one dislocation per crystallite
face (ρ = n/D², n = 1). It is what most thin-film papers report, so VIDUR computes it
from each available D, labelled with the D it came from.

### 7.2 From size and strain

Williamson & Smallman (1956):

  ρ = 2√3 ε / (D b)

with b the Burgers vector magnitude. This is computed only when the caller gives
`burgers_nm`, since b depends on the material and slip system.

### 7.3 How much to trust it

Gubicza (2022, *Eur. Phys. J. Spec. Top.*) shows that peak breadth alone cannot fix
the dislocation density: the breadth depends on how dislocations are arranged
(dipoles and walls narrow the peaks) as well as how many there are, and
breadth-based values systematically disagree with whole-pattern (CMWP) fits. It also
warns that the X-ray "crystallite" is a coherently scattering domain (often a
sub-grain), typically 3–7× smaller than grain sizes from microscopy above ~20 nm.

So δ = 1/D² is a **conventional index for comparing samples measured the same way**,
not a measured dislocation density. VIDUR labels it that way in the output.

---

## 8. What VIDUR reports per file

- `peaks.csv`: one row per candidate. Columns: 2θ, d, height, area, FWHM (measured and
  sample), integral breadths, fitted σ and γ, instrument widths, fit χ²_red, Scherrer D
  (FWHM and integral breadth), Voigt size and strain, δ, each with its standard error,
  plus `status` (`ok`, `unresolved`, `not_fitted`, `not_significant`, `ka2_partner`,
  `suspect`), `reason` and `flags` columns.
- `fit.csv`: the fit next to the data, with columns 2θ, raw (unaltered), model,
  background and residual (raw − model).
- `microstructure.json`: both WH variants and the SSP regression (D, ε, standard errors, R², n, which
  peaks were used), every constant used and where it came from (file or caller), and
  any open questions.

---

## 9. Not done (and why)

- **Rietveld / whole-pattern fitting** needs a structure model. For quantitative work
  GSAS-II, FullProf or TOPAS are the right tools.
- **Warren–Averbach / Fourier analysis** needs at least two orders of one reflection,
  with clean tails.
- **CMWP / modified Williamson–Hall with contrast factors** (Ungár) is needed for
  anisotropic dislocation broadening; it needs elastic constants and slip systems.
- **Phase identification** needs a reference database.
- **Texture coefficient** (Harris 1952) needs reference intensities I₀(hkl) from a
  powder pattern or database for the material.
- **USDM/UDEDM** needs E_hkl (§6.1).

---

## 10. Measured on real files (2026-09-25, no instrument standard available)

- `XRD-918-16_10.xrdml` (PANalytical Empyrean, multiphase powder, Kα2 ratio 0.5 read
  from the file): 26 reflections fitted. Uncorrected Scherrer sizes are 60–195 nm, and
  the Williamson–Hall R² is 0.07. This is what instrument-limited peaks look like: the
  sizes are lower bounds, and they say more about the diffractometer than the sample.
  The strongest reflection (26.6°, 10⁵ counts) has χ²_red ≈ 3.6×10⁴, and its peak is
  visibly asymmetric. It is flagged, not removed.
- `m54313_om2th_10.xrdml` (X'Pert MRD, 2θ–ω scan of a film on a single-crystal
  substrate): the substrate lines are 1 point wide at a 0.03° step and are reported as
  `not_fitted`. The substrate line at 31.7° is masked out of the broad film peak's fit
  at 33.0°. Two film reflections are fitted, so WH and SSP are correctly not computed.
- `test_scan.xrdml` (ratio 0 in the file, so Kα1 only): four reflections are fitted
  with no doublet.

The detection threshold was not tuned on these files. It follows from the false-alarm
rate (§2).

## References

Provenance. The formulas and numbers for 11, 13, 15, 16, 17 and 18 were read from the
source during this survey. Refs 1 and 7 were checked only through secondary pages, and
the rest are cited from standard bibliographic knowledge. Volume and page numbers not
marked as read should be checked before any of these references goes into a paper.

1. P. Scherrer, *Nachr. Ges. Wiss. Göttingen* 26, 98 (1918).
2. A. R. Stokes, A. J. C. Wilson, *Proc. Camb. Phil. Soc.* 38, 313 (1942): volume-weighted size from integral breadth.
3. G. K. Williamson, W. H. Hall, *Acta Metall.* 1, 22 (1953).
4. G. K. Williamson, R. E. Smallman, *Phil. Mag.* 1, 34 (1956), doi:10.1080/14786435608238074.
5. G. Caglioti, A. Paoletti, F. P. Ricci, *Nucl. Instrum.* 3, 223 (1958).
6. H. P. Klug, L. E. Alexander, *X-ray Diffraction Procedures*, 2nd ed., Wiley (1974).
7. J. I. Langford, A. J. C. Wilson, *J. Appl. Cryst.* 11, 102 (1978): Scherrer constants.
8. J. J. Olivero, R. L. Longbothum, *J. Quant. Spectrosc. Radiat. Transf.* 17, 233 (1977): Voigt FWHM.
9. Th. H. de Keijser, J. I. Langford, E. J. Mittemeijer, A. B. P. Vogels, *J. Appl. Cryst.* 15, 308 (1982), doi:10.1107/S0021889882012035: single-line Voigt method.
10. P. Thompson, D. E. Cox, J. B. Hastings, *J. Appl. Cryst.* 20, 79 (1987): TCH pseudo-Voigt.
11. D. Balzar, "Voigt-function model in diffraction line-broadening analysis", in *Microstructure Analysis from Diffraction*, IUCr (1999).
12. G. Hölzer et al., *Phys. Rev. A* 56, 4554 (1997): Cu Kα lines.
13. M. H. Mendenhall et al., *J. Phys. B* 50, 115004 (2017): Cu Kα, NIST.
14. M. A. Tagliente, M. Massaro, *Nucl. Instrum. Methods B* 266, 1055 (2008): size–strain plot.
15. V. D. Mote, Y. Purushotham, B. N. Dole, *J. Theor. Appl. Phys.* 6, 6 (2012): W–H UDM/USDM/UDEDM and SSP on ZnO.
16. R. E. Dinnebier, P. Scardi, *J. Appl. Cryst.* 54, 1811 (2021): Bragg peak profiles (TCH coefficients).
17. NIST SRM 660c certificate: LaB₆ line position and line shape standard.
18. J. Gubicza, *Eur. Phys. J. Spec. Top.* 231, 4153 (2022), doi:10.1140/epjs/s11734-022-00572-z: reliability of line-profile parameters.
