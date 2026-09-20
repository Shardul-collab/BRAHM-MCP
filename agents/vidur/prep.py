"""Derived columns for plot-ready CSVs.

The contract (VIDUR-DESIGN.md): the raw column is never altered. Everything here
ADDS a column named for exactly what it did. Nothing removes points.

Two kinds of step, and only the first runs by default:

  rescaling      reversible, loses nothing   normalisation, unit conversion
  reconstruction alters/removes points       background, smoothing, spike removal

Reconstruction is not implemented here. When it is, it must follow the same rule:
an extra column, never in place, and off unless asked for.
"""
from __future__ import annotations

import numpy as np

H_C_EV_NM = 1239.84158  # hc in eV*nm; hv(eV) = H_C_EV_NM / lambda(nm)
CU_KA1_A = 1.5405980    # only ever used when the caller supplies it explicitly


# ── normalisation (rescaling: safe, on by default) ───────────────────────────

def norm_max(y: np.ndarray) -> np.ndarray:
    """y / max(y). The usual one for comparing XRD patterns or XPS regions on
    one axis -- without it a strong and a weak scan cannot be read together."""
    m = np.nanmax(np.abs(y))
    return y / m if m else y.copy()


def norm_area(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y / integral(y dx). Area-under-curve normalisation."""
    a = np.trapezoid(np.abs(y), x) if hasattr(np, "trapezoid") else np.trapz(np.abs(y), x)
    return y / a if a else y.copy()


def norm_vector(y: np.ndarray) -> np.ndarray:
    """y / ||y||. Vector (unit-norm) normalisation."""
    n = np.sqrt(np.nansum(y ** 2))
    return y / n if n else y.copy()


def norm_reference_band(y: np.ndarray, x: np.ndarray, centre: float,
                        window: float = 5.0) -> np.ndarray:
    """y / max(y within centre +/- window). The internal-standard normalisation,
    e.g. Raman against the Si band at 520.7 cm-1."""
    sel = (x >= centre - window) & (x <= centre + window)
    if not sel.any():
        raise ValueError(f"no points within {window} of {centre}")
    m = np.nanmax(y[sel])
    return y / m if m else y.copy()


def offset(y: np.ndarray, step: float, index: int) -> np.ndarray:
    """Additive offset for stacked/waterfall figures. Figure-only: it changes
    the values, so it is its own column and never replaces the raw one."""
    return y + step * index


# ── unit conversion (rescaling: safe, but needs a stated constant) ───────────

def two_theta_to_d(two_theta_deg: np.ndarray, wavelength_a: float) -> np.ndarray:
    """Bragg: d = lambda / (2 sin(theta)). Needs the wavelength -- .xrdml carries
    it, ASCII usually does not, so the caller must supply it rather than have a
    Cu Ka1 value assumed silently."""
    theta = np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return wavelength_a / (2.0 * np.sin(theta))


def two_theta_to_q(two_theta_deg: np.ndarray, wavelength_a: float) -> np.ndarray:
    """q = 4 pi sin(theta) / lambda, in A^-1."""
    theta = np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)
    return 4.0 * np.pi * np.sin(theta) / wavelength_a


def wavelength_to_photon_energy(wl_nm: np.ndarray) -> np.ndarray:
    """hv (eV) from wavelength (nm)."""
    wl = np.asarray(wl_nm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return H_C_EV_NM / wl


def transmittance_to_absorbance(pct_t: np.ndarray) -> np.ndarray:
    """A = 2 - log10(%T)."""
    t = np.asarray(pct_t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 - np.log10(t)


def kubelka_munk(reflectance: np.ndarray) -> np.ndarray:
    """F(R) = (1-R)^2 / 2R, for diffuse reflectance."""
    r = np.asarray(reflectance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1.0 - r) ** 2 / (2.0 * r)


def tauc(alpha: np.ndarray, hv_ev: np.ndarray, n: float) -> np.ndarray:
    """(alpha*hv)^n. n = 2 for a direct allowed transition, 1/2 for indirect.

    alpha must be a real absorption coefficient, which needs the film thickness
    (alpha = 2.303 A / t). Substituting absorbance for alpha is common and the
    literature warns against it, so callers that have no thickness must not get
    this column at all rather than get a wrong one.
    """
    return (np.asarray(alpha, dtype=float) * np.asarray(hv_ev, dtype=float)) ** n


def absorbance_to_alpha(absorbance: np.ndarray, thickness_cm: float) -> np.ndarray:
    """alpha (cm^-1) = 2.303 * A / t."""
    if not thickness_cm or thickness_cm <= 0:
        raise ValueError("film thickness (cm) is required to compute alpha")
    return 2.303 * np.asarray(absorbance, dtype=float) / thickness_cm


# ── per-technique column sets ────────────────────────────────────────────────

def derive(technique: str, axis: np.ndarray, y: np.ndarray,
           params: dict | None = None) -> tuple[dict, list[str]]:
    """Return ({column_name: values}, [notes]) for one parsed spectrum.

    Only rescaling steps run. Anything needing a constant the file does not
    carry (wavelength, thickness) is skipped, and the reason is returned as a
    note so the caller can ask for it instead of assuming a default.
    """
    p = params or {}
    cols: dict[str, np.ndarray] = {"raw": y}
    notes: list[str] = []
    axis = np.asarray(axis, dtype=float)
    y = np.asarray(y, dtype=float)

    cols["norm_max"] = norm_max(y)

    if technique == "XRD":
        lam = p.get("wavelength_a")
        if p.get("axis_is_two_theta") is False:
            # The file declares a q axis, not degrees. Bragg does not apply.
            notes.append("x axis is q as declared by the file, not 2theta: "
                         "d-spacing and q columns skipped")
            lam = None
        if lam:
            cols["d_A"] = two_theta_to_d(axis, lam)
            cols["q_invA"] = two_theta_to_q(axis, lam)
        else:
            notes.append("no wavelength: d-spacing and q columns skipped "
                         "(Cu Ka1 = 1.5406 A is not assumed)")

    elif technique == "Raman":
        cols["norm_area"] = norm_area(y, axis)
        cols["norm_vector"] = norm_vector(y)
        ref = p.get("reference_band_cm1")
        if ref:
            try:
                cols[f"norm_ref_{ref:g}"] = norm_reference_band(y, axis, float(ref))
            except ValueError as exc:
                notes.append(f"reference-band normalisation skipped: {exc}")

    elif technique == "UV-Vis":
        if p.get("axis_is_nm", True):
            cols["photon_energy_eV"] = wavelength_to_photon_energy(axis)
        kind = p.get("y_kind", "absorbance")
        if kind == "transmittance_pct":
            cols["absorbance"] = transmittance_to_absorbance(y)
        elif kind == "reflectance":
            cols["kubelka_munk"] = kubelka_munk(y)
        thickness = p.get("thickness_cm")
        if thickness and "photon_energy_eV" in cols:
            a = cols.get("absorbance", y if kind == "absorbance" else None)
            if a is not None:
                alpha = absorbance_to_alpha(a, thickness)
                cols["alpha_invcm"] = alpha
                cols["tauc_direct"] = tauc(alpha, cols["photon_energy_eV"], 2.0)
                cols["tauc_indirect"] = tauc(alpha, cols["photon_energy_eV"], 0.5)
        else:
            notes.append("no film thickness: alpha and Tauc columns skipped "
                         "(absorbance is not substituted for alpha)")

    return cols, notes
