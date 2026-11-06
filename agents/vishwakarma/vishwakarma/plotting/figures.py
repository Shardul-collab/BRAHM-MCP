"""
Figures from a BandGrid: a 2D density of states and 3D band surfaces over a
Brillouin-zone plane. Nothing is material-specific: the zone shape comes from
the reciprocal lattice (Wigner-Seitz test), axes are in 1/Angstrom.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from .bandgrid import BandGrid


def _ref_energy(g: BandGrid) -> tuple[float, str]:
    """Energy zero: the Fermi level if the run has one, else the valence-band maximum."""
    if g.fermi_eV is not None:
        return g.fermi_eV, "E$_F$"
    s = g.gap_summary()
    if s.get("has_conduction_bands"):
        return s["vbm_eV"], "E$_{VBM}$"
    return float(g.energies.max()), "E$_{max}$"


def compute_dos(g: BandGrid, sigma_eV: float = 0.05, window_eV: tuple = (-14.0, 10.0), npts: int = 4000):
    """Gaussian-broadened DOS (states/eV/cell, both spins) on the unfolded grid."""
    ref, label = _ref_energy(g)
    x = np.linspace(window_eV[0], window_eV[1], npts)
    e = g.energies.reshape(-1, g.nbnd).ravel() - ref
    e = e[(e > x[0] - 8 * sigma_eV) & (e < x[-1] + 8 * sigma_eV)]
    dos = np.zeros_like(x)
    # histogram then convolve: exact for a gaussian kernel and far faster than a double loop
    dx = x[1] - x[0]
    hist, _ = np.histogram(e, bins=np.append(x - dx / 2, x[-1] + dx / 2))
    k = np.arange(-int(8 * sigma_eV / dx), int(8 * sigma_eV / dx) + 1) * dx
    kern = np.exp(-0.5 * (k / sigma_eV) ** 2)
    kern /= kern.sum()
    dos = np.convolve(hist, kern, mode="same") * 2.0 / (np.prod(g.mesh) * dx)
    return x, dos, ref, label


def plot_dos(g: BandGrid, out_png: Path, sigma_eV: float = 0.05, window_eV: tuple = (-14.0, 10.0),
             title: str = "") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, dos, ref, label = compute_dos(g, sigma_eV, window_eV)
    s = g.gap_summary()
    occ_mask = x < (s["vbm_eV"] - ref + 1e-9) if s.get("has_conduction_bands") else x < 0
    n_el = float(np.trapezoid(dos[occ_mask], x[occ_mask])) if occ_mask.any() else float("nan")
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.fill_between(x, dos, color="#3b6fb6", alpha=0.3)
    ax.plot(x, dos, color="#3b6fb6", lw=1.2)
    if s.get("has_conduction_bands") and not s["metallic"]:
        ax.axvspan(s["vbm_eV"] - ref, s["cbm_eV"] - ref, color="#f1c40f", alpha=0.25,
                   label=f"gap {s['gap_eV']:.2f} eV (k-grid)")
        ax.legend(frameon=False, fontsize=8)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlim(*window_eV)
    ax.set_xlabel(f"E − {label} (eV)")
    ax.set_ylabel("DOS (states/eV/cell)")
    ax.set_title(title or f"Electronic DOS (Gaussian σ = {sigma_eV:g} eV, {'x'.join(map(str, g.mesh))} k-mesh)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return {"file": str(out_png), "sigma_eV": sigma_eV, "reference_eV": ref,
            "reference": "fermi" if g.fermi_eV is not None else "vbm",
            "electrons_below_gap": n_el, "nelec_in_run": g.nelec}


def _in_first_bz(kxy: np.ndarray, recip: np.ndarray) -> np.ndarray:
    """Wigner-Seitz test: k is in the first zone if Gamma is its nearest reciprocal-lattice point."""
    gs = [np.array(c) @ recip for c in itertools.product(range(-2, 3), repeat=3) if any(c)]
    d0 = np.einsum("ij,ij->i", kxy, kxy)
    ok = np.ones(len(kxy), bool)
    for G in gs:
        ok &= d0 <= np.einsum("ij,ij->i", kxy - G, kxy - G) + 1e-12
    return ok


def band_surfaces(g: BandGrid, bands: tuple | None = None, n: int = 161, kz: float = 0.0):
    """Interpolate bands over the cartesian plane k_z = kz (1/Angstrom), masked to the first zone."""
    from scipy.interpolate import RegularGridInterpolator

    s = g.gap_summary()
    if bands is None:
        if not s.get("has_conduction_bands"):
            raise ValueError("run has no conduction bands; pass bands=(i, j) explicitly (0-based)")
        bands = (g.n_occ - 1, g.n_occ)
    pad = 2
    grids = []
    axes = []
    for ax_i, (m, sh) in enumerate(zip(g.mesh, g.shift)):
        axes.append((np.arange(-pad, m + pad) + 0.5 * sh) / m)
    L = 0.9 * float(np.max(np.linalg.norm(g.recip_ang, axis=1)))
    ku = np.linspace(-L, L, n)
    KX, KY = np.meshgrid(ku, ku)
    kxy = np.stack([KX.ravel(), KY.ravel(), np.full(n * n, kz)], 1)
    inbz = _in_first_bz(kxy, g.recip_ang)
    cry = (kxy @ np.linalg.inv(g.recip_ang)) % 1.0    # fractional coordinates of cartesian k, folded
    surf = []
    for b in bands:
        arr = np.pad(g.energies[..., b], pad, mode="wrap")
        itp = RegularGridInterpolator(tuple(axes), arr, method="cubic", bounds_error=False, fill_value=np.nan)
        z = itp(cry).reshape(n, n)
        surf.append(np.where(inbz.reshape(n, n), z, np.nan))
    return KX, KY, surf, bands


def plot_bands3d(g: BandGrid, out_png: Path, out_html: Path | None, bands: tuple | None = None,
                 title: str = "") -> dict:
    KX, KY, surf, bands = band_surfaces(g, bands)
    ref, label = _ref_energy(g)
    files = {}
    zscale = 0.3
    ttl = title or f"Bands {bands[0]+1} and {bands[1]+1} over the k$_z$ = 0 plane"
    # --- static PNG: PyVista if it can render offscreen, matplotlib 3D otherwise ---
    engine = "pyvista"
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
        pl = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        pl.set_background("white")
        for Z, cm in zip(surf, ("Blues_r", "Reds")):
            grid = pv.StructuredGrid(KX, KY, (Z - ref) * zscale)
            grid["E - ref (eV)"] = (Z - ref).ravel(order="F")
            pl.add_mesh(grid.threshold(value=-1e9, scalars="E - ref (eV)"), scalars="E - ref (eV)",
                        cmap=cm, show_scalar_bar=False, smooth_shading=True)
        pl.add_axes()
        pl.camera_position = "iso"
        pl.reset_camera()
        pl.camera.zoom(0.85)
        pl.add_text(ttl.replace("$", ""), font_size=11, color="black", position="upper_edge")
        pl.screenshot(str(out_png))
        pl.close()
    except Exception as exc:                       # no VTK / no offscreen context
        engine = f"matplotlib (pyvista unavailable: {type(exc).__name__})"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6), dpi=150)
        ax = fig.add_subplot(projection="3d")
        for Z, cm in zip(surf, ("Blues_r", "Reds")):
            ax.plot_surface(KX, KY, Z - ref, cmap=cm, linewidth=0, antialiased=True)
        ax.set_xlabel("k$_x$ (1/Å)"); ax.set_ylabel("k$_y$ (1/Å)"); ax.set_zlabel(f"E − {label} (eV)")
        ax.set_title(ttl)
        fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
    files["png"] = str(out_png)
    files["png_engine"] = engine
    # --- interactive HTML (rotate / zoom / hover) ---
    if out_html is not None:
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            for Z, cs, nm in zip(surf, ("Blues", "Reds"), (f"band {bands[0]+1}", f"band {bands[1]+1}")):
                fig.add_trace(go.Surface(x=KX[0], y=KY[:, 0], z=Z - ref, colorscale=cs, showscale=False, name=nm))
            fig.update_layout(title=ttl.replace("$", ""), scene=dict(
                xaxis_title="kx (1/Å)", yaxis_title="ky (1/Å)", zaxis_title=f"E − {label.replace('$', '').replace('_', '')} (eV)"),
                margin=dict(l=0, r=0, t=40, b=0))
            fig.write_html(str(out_html), include_plotlyjs=True)
            files["html"] = str(out_html)
        except Exception as exc:
            files["html_error"] = f"{type(exc).__name__}: {exc}"
    z0 = [float(np.nanmin(z)) for z in surf], [float(np.nanmax(z)) for z in surf]
    files["bands_plotted"] = [int(b) + 1 for b in bands]
    files["plane"] = "k_z = 0 (cartesian), first Brillouin zone"
    files["energy_range_eV"] = {"min": z0[0], "max": z0[1]}
    return files
