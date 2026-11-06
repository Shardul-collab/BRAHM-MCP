"""
Valence charge density from a QE run (charge-density.hdf5) as a 3D isosurface.

QE stores rho(G) as Fourier coefficients: MillerIndices (ngm, 3) and rhotot_g
(real and imaginary parts interleaved), in units of e/bohr^3. The real-space
density on the primitive cell is rebuilt with an inverse FFT on a grid that is
zero-padded (exact Fourier interpolation) so the surface is smooth.

Checks reported with every result, so a wrong reconstruction shows up as a
number and not as a pretty picture:
    - rho(G=0) * volume equals the electron count in the run,
    - the imaginary part of rho(r) is zero to rounding (rho(-G) = conj rho(G)),
    - the real-space mean equals rho(G=0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bandgrid import BOHR_ANG, BandGridError, read_xml


@dataclass
class Density:
    rho: np.ndarray               # (n1, n2, n3) e/Angstrom^3
    cell_ang: np.ndarray          # (3,3) rows a_i in Angstrom
    atoms: list                   # [(symbol, cartesian xyz Angstrom)]
    nelec: float
    volume_ang3: float
    checks: dict
    source: str = ""


def load_density(save_dir, oversample: float = 2.0) -> Density:
    try:
        import h5py
        from scipy.fft import next_fast_len
    except ImportError as exc:
        raise BandGridError(f"density plots need h5py and scipy ({exc})")
    p = Path(save_dir)
    h5 = p / "charge-density.hdf5"
    xml = p / "data-file-schema.xml"
    if not h5.is_file():
        raise BandGridError(f"no charge-density.hdf5 in {p} (was disk_io='none', or an old binary .dat format?)")
    d = read_xml(xml, need_mesh=None)
    cell_ang = d["cell"] * d["alat_bohr"] * BOHR_ANG
    vol = float(abs(np.linalg.det(cell_ang)))

    with h5py.File(h5, "r") as f:
        if f.attrs.get("gamma_only", b".FALSE.") in (b".TRUE.", ".TRUE."):
            raise BandGridError("gamma-only charge densities are not supported yet")
        if int(f.attrs.get("nspin", 1)) != 1:
            raise BandGridError("spin-polarised densities are not supported yet")
        mill = f["MillerIndices"][:].astype(int)
        ri = f["rhotot_g"][:]
    c = ri[0::2] + 1j * ri[1::2]
    if len(c) != len(mill):
        raise BandGridError("rhotot_g and MillerIndices disagree in length")

    n = [next_fast_len(int(np.ceil(oversample * (2 * np.abs(mill[:, i]).max() + 1)))) for i in range(3)]
    cube = np.zeros(n, dtype=complex)
    cube[mill[:, 0] % n[0], mill[:, 1] % n[1], mill[:, 2] % n[2]] = c
    rho_bohr = np.fft.ifftn(cube) * cube.size                 # sum_G c_G exp(2 pi i G.r)
    im = float(np.abs(rho_bohr.imag).max())
    re = rho_bohr.real
    g0 = c[np.all(mill == 0, axis=1)]
    rho0 = float(g0[0].real) if len(g0) else float("nan")
    vol_bohr3 = vol / BOHR_ANG ** 3
    scale = 1.0 / BOHR_ANG ** 3                                # e/bohr^3 -> e/Angstrom^3
    rho = re * scale
    checks = {
        "grid": n,
        "electrons_from_rho_G0": rho0 * vol_bohr3,
        "electrons_in_run": d["nelec"],
        "electrons_from_real_space_mean": float(re.mean() * vol_bohr3),
        "max_abs_imag_over_max_real": im / float(np.abs(re).max()),
        "rho_min_e_per_A3": float(rho.min()), "rho_max_e_per_A3": float(rho.max()),
        "rho_mean_e_per_A3": float(rho.mean()),
    }
    if abs(checks["electrons_from_rho_G0"] - d["nelec"]) > 0.01 * d["nelec"]:
        raise BandGridError(
            f"charge density integrates to {checks['electrons_from_rho_G0']:.3f} electrons, "
            f"run has {d['nelec']}: refusing to plot (units or file format not understood)")
    if checks["max_abs_imag_over_max_real"] > 1e-6:
        raise BandGridError("rho(r) has an imaginary part: coefficients are not Hermitian, refusing to plot")
    atoms = [(nm, np.array(x) * BOHR_ANG) for nm, x in d["atoms"]]
    return Density(rho=rho, cell_ang=cell_ang, atoms=atoms, nelec=d["nelec"], volume_ang3=vol,
                   checks=checks, source=str(h5))


def _periodic(rho: np.ndarray) -> np.ndarray:
    return np.pad(rho, ((0, 1), (0, 1), (0, 1)), mode="wrap")


def _contour(dn: Density, iso: float):
    import pyvista as pv
    n = dn.rho.shape
    f = [np.linspace(0, 1, m + 1) for m in n]
    X, Y, Z = np.meshgrid(*f, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) @ dn.cell_ang
    grid = pv.StructuredGrid(pts[:, 0].reshape(X.shape), pts[:, 1].reshape(X.shape), pts[:, 2].reshape(X.shape))
    grid["rho"] = _periodic(dn.rho).ravel(order="F")
    return grid, grid.contour([iso], scalars="rho").triangulate()


def default_isovalue(dn: Density, volume_fraction: float = 0.10) -> float:
    """
    The density level above which `volume_fraction` of the cell lies.

    A fixed fraction of the maximum picks out only the highest peaks (for Si
    with an ultrasoft pseudopotential that is the ion position, so no bonds
    show). A volume fraction adapts to any material and reads as "the densest
    10 % of the cell". For Si it gives 0.40 e/A^3, between the empty
    interstitial (0.02) and the bond centre (0.58).
    """
    return float(np.percentile(dn.rho, 100.0 * (1.0 - volume_fraction)))


_COLORS = {"Si": "#8b8fa3", "Zn": "#7c9cc9", "Se": "#d9a441", "In": "#a67db8", "Mo": "#5aa7a0", "S": "#e0c341"}


def plot_density(dn: Density, out_png: Path, out_html: Path | None, isovalue: float | None = None,
                 title: str = "") -> dict:
    iso = float(isovalue) if isovalue is not None else default_isovalue(dn)
    lo, hi = dn.checks["rho_min_e_per_A3"], dn.checks["rho_max_e_per_A3"]
    if not (lo < iso < hi):
        raise BandGridError(f"isovalue {iso:.4g} e/Å³ is outside the density range [{lo:.4g}, {hi:.4g}]")
    import pyvista as pv
    grid, surf = _contour(dn, iso)
    if surf.n_points == 0:
        raise BandGridError("the isosurface is empty at this isovalue")
    ttl = title or f"Valence charge density, isosurface at {iso:.3f} e/Å³"
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ dn.cell_ang
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    out = {"isovalue_e_per_A3": iso, "checks": dn.checks, "mesh_triangles": int(surf.n_cells),
           "cell_fraction_above_isovalue": float((dn.rho > iso).mean())}
    engine = "pyvista"
    try:
        pv.OFF_SCREEN = True
        pl = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        pl.set_background("white")
        pl.add_mesh(surf, color="#e8793a", opacity=0.75, smooth_shading=True)
        for a, b in edges:
            pl.add_mesh(pv.Line(corners[a], corners[b]), color="#555555", line_width=2)
        for sym, xyz in dn.atoms:
            pl.add_mesh(pv.Sphere(radius=0.32, center=xyz), color=_COLORS.get(sym, "#888888"))
        pl.add_axes()
        pl.camera_position = "iso"
        pl.reset_camera()
        pl.add_text(ttl, font_size=11, color="black", position="upper_edge")
        pl.screenshot(str(out_png))
        pl.close()
    except Exception as exc:
        engine = f"unavailable: {type(exc).__name__}: {exc}"
    out["png"] = str(out_png) if engine == "pyvista" else None
    out["png_engine"] = engine
    if out_html is not None:
        try:
            import plotly.graph_objects as go
            v = np.asarray(surf.points)
            tri = np.asarray(surf.faces).reshape(-1, 4)[:, 1:]
            fig = go.Figure()
            fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                                    color="#e8793a", opacity=0.75, name=f"ρ = {iso:.3f} e/Å³", showscale=False))
            xs, ys, zs = [], [], []
            for a, b in edges:
                xs += [corners[a][0], corners[b][0], None]
                ys += [corners[a][1], corners[b][1], None]
                zs += [corners[a][2], corners[b][2], None]
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#555", width=3),
                                       name="cell", hoverinfo="skip"))
            for sym in sorted({s for s, _ in dn.atoms}):
                pts = np.array([x for s, x in dn.atoms if s == sym])
                fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=sym,
                                           marker=dict(size=9, color=_COLORS.get(sym, "#888"))))
            fig.update_layout(title=ttl, scene=dict(aspectmode="data", xaxis_title="x (Å)",
                                                    yaxis_title="y (Å)", zaxis_title="z (Å)"),
                              margin=dict(l=0, r=0, t=40, b=0))
            fig.write_html(str(out_html), include_plotlyjs=True)
            out["html"] = str(out_html)
        except Exception as exc:
            out["html_error"] = f"{type(exc).__name__}: {exc}"
    return out
