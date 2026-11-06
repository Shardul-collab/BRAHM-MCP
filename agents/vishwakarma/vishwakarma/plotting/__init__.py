"""
vishwakarma.plotting: visualise results of a finished Vishwakarma DFT job.

    plot_job(job_id="1ef22e16-...", workdir=QE_WORKDIR, kind="all")

kinds:
    dos      2D density of states, built from the eigenvalues on the full k-grid
             (so the broadening is explicit and chosen here, not inherited
             from dos.x's degauss, which is in Ry).
    bands3d  3D surfaces of the highest valence and lowest conduction band over
             the k_z = 0 plane of the first Brillouin zone (PNG + interactive HTML).
    bands    2D band structure along a k-path (a 'bands' NSCF run).
    density  3D isosurface of the valence charge density rebuilt from
             charge-density.hdf5 (any finished pw.x run; PNG + interactive HTML).
             isovalue is in e/Angstrom^3; default is the level above which 10 % of the cell lies.
    all      what fits the run: dos + bands3d for a uniform mesh, bands for a k-path.

dos and bands3d need a uniform-mesh NSCF (or SCF); bands needs a k-path run.
Output goes to <job folder>/plots/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .bandgrid import BandGrid, BandGridError, load_band_grid, run_kind

KINDS = ("dos", "bands3d", "bands", "density", "all")


def plot_job(job_id: Optional[str] = None, workdir: Optional[str] = None,
             save_dir: Optional[str] = None, kind: str = "all",
             out_dir: Optional[str] = None, sigma_eV: float = 0.05,
             window_eV: tuple = (-14.0, 10.0), bands: Optional[tuple] = None,
             labels: Optional[list] = None, isovalue: Optional[float] = None) -> dict:
    """Render plots for a job; returns file paths and the numbers behind them."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    from vishwakarma.device import transport
    from . import figures

    sd = transport.locate_save_dir(job_id=job_id, workdir=workdir, save_dir=save_dir)
    xml = Path(sd) / "data-file-schema.xml"
    rk = run_kind(xml)

    if kind == "all":
        todo = ["bands"] if rk == "path" else ["dos", "bands3d"]
    else:
        todo = [kind]
    if kind == "density":
        pass                                   # the density does not depend on the k-point set
    elif rk == "path" and todo != ["bands"]:
        raise BandGridError("this run is a k-path: use kind='bands' (DOS and 3D bands need a uniform mesh)")
    if rk == "mesh" and todo == ["bands"]:
        raise BandGridError("this run is a uniform mesh, not a k-path: use kind='dos' or 'bands3d'")
    if kind == "density":
        from .density import load_density, plot_density
        out = Path(out_dir) if out_dir else (Path(workdir) / job_id / "plots" if job_id and workdir
                                              else Path(sd).parent / "plots")
        out.mkdir(parents=True, exist_ok=True)
        tag = job_id[:8] if job_id else Path(sd).parent.name
        dn = load_density(sd)
        res = plot_density(dn, out / f"density_{tag}.png", out / f"density_{tag}.html", isovalue=isovalue)
        files = {k: v for k, v in (("density_png", res.get("png")), ("density_html", res.get("html"))) if v}
        return {"source": str(sd), "run_type": "k-path" if rk == "path" else "uniform mesh",
                "electrons": dn.nelec, "files": files, "density": res}

    if out_dir:
        out = Path(out_dir)
    elif job_id and workdir:
        out = Path(workdir) / job_id / "plots"
    else:
        out = Path(sd).parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    tag = job_id[:8] if job_id else Path(sd).parent.name
    result = {"source": str(sd), "run_type": "k-path" if rk == "path" else "uniform mesh", "files": {}}

    if rk == "mesh":
        grid = load_band_grid(sd)
        result.update({"mesh": list(grid.mesh), "bands_in_file": grid.nbnd, "electrons": grid.nelec,
                       "symmetry_ops_used": grid.n_sym_ops, "band_edges": grid.gap_summary()})
        if "dos" in todo:
            result["dos"] = figures.plot_dos(grid, out / f"dos_{tag}.png", sigma_eV=sigma_eV,
                                             window_eV=tuple(window_eV))
            result["files"]["dos_png"] = result["dos"]["file"]
        if "bands3d" in todo:
            b3 = figures.plot_bands3d(grid, out / f"bands3d_{tag}.png", out / f"bands3d_{tag}.html",
                                      bands=tuple(bands) if bands else None)
            result["bands3d"] = b3
            result["files"]["bands3d_png"] = b3["png"]
            if "html" in b3:
                result["files"]["bands3d_html"] = b3["html"]
    else:
        from .bandpath import load_band_path, plot_bands2d
        bp = load_band_path(sd, labels=labels)
        win = tuple(window_eV) if tuple(window_eV) != (-14.0, 10.0) else (-12.0, 8.0)
        result["bands2d"] = plot_bands2d(bp, out / f"bands2d_{tag}.png", window_eV=win)
        result["files"]["bands2d_png"] = result["bands2d"]["file"]
        result["electrons"] = bp.nelec
        result["band_edges"] = bp.edges()
    return result


__all__ = ["plot_job", "load_band_grid", "BandGrid", "BandGridError", "KINDS"]
