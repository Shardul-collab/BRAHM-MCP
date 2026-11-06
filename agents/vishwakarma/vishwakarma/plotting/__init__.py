"""
vishwakarma.plotting: visualise results of a finished Vishwakarma DFT job.

    plot_job(job_id="1ef22e16-...", workdir=QE_WORKDIR, kind="all")

kinds:
    dos      2D density of states, built from the eigenvalues on the full k-grid
             (so the broadening is explicit and chosen here, not inherited
             from dos.x's degauss, which is in Ry).
    bands3d  3D surfaces of the highest valence and lowest conduction band over
             the k_z = 0 plane of the first Brillouin zone (PNG + interactive HTML).
    all      both.

Needs a uniform-mesh NSCF (or SCF) job. Output goes to <job folder>/plots/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .bandgrid import BandGrid, BandGridError, load_band_grid

KINDS = ("dos", "bands3d", "all")


def plot_job(job_id: Optional[str] = None, workdir: Optional[str] = None,
             save_dir: Optional[str] = None, kind: str = "all",
             out_dir: Optional[str] = None, sigma_eV: float = 0.05,
             window_eV: tuple = (-14.0, 10.0), bands: Optional[tuple] = None) -> dict:
    """Render plots for a job; returns file paths and the numbers behind them."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    from vishwakarma.device import transport
    from . import figures

    sd = transport.locate_save_dir(job_id=job_id, workdir=workdir, save_dir=save_dir)
    grid = load_band_grid(sd)

    if out_dir:
        out = Path(out_dir)
    elif job_id and workdir:
        out = Path(workdir) / job_id / "plots"
    else:
        out = Path(sd).parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    tag = job_id[:8] if job_id else Path(sd).parent.name

    result = {
        "source": str(sd), "mesh": list(grid.mesh), "bands_in_file": grid.nbnd,
        "electrons": grid.nelec, "symmetry_ops_used": grid.n_sym_ops,
        "band_edges": grid.gap_summary(), "files": {},
    }
    if kind in ("dos", "all"):
        result["dos"] = figures.plot_dos(grid, out / f"dos_{tag}.png", sigma_eV=sigma_eV, window_eV=tuple(window_eV))
        result["files"]["dos_png"] = result["dos"]["file"]
    if kind in ("bands3d", "all"):
        b3 = figures.plot_bands3d(grid, out / f"bands3d_{tag}.png", out / f"bands3d_{tag}.html",
                                  bands=tuple(bands) if bands else None)
        result["bands3d"] = b3
        result["files"]["bands3d_png"] = b3["png"]
        if "html" in b3:
            result["files"]["bands3d_html"] = b3["html"]
    return result


__all__ = ["plot_job", "load_band_grid", "BandGrid", "BandGridError", "KINDS"]
