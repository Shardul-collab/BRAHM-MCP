"""
Band energies along a k-path (a QE 'bands' NSCF run), read from
data-file-schema.xml, and a 2D band-structure figure.

The path has no mesh to unfold, so this is separate from bandgrid.py. Path
nodes (where the plot gets vertical lines and labels) are found from the
k-points themselves: a node is where the direction of travel changes or the
path jumps. Labels are filled in for the fcc special points when the lattice
is recognised as primitive fcc, otherwise only Gamma is named; pass
labels=[...] to override.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bandgrid import BandGridError, BOHR_ANG, read_xml_path


@dataclass
class BandPath:
    x: np.ndarray                 # cumulative path length, 1/Angstrom (jumps contribute 0)
    energies: np.ndarray          # (nk, nbnd) eV
    k_cart_ang: np.ndarray        # (nk, 3) 1/Angstrom
    nodes: list                   # indices into x of path nodes
    labels: list                  # one label per node
    nelec: float
    cell_alat: np.ndarray
    alat_ang: float
    source: str = ""

    @property
    def n_occ(self) -> int:
        return int(round(self.nelec / 2.0))

    def edges(self) -> dict:
        nocc = self.n_occ
        if self.energies.shape[1] <= nocc:
            return {"has_conduction_bands": False}
        vb, cb = self.energies[:, nocc - 1], self.energies[:, nocc]
        iv, ic = int(vb.argmax()), int(cb.argmin())
        return {"has_conduction_bands": True, "vbm_eV": float(vb[iv]), "cbm_eV": float(cb[ic]),
                "gap_eV": float(cb[ic] - vb[iv]), "direct_gap_min_eV": float((cb - vb).min()),
                "vbm_index": iv, "cbm_index": ic,
                "note": "gap along this path only; the true minimum may lie off the path"}


_FCC_POINTS = {            # sorted |components| in units of 2*pi/a_cubic
    "Γ": (0.0, 0.0, 0.0), "X": (0.0, 0.0, 1.0), "L": (0.5, 0.5, 0.5),
    "W": (0.0, 0.5, 1.0), "K": (0.0, 0.75, 0.75), "U": (0.25, 0.25, 1.0),
}


def _is_fcc_primitive(cell_alat: np.ndarray) -> bool:
    a = cell_alat
    n = np.linalg.norm(a, axis=1)
    if not np.allclose(n, n[0], rtol=1e-4):
        return False
    c = [a[i] @ a[j] / (n[0] ** 2) for i, j in ((0, 1), (0, 2), (1, 2))]
    return bool(np.allclose(c, 0.5, atol=1e-3))


def _auto_labels(k_cart_ang: np.ndarray, node_idx: list, cell_alat: np.ndarray, alat_ang: float) -> list:
    labels = []
    fcc = _is_fcc_primitive(cell_alat)
    a_cubic = alat_ang * np.sqrt(2.0)
    for i in node_idx:
        k = k_cart_ang[i]
        if np.linalg.norm(k) < 1e-6:
            labels.append("Γ")
            continue
        name = ""
        if fcc:
            v = np.sort(np.abs(k) * a_cubic / (2 * np.pi))
            for nm, ref in _FCC_POINTS.items():
                if np.allclose(v, ref, atol=2e-3):
                    name = nm
                    break
        labels.append(name or f"k{len(labels) + 1}")
    return labels


def load_band_path(save_dir, labels: list | None = None) -> BandPath:
    p = Path(save_dir)
    xml = p if p.is_file() else p / "data-file-schema.xml"
    if not xml.is_file():
        raise BandGridError(f"no data-file-schema.xml at {xml}")
    d = read_xml_path(xml)
    cell = d["cell"]
    alat_ang = d["alat_bohr"] * BOHR_ANG
    recip = 2 * np.pi * np.linalg.inv(cell * alat_ang).T
    # k in the XML is cartesian in units of 2*pi/alat
    k_ang = d["k_cart_alat"] * (2 * np.pi / alat_ang)
    if len(k_ang) < 3:
        raise BandGridError("a band path needs at least 3 k-points")
    dk = np.diff(k_ang, axis=0)
    step = np.linalg.norm(dk, axis=1)
    med = float(np.median(step[step > 1e-9])) if np.any(step > 1e-9) else 1.0
    jump = step > 3.0 * med                       # discontinuity between segments
    dist = np.where(jump, 0.0, step)
    x = np.concatenate([[0.0], np.cumsum(dist)])
    unit = dk / np.maximum(step, 1e-12)[:, None]
    nodes = [0]
    for i in range(1, len(dk)):
        if jump[i - 1] or jump[i]:
            nodes.append(i)
        elif np.linalg.norm(unit[i] - unit[i - 1]) > 1e-3:
            nodes.append(i)
    nodes.append(len(k_ang) - 1)
    nodes = sorted(set(nodes))
    if labels is not None and len(labels) != len(nodes):
        raise BandGridError(f"got {len(labels)} labels for {len(nodes)} path nodes")
    lab = list(labels) if labels is not None else _auto_labels(k_ang, nodes, cell, alat_ang)
    return BandPath(x=x, energies=d["E_eV"], k_cart_ang=k_ang, nodes=nodes, labels=lab,
                    nelec=d["nelec"], cell_alat=cell, alat_ang=alat_ang, source=str(xml))


def plot_bands2d(bp: BandPath, out_png: Path, window_eV: tuple = (-12.0, 8.0), title: str = "") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = bp.edges()
    ref = e["vbm_eV"] if e.get("has_conduction_bands") else float(bp.energies[:, bp.n_occ - 1].max())
    # merge a node pair that sit at the same x (a jump) into one tick label
    ticks, tl = [], []
    for i, lab in zip(bp.nodes, bp.labels):
        xv = bp.x[i]
        if ticks and abs(ticks[-1] - xv) < 1e-9:
            if tl[-1] != lab:
                tl[-1] = f"{tl[-1]}|{lab}"
        else:
            ticks.append(xv)
            tl.append(lab)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    nocc = bp.n_occ
    for b in range(bp.energies.shape[1]):
        ax.plot(bp.x, bp.energies[:, b] - ref, color="#c0392b" if b >= nocc else "#2c5aa0", lw=1.3)
    for t in ticks:
        ax.axvline(t, color="#888", lw=0.6)
    ax.axhline(0, color="k", ls="--", lw=0.7)
    if e.get("has_conduction_bands"):
        iv, ic = e["vbm_index"], e["cbm_index"]
        ax.plot([bp.x[iv]], [0.0], "o", color="#2c5aa0", ms=6)
        ax.plot([bp.x[ic]], [e["cbm_eV"] - ref], "o", color="#c0392b", ms=6)
        ax.text(0.98, 0.04, f"gap along path: {e['gap_eV']:.2f} eV (indirect)" if abs(bp.x[iv] - bp.x[ic]) > 1e-6
                else f"gap along path: {e['gap_eV']:.2f} eV (direct)",
                transform=ax.transAxes, ha="right", fontsize=9)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tl)
    ax.set_xlim(bp.x[0], bp.x[-1])
    ax.set_ylim(*window_eV)
    ax.set_ylabel("E − E$_{VBM}$ (eV)")
    ax.set_title(title or "Band structure along the k-path")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return {"file": str(out_png), "n_kpoints": int(len(bp.x)), "n_bands": int(bp.energies.shape[1]),
            "path_nodes": dict(zip(range(len(bp.labels)), bp.labels)), "band_edges": e}
