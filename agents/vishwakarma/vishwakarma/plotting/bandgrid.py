"""
Read band energies from a Quantum ESPRESSO data-file-schema.xml and unfold the
irreducible k-points onto the full uniform Monkhorst-Pack grid.

Everything is read from the XML (cell, atoms, mesh, electron count, Fermi
level) so nothing here is specific to one material. Unfolding applies the
space-group point operations found by spglib plus time reversal, and checks
that every grid point is filled and that images of one k-point agree; a run
that is not a uniform-mesh NSCF (e.g. a bands path) is rejected with a clear
error instead of being plotted wrongly.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HARTREE_EV = 27.211386245988
BOHR_ANG = 0.529177210903


class BandGridError(ValueError):
    """The XML cannot be turned into a full-grid band set."""


@dataclass
class BandGrid:
    energies: np.ndarray            # (n1, n2, n3, nbnd) eV, full grid
    mesh: tuple                     # (n1, n2, n3)
    shift: tuple                    # (s1, s2, s3), 0 or 1 (half-step)
    cell_alat: np.ndarray           # (3,3) rows a_i in units of alat
    alat_ang: float
    recip_ang: np.ndarray           # (3,3) rows b_i in 1/Angstrom (with 2 pi)
    nelec: float
    fermi_eV: float | None
    n_sym_ops: int
    source: str = ""
    notes: list = field(default_factory=list)

    @property
    def nbnd(self) -> int:
        return self.energies.shape[-1]

    @property
    def n_occ(self) -> int:
        return int(round(self.nelec / 2.0))

    def gap_summary(self) -> dict:
        """Band edges from the grid (the true extrema may lie between grid points)."""
        nocc = self.n_occ
        if self.nbnd <= nocc:
            return {"has_conduction_bands": False, "n_occupied_bands": nocc}
        vb = self.energies[..., nocc - 1]
        cb = self.energies[..., nocc]
        iv = np.unravel_index(int(vb.argmax()), vb.shape)
        ic = np.unravel_index(int(cb.argmin()), cb.shape)
        vbm, cbm = float(vb.max()), float(cb.min())
        frac = lambda idx: [float((i + 0.5 * s) / n) for i, n, s in zip(idx, self.mesh, self.shift)]
        return {
            "has_conduction_bands": True,
            "n_occupied_bands": nocc,
            "vbm_eV": vbm, "cbm_eV": cbm,
            "gap_eV": cbm - vbm,
            "direct_gap_min_eV": float((cb - vb).min()),
            "vbm_k_crystal": frac(iv), "cbm_k_crystal": frac(ic),
            "metallic": bool(cbm <= vbm),
            "note": "gap from the k-grid only; true band edges may fall between grid points",
        }


def _floats(text: str) -> list:
    return [float(x) for x in text.split()]


def read_xml(path) -> dict:
    root = ET.parse(str(path)).getroot()
    out = root.find("output")
    if out is None:
        raise BandGridError("no <output> element: not a finished QE run")
    bs = out.find("band_structure")
    if bs is None:
        raise BandGridError("no <band_structure> in XML")
    if bs.findtext("lsda", "false").strip() == "true" or bs.findtext("noncolin", "false").strip() == "true":
        raise BandGridError("spin-polarised / noncollinear runs are not supported yet")

    st = out.find("atomic_structure")
    alat_bohr = float(st.attrib["alat"])
    cell = np.array([_floats(st.find(f"cell/a{i}").text) for i in (1, 2, 3)]) / alat_bohr
    atoms = [(a.attrib["name"], _floats(a.text)) for a in st.findall("atomic_positions/atom")]
    mp = bs.find("starting_k_points/monkhorst_pack")
    if mp is None:
        raise BandGridError(
            "k-points are not a Monkhorst-Pack mesh (a bands/path run?). "
            "Plotting needs a uniform-mesh NSCF.")
    mesh = (int(mp.attrib["nk1"]), int(mp.attrib["nk2"]), int(mp.attrib["nk3"]))
    shift = (int(mp.attrib["k1"]), int(mp.attrib["k2"]), int(mp.attrib["k3"]))
    ks, en = [], []
    for e in bs.findall("ks_energies"):
        ks.append(_floats(e.find("k_point").text))
        en.append(_floats(e.find("eigenvalues").text))
    fermi = bs.findtext("fermi_energy")
    if fermi is None:
        fermi = bs.findtext("highestOccupiedLevel")
    return {
        "alat_bohr": alat_bohr, "cell": cell, "atoms": atoms, "mesh": mesh, "shift": shift,
        "k_cart_alat": np.array(ks), "E_eV": np.array(en) * HARTREE_EV,
        "nelec": float(bs.findtext("nelec")),
        "fermi_eV": float(fermi) * HARTREE_EV if fermi is not None else None,
    }


def load_band_grid(save_dir) -> BandGrid:
    """Load <save_dir>/data-file-schema.xml (or the xml itself) and unfold it onto the full mesh."""
    import spglib

    p = Path(save_dir)
    xml = p if p.is_file() else p / "data-file-schema.xml"
    if not xml.is_file():
        raise BandGridError(f"no data-file-schema.xml at {xml}")
    d = read_xml(xml)
    cell, mesh, shift = d["cell"], d["mesh"], d["shift"]

    # symmetry from the actual structure (atom positions are cartesian, in bohr)
    cell_bohr = cell * d["alat_bohr"]
    names = sorted({n for n, _ in d["atoms"]})
    numbers = [names.index(n) + 1 for n, _ in d["atoms"]]
    frac = np.array([np.array(x) @ np.linalg.inv(cell_bohr) for _, x in d["atoms"]])
    sym = spglib.get_symmetry((cell_bohr, frac, numbers), symprec=1e-4)
    if sym is None:
        raise BandGridError("spglib could not find the symmetry of this structure")
    rots = np.unique(sym["rotations"], axis=0)

    kc = d["k_cart_alat"] @ cell.T            # k in reciprocal crystal coordinates
    E = d["E_eV"]
    n = np.array(mesh)
    half = np.array(shift) * 0.5
    full = np.full(tuple(mesh) + (E.shape[1],), np.nan)
    for sign in (1, -1):                      # time reversal
        for R in rots:
            kp = sign * (kc @ np.linalg.inv(R))   # k' = R^-T k, written for row vectors
            x = kp * n - half
            idx = np.rint(x).astype(int)
            if not np.allclose(idx, x, atol=1e-5):
                raise BandGridError("a k-point does not sit on the Monkhorst-Pack grid")
            idx %= n
            for (i, j, l), e in zip(idx, E):
                cur = full[i, j, l]
                if np.isnan(cur[0]):
                    full[i, j, l] = e
                elif np.abs(cur - e).max() > 5e-3:     # eV; images must agree
                    raise BandGridError(
                        f"symmetry images disagree by {np.abs(cur - e).max():.4f} eV; "
                        "the k-point set is not a symmetry-reduced uniform mesh")
    missing = int(np.isnan(full[..., 0]).sum())
    if missing:
        raise BandGridError(f"{missing} of {int(np.prod(mesh))} grid points not reached by symmetry")

    alat_ang = d["alat_bohr"] * BOHR_ANG
    recip = 2 * np.pi * np.linalg.inv(cell * alat_ang).T   # rows b_i, 1/Angstrom
    return BandGrid(energies=full, mesh=tuple(mesh), shift=tuple(shift), cell_alat=cell,
                    alat_ang=alat_ang, recip_ang=recip, nelec=d["nelec"],
                    fermi_eV=d["fermi_eV"], n_sym_ops=len(rots), source=str(xml))
