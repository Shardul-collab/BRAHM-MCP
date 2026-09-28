"""VIDUR Rietveld refinement through FullProf (fp2k).

Decided 2026-09-27 (Shardul): FullProf is the engine -- the RRD-A reference
refinements (002, 026-028) are FullProf, so results stay comparable -- with
strict CIF identity validation and GSAS-II kept for later cross-validation.

What this module does, and nothing more:
  1. read_cif()        - a small CIF reader (cell, space group, formula, Z,
                         symmetry operators, atom sites). No dependency.
  2. validate_phase()  - STRICT identity check of a CIF against the phase the
                         caller says it is. A mismatch is a rejection, never a
                         warning: a wrong structural model refines to numbers
                         that look fine and mean nothing.
  3. write_pcr()       - a FullProf PCR for one constant-wavelength X-ray
                         pattern (Ins=10 x,y,sigma), TCH pseudo-Voigt (Npr=7),
                         optional instrument resolution file (Res=1),
                         Chebyshev background, Rietveld (Jbt=0) or Le Bail
                         (Jbt=2) phases.
  4. refine()          - runs fp2k in stages (which parameter groups are free
                         at each stage is the caller's choice), reading the
                         refined values back from the PCR FullProf rewrites.
  5. refine_sequence() - the same over a list of patterns, each starting from
                         the previous pattern's result.

Every substitution this module makes in a structural model (a non-positive
or missing Biso set to 1.0, as FullProf's own CIFs_to_PCR does) is listed in
the report under "model_changes". Nothing is changed silently.
"""
from __future__ import annotations

import copy
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── atomic weights (IUPAC 2021 abridged/conventional values) ───────────────────
ATOMIC_WEIGHT = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.95,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "Sm": 150.36,
    "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93, "Er": 167.26,
    "Tm": 168.93, "Yb": 173.05, "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84,
    "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Th": 232.04, "U": 238.03,
}

_NUM = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?:\((\d+)\))?$")


def cif_number(tok: str) -> float | None:
    """'10.3328(3)' -> 10.3328; '?' / '.' -> None."""
    if tok is None:
        return None
    m = _NUM.match(tok.strip())
    return float(m.group(1)) if m else None


# ── CIF reading ───────────────────────────────────────────────────────────────

def _cif_tokens(text: str):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            buf = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            yield "\n".join(buf).strip()
            i += 1
            continue
        for m in re.finditer(r"'(?:[^']|'(?=\S))*'(?=\s|$)|\"[^\"]*\"|#.*|\S+", line):
            t = m.group(0)
            if t.startswith("#"):
                break
            if (t[0] == t[-1]) and t[0] in "'\"" and len(t) >= 2:
                t = t[1:-1]
            yield t
        i += 1


def parse_cif_items(text: str) -> tuple[dict, list[dict]]:
    """First data block only: (items, loops). loops are {tag: [values]}."""
    items, loops = {}, []
    toks = list(_cif_tokens(text))
    i, seen_block = 0, False
    while i < len(toks):
        t = toks[i]
        low = t.lower()
        if low.startswith("data_"):
            if seen_block:
                break
            seen_block = True
            i += 1
        elif low == "loop_":
            i += 1
            tags = []
            while i < len(toks) and toks[i].startswith("_"):
                tags.append(toks[i].lower())
                i += 1
            vals = []
            while i < len(toks) and not toks[i].startswith("_") and toks[i].lower() != "loop_" \
                    and not toks[i].lower().startswith("data_"):
                vals.append(toks[i])
                i += 1
            n = len(tags)
            if n and len(vals) % n == 0:
                loops.append({tag: vals[k::n] for k, tag in enumerate(tags)})
        elif t.startswith("_"):
            items[low] = toks[i + 1] if i + 1 < len(toks) else None
            i += 2
        else:
            i += 1
    return items, loops


def _loop_with(loops, tag):
    for lp in loops:
        if tag in lp:
            return lp
    return None


def _element(type_symbol: str) -> str:
    m = re.match(r"([A-Z][a-z]?)", type_symbol.strip())
    return m.group(1) if m else type_symbol


def _parse_op(op: str):
    """'-x+1/2, y, z+1/2' -> (R 3x3, t 3)."""
    R, t = np.zeros((3, 3)), np.zeros(3)
    for row, part in enumerate(op.lower().replace(" ", "").split(",")):
        for sign, term in re.findall(r"([+-]?)([^+-]+)", part):
            s = -1.0 if sign == "-" else 1.0
            if term in ("x", "y", "z"):
                R[row, "xyz".index(term)] = s
            elif re.fullmatch(r"[\d.]+/[\d.]+", term):
                a, b = term.split("/")
                t[row] += s * float(a) / float(b)
            else:
                t[row] += s * float(term)
    return R, t


def site_multiplicity(xyz, ops, tol=1e-3) -> int:
    pts = []
    p = np.asarray(xyz, float)
    for R, t in ops:
        q = (R @ p + t) % 1.0
        if not any(np.all(np.minimum(abs(q - r), 1 - abs(q - r)) < tol) for r in pts):
            pts.append(q)
    return len(pts)


@dataclass
class Cif:
    path: str
    items: dict
    formula_sum: str | None
    spacegroup: str | None
    cell: tuple | None
    z: float | None
    ops: list
    atoms: list = field(default_factory=list)   # dicts: label, element, x,y,z, occ, biso, mult
    source_id: str | None = None


def read_cif(path: str) -> Cif:
    text = Path(path).read_text(errors="ignore")
    items, loops = parse_cif_items(text)
    sg = (items.get("_space_group_name_h-m_alt") or items.get("_symmetry_space_group_name_h-m"))
    cell = tuple(cif_number(items.get(k)) for k in (
        "_cell_length_a", "_cell_length_b", "_cell_length_c",
        "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"))
    if any(v is None for v in cell):
        cell = None
    lp = _loop_with(loops, "_space_group_symop_operation_xyz") or \
        _loop_with(loops, "_symmetry_equiv_pos_as_xyz")
    ops_txt = (lp.get("_space_group_symop_operation_xyz") or lp.get("_symmetry_equiv_pos_as_xyz")) if lp else []
    ops = [_parse_op(o) for o in ops_txt]
    atoms = []
    al = _loop_with(loops, "_atom_site_fract_x")
    if al:
        n = len(al["_atom_site_fract_x"])
        for k in range(n):
            label = al.get("_atom_site_label", ["X%d" % k] * n)[k]
            ts = al.get("_atom_site_type_symbol", [label] * n)[k]
            occ = cif_number(al.get("_atom_site_occupancy", ["1"] * n)[k])
            b = None
            if "_atom_site_b_iso_or_equiv" in al:
                b = cif_number(al["_atom_site_b_iso_or_equiv"][k])
            elif "_atom_site_u_iso_or_equiv" in al:
                u = cif_number(al["_atom_site_u_iso_or_equiv"][k])
                b = None if u is None else 8 * math.pi ** 2 * u
            xyz = [cif_number(al[f"_atom_site_fract_{c}"][k]) for c in "xyz"]
            atoms.append({"label": label, "element": _element(ts), "x": xyz[0], "y": xyz[1],
                          "z": xyz[2], "occ": 1.0 if occ is None else occ, "biso": b,
                          "mult": site_multiplicity(xyz, ops) if ops else None})
    src = items.get("_cod_database_code")
    if not src:
        m = re.search(r"(mp-\d+|\d{7})", Path(path).stem)
        src = m.group(1) if m else None
    return Cif(path=str(path), items=items, formula_sum=items.get("_chemical_formula_sum"),
               spacegroup=sg, cell=cell, z=cif_number(items.get("_cell_formula_units_z")),
               ops=ops, atoms=atoms, source_id=src)


# ── formulas ──────────────────────────────────────────────────────────────────

def parse_formula(f: str) -> dict:
    """'LiFePO4', 'Fe Li O4 P', 'Ca5(PO4)3(OH)', 'Li1.33 O4 Ti1.67' -> {el: n}."""
    f = f.replace(" ", "")

    def parse(s, i):
        out = {}
        while i < len(s):
            c = s[i]
            if c == "(":
                sub, i = parse(s, i + 1)
                m = re.match(r"[\d.]+", s[i:])
                k = float(m.group(0)) if m else 1.0
                i += len(m.group(0)) if m else 0
                for e, n in sub.items():
                    out[e] = out.get(e, 0) + n * k
            elif c == ")":
                return out, i + 1
            else:
                m = re.match(r"([A-Z][a-z]?)([\d.]*)", s[i:])
                if not m:
                    raise ValueError(f"cannot parse formula {f!r} at {s[i:]!r}")
                e, k = m.group(1), float(m.group(2) or 1)
                out[e] = out.get(e, 0) + k
                i += len(m.group(0))
        return out, i
    return parse(f, 0)[0]


def _normalised(comp: dict) -> dict:
    tot = sum(comp.values())
    return {e: n / tot for e, n in comp.items() if n}


def same_composition(a: dict, b: dict, rtol=0.02) -> bool:
    na, nb = _normalised(a), _normalised(b)
    return set(na) == set(nb) and all(abs(na[e] - nb[e]) <= rtol * max(nb[e], 1e-9) + 1e-4 for e in na)


def molar_mass(comp: dict) -> float:
    return sum(ATOMIC_WEIGHT[e] * n for e, n in comp.items())


# ── strict identity validation ────────────────────────────────────────────────

@dataclass
class Phase:
    """A phase the caller asks for, and the CIF offered for it."""
    name: str
    cif: str
    formula: str                   # what the phase IS (e.g. "LiFePO4")
    source_id: str | None = None   # e.g. "1101111" or "mp-24941": must match the CIF
    method: str = "rietveld"       # or "lebail"


def validate_phase(ph: Phase) -> dict:
    """Reject a structural model whose identity does not match the phase named.

    Rejections (any one refuses the phase):
      - the CIF has no cell or no space group;
      - the CIF formula differs in elements or ratios from the phase's formula;
      - (Rietveld) the atom sites, weighted by occupancy x multiplicity, do not
        reproduce the CIF's own formula -- the file contradicts itself;
      - a source id was given and the CIF carries a different one.
    Warnings do not refuse: non-positive/missing Biso, origin-choice suffix,
    partial occupancies, no symmetry operators to check multiplicities."""
    rej, warn = [], []
    try:
        c = read_cif(ph.cif)
    except Exception as exc:  # unreadable is a rejection
        return {"phase": ph.name, "ok": False, "rejections": [f"CIF unreadable: {exc}"],
                "warnings": [], "cif": None}
    want = parse_formula(ph.formula)
    if not c.cell:
        rej.append("no complete cell in the CIF")
    if not c.spacegroup:
        rej.append("no space group symbol in the CIF")
    if not c.formula_sum:
        rej.append("no _chemical_formula_sum in the CIF")
    else:
        have = parse_formula(c.formula_sum)
        if set(have) != set(want):
            rej.append(f"elements differ: CIF has {sorted(have)}, phase {ph.name} needs {sorted(want)}")
        elif not same_composition(have, want):
            rej.append(f"composition differs: CIF {c.formula_sum!r} vs phase formula {ph.formula!r}")
    if ph.source_id and c.source_id and ph.source_id != c.source_id:
        rej.append(f"source id mismatch: expected {ph.source_id}, CIF is {c.source_id}")
    if ph.source_id and not c.source_id:
        warn.append(f"CIF carries no source id to check against {ph.source_id}")
    if ph.method == "rietveld":
        if not c.atoms:
            rej.append("no atom sites: a Rietveld phase needs a structural model")
        elif not c.ops:
            warn.append("no symmetry operators in the CIF: site multiplicities not checked")
        elif c.formula_sum and not rej:
            sites = {}
            for a in c.atoms:
                sites[a["element"]] = sites.get(a["element"], 0) + a["occ"] * a["mult"]
            if not same_composition(sites, parse_formula(c.formula_sum), rtol=0.03):
                rej.append(f"atom sites give {({e: round(n, 3) for e, n in sites.items()})}, "
                           f"inconsistent with the CIF formula {c.formula_sum!r}")
            elif c.z:
                per_z = {e: n / c.z for e, n in sites.items()}
                if not same_composition(per_z, parse_formula(c.formula_sum), rtol=0.03):
                    rej.append("Z inconsistent with the atom sites")
        for a in c.atoms if not rej else ():   # a rejected model is not refined: no model notes
            if a["biso"] is None or a["biso"] <= 0:
                warn.append(f"{a['label']}: Biso {a['biso']} is not physical; set to 1.0 "
                            "(FullProf CIFs_to_PCR convention)")
            if a["occ"] < 0.999:
                warn.append(f"{a['label']}: partial occupancy {a['occ']}")
    if c.spacegroup and ":" in c.spacegroup:
        warn.append(f"space group {c.spacegroup!r} carries a setting suffix; FullProf's default "
                    "setting is used -- check the origin choice")
    return {"phase": ph.name, "ok": not rej, "rejections": rej, "warnings": warn, "cif": c}


# ── FullProf input ────────────────────────────────────────────────────────────

GROUPS = ("scale", "zero", "background", "cell", "strain", "size")
SIG_SCALE = 3.0    # a phase is "present" when its scale exceeds 3 sigma (stated default)
DAMPED_RELAX = 0.3  # relaxation factor for the one retry of a failed step (stated default)
MIN_WT_PCT = 3.0   # below this weight %, a phase's cell and breadth are held (stated default)
CHI2_RISE_TOL = 0.02  # a step that raises chi2 by more than 2 % is not accepted (stated default)
PAPER_STEPS = [("scale",), ("zero",), ("background",), ("cell",), ("strain",), ("size",)]


@dataclass
class Instrument:
    wavelength: float
    irf: str | None = None           # FullProf IRF (Res=1)
    cthm: float = 0.0                # monochromator polarisation term
    ilo: int = 0                     # 3 = synchrotron x,y,sigma polarisation


def read_irf(path: str) -> dict:
    out = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        p = line.split()
        if not p or p[0].startswith("!"):
            continue
        if p[0] == "WAVE":
            out["wavelength"] = float(p[1])
        elif p[0] == "CTHM":
            out["cthm"] = float(p[1])
    return out


def _sg_for_fullprof(sg: str) -> str:
    return re.sub(r"\s*:\s*\S+$", "", sg).strip()


def _cell_codes(sg: str, cell) -> list:
    """Equal refinement-parameter ids for symmetry-equal cell edges; angles free
    only where the lattice allows. Returns six group labels (None = fixed)."""
    s = sg.replace(" ", "").upper()
    a, b, c, al, be, ga = cell
    if s[1:].startswith(("M-3", "M3", "N-3", "N3", "D-3", "A-3", "-43", "432", "23", "-3")) and \
            abs(a - b) < 1e-4 and abs(a - c) < 1e-4 and abs(al - 90) < 1e-3 and abs(ga - 90) < 1e-3:
        return ["a", "a", "a", None, None, None]          # cubic
    if abs(a - b) < 1e-4 and abs(ga - 120) < 1e-3:
        return ["a", "a", "c", None, None, None]          # hexagonal/trigonal (hex axes)
    if abs(a - b) < 1e-4 and abs(ga - 90) < 1e-3 and abs(al - 90) < 1e-3 and abs(be - 90) < 1e-3:
        return ["a", "a", "c", None, None, None]          # tetragonal
    if abs(al - 90) < 1e-3 and abs(be - 90) < 1e-3 and abs(ga - 90) < 1e-3:
        return ["a", "b", "c", None, None, None]          # orthorhombic
    if abs(al - 90) < 1e-3 and abs(ga - 90) < 1e-3:
        return ["a", "b", "c", None, "be", None]          # monoclinic b-unique
    return ["a", "b", "c", "al", "be", "ga"]


@dataclass
class PhaseState:
    name: str
    method: str
    cif: Cif
    formula: str
    scale: float = 1e-3
    uvwxy: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # U V W X Y GauSiz LorSiz
    cell: list = field(default_factory=list)


@dataclass
class State:
    zero: float = 0.0
    background: list = field(default_factory=lambda: [0.0] * 24)
    phases: list = field(default_factory=list)


def _fmt(v, w=10, d=5):
    return f"{v:{w}.{d}f}"


def write_pcr(path: Path, title: str, datafile: str, x, inst: Instrument, st: State,
              free: set, n_bkg: int = 12, excluded=(), ncy: int = 30,
              held: set = frozenset(), relax: float = 1.0) -> dict:
    """Write a single-pattern PCR. `free` = parameter groups refined in this run;
    `held` = 1-based phase numbers whose cell, strain and size stay fixed (their
    scale stays free, so a phase can still grow in).
    Returns {"n_refined": n, "model_changes": [...]}"""
    changes = []
    code_no = [0]
    ids: dict = {}

    def code(key, on):
        if not on:
            return 0.0
        if key not in ids:
            code_no[0] += 1
            ids[key] = code_no[0]
        return 10.0 * ids[key] + 1.0

    L = []
    L.append(f"COMM {title}")
    L.append(f"! Files => DAT-file: {datafile},  PCR-file: {path.stem}")
    L.append("!Job Npr Nph Nba Nex Nsc Nor Dum Iwg Ilo Ias Res Ste Nre Cry Uni Cor Opt Aut")
    res = 1 if inst.irf else 0
    ias = 1 if any(p.method == "lebail" for p in st.phases) else 0
    L.append(f"   0   7 {len(st.phases):3d}  -5 {max(len(excluded), 0):3d}   0   1   1   0 {inst.ilo:3d} "
             f"{ias:3d} {res:3d}   0   0   0   0   0   0   0")
    if inst.irf:
        L.append("!  Resolution file for Pattern#   1")
        L.append(Path(inst.irf).name)
    L.append("!Ipr Ppl Ioc Mat Pcr Ls1 Ls2 Ls3 NLI Prf Ins Rpa Sym Hkl Fou Sho Ana")
    L.append("   0   0   1   0   1   0   0   0   0   3  10   0   0   0   0   0   0")
    L.append("!")
    L.append("! Lambda1  Lambda2    Ratio    Bkpos    Wdt    Cthm     muR   AsyLim   Rpolarz  2nd-muR -> Patt# 1")
    L.append(f" {inst.wavelength:8.6f} {inst.wavelength:8.6f}  0.00000   40.000  15.000  {inst.cthm:6.4f}  "
             "0.0000  180.00    0.0000  0.0000")
    L.append("!")
    L.append("!NCY  Eps  R_at  R_an  R_pr  R_gl     Thmin       Step       Thmax    PSD    Sent0")
    step = float(np.median(np.diff(x)))
    L.append(f" {ncy:3d}  0.10  {relax:4.2f}  {relax:4.2f}  {relax:4.2f}  {relax:4.2f}  "
             f"{x[0]:10.4f} {step:10.6f} {x[-1]:10.4f}   0.000   0.000")
    L.append("!")
    L.append("! Excluded regions (LowT  HighT) for Pattern#  1")
    for lo, hi in excluded:
        L.append(f"  {lo:10.4f}  {hi:10.4f}")
    L.append("!")
    body = []
    # zero
    body.append("!  Zero    Code    SyCos    Code   SySin    Code  Lambda     Code MORE ->Patt# 1")
    body.append(f" {st.zero:8.5f} {code('zero', 'zero' in free):6.1f}  0.00000    0.0  0.00000    0.0 0.000000    0.00   0")
    body.append("!   Background coefficients/codes  for Pattern#  1  (Chebychev polynomials, up to 24 coefficients)")
    for r in range(4):
        vals = st.background[6 * r: 6 * r + 6]
        body.append(" ".join(f"{v:11.3f}" for v in vals))
        body.append(" ".join(f"{code(('bkg', 6 * r + k), 'background' in free and 6 * r + k < n_bkg):11.2f}"
                             for k in range(6)))
    body.append("!-------------------------------------------------------------------------------")
    for n, ph in enumerate(st.phases, 1):
        c = ph.cif
        body.append(f"!  Data for PHASE number: {n:3d}")
        body.append("!-------------------------------------------------------------------------------")
        body.append(ph.name)
        body.append("!")
        body.append("!Nat Dis Ang Pr1 Pr2 Pr3 Jbt Irf Isy Str Furth       ATZ    Nvk Npr More")
        comp = parse_formula(c.formula_sum or ph.formula)
        atz = (c.z or 0) * molar_mass(comp) if ph.method == "rietveld" else 0.0
        nat = len(c.atoms) if ph.method == "rietveld" else 0
        jbt = 0 if ph.method == "rietveld" else 2
        body.append(f" {nat:3d}   0   0 0.0 0.0 1.0 {jbt:3d}   0   0   0   0 {atz:14.3f}   0   7   0")
        body.append("!")
        body.append(f"{_sg_for_fullprof(c.spacegroup):<24} <--Space group symbol")
        if ph.method == "rietveld":
            body.append("!Atom   Typ       X        Y        Z     Biso       Occ     In Fin N_t Spc /Codes")
            nops = len(c.ops) or 1
            for a in c.atoms:
                b = a["biso"]
                if b is None or b <= 0:
                    changes.append(f"{ph.name}/{a['label']}: Biso {b} -> 1.0")
                    b = 1.0
                occ = a["occ"] * (a["mult"] or nops) / nops
                body.append(f"{a['label']:<6} {a['element']:<4} {a['x']:9.5f}{a['y']:9.5f}{a['z']:9.5f}"
                            f"{b:9.5f}{occ:10.5f}   0   0   0    0")
                body.append("                  0.00     0.00     0.00     0.00      0.00")
        body.append(f"!-------> Profile Parameters for Pattern #   1  ----> Phase # {n:3d}")
        body.append("!  Scale          Shape1      Bov      Str1      Str2      Str3   Strain-Model")
        body.append(f" {ph.scale:14.7g}   0.00000   0.00000   0.00000   0.00000   0.00000       0")
        sc = code(("scale", n), "scale" in free and ph.method == "rietveld")
        body.append(f"  {sc:12.5f}     0.000     0.000     0.000     0.000     0.000")
        body.append("!       U            V            W             X            Y         GauSiz      LorSiz    Size-Model")
        body.append("".join(f"{v:13.6f}" for v in ph.uvwxy) + "       0")
        cx = code(("X", n), "strain" in free and n not in held)
        cy = code(("Y", n), "size" in free and n not in held)
        body.append(f"{0:13.2f}{0:13.2f}{0:13.2f}{cx:13.2f}{cy:13.2f}{0:13.2f}{0:13.2f}")
        body.append("!     a          b         c        alpha      beta       gamma      # Cell Info")
        body.append("".join(f"{v:11.6f}" for v in ph.cell))
        groups = _cell_codes(c.spacegroup, ph.cell)
        body.append("".join(f"{code(('cell', n, g), 'cell' in free and g is not None and n not in held):11.5f}"
                            for g in groups))
        body.append("!  Pref1    Pref2      Asy1     Asy2     Asy3     Asy4      S_L      D_L")
        body.append("  1.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000  0.00000")
        body.append("     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00")
    L.append(f"   {code_no[0]:4d}    !Number of refined parameters")
    L.append("!")
    path.write_text("\n".join(L + body) + "\n")
    return {"n_refined": code_no[0], "model_changes": changes}


# ── reading FullProf's rewritten PCR back into the state ─────────────────────

def _floats(line):
    return [float(t) for t in line.split()]


def read_state(pcr: Path, st: State) -> State:
    lines = pcr.read_text(errors="ignore").splitlines()
    ph_i = -1
    i = 0
    while i < len(lines):
        s = lines[i]
        if s.startswith("!  Zero"):
            st.zero = _floats(lines[i + 1])[0]
        elif "Background coefficients" in s:
            vals = []
            for r in range(4):
                vals += _floats(lines[i + 1 + 2 * r])
            st.background = vals[:24]
        elif s.startswith("!-------> Profile Parameters"):
            ph_i += 1
        elif s.startswith("!  Scale") and ph_i >= 0:
            st.phases[ph_i].scale = _floats(lines[i + 1])[0]
        elif s.startswith("!       U") and ph_i >= 0:
            st.phases[ph_i].uvwxy = _floats(lines[i + 1])[:7]
        elif s.startswith("!     a") and ph_i >= 0:
            st.phases[ph_i].cell = [float(t) for t in lines[i + 1].split()[:6]]
        i += 1
    return st


# ── FullProf output ───────────────────────────────────────────────────────────

def parse_sum(path: Path) -> dict:
    """Global R-factors, per-phase cell (+esd) and scale, and the Bragg R /
    weight-fraction table. Weight fractions are relative to the Rietveld
    phases; a Le Bail phase always reports 0 (it has no structure factor)."""
    t = path.read_text(errors="ignore")
    out = {"phases": []}
    m = re.search(r"Rp:\s*([\d.]+)\s+Rwp:\s*([\d.]+)\s+Rexp:\s*([\d.]+)\s+Chi2:\s*([\d.Ee+-]+)", t)
    if m:
        out.update(Rp=float(m.group(1)), Rwp=float(m.group(2)), Rexp=float(m.group(3)),
                   chi2=float(m.group(4)))
    head, _, bragg = t.partition("BRAGG R-Factors and weight fractions")
    for blk in re.split(r"=> Phase No\.\s+", head)[1:]:
        ph = {"name": blk.split()[1] if len(blk.split()) > 1 else None}
        cm = re.search(r"=> Cell parameters\s*:\s*((?:\s*[-\d.]+\s+[-\d.]+\s*\n){6})", blk)
        if cm:
            rows = [r.split() for r in cm.group(1).strip().splitlines()]
            ph["cell"] = [float(r[0]) for r in rows]
            ph["cell_esd"] = [float(r[1]) for r in rows]
        sm = re.search(r"=> Overall scale factor\s*:\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)", blk, re.I)
        if sm:
            ph["scale"], ph["scale_esd"] = float(sm.group(1)), float(sm.group(2))
        out["phases"].append(ph)
    for n, rb, vol, vole, fr, fre in re.findall(
            r"=> Phase:\s*(\d+).*?\n\s*=> Bragg R-factor:\s*([\d.]+)\s+Vol:\s*([\d.]+)\(\s*([\d.]+)\)"
            r"\s+Fract\(%\):\s*([\d.]+)\(\s*([\d.]+)\)", bragg):
        k = int(n) - 1
        if k < len(out["phases"]):
            out["phases"][k].update(R_bragg=float(rb), volume=float(vol), volume_esd=float(vole),
                                    weight_pct=float(fr), weight_pct_esd=float(fre))
    return out


def fullprof_exe() -> str:
    home = os.environ.get("FULLPROF") or str(Path.home() / "FullProf_Suite")
    exe = Path(home) / "fp2k"
    if not exe.is_file():
        raise FileNotFoundError(f"FullProf fp2k not found at {exe}; set FULLPROF")
    return str(exe)


def run_fp2k(workdir: Path, code: str, datafile: str, timeout: int = 300) -> str:
    env = {**os.environ, "FULLPROF": str(Path(fullprof_exe()).parent)}
    p = subprocess.run([fullprof_exe(), code, datafile], cwd=workdir, env=env,
                       input="\n\n", capture_output=True, text=True, timeout=timeout)
    return p.stdout + p.stderr


# ── data ──────────────────────────────────────────────────────────────────────

def write_ins10(path: Path, x, y, s, header: str):
    lines = [f"! {header}"[:120]] + ["!"] * 5
    lines += [f"{a:12.5f} {b:14.4f} {c:12.4f}" for a, b, c in zip(x, y, s)]
    path.write_text("\n".join(lines) + "\n")


def load_xye(path: str):
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        if not line.strip() or line.lstrip()[0] in "#!":
            continue
        p = line.split()
        if len(p) >= 3:
            rows.append((float(p[0]), float(p[1]), float(p[2])))
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]


# ── driver ────────────────────────────────────────────────────────────────────

def initial_state(phases_ok: list) -> State:
    st = State()
    st.background[0] = 0.0
    for ph, v in phases_ok:
        c = v["cif"]
        st.phases.append(PhaseState(name=ph.name, method=ph.method, cif=c, formula=ph.formula,
                                    cell=list(c.cell)))
    return st


def refine(data: str, phases: list, inst: Instrument, workdir: str,
           steps=PAPER_STEPS, n_bkg: int = 12, state: State | None = None,
           code: str = "vidur") -> dict:
    """Stepwise Rietveld refinement of one pattern. Refuses to run if any phase
    fails validation. Groups freed at a step stay free in later steps."""
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    checks = [validate_phase(p) for p in phases]
    report = {"data": data, "validation": [{k: v for k, v in c.items() if k != "cif"} for c in checks],
              "steps": [], "model_changes": []}
    if not all(c["ok"] for c in checks):
        report["status"] = "rejected"
        return report
    x, y, s = load_xye(data)
    bad = s <= 0
    report["points"] = int(len(x))
    report["points_sigma_nonpositive"] = int(bad.sum())
    excluded = []
    if bad.any():
        # points FullProf cannot weight; excluded (listed), never altered or dropped
        for xv in x[bad]:
            excluded.append((xv - 1e-4, xv + 1e-4))
    datafile = f"{code}.dat"
    write_ins10(wd / datafile, x, y, s, f"VIDUR from {Path(data).name}")
    if inst.irf:
        shutil.copy(inst.irf, wd / Path(inst.irf).name)
    st = state or initial_state([(p, c) for p, c in zip(phases, checks)])
    if state is None:
        st.background[0] = float(np.median(y))
    free: set = set()
    held: set = set()
    dropped: set = set()
    prev_chi2 = None
    last_res = None
    for k, stp in enumerate(steps, 1):
        new_groups = set(stp) - free - dropped
        free |= new_groups
        before = copy.deepcopy(st)
        entry = {"step": k, "free": sorted(free)}
        outcome = None
        for relax in (1.0, DAMPED_RELAX):
            st_try = copy.deepcopy(before)
            info = write_pcr(wd / f"{code}.pcr", f"VIDUR {Path(data).stem} step {k}", datafile, x,
                             inst, st_try, free, n_bkg=n_bkg, excluded=excluded, held=held,
                             relax=relax)
            if k == 1:
                report["model_changes"] = info["model_changes"]
            log = run_fp2k(wd, code, datafile)
            sumf = wd / f"{code}.sum"
            if not sumf.is_file() or "END" not in log:
                entry["error"] = log[-800:]
                report["steps"].append(entry)
                report["status"] = "failed"
                return report
            res = parse_sum(sumf)
            chi2 = res.get("chi2")
            finite = chi2 is not None and math.isfinite(chi2)
            worse = finite and prev_chi2 is not None and chi2 > prev_chi2 * (1 + CHI2_RISE_TOL)
            entry.update(n_refined=info["n_refined"], relax=relax)
            if finite and not worse:
                read_state(wd / f"{code}.pcr", st_try)
                st = st_try
                outcome = "accepted" if relax == 1.0 else "accepted_after_damped_retry"
                break
            # Diverged (NaN) or chi2 rose: least squares did not improve the fit.
            # The PCR FullProf rewrote holds those values, so the state is not
            # read from it. Retry once damped from the same start.
            outcome = "diverged" if not finite else "chi2_increased"
            entry["first_attempt"] = {"outcome": outcome, "chi2": chi2}
        entry["outcome"] = outcome
        if outcome in ("diverged", "chi2_increased"):
            # Keep the previous step's model and do not free these groups again
            # for this pattern. Reported, never hidden.
            st = before
            free -= new_groups
            dropped |= new_groups
            entry["free"] = sorted(free)
            entry["groups_dropped"] = sorted(new_groups)
            entry.update({m: (last_res or {}).get(m) for m in ("Rp", "Rwp", "Rexp", "chi2")})
            entry["held_phases"] = sorted(st.phases[i - 1].name for i in held)
            report["steps"].append(entry)
            if last_res is None:
                report["status"] = "diverged"
                report["result"] = res
                return report
            continue
        entry["held_phases"] = sorted(st.phases[i - 1].name for i in held)
        entry.update({m: res.get(m) for m in ("Rp", "Rwp", "Rexp", "chi2")})
        report["steps"].append(entry)
        last_res, prev_chi2 = res, res["chi2"]
        # A Rietveld phase whose scale is not significant (< SIG_SCALE sigma) is
        # not seen in this pattern: its cell and broadening are undetermined and
        # refining them makes the normal matrix singular (measured: frame 1 of
        # the FullProfAPP LiFePO4 sequence went NaN the moment FePO4's strain was
        # freed while FePO4 was absent). Hold them; keep its scale free.
        # A minor phase is held too: below MIN_WT_PCT its reflections are too weak
        # to fix a cell or a breadth, and a free breadth lets it turn into a broad
        # hump that absorbs background (measured 09-27 v2: LiFePO4 at 0.2 wt% went
        # to TCH X = 9.0 and each FullProf run slowed from ~3 s to ~2 min).
        held = {i for i, p in enumerate(res.get("phases", []), 1)
                if i <= len(st.phases) and st.phases[i - 1].method == "rietveld"
                and ((p.get("scale_esd") and abs(p.get("scale", 0.0)) < SIG_SCALE * p["scale_esd"])
                     or (p.get("weight_pct") is not None and p["weight_pct"] < MIN_WT_PCT))}
    # The reported result (values AND esds) is the last ACCEPTED FullProf run.
    report["result"] = last_res
    report["groups_dropped"] = sorted(dropped)
    if dropped:
        # The files on disk belong to a rejected attempt: recompute the accepted
        # model (nothing refined) so the .prf/.pcr match it. Its .sum carries no
        # esds -- they are taken from the accepted run above (bug fixed 09-27:
        # the esds used to be read from this recomputation and came out 0).
        write_pcr(wd / f"{code}.pcr", f"VIDUR {Path(data).stem} final", datafile, x, inst, st,
                  set(), n_bkg=n_bkg, excluded=excluded, held=held)
        run_fp2k(wd, code, datafile)
        report["files_note"] = ("files recomputed from the accepted model with nothing refined; "
                                "esds are from the last accepted run")
    report["status"] = "ok"
    report["state"] = st
    return report


def summary(report: dict) -> dict:
    """JSON-safe view of a refine() report (the live State is dropped)."""
    out = {k: v for k, v in report.items() if k not in ("state",)}
    st = report.get("state")
    if st is not None:
        out["final_parameters"] = {
            "zero": st.zero,
            "phases": [{"name": p.name, "method": p.method, "scale": p.scale,
                        "cell": p.cell, "X_strain": p.uvwxy[3], "Y_size": p.uvwxy[4]}
                       for p in st.phases]}
    return out


def refine_sequence(datafiles: list, phases: list, inst: Instrument, workdir: str,
                    steps=PAPER_STEPS, n_bkg: int = 12) -> list:
    """Sequential refinement: pattern i starts from pattern i-1's result (the
    first from the CIFs). Each pattern runs the full step list."""
    import copy
    out, st = [], None
    for f in datafiles:
        start = copy.deepcopy(st)          # a failed frame must not corrupt the next start
        r = refine(f, phases, inst, str(Path(workdir) / Path(f).stem), steps, n_bkg, start)
        r["started_from"] = "previous frame" if st is not None else "CIF models"
        out.append(r)
        if r.get("status") == "ok":
            st = r["state"]
    return out
