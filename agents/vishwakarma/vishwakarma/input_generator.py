# vishwakarma/input_generator.py
#
# Generate Quantum ESPRESSO input files for:
#   pw.x   → scf | nscf | relax | vc-relax | bands | md
#   ph.x   → phonon (DFPT)
#   pp.x   → post-processing (charge density, potential, STM)
#   dos.x  → density of states
#   bands.x→ band structure reordering
#   neb.x  → nudged elastic band
#   cp.x   → Car-Parrinello MD
#   hp.x   → Hubbard U parameters
#
# All generators return a plain string (the .in file content).
# Structure is passed as a dict; see _STRUCTURE_SCHEMA below for keys.

import logging
from typing import Any

logger = logging.getLogger("vishwakarma.input_generator")

# ── Structure dict schema (reference) ────────────────────────────────────────
#
# structure = {
#     "prefix":          str,           # job label, e.g. "silicon"
#     "ibrav":           int,           # Bravais lattice index (0 = free cell)
#     "cell_parameters": list[list[float]],  # 3x3 matrix in Ångström (if ibrav=0)
#     "celldm":          list[float],   # celldm(1..6) in Bohr (if ibrav != 0)
#     "nat":             int,           # number of atoms
#     "ntyp":            int,           # number of species
#     "atomic_species":  [             # list, one per species
#         {"symbol": str, "mass": float, "pseudo": str}
#     ],
#     "atomic_positions": [            # list, one per atom
#         {"symbol": str, "x": float, "y": float, "z": float,
#          "fixed": [bool, bool, bool]}  # optional, for constrained relax
#     ],
#     "kpoints": {
#         "mode":   "automatic" | "gamma" | "crystal" | "tpiba_b",
#         "mesh":   [int, int, int],      # for automatic
#         "shift":  [int, int, int],      # for automatic
#         "nk":     int,                  # for explicit k-path
#         "points": [[kx, ky, kz, weight], ...],  # for explicit
#     },
# }
#
# calc_params = {
#     "ecutwfc":   float,   # plane-wave cutoff (Ry)
#     "ecutrho":   float,   # charge density cutoff (Ry), default 4*ecutwfc
#     "occupations": str,   # "smearing" | "fixed" | "tetrahedra"
#     "smearing":  str,     # "gaussian" | "methfessel-paxton" | "marzari-vanderbilt"
#     "degauss":   float,   # smearing width (Ry)
#     "nspin":     int,     # 1 | 2 | 4
#     "nstep":     int,     # max ionic steps
#     "conv_thr":  float,   # SCF convergence (Ry)
#     "forc_conv_thr": float,  # force convergence (a.u.)
#     "etot_conv_thr": float,  # energy convergence (Ry)
#     "mixing_beta": float,
#     "electron_maxstep": int,
#     "restart_mode": "from_scratch" | "restart",
#     "pseudo_dir": str,
#     "outdir":    str,
#     "disk_io":   str,     # "none" | "low" | "medium" | "high"
#     "verbosity": str,     # "low" | "high"
#     "dft_d3":    bool,    # DFT-D3 van der Waals correction
#     "hubbard_u": {symbol: U_value},   # DFT+U
#     "lda_plus_u": bool,
#     "input_dft": str,     # override XC functional
#     "nbnd":      int,     # number of bands
#     "tot_charge": float,
#     "tot_magnetization": float,
#     "starting_magnetization": {symbol: float},
# }


# ─── Public generators ────────────────────────────────────────────────────────

def scf(structure: dict, calc_params: dict) -> str:
    """Generate pw.x input for a self-consistent field calculation."""
    return _pw_input(structure, calc_params, calculation="scf")


def nscf(structure: dict, calc_params: dict) -> str:
    """Generate pw.x input for a non-self-consistent calculation (needed before DOS/bands)."""
    return _pw_input(structure, calc_params, calculation="nscf")


def relax(structure: dict, calc_params: dict, vc: bool = False) -> str:
    """Generate pw.x input for ionic (vc=False) or variable-cell (vc=True) relaxation."""
    calc = "vc-relax" if vc else "relax"
    return _pw_input(structure, calc_params, calculation=calc)


def bands(structure: dict, calc_params: dict) -> str:
    """Generate pw.x input for band structure (reads SCF charge density)."""
    p = dict(calc_params)
    p.setdefault("nosym", True)
    return _pw_input(structure, p, calculation="bands")


def bands_pp(prefix: str, outdir: str, filband: str = "bands.dat") -> str:
    """Generate bands.x post-processing input."""
    return (
        f"&BANDS\n"
        f"  prefix   = '{prefix}'\n"
        f"  outdir   = '{outdir}'\n"
        f"  filband  = '{filband}'\n"
        f"/\n"
    )


def dos(prefix: str, outdir: str,
        emin: float = -20.0, emax: float = 20.0,
        deltaE: float = 0.01, fildos: str = "dos.dat") -> str:
    """Generate dos.x input."""
    return (
        f"&DOS\n"
        f"  prefix  = '{prefix}'\n"
        f"  outdir  = '{outdir}'\n"
        f"  Emin    = {emin}\n"
        f"  Emax    = {emax}\n"
        f"  DeltaE  = {deltaE}\n"
        f"  fildos  = '{fildos}'\n"
        f"/\n"
    )


def projwfc(prefix: str, outdir: str,
            emin: float = -20.0, emax: float = 20.0,
            deltaE: float = 0.01, filpdos: str = "pdos") -> str:
    """Generate projwfc.x input for projected / partial DOS."""
    return (
        f"&PROJWFC\n"
        f"  prefix   = '{prefix}'\n"
        f"  outdir   = '{outdir}'\n"
        f"  Emin     = {emin}\n"
        f"  Emax     = {emax}\n"
        f"  DeltaE   = {deltaE}\n"
        f"  filpdos  = '{filpdos}'\n"
        f"/\n"
    )


def pp(prefix: str, outdir: str,
       plot_num: int = 0,
       fileout: str = "charge.xsf",
       iflag: int = 3,
       output_format: int = 5) -> str:
    """
    Generate pp.x post-processing input.

    plot_num codes (common):
      0  — charge density
      1  — total potential
      2  — local ionic potential
      3  — LDOS
      6  — STM
      7  — |psi|²
      10 — ILDOS
      13 — ELF
    """
    return (
        f"&INPUTPP\n"
        f"  prefix      = '{prefix}'\n"
        f"  outdir      = '{outdir}'\n"
        f"  plot_num    = {plot_num}\n"
        f"  filplot     = 'tmp_plot'\n"
        f"/\n"
        f"&PLOT\n"
        f"  iflag        = {iflag}\n"
        f"  output_format= {output_format}\n"
        f"  fileout      = '{fileout}'\n"
        f"/\n"
    )


def phonon(prefix: str, outdir: str,
           tr2_ph: float = 1e-14,
           qpoints: list | None = None,
           ldisp: bool = False,
           nq: tuple = (4, 4, 4),
           fildyn: str = "dyn",
           epsil: bool = False,
           lraman: bool = False,
           recover: bool = False,
           qplot: bool | None = None) -> str:
    """
    Generate ph.x input for DFPT phonon calculation.

    qpoints: explicit list of [[qx,qy,qz], ...] — used when ldisp=False.
             A single q-point ([[qx,qy,qz]] or None, which defaults to
             Gamma) is written per QE's plain single-q-point card: one
             bare "xq1 xq2 xq3" line, no count, no weight.
             Multiple q-points require qplot=.true. in the namelist and a
             count line + "xq1 xq2 xq3 weight" per point — this is a
             DIFFERENT card format, only valid when qplot is actually set.
    ldisp:   Monkhorst-Pack q-grid (nq1 x nq2 x nq3) — no q-point card at
             all after the namelist; QE generates the grid itself.
    qplot:   force the qplot-style (count + weighted-list) card even for
             a single q-point, or force it off for multiple q-points (not
             recommended). Defaults to None, which auto-selects: False for
             a single q-point, True for more than one.
    epsil:   also compute dielectric tensor + Born charges
    lraman:  compute Raman tensors (requires epsil=True)
    recover: resume from a previously interrupted run using the .recover
             file ph.x writes under outdir/_ph0/. Requires the SAME outdir
             as the original (killed) run so ph.x finds its own scratch data.
    """
    if not ldisp:
        if qpoints is None:
            qpoints = [[0.0, 0.0, 0.0]]
        if qplot is None:
            qplot = len(qpoints) > 1
        if qplot and len(qpoints) < 2:
            logger.warning(
                "phonon(): qplot=True requested with only %d q-point(s); "
                "QE's qplot card format still applies but a single bare "
                "q-point line (qplot=False) is simpler and less error-prone.",
                len(qpoints),
            )
    else:
        qplot = False  # ldisp grids never use an explicit q-point card

    lines = [
        "Phonon calculation",
        "&INPUTPH",
        f"  prefix    = '{prefix}'",
        f"  outdir    = '{outdir}'",
        f"  tr2_ph    = {tr2_ph:.2e}",
        f"  fildyn    = '{fildyn}'",
        f"  epsil     = .{'true' if epsil else 'false'}.",
        f"  lraman    = .{'true' if lraman else 'false'}.",
        f"  recover   = .{'true' if recover else 'false'}.",
    ]
    if ldisp:
        nq1, nq2, nq3 = nq
        lines += [
            f"  ldisp     = .true.",
            f"  nq1 = {nq1}, nq2 = {nq2}, nq3 = {nq3}",
        ]
    elif qplot:
        lines.append(f"  qplot     = .true.")
    lines.append("/")

    if not ldisp:
        if qplot:
            # qplot card: count line + "xq1 xq2 xq3 weight" per point.
            lines.append(str(len(qpoints)))
            for q in qpoints:
                lines.append(f"  {q[0]:.6f}  {q[1]:.6f}  {q[2]:.6f}  1.0")
        else:
            # Plain single-q-point card: one bare line, no count, no weight.
            q = qpoints[0]
            lines.append(f"  {q[0]:.6f}  {q[1]:.6f}  {q[2]:.6f}")

    return "\n".join(lines) + "\n"


def neb(images: list[dict], calc_params: dict,
        num_of_images: int = 7,
        ci_scheme: str = "no-CI",
        opt_scheme: str = "broyden",
        nstep_path: int = 200,
        path_thr: float = 0.05) -> str:
    """
    Generate neb.x input for minimum energy path / transition state search.

    images: list of structure dicts — first = reactant, last = product.
    ci_scheme: "no-CI" | "auto" | "manual"
    opt_scheme: "broyden" | "sd" | "lbfgs"
    """
    lines = [
        "BEGIN",
        "BEGIN_PATH_INPUT",
        "&PATH",
        f"  restart_mode   = 'from_scratch'",
        f"  string_method  = 'neb'",
        f"  nstep_path     = {nstep_path}",
        f"  num_of_images  = {num_of_images}",
        f"  opt_scheme     = '{opt_scheme}'",
        f"  ci_scheme      = '{ci_scheme}'",
        f"  path_thr       = {path_thr}",
        "/",
        "END_PATH_INPUT",
    ]

    for i, img in enumerate(images):
        label = "FIRST_IMAGE" if i == 0 else ("LAST_IMAGE" if i == len(images) - 1 else "INTERMEDIATE_IMAGE")
        lines.append(f"BEGIN_{label}")
        lines.append(_pw_input(img, calc_params, calculation="relax"))
        lines.append(f"END_{label}")

    lines.append("END")
    return "\n".join(lines) + "\n"


def hp(prefix: str, outdir: str,
       nq: tuple = (2, 2, 2),
       conv_thr_chi: float = 1e-5,
       iverbosity: int = 1) -> str:
    """Generate hp.x input for computing Hubbard U parameters from linear response."""
    nq1, nq2, nq3 = nq
    return (
        f"&INPUT\n"
        f"  prefix        = '{prefix}'\n"
        f"  outdir        = '{outdir}'\n"
        f"  iverbosity    = {iverbosity}\n"
        f"  conv_thr_chi  = {conv_thr_chi:.2e}\n"
        f"  nq1 = {nq1}, nq2 = {nq2}, nq3 = {nq3}\n"
        f"/\n"
    )


def cp(structure: dict, calc_params: dict,
       dt: float = 5.0,
       nstep: int = 1000,
       ion_dynamics: str = "verlet",
       electron_dynamics: str = "damp") -> str:
    """Generate cp.x (Car-Parrinello MD) input."""
    p = dict(calc_params)
    p["calculation"] = "cp"
    p["dt"]          = dt
    p["nstep"]       = nstep
    p["ion_dynamics"] = ion_dynamics
    p["electron_dynamics"] = electron_dynamics
    return _pw_input(structure, p, calculation="cp")


# ─── Core pw.x builder ───────────────────────────────────────────────────────

def _pw_input(structure: dict, calc_params: dict, calculation: str) -> str:
    """
    Assemble a complete pw.x input file from structure + params dicts.
    """
    p   = calc_params
    s   = structure
    nat  = s.get("nat",  len(s.get("atomic_positions", [])))
    ntyp = s.get("ntyp", len(s.get("atomic_species", [])))

    lines = []

    # ── &CONTROL ─────────────────────────────────────────────────────────────
    lines.append("&CONTROL")
    lines.append(f"  calculation   = '{calculation}'")
    lines.append(f"  prefix        = '{s.get('prefix', 'pwscf')}'")
    lines.append(f"  outdir        = '{p.get('outdir', './out')}'")
    lines.append(f"  pseudo_dir    = '{p.get('pseudo_dir', './pseudo')}'")
    lines.append(f"  verbosity     = '{p.get('verbosity', 'low')}'")
    lines.append(f"  restart_mode  = '{p.get('restart_mode', 'from_scratch')}'")
    if p.get("disk_io"):
        lines.append(f"  disk_io       = '{p['disk_io']}'")
    if calculation in ("relax", "vc-relax", "md", "cp"):
        lines.append(f"  nstep         = {p.get('nstep', 200)}")
        if p.get("forc_conv_thr"):
            lines.append(f"  forc_conv_thr = {p['forc_conv_thr']:.2e}")
        if p.get("etot_conv_thr"):
            lines.append(f"  etot_conv_thr = {p['etot_conv_thr']:.2e}")
    lines.append("/\n")

    # ── &SYSTEM ──────────────────────────────────────────────────────────────
    lines.append("&SYSTEM")
    lines.append(f"  ibrav         = {s.get('ibrav', 0)}")
    lines.append(f"  nat           = {nat}")
    lines.append(f"  ntyp          = {ntyp}")
    lines.append(f"  ecutwfc       = {p.get('ecutwfc', 60.0)}")
    lines.append(f"  ecutrho       = {p.get('ecutrho', p.get('ecutwfc', 60.0) * 8)}")

    occ = p.get("occupations", "smearing")
    lines.append(f"  occupations   = '{occ}'")
    if occ == "smearing":
        lines.append(f"  smearing      = '{p.get('smearing', 'gaussian')}'")
        lines.append(f"  degauss       = {p.get('degauss', 0.02)}")

    if p.get("nspin", 1) != 1:
        lines.append(f"  nspin         = {p['nspin']}")

    sm = p.get("starting_magnetization", {})
    for sym, mag in sm.items():
        lines.append(f"  starting_magnetization({_species_index(s, sym)}) = {mag}")

    if p.get("tot_charge"):
        lines.append(f"  tot_charge    = {p['tot_charge']}")
    if p.get("tot_magnetization") is not None:
        lines.append(f"  tot_magnetization = {p['tot_magnetization']}")
    if p.get("nbnd"):
        lines.append(f"  nbnd          = {p['nbnd']}")
    if p.get("input_dft"):
        lines.append(f"  input_dft     = '{p['input_dft']}'")
    if p.get("lda_plus_u") or p.get("hubbard_u"):
        lines.append(f"  lda_plus_u    = .true.")
        for sym, u in p.get("hubbard_u", {}).items():
            lines.append(f"  Hubbard_U({_species_index(s, sym)}) = {u}")
    if p.get("nosym"):
        lines.append(f"  nosym         = .true.")
    if p.get("dft_d3"):
        lines.append(f"  vdw_corr      = 'DFT-D3'")
    if s.get("ibrav", 0) != 0:
        # ibrav=0 uses CELL_PARAMETERS {angstrom} — celldm must NOT be set
        for i, v in enumerate(s.get("celldm", []), start=1):
            lines.append(f"  celldm({i})     = {v}")
    lines.append("/\n")

    # ── &ELECTRONS ───────────────────────────────────────────────────────────
    lines.append("&ELECTRONS")
    lines.append(f"  conv_thr        = {p.get('conv_thr', 1e-8):.2e}")
    lines.append(f"  mixing_beta     = {p.get('mixing_beta', 0.7)}")
    lines.append(f"  electron_maxstep= {p.get('electron_maxstep', 200)}")
    if p.get("startingwfc"):
        lines.append(f"  startingwfc     = '{p['startingwfc']}'")
    if p.get("diagonalization"):
        lines.append(f"  diagonalization = '{p['diagonalization']}'")
    lines.append("/\n")

    # ── &IONS (relax / md) ───────────────────────────────────────────────────
    if calculation in ("relax", "vc-relax", "md", "cp"):
        lines.append("&IONS")
        lines.append(f"  ion_dynamics  = '{p.get('ion_dynamics', 'bfgs')}'")
        lines.append("/\n")

    # ── &CELL (vc-relax) ─────────────────────────────────────────────────────
    if calculation == "vc-relax":
        lines.append("&CELL")
        lines.append(f"  cell_dynamics = '{p.get('cell_dynamics', 'bfgs')}'")
        if p.get("press"):
            lines.append(f"  press         = {p['press']}")
        if p.get("cell_dofree"):
            lines.append(f"  cell_dofree   = '{p['cell_dofree']}'")
        lines.append("/\n")

    # ── ATOMIC_SPECIES ────────────────────────────────────────────────────────
    lines.append("ATOMIC_SPECIES")
    for sp in s.get("atomic_species", []):
        lines.append(f"  {sp['symbol']:<4}  {sp['mass']:.6f}  {sp['pseudo']}")
    lines.append("")

    # ── ATOMIC_POSITIONS ─────────────────────────────────────────────────────
    pos_units = s.get("position_units", "crystal")
    lines.append(f"ATOMIC_POSITIONS {{{pos_units}}}")
    for atom in s.get("atomic_positions", []):
        fx, fy, fz = "", "", ""
        if "fixed" in atom:
            fx = 0 if atom["fixed"][0] else 1
            fy = 0 if atom["fixed"][1] else 1
            fz = 0 if atom["fixed"][2] else 1
            constraint = f"  {fx} {fy} {fz}"
        else:
            constraint = ""
        lines.append(
            f"  {atom['symbol']:<4}  "
            f"{atom['x']:>14.9f}  {atom['y']:>14.9f}  {atom['z']:>14.9f}"
            f"{constraint}"
        )
    lines.append("")

    # ── K_POINTS ─────────────────────────────────────────────────────────────
    kp = s.get("kpoints", {"mode": "automatic", "mesh": [4, 4, 4], "shift": [0, 0, 0]})
    mode = kp.get("mode", "automatic")
    lines.append(f"K_POINTS {{{mode}}}")
    if mode == "automatic":
        m = kp.get("mesh", [4, 4, 4])
        sh = kp.get("shift", [0, 0, 0])
        lines.append(f"  {m[0]} {m[1]} {m[2]}  {sh[0]} {sh[1]} {sh[2]}")
    elif mode == "gamma":
        pass   # no extra lines
    elif mode in ("crystal", "tpiba", "crystal_b", "tpiba_b"):
        pts = kp.get("points", [])
        lines.append(f"  {len(pts)}")
        for pt in pts:
            lines.append(f"  {pt[0]:.6f}  {pt[1]:.6f}  {pt[2]:.6f}  {pt[3]:.4f}")
    lines.append("")

    # ── CELL_PARAMETERS (ibrav=0 only) ────────────────────────────────────────
    if s.get("ibrav", 0) == 0 and s.get("cell_parameters"):
        lines.append("CELL_PARAMETERS {angstrom}")
        for vec in s["cell_parameters"]:
            lines.append(f"  {vec[0]:>14.9f}  {vec[1]:>14.9f}  {vec[2]:>14.9f}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _species_index(structure: dict, symbol: str) -> int:
    """Return 1-based index of species in atomic_species list."""
    for i, sp in enumerate(structure.get("atomic_species", []), start=1):
        if sp["symbol"] == symbol:
            return i
    return 1
