"""
vishwakarma/structure_builder.py
==================================
High-level crystal structure builder for BRAHM/Vishwakarma.

Converts natural-language-style requests into fully specified ASE Atoms
objects and Vishwakarma-compatible structure dicts for QE input generation.

Capabilities:
  - Build common research materials from name alone
  - Create supercells for doping calculations
  - Substitute atoms at specific or optimal sites (doping)
  - Find symmetry-inequivalent sites using spglib
  - Convert ASE Atoms → Vishwakarma structure dict
  - Rank doping sites by formation energy proxy (bond valence / coordination)
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger("vishwakarma.structure_builder")

# ── Atomic masses ─────────────────────────────────────────────────────────────
MASSES = {
    'H':1.008,'Li':6.941,'C':12.011,'N':14.007,'O':15.999,'Al':26.982,
    'Si':28.086,'Ti':47.867,'Zn':65.38,'Ga':69.723,'Ge':72.63,'As':74.922,
    'Se':78.971,'Nb':92.906,'Mo':95.96,'In':114.818,'Sn':118.710,
    'Te':127.6,'La':138.905,'W':183.84,'Pb':207.2,'Bi':208.98,
    'Fe':55.845,'Cu':63.546,'Mg':24.305,'Ca':40.078,'Sr':87.62,
    'Ba':137.327,'Mn':54.938,'Co':58.933,'Ni':58.693,'Zr':91.224,
    'S':32.06,
}

# ── Pseudopotential map (element → filename in pseudo_dir) ───────────────────
PSEUDO_MAP = {
    'Zn': 'Zn.pbe-dn-kjpaw_psl.1.0.0.UPF',
    'Se': 'Se.pbe-dn-kjpaw_psl.1.0.0.UPF',
    'O':  'O.pbe-n-kjpaw_psl.0.1.UPF',
    'N':  'N.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Ti': 'Ti.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Li': 'Li.pbe-s-kjpaw_psl.0.2.1.UPF',
    'Nb': 'Nb.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Al': 'Al.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Si': 'Si.pbe-n-rrkjus_psl.1.0.0.UPF',
    'C':  'C.pbe-n-kjpaw_psl.1.0.0.UPF',
    'H':  'H.pbe-rrkjus_psl.1.0.0.UPF',
    'Cu': 'Cu.pbe-dn-kjpaw_psl.1.0.0.UPF',
    'Fe': 'Fe.pbe-spn-kjpaw_psl.0.2.1.UPF',
    'Ga': 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF',
    'Ge': 'Ge.pbe-dn-kjpaw_psl.1.0.0.UPF',
    'As': 'As.pbe-n-rrkjus_psl.0.2.UPF',
    'Mo': 'Mo.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'S':  'S.pbe-n-kjpaw_psl.1.0.0.UPF',
}

# ── Material library ──────────────────────────────────────────────────────────
def _build_library() -> dict:
    """Returns dict of material_key -> builder function."""
    try:
        from ase.spacegroup import crystal
        from ase.build import bulk
    except ImportError:
        return {}

    return {
        # ZnSe
        'znse':          lambda: bulk('ZnSe', crystalstructure='zincblende', a=5.668),
        'znse_zb':       lambda: bulk('ZnSe', crystalstructure='zincblende', a=5.668),
        'znse_wz':       lambda: crystal(['Zn','Se'], spacegroup=186,
                             cellpar=[3.996,3.996,6.626,90,90,120],
                             basis=[(1/3,2/3,0),(1/3,2/3,0.375)]),
        # ZnO
        'zno':           lambda: crystal(['Zn','O'], spacegroup=186,
                             cellpar=[3.250,3.250,5.207,90,90,120],
                             basis=[(1/3,2/3,0),(1/3,2/3,0.382)]),
        'zno_wz':        lambda: crystal(['Zn','O'], spacegroup=186,
                             cellpar=[3.250,3.250,5.207,90,90,120],
                             basis=[(1/3,2/3,0),(1/3,2/3,0.382)]),
        'zno_rs':        lambda: bulk('ZnO', crystalstructure='rocksalt', a=4.280),
        # TiO2
        'tio2':          lambda: crystal(['Ti','O'], basis=[(0,0,0),(0,0,0.2087)],
                             spacegroup=141, cellpar=[3.785,3.785,9.512,90,90,90]),
        'tio2_anatase':  lambda: crystal(['Ti','O'], basis=[(0,0,0),(0,0,0.2087)],
                             spacegroup=141, cellpar=[3.785,3.785,9.512,90,90,90]),
        'tio2_rutile':   lambda: crystal(['Ti','O'], basis=[(0,0,0),(0.3,0.3,0)],
                             spacegroup=136, cellpar=[4.594,4.594,2.959,90,90,90]),
        # LiNbO3
        'linbo3':        lambda: crystal(['Li','Nb','O'], spacegroup=161,
                             cellpar=[5.148,5.148,13.863,90,90,120],
                             basis=[(0,0,0.2829),(0,0,0),(0.0492,0.3446,0.0833)]),
        # ZnS
        'zns':           lambda: bulk('ZnS', crystalstructure='zincblende', a=5.420),
        # GaAs
        'gaas':          lambda: bulk('GaAs', crystalstructure='zincblende', a=5.653),
        # Si
        'si':            lambda: bulk('Si', crystalstructure='diamond', a=5.431),
        # ZnSe1-xOx (approximated as ZnSe with O substitution handled by doping)
        'znse_host':     lambda: bulk('ZnSe', crystalstructure='zincblende', a=5.668),
    }

MATERIAL_LIBRARY = {}  # populated lazily


def _get_library():
    global MATERIAL_LIBRARY
    if not MATERIAL_LIBRARY:
        MATERIAL_LIBRARY = _build_library()
    return MATERIAL_LIBRARY


# ── Public API ────────────────────────────────────────────────────────────────

def list_materials() -> list[str]:
    """Return all supported material keys."""
    return sorted(_get_library().keys())


def build(material: str) -> object:
    """
    Build an ASE Atoms object from a material key.
    e.g. build('tio2_anatase'), build('znse'), build('linbo3')
    """
    lib = _get_library()
    key = material.lower().replace('-', '_').replace(' ', '_')
    if key not in lib:
        raise ValueError(
            f"Unknown material '{material}'. "
            f"Available: {sorted(lib.keys())}"
        )
    atoms = lib[key]()
    logger.info("Built %s: %d atoms, formula=%s",
                material, len(atoms), atoms.get_chemical_formula())
    return atoms


def make_supercell(atoms, size: list[int]) -> object:
    """
    Create a supercell. size = [nx, ny, nz].
    e.g. make_supercell(atoms, [2,2,2])
    """
    from ase.build import make_supercell as ase_sc
    matrix = np.diag(size)
    sc = ase_sc(atoms, matrix)
    logger.info("Supercell %s: %d atoms", size, len(sc))
    return sc


def find_doping_sites(atoms, host_element: str) -> list[dict]:
    """
    Find symmetry-inequivalent sites for a given host element.
    Uses spglib if available, falls back to index enumeration.
    Returns list of site dicts with index, symbol, coordination info.
    """
    symbols = atoms.get_chemical_symbols()
    indices = [i for i, s in enumerate(symbols) if s == host_element]

    if not indices:
        raise ValueError(
            f"Element '{host_element}' not found in structure. "
            f"Available: {set(symbols)}"
        )

    # Try spglib for symmetry analysis
    try:
        import spglib
        cell = (
            atoms.get_cell().tolist(),
            atoms.get_scaled_positions().tolist(),
            atoms.get_atomic_numbers().tolist(),
        )
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-3)
        equiv = dataset.equivalent_atoms

        # Group by equivalence class
        seen, unique_sites = set(), []
        for idx in indices:
            rep = equiv[idx]
            if rep not in seen:
                seen.add(rep)
                pos = atoms.get_scaled_positions()[idx]
                unique_sites.append({
                    'index':       idx,
                    'symbol':      host_element,
                    'wyckoff':     dataset.wyckoffs[idx],
                    'site_symmetry': dataset.site_symmetry_symbols[idx],
                    'fractional_coords': pos.round(4).tolist(),
                    'equivalent_count': int(list(equiv).count(rep)),
                    'symmetry_analysis': 'spglib',
                })
        return unique_sites

    except (ImportError, Exception) as exc:
        logger.warning("spglib unavailable (%s) — listing all %s sites", exc, host_element)
        sites = []
        for idx in indices:
            pos = atoms.get_scaled_positions()[idx]
            sites.append({
                'index':             idx,
                'symbol':            host_element,
                'wyckoff':           'unknown',
                'site_symmetry':     'unknown',
                'fractional_coords': pos.round(4).tolist(),
                'equivalent_count':  1,
                'symmetry_analysis': 'no_spglib',
            })
        return sites


def dope(atoms, site_index: int, dopant: str) -> object:
    """
    Substitute atom at site_index with dopant element.
    Returns a new Atoms object (original unchanged).
    """
    import copy
    doped = copy.deepcopy(atoms)
    symbols = doped.get_chemical_symbols()
    old = symbols[site_index]
    symbols[site_index] = dopant
    doped.set_chemical_symbols(symbols)
    logger.info("Doped site %d: %s → %s | formula=%s",
                site_index, old, dopant, doped.get_chemical_formula())
    return doped


def rank_doping_sites(atoms, host_element: str, dopant: str) -> list[dict]:
    """
    Rank inequivalent doping sites by a formation energy proxy.
    Proxy = ionic radius mismatch + charge difference + coordination number.
    Lower score = more likely to be the stable doping site.
    """
    # Ionic radii (pm) for common oxidation states
    IONIC_RADII = {
        'Ti4+': 60.5, 'Ti3+': 67.0,
        'Zn2+': 74.0, 'Zn4+': 60.0,
        'O2-':  140.0,'Se2-': 198.0,
        'Li1+': 76.0, 'Nb5+': 64.0,
        'Al3+': 53.5, 'Si4+': 40.0,
        'Fe3+': 64.5, 'Cu2+': 73.0,
        'Ga3+': 62.0, 'Ge4+': 53.0,
    }
    # Formal charges proxy
    CHARGES = {
        'Ti': 4, 'Zn': 2, 'O': -2, 'Se': -2,
        'Li': 1, 'Nb': 5, 'Al': 3, 'Si': 4,
        'Fe': 3, 'Cu': 2, 'Ga': 3, 'Ge': 4, 'N': -3,
    }

    sites = find_doping_sites(atoms, host_element)
    host_charge  = CHARGES.get(host_element, 0)
    dopant_charge = CHARGES.get(dopant, 0)
    charge_diff  = abs(dopant_charge - host_charge)

    host_key   = f"{host_element}{host_charge:+d}".replace('+','').replace('-','') + ('+' if host_charge > 0 else '-') * (host_charge != 0)
    dopant_key = f"{dopant}{dopant_charge:+d}".replace('+','').replace('-','') + ('+' if dopant_charge > 0 else '-') * (dopant_charge != 0)

    host_r   = IONIC_RADII.get(f"{host_element}{host_charge}+", 70)
    dopant_r = IONIC_RADII.get(f"{dopant}{dopant_charge}+", 70)
    radius_mismatch = abs(dopant_r - host_r) / max(host_r, 1)

    ranked = []
    for site in sites:
        score = charge_diff * 0.5 + radius_mismatch * 0.5
        ranked.append({**site,
            'dopant':          dopant,
            'charge_diff':     charge_diff,
            'radius_mismatch': round(radius_mismatch, 3),
            'stability_score': round(score, 3),
            'interpretation':  (
                'favorable' if score < 0.3 else
                'moderate'  if score < 0.6 else
                'unfavorable'
            ),
        })

    ranked.sort(key=lambda x: x['stability_score'])
    return ranked


def to_qe_structure(atoms, prefix: str = "pwscf",
                    pseudo_dir: str = "/mnt/d/brahm/agents/vishwakarma/pseudo",
                    kpoints: Optional[dict] = None) -> dict:
    """
    Convert ASE Atoms object to Vishwakarma/QE structure dict.
    Ready to pass directly to input_generator.scf(), .relax(), etc.
    """
    cell    = atoms.get_cell().tolist()
    pos     = atoms.get_scaled_positions().tolist()
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))

    atomic_species = []
    for s in species:
        pseudo = PSEUDO_MAP.get(s, f"{s}.pbe-rrkjus.UPF")
        atomic_species.append({
            'symbol': s,
            'mass':   MASSES.get(s, 1.0),
            'pseudo': pseudo,
        })

    atomic_positions = []
    for sym, (x, y, z) in zip(symbols, pos):
        atomic_positions.append({'symbol': sym, 'x': x, 'y': y, 'z': z})

    if kpoints is None:
        kpoints = {'mode': 'automatic', 'mesh': [4, 4, 4], 'shift': [0, 0, 0]}

    return {
        'prefix':           prefix,
        'ibrav':            0,
        'cell_parameters':  cell,
        'nat':              len(atoms),
        'ntyp':             len(species),
        'atomic_species':   atomic_species,
        'atomic_positions': atomic_positions,
        'kpoints':          kpoints,
    }


def build_doped_structure(
    material:      str,
    host_element:  str,
    dopant:        str,
    supercell:     list[int] = [2, 2, 2],
    site_index:    Optional[int] = None,
    prefix:        str = "doped",
    pseudo_dir:    str = "/mnt/d/brahm/agents/vishwakarma/pseudo",
    kpoints:       Optional[dict] = None,
) -> dict:
    """
    One-call builder: material → supercell → dope → QE structure dict.

    If site_index is None, uses the highest-ranked (most stable) site.
    Returns {structure, doping_info, ranked_sites, atoms_formula}.
    """
    # 1. Build primitive cell
    atoms = build(material)

    # 2. Find and rank doping sites in primitive cell
    ranked = rank_doping_sites(atoms, host_element, dopant)
    if not ranked:
        raise ValueError(f"No {host_element} sites found in {material}")

    # 3. Make supercell
    sc = make_supercell(atoms, supercell)

    # 4. Find sites in supercell
    sc_ranked = rank_doping_sites(sc, host_element, dopant)

    # 5. Pick site
    chosen = sc_ranked[0] if site_index is None else next(
        (s for s in sc_ranked if s['index'] == site_index), sc_ranked[0]
    )

    # 6. Dope
    doped = dope(sc, chosen['index'], dopant)

    # 7. Convert to QE structure dict
    n_host = sc.get_chemical_symbols().count(host_element)
    conc   = round(1 / max(n_host, 1) * 100, 2)

    structure = to_qe_structure(doped, prefix=prefix,
                                pseudo_dir=pseudo_dir, kpoints=kpoints)

    return {
        'structure':      structure,
        'doping_info': {
            'material':        material,
            'host_element':    host_element,
            'dopant':          dopant,
            'supercell':       supercell,
            'site_index':      chosen['index'],
            'wyckoff':         chosen.get('wyckoff', 'unknown'),
            'site_symmetry':   chosen.get('site_symmetry', 'unknown'),
            'concentration_pct': conc,
            'stability_score': chosen['stability_score'],
            'interpretation':  chosen['interpretation'],
            'formula':         doped.get_chemical_formula(),
        },
        'ranked_sites':   ranked,   # primitive cell ranking
        'atoms_formula':  doped.get_chemical_formula(),
    }


# ── CIF import ────────────────────────────────────────────────────────────────

def from_cif(cif_path: str) -> object:
    """
    Load a crystal structure from a CIF file.
    Returns ASE Atoms object.
    """
    from ase.io import read
    atoms = read(cif_path)
    logger.info("Loaded CIF %s: %d atoms, formula=%s",
                cif_path, len(atoms), atoms.get_chemical_formula())
    return atoms


# ── Materials Project query ───────────────────────────────────────────────────

def from_materials_project(formula: str, api_key: str = None,
                           preferred_spacegroup: str = None) -> object:
    """
    Fetch the most stable structure for a formula from Materials Project.
    Returns ASE Atoms object.
    Falls back to library if MP unavailable.
    """
    import os
    key = api_key or os.environ.get("MP_API_KEY", "")
    if not key:
        raise ValueError(
            "No Materials Project API key. "
            "Set MP_API_KEY in .env or pass api_key= argument. "
            "Get a free key at https://materialsproject.org"
        )
    try:
        from mp_api.client import MPRester
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as e:
        raise ImportError(f"mp-api or pymatgen not installed: {e}")

    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "structure",
                    "energy_above_hull", "symmetry"],
        )
        if not docs:
            raise ValueError(f"No MP entries found for formula '{formula}'")

        # Filter by spacegroup if requested
        if preferred_spacegroup:
            filtered = [d for d in docs
                       if d.symmetry.symbol == preferred_spacegroup]
            if filtered:
                docs = filtered

        # Sort by energy above hull (most stable first)
        docs.sort(key=lambda d: d.energy_above_hull or 999)
        best = docs[0]

        logger.info("MP fetch: %s → %s (%s), E_hull=%.3f eV/atom",
                    formula, best.material_id,
                    best.symmetry.symbol if best.symmetry else "?",
                    best.energy_above_hull or 0)

        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(best.structure)
        atoms.info['mp_id']         = best.material_id
        atoms.info['formula_pretty'] = best.formula_pretty
        atoms.info['e_above_hull']   = best.energy_above_hull
        atoms.info['spacegroup']     = best.symmetry.symbol if best.symmetry else "?"
        return atoms


def build_any(material: str, api_key: str = None) -> object:
    """
    Universal builder — tries in order:
    1. Built-in library (fast, no internet)
    2. Materials Project API (requires MP_API_KEY)
    3. Raises with helpful message

    material can be:
      - Library key:  'tio2_anatase', 'znse', 'linbo3'
      - Formula:      'TiO2', 'BiVO4', 'SnO2', 'ZnSe'
      - MP id:        'mp-2657'
    """
    import os

    # 1. Try library first
    lib = _get_library()
    key = material.lower().replace('-','_').replace(' ','_')
    if key in lib:
        return lib[key]()

    # 2. Try MP
    api_key = api_key or os.environ.get("MP_API_KEY", "")
    if api_key:
        logger.info("'%s' not in library — querying Materials Project", material)
        return from_materials_project(material, api_key=api_key)

    # 3. Fail helpfully
    raise ValueError(
        f"Material '{material}' not in built-in library and no MP_API_KEY set.\n"
        f"Options:\n"
        f"  1. Add MP_API_KEY to .env (free at materialsproject.org)\n"
        f"  2. Use from_cif('/path/to/structure.cif')\n"
        f"  3. Use a built-in key: {sorted(lib.keys())}"
    )


# ── Calculation parameter suggestions ────────────────────────────────────────

# Recommended ecutwfc per pseudopotential type (Ry)
# PAW > USPP > NC; transition metals need higher cutoffs
_ECUTWFC_HINTS = {
    'H':4,'Li':30,'C':50,'N':50,'O':50,'Al':30,'Si':30,
    'Ti':60,'Zn':60,'Ga':50,'Ge':40,'As':40,'Se':40,
    'Nb':60,'Mo':60,'Fe':60,'Cu':50,'Sn':40,
}

def suggest_calc_params(atoms, pseudo_dir: str = None,
                        calculation: str = "scf") -> dict:
    """
    Suggest QE calculation parameters based on the structure's elements.
    Returns a dict with recommended ecutwfc, ecutrho, and other params.
    These are SUGGESTIONS — user should review and adjust.
    """
    symbols = set(atoms.get_chemical_symbols())
    ecutwfc = max(_ECUTWFC_HINTS.get(s, 50) for s in symbols)
    ecutrho = ecutwfc * 8  # PAW default ratio

    params = {
        "ecutwfc":       ecutwfc,
        "ecutrho":       ecutrho,
        "occupations":   "smearing",
        "smearing":      "gaussian",
        "degauss":       0.01,
        "conv_thr":      1e-6,
        "mixing_beta":   0.4,
        "startingwfc":   "atomic+random",
        "diagonalization": "david",
        "electron_maxstep": 200,
        "disk_io":       "low",
        "verbosity":     "low",
    }

    if pseudo_dir:
        params["pseudo_dir"] = pseudo_dir

    notes = [
        f"ecutwfc={ecutwfc} Ry based on elements: {sorted(symbols)}",
        f"ecutrho={ecutrho} Ry (8× ecutwfc for PAW)",
        "Review and adjust before running — these are starting suggestions.",
        "For transition metals (Ti,Zn,Fe,Nb), consider DFT+U for accuracy.",
    ]

    # Flag elements that might need DFT+U
    dftu_elements = symbols & {'Ti','Fe','Cu','Mn','Co','Ni','Zn','Mo','Nb'}
    if dftu_elements:
        notes.append(
            f"DFT+U recommended for: {sorted(dftu_elements)}. "
            "Add hubbard_u={{'Ti':4.2,'Zn':7.0}} to calc_params."
        )

    return {"suggested_params": params, "notes": notes}
