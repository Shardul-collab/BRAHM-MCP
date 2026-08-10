# vishwakarma/pseudo_manager.py
#
# Manage UPF pseudopotential files for Quantum ESPRESSO.
# Supports discovery from local directories, type detection, and
# automatic selection recommendations.
#
# Pseudopotential libraries supported:
#   - SSSP (Standard Solid-State Pseudopotentials) — recommended for most work
#   - PseudoDojo — norm-conserving, high accuracy
#   - GBRV — ultrasoft, fast
#   - QE default collection (upf_files/)

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vishwakarma.pseudo_manager")

# ── Known UPF filename patterns ────────────────────────────────────────────────
# Format: {element}.{library}-{type}.UPF  (case-insensitive)
# type codes: nc = norm-conserving, us = ultrasoft, paw = projector-augmented-wave

_UPF_PATTERN = re.compile(
    r"^([A-Z][a-z]?)\.(.+)\.upf$", re.IGNORECASE
)

# Functional tags seen in common filenames
_FUNCTIONAL_TAGS = {
    "pbe":   "GGA-PBE",
    "pbesol":"GGA-PBEsol",
    "lda":   "LDA",
    "pz":    "LDA-PZ",
    "blyp":  "GGA-BLYP",
    "hse":   "Hybrid-HSE",
    "scan":  "meta-GGA-SCAN",
}

_TYPE_TAGS = {
    "nc":       "norm-conserving",
    "ncsr":     "norm-conserving (scalar relativistic)",
    "us":       "ultrasoft",
    "kjpaw":    "PAW (Kresse-Joubert)",
    "bpaw":     "PAW (Blochl)",
    "rrkjus":   "ultrasoft (Rappe-Rabe-Kaxiras-Joannopoulos)",
    "rrkj":     "norm-conserving (RRKJ)",
}


# ─── Discovery ────────────────────────────────────────────────────────────────

def discover(pseudo_dirs: list[str]) -> list[dict]:
    """
    Scan directories for UPF files and return structured metadata.

    Returns:
        List of dicts:
        {
            "path":      str,
            "filename":  str,
            "element":   str,
            "functional": str,
            "type":      str,
            "library":   str,
            "size_kb":   float,
        }
    """
    found = []
    for d in pseudo_dirs:
        p = Path(d)
        if not p.exists():
            logger.warning("Pseudo dir not found: %s", d)
            continue
        for f in p.rglob("*.upf"):
            meta = _parse_upf_filename(f)
            if meta:
                found.append(meta)
        for f in p.rglob("*.UPF"):
            meta = _parse_upf_filename(f)
            if meta:
                found.append(meta)

    logger.info("Discovered %d pseudopotentials across %d directories", len(found), len(pseudo_dirs))
    return found


def list_for_structure(structure: dict, pseudo_dirs: list[str],
                       preferred_functional: str = "pbe",
                       preferred_type: str = "us") -> dict:
    """
    For each species in structure, find the best available pseudopotential.

    Returns:
        {
            element: {
                "recommended": filename | None,
                "alternatives": [filename, ...],
                "missing": bool,
            }
        }
    """
    all_upf = discover(pseudo_dirs)
    result  = {}

    for sp in structure.get("atomic_species", []):
        sym = sp["symbol"]
        candidates = [u for u in all_upf if u["element"].upper() == sym.upper()]

        if not candidates:
            result[sym] = {"recommended": None, "alternatives": [], "missing": True}
            continue

        # Rank: prefer functional match, then type match
        def _score(u: dict) -> tuple:
            func_ok = int(preferred_functional.lower() in u["functional"].lower())
            type_ok = int(preferred_type.lower() in u["type"].lower())
            return (-func_ok, -type_ok)   # lower is better

        ranked = sorted(candidates, key=_score)
        result[sym] = {
            "recommended":  ranked[0]["filename"],
            "alternatives": [c["filename"] for c in ranked[1:5]],
            "missing":       False,
        }

    return result


def validate(pseudo_path: str) -> dict:
    """
    Basic validation of a UPF file (check XML structure and element tag).

    Returns:
        {"valid": bool, "element": str|None, "version": str|None, "error": str|None}
    """
    result = {"valid": False, "element": None, "version": None, "error": None}
    p = Path(pseudo_path)
    if not p.exists():
        result["error"] = f"File not found: {pseudo_path}"
        return result
    try:
        text = p.read_text(errors="replace")
        # UPF v2 is XML-based
        m_el = re.search(r'element\s*=\s*["\']?\s*(\w+)', text, re.IGNORECASE)
        m_ver = re.search(r'version\s*=\s*["\']([^"\']+)', text, re.IGNORECASE)
        if m_el:
            result["element"] = m_el.group(1).strip()
            result["valid"]   = True
        if m_ver:
            result["version"] = m_ver.group(1).strip()
        if not result["valid"]:
            result["error"] = "Could not find element tag — may not be a valid UPF file"
    except Exception as exc:
        result["error"] = str(exc)
    return result


def check_pseudo_dir(pseudo_dir: str) -> dict:
    """
    Quick health check: does the directory exist and how many UPF files are in it?
    """
    p = Path(pseudo_dir)
    if not p.exists():
        return {"exists": False, "upf_count": 0, "path": pseudo_dir}
    upf_count = len(list(p.rglob("*.upf"))) + len(list(p.rglob("*.UPF")))
    return {
        "exists":    True,
        "upf_count": upf_count,
        "path":      str(p.resolve()),
    }


# ─── SSSP database (element → recommended file patterns) ─────────────────────
# Source: materials cloud SSSP efficiency library (PBE)
# These are the SSSP efficiency recommendations — update if using precision library.

SSSP_EFFICIENCY_PBE = {
    "H":  "H.pbe-rrkjus_psl.1.0.0.UPF",
    "Li": "li_pbe_v1.4.uspp.F.UPF",
    "Be": "be_pbe_v1.4.uspp.F.UPF",
    "B":  "b_pbe_v1.4.uspp.F.UPF",
    "C":  "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N":  "N.pbe-n-radius_5.UPF",
    "O":  "O.pbe-n-kjpaw_psl.0.1.UPF",
    "F":  "f_pbe_v1.4.uspp.F.UPF",
    "Na": "na_pbe_v1.5.uspp.F.UPF",
    "Mg": "mg_pbe_v1.4.uspp.F.UPF",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
    "P":  "P.pbe-n-rrkjus_psl.1.0.0.UPF",
    "S":  "s_pbe_v1.4.uspp.F.UPF",
    "Cl": "Cl.pbe-n-rrkjus_psl.1.0.0.UPF",
    "K":  "K.pbe-n-rkjus_psl.0.2.3.upf",
    "Ca": "Ca_pbe_v1.uspp.F.UPF",
    "Ti": "ti_pbe_v1.4.uspp.F.UPF",
    "V":  "v_pbe_v1.4.uspp.F.UPF",
    "Cr": "cr_pbe_v1.5.uspp.F.UPF",
    "Mn": "mn_pbe_v1.5.uspp.F.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
    "Co": "Co.pbe-spn-kjpaw_psl.0.3.1.UPF",
    "Ni": "ni_pbe_v1.4.uspp.F.UPF",
    "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Zn": "Zn.pbe-dnl-kjpaw_psl.0.2.2.UPF",
    "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Ge": "Ge.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "As": "As.pbe-n-rrkjus_psl.0.2.UPF",
    "Se": "Se_pbe_v1.uspp.F.UPF",
    "Br": "br_pbe_v1.4.uspp.F.UPF",
    "Mo": "Mo_ONCV_PBE-1.0.oncvpsp.upf",
    "Ag": "Ag_ONCV_PBE-1.0.oncvpsp.upf",
    "Sn": "Sn_pbe_v1.uspp.F.UPF",
    "I":  "I.pbe-n-kjpaw_psl.0.2.UPF",
    "Pt": "pt_pbe_v1.4.uspp.F.UPF",
    "Au": "Au_ONCV_PBE-1.0.oncvpsp.upf",
    "Pb": "Pb.pbe-dn-kjpaw_psl.0.2.2.UPF",
    "Bi": "Bi_pbe_v1.uspp.F.UPF",
}


def sssp_recommendation(element: str) -> Optional[str]:
    """Return the SSSP efficiency PBE recommendation for an element, or None."""
    return SSSP_EFFICIENCY_PBE.get(element)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _parse_upf_filename(path: Path) -> Optional[dict]:
    m = _UPF_PATTERN.match(path.name)
    if not m:
        return None
    element = m.group(1).capitalize()
    rest    = m.group(2).lower()

    # Detect functional
    functional = "unknown"
    for tag, label in _FUNCTIONAL_TAGS.items():
        if tag in rest:
            functional = label
            break

    # Detect type
    ptype = "unknown"
    for tag, label in _TYPE_TAGS.items():
        if tag in rest:
            ptype = label
            break

    # Guess library
    library = "unknown"
    if "psl" in rest:
        library = "PSLibrary"
    elif "oncv" in rest:
        library = "ONCVPSP"
    elif "sssp" in rest or "efficiency" in rest or "precision" in rest:
        library = "SSSP"
    elif "gbrv" in rest:
        library = "GBRV"
    elif "sg15" in rest:
        library = "SG15"

    try:
        size_kb = path.stat().st_size / 1024
    except Exception:
        size_kb = 0.0

    return {
        "path":       str(path),
        "filename":   path.name,
        "element":    element,
        "functional": functional,
        "type":       ptype,
        "library":    library,
        "size_kb":    round(size_kb, 1),
    }
