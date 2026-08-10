# vishwakarma/output_parser.py
#
# Parse Quantum ESPRESSO output files into structured Python dicts.
# All parsers accept raw text (str) and return dicts.
# No external deps beyond the stdlib + numpy.

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger("vishwakarma.output_parser")

# ── Physical constants ────────────────────────────────────────────────────────
RY_TO_EV  = 13.605693122994
BOHR_TO_ANG = 0.529177210903


# ─── Top-level dispatcher ─────────────────────────────────────────────────────

def parse(output_text: str, code: str = "pw") -> dict:
    """
    Parse QE output text.  Dispatch by code type.

    Args:
        output_text: raw content of *.out file
        code: "pw" | "ph" | "dos" | "bands" | "pp" | "neb"

    Returns:
        Structured dict — keys depend on code, always includes "converged" and "warnings".
    """
    dispatch = {
        "pw":    parse_pw,
        "ph":    parse_ph,
        "dos":   parse_dos,
        "bands": parse_bands,
        "neb":   parse_neb,
    }
    fn = dispatch.get(code, parse_pw)
    return fn(output_text)


# ─── pw.x parser ─────────────────────────────────────────────────────────────

def parse_pw(text: str) -> dict:
    """
    Parse pw.x output — works for scf, nscf, relax, vc-relax, md.
    """
    result = {
        "code":             "pw",
        "converged":        False,
        "total_energy_ry":  None,
        "total_energy_ev":  None,
        "fermi_energy_ev":  None,
        "homo_ev":          None,
        "gap_ev":           None,
        "forces_ry_au":     None,      # list of [fx,fy,fz] per atom
        "max_force_ry_au":  None,
        "stress_kbar":      None,      # 3x3 matrix
        "pressure_kbar":    None,
        "final_cell_ang":   None,      # 3x3 after relax
        "final_positions":  None,      # list of [x,y,z] in Å
        "scf_iterations":   None,
        "wall_time_s":      None,
        "warnings":         [],
        "errors":           [],
        "calculation_type": None,
    }

    # Calculation type
    m = re.search(r"calculation\s*=\s*'([^']+)'", text)
    if m:
        result["calculation_type"] = m.group(1).strip()

    # Convergence
    if "convergence has been achieved" in text or "SCF Done" in text:
        result["converged"] = True

    # Total energy (last occurrence — relevant for relax)
    for pat in [
        r"!\s+total energy\s*=\s*([-\d.]+)\s*Ry",
        r"total energy\s*=\s*([-\d.]+)\s*Ry",
    ]:
        matches = re.findall(pat, text)
        if matches:
            e = float(matches[-1])
            result["total_energy_ry"] = e
            result["total_energy_ev"] = round(e * RY_TO_EV, 6)
            break

    # Fermi energy
    m = re.search(r"the Fermi energy is\s+([-\d.]+)\s*ev", text, re.IGNORECASE)
    if m:
        result["fermi_energy_ev"] = float(m.group(1))

    # Highest occupied / HOMO + gap
    m = re.search(
        r"highest occupied(?:, lowest unoccupied)? level[s]?\s*\(ev\)\s*:\s*([-\d.\s]+)",
        text, re.IGNORECASE
    )
    if m:
        vals = [float(x) for x in m.group(1).split()]
        if vals:
            result["homo_ev"] = vals[0]
        if len(vals) >= 2:
            result["gap_ev"] = round(vals[1] - vals[0], 4)

    # SCF iterations
    m = re.search(r"convergence has been achieved in\s+(\d+)\s+iterations", text)
    if m:
        result["scf_iterations"] = int(m.group(1))

    # Forces
    force_block = re.search(
        r"Forces acting on atoms.*?\n((?:\s+atom.*?\n)+)", text, re.DOTALL
    )
    if force_block:
        forces = []
        for line in force_block.group(1).splitlines():
            fm = re.search(r"force\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
            if fm:
                forces.append([float(fm.group(i)) for i in (1, 2, 3)])
        if forces:
            result["forces_ry_au"] = forces
            mags = [sum(f**2 for f in fvec)**0.5 for fvec in forces]
            result["max_force_ry_au"] = round(max(mags), 8)

    # Stress tensor + pressure
    stress_m = re.search(
        r"total\s+stress.*?P=\s*([-\d.]+).*?\n((?:\s*[-\d.]+.*?\n){3})",
        text, re.DOTALL
    )
    if stress_m:
        result["pressure_kbar"] = float(stress_m.group(1))
        rows = []
        for line in stress_m.group(2).strip().splitlines():
            vals = line.split()
            if len(vals) >= 3:
                rows.append([float(v) for v in vals[:3]])
        if rows:
            result["stress_kbar"] = rows

    # Final cell parameters (after relax/vc-relax)
    cell_block = re.findall(
        r"CELL_PARAMETERS\s*\(angstrom\)\s*\n((?:\s*[-\d.]+.*?\n){3})", text
    )
    if cell_block:
        rows = []
        for line in cell_block[-1].strip().splitlines():
            vals = line.split()
            if len(vals) >= 3:
                rows.append([float(v) for v in vals[:3]])
        result["final_cell_ang"] = rows

    # Final atomic positions (after relax)
    pos_block = re.findall(
        r"ATOMIC_POSITIONS\s*\{?angstrom\}?\s*\n((?:\s*\w+\s+[-\d.]+.*?\n)+)", text
    )
    if pos_block:
        positions = []
        for line in pos_block[-1].strip().splitlines():
            parts = line.split()
            if len(parts) >= 4:
                positions.append({
                    "symbol": parts[0],
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                })
        result["final_positions"] = positions

    # Wall time
    m = re.search(r"PWSCF\s+:.*?wall\s+time\s*=\s*([\d.:]+)", text)
    if m:
        result["wall_time_s"] = _parse_wall_time(m.group(1))

    # Warnings and errors
    result["warnings"] = re.findall(r"Warning:(.+)", text, re.IGNORECASE)
    result["errors"]   = re.findall(r"Error in routine (.+)", text, re.IGNORECASE)

    return result


# ─── ph.x parser ─────────────────────────────────────────────────────────────

def parse_ph(text: str) -> dict:
    result = {
        "code":           "ph",
        "converged":      "Phonon calculation" in text and "end of run" in text.lower(),
        "frequencies_cm": [],     # list of lists (per q-point)
        "q_points":       [],
        "dielectric_tensor": None,
        "born_charges":   None,
        "warnings":       [],
    }

    # Frequencies per q-point — handles QE 6.x and 7.x output formats
    # Format 1 (QE 7.x):  freq (    1) =   -0.5 [THz] =   -0.5 [cm-1]
    # Format 2 (QE 6.x):  omega(  1) =    -0.5 cm-1
    # Format 3 (bracket):  omega(1) = [  252.0 cm-1]

    # First try to find q-point blocks with any freq line style
    q_blocks = re.findall(
        r"q\s*=\s*([-\d.\s]+)\n(.*?)(?=\n\s*q\s*=|\Z)",
        text, re.DOTALL
    )
    if not q_blocks:
        # fallback: single q-point without explicit block boundary
        q_blocks = re.findall(r"q\s*=\s*([-\d.\s]+)", text)
        q_blocks = [(q, text) for q in q_blocks]

    for q_str, block in q_blocks:
        q = [float(x) for x in q_str.split() if x.replace("-","").replace(".","").isdigit() or x.lstrip("-").replace(".","").isdigit()]
        if not q:
            continue
        # Try all three frequency formats
        freqs = (
            re.findall(r"\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]", block) or
            re.findall(r"\[\s*([-\d.]+)\s*cm-1\]", block) or
            re.findall(r"omega\s*\(.*?\)\s*=\s*([-\d.]+)\s*cm-1", block) or
            re.findall(r"freq\s*\(.*?\)\s*=.*?=\s*([-\d.]+)\s*\[cm-1\]", block)
        )
        if freqs:
            result["q_points"].append(q)
            result["frequencies_cm"].append([float(f) for f in freqs])

    # Dielectric tensor
    diel = re.search(
        r"Dielectric constant in cartesian axis.*?\n((?:\s*[-\d.]+.*?\n){3})",
        text, re.DOTALL
    )
    if diel:
        rows = []
        for line in diel.group(1).strip().splitlines():
            vals = [float(x) for x in line.split()]
            if len(vals) >= 3:
                rows.append(vals[:3])
        result["dielectric_tensor"] = rows

    # Born effective charges (simplified — just atom count)
    born_count = len(re.findall(r"Born effective charges.*?atom\s+\d+", text, re.DOTALL))
    if born_count:
        result["born_charges"] = f"{born_count} atoms parsed"

    result["warnings"] = re.findall(r"Warning:(.+)", text, re.IGNORECASE)
    return result


# ─── dos.x parser ─────────────────────────────────────────────────────────────

def parse_dos(text: str) -> dict:
    """Parse dos.x stdout (minimal — actual DOS data is in fildos file)."""
    result = {
        "code":        "dos",
        "converged":   "end of run" in text.lower(),
        "fermi_ev":    None,
        "warnings":    [],
    }
    m = re.search(r"Fermi energy\s*=\s*([-\d.]+)\s*eV", text, re.IGNORECASE)
    if m:
        result["fermi_ev"] = float(m.group(1))
    result["warnings"] = re.findall(r"Warning:(.+)", text, re.IGNORECASE)
    return result


def parse_dos_file(dos_text: str) -> dict:
    """
    Parse the actual DOS data file produced by dos.x (fildos).
    Returns energy axis and DOS values.
    """
    energies, dos_up, dos_dn = [], [], []
    for line in dos_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            energies.append(float(parts[0]))
            dos_up.append(float(parts[1]))
            if len(parts) >= 3:
                dos_dn.append(float(parts[2]))
        except (ValueError, IndexError):
            continue
    return {
        "energy_ev": energies,
        "dos_up":    dos_up,
        "dos_dn":    dos_dn if dos_dn else None,
    }


# ─── bands.x parser ──────────────────────────────────────────────────────────

def parse_bands(text: str) -> dict:
    result = {
        "code":       "bands",
        "converged":  "end of run" in text.lower(),
        "nbnd":       None,
        "nks":        None,
        "warnings":   [],
    }
    m = re.search(r"Number of k-points.*?=\s*(\d+)", text)
    if m:
        result["nks"] = int(m.group(1))
    m = re.search(r"Number of bands.*?=\s*(\d+)", text)
    if m:
        result["nbnd"] = int(m.group(1))
    result["warnings"] = re.findall(r"Warning:(.+)", text, re.IGNORECASE)
    return result


def parse_bands_file(bands_text: str) -> dict:
    """
    Parse the bands data file produced by bands.x (filband).
    Returns k-points and eigenvalue arrays.
    """
    kpoints  = []
    bands_arr = []
    current_band = []

    for line in bands_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.lower().startswith("nbnd"):
            continue
        parts = line.split()
        try:
            vals = [float(x) for x in parts]
            current_band.extend(vals)
        except ValueError:
            continue

    # bands.x filband format: nks*nbnd eigenvalues in column order
    # We return the raw flat array; caller slices by nbnd
    return {
        "eigenvalues_ev": current_band,
        "note": "Slice by nbnd to get per-k eigenvalues",
    }


# ─── neb.x parser ────────────────────────────────────────────────────────────

def parse_neb(text: str) -> dict:
    result = {
        "code":           "neb",
        "converged":      "neb: convergence achieved" in text.lower(),
        "activation_ev":  None,
        "reaction_ev":    None,
        "path_energies_ev": [],
        "warnings":       [],
    }

    # Path energies (in eV)
    energy_block = re.search(
        r"activation energy.*?\(([-\d.]+)\s*eV\)", text, re.IGNORECASE
    )
    if energy_block:
        result["activation_ev"] = float(energy_block.group(1))

    path_e = re.findall(r"image:\s*\d+\s+([-\d.]+)\s*eV", text)
    if path_e:
        result["path_energies_ev"] = [float(e) for e in path_e]
        if len(path_e) >= 2:
            result["reaction_ev"] = round(
                float(path_e[-1]) - float(path_e[0]), 6
            )

    result["warnings"] = re.findall(r"Warning:(.+)", text, re.IGNORECASE)
    return result


# ─── Convergence history ─────────────────────────────────────────────────────

def parse_scf_convergence(text: str) -> list[dict]:
    """
    Extract per-iteration SCF convergence data.
    Returns list of {iteration, total_energy_ry, delta_e_ry}.
    """
    history = []
    for m in re.finditer(
        r"iter\s*#\s*(\d+).*?total energy\s*=\s*([-\d.]+)\s*Ry\s*delta\s*E\s*=\s*([-\d.eE+]+)\s*Ry",
        text, re.DOTALL
    ):
        history.append({
            "iteration":       int(m.group(1)),
            "total_energy_ry": float(m.group(2)),
            "delta_e_ry":      float(m.group(3)),
        })
    return history


def parse_relax_steps(text: str) -> list[dict]:
    """
    Extract per-ionic-step data from relax/vc-relax output.
    Returns list of {step, energy_ry, max_force_ry_au}.
    """
    steps = []
    for m in re.finditer(
        r"number of scf cycles\s*=\s*(\d+).*?"
        r"total energy\s*=\s*([-\d.]+)\s*Ry.*?"
        r"convergence has been achieved",
        text, re.DOTALL
    ):
        steps.append({
            "step":          int(m.group(1)),
            "energy_ry":     float(m.group(2)),
        })
    return steps


# ─── Utilities ───────────────────────────────────────────────────────────────

def _parse_wall_time(s: str) -> Optional[float]:
    """Convert QE wall-time string like '1h23m45.6s' or '45.6s' to seconds."""
    total = 0.0
    for unit, mult in (("h", 3600), ("m", 60), ("s", 1)):
        m = re.search(rf"(\d+\.?\d*){unit}", s)
        if m:
            total += float(m.group(1)) * mult
    return total if total else None


def check_job_success(output_text: str) -> tuple[bool, str]:
    """
    Quick success/fail check without full parse.
    Returns (success: bool, reason: str).
    """
    if not output_text:
        return False, "empty output"
    if "JOB DONE" in output_text:
        return True, "JOB DONE found"
    if "convergence has been achieved" in output_text:
        return True, "SCF converged"
    if "Error in routine" in output_text:
        errors = re.findall(r"Error in routine (.+)", output_text)
        return False, f"QE error: {errors[0] if errors else 'unknown'}"
    if "stopping ..." in output_text.lower():
        return False, "QE stopped abnormally"
    return False, "JOB DONE not found"
