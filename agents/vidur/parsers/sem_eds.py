# parsers/sem_eds.py
#
# SEM / EDS (EDX) parser.
# Handles: Oxford EMSA/MSA, EDAX SPC, Bruker SPX XML, generic ASCII.

import os
import numpy as np


# ── scoring keywords ──────────────────────────────────────────────────────────

# Element symbols frequently seen in EDS composition tables
_ELEMENT_SYMBOLS = [
    "si", "fe", "al", "ca", "mg", "ti", "cu", "zn", "ni", "cr",
    "mn", "pb", "sn", "au", "ag", "pt", "o", "n", "c", "na", "k",
]

_STRONG_KEYWORDS = [
    "kev", "keV", "eds", "edx", "eds spectrum", "energy dispersive",
    "sem", "scanning electron", "microstructure", "composition",
    "weight %", "atomic %", "wt%", "at%", "elemental analysis",
    "magnification", "acceleration voltage",
]
_WEAK_KEYWORDS = [
    "counts", "x-ray", "characteristic", "emission", "fluorescence",
    "detector", "accelerating", "backscatter",
]

# EDS energy axis: 0–20 keV typically
_AXIS_MAX_KEV = 20.0


def can_parse(data: dict) -> tuple[float, list]:
    """
    Score how likely this file is SEM/EDS data.

    Scoring layers (in order of strength):
      1. Extension / magic bytes      (0.5–0.55)
      2. Keywords in text             (0.05–0.15 each)
      3. Element symbol hits          (0.15)
      4. Numeric axis range (keV)     (0.25)
      5. [NEW] Image signals          (up to 0.20)
      6. [NEW] Table signals          (up to 0.40)

    Total is capped at 1.0.

    Returns:
        (score: float 0–1, signals: list of matched signals)
    """
    signals = []
    score   = 0.0
    text    = data.get("text", "")
    ext     = data.get("extension", "")
    magic   = data.get("magic_bytes", b"")
    numeric = data.get("numeric_data")

    # ── 1. Extension / magic ──────────────────────────────────────────────────
    if ext in (".emsa", ".msa"):
        signals.append(f"extension:{ext} (EMSA/MSA)")
        score += 0.55
    if ext == ".spx":
        signals.append("extension:.spx (Bruker SPX)")
        score += 0.5
    if ext == ".eds":
        signals.append("extension:.eds")
        score += 0.5
    if magic[:7] == b"#FORMAT":
        signals.append("magic:#FORMAT (EMSA)")
        score += 0.55

    # ── 2. Keyword scoring ────────────────────────────────────────────────────
    for kw in _STRONG_KEYWORDS:
        if kw.lower() in text:
            signals.append(f"keyword:{kw}")
            score += 0.15
    for kw in _WEAK_KEYWORDS:
        if kw.lower() in text:
            signals.append(f"weak_keyword:{kw}")
            score += 0.05

    # ── 3. Element symbols in text (compositional data indicator) ─────────────
    elem_hits = [el for el in _ELEMENT_SYMBOLS if f" {el} " in f" {text} "]
    if len(elem_hits) >= 3:
        signals.append(f"element_symbols_in_text:{','.join(elem_hits[:5])}")
        score += 0.15

    # ── 4. Numeric axis range: EDS is 0–20 keV ───────────────────────────────
    if numeric is not None and numeric.shape[1] >= 2:
        x = numeric[:, 0]
        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        if 0 <= x_min and x_max <= _AXIS_MAX_KEV:
            signals.append(f"axis_range:[{x_min:.2f}, {x_max:.2f}] keV (EDS)")
            score += 0.25

    # ── 5. [UPGRADE A] Image signals ──────────────────────────────────────────
    image_count  = data.get("image_count", 0)
    img_sigs     = data.get("image_signals", {})

    if image_count >= 1:
        signals.append(f"has_images:{image_count}")
        score += 0.10

    if image_count >= 3:
        # Image-heavy document → strong SEM hint
        signals.append(f"image_heavy:{image_count} images")
        score += 0.10

    if img_sigs.get("microscopy_like"):
        signals.append("image:microscopy_like (grayscale/low-diversity)")
        score += 0.10
    elif img_sigs.get("is_grayscale"):
        signals.append("image:grayscale")
        score += 0.05

    # ── 6. [UPGRADE B] Table signals ──────────────────────────────────────────
    tsigs = data.get("table_signals", {})

    if data.get("has_table"):
        signals.append("table_detected")
        score += 0.10

    if tsigs.get("has_percentage_headers"):
        signals.append("table:percentage_headers (wt%/at%)")
        score += 0.20

    if tsigs.get("has_element_columns"):
        elem_cols = tsigs.get("element_columns", [])
        signals.append(f"table:element_columns:{','.join(elem_cols[:5])}")
        score += 0.20

    if tsigs.get("percentages_sum_100"):
        signals.append("table:composition_percentages_sum~100")
        score += 0.10

    return (min(score, 1.0), signals)


def parse(data: dict) -> dict:
    """
    Parse SEM/EDS data.

    Returns:
        {
            "technique": "SEM_EDX",
            "axis_name": "Energy_keV",
            "axis": [...],
            "intensity": [...],
            "metadata": {...},
        }
    """
    path = data["file_path"]
    ext  = data["extension"]

    if ext in (".emsa", ".msa"):
        return _parse_emsa(path)
    if ext == ".spx":
        return _parse_bruker_spx(path)
    if ext == ".spc":
        result = _parse_edax_spc(path)
        if result:
            return result
    return _parse_ascii(path)


# ── sub-parsers ───────────────────────────────────────────────────────────────

def _parse_emsa(path: str) -> dict:
    eV_per_ch = 10.0
    zero_ch   = 0
    counts    = []
    in_data   = False

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("#SPECTRUM"):
                in_data = True
                continue
            if line.upper().startswith("#ENDOFDATA"):
                break
            if line.upper().startswith("#EVCHANNEL"):
                try:
                    eV_per_ch = float(line.split(":")[1].split()[0])
                except Exception:
                    pass
            if line.upper().startswith("#CHOFFSET"):
                try:
                    zero_ch = int(float(line.split(":")[1].split()[0]))
                except Exception:
                    pass
            if in_data and not line.startswith("#"):
                parts = line.replace(",", " ").split()
                try:
                    counts.append(float(parts[-1]))
                except (ValueError, IndexError):
                    pass

    if not counts:
        raise ValueError("No data found in EMSA file")

    c   = np.array(counts, dtype=np.float64)
    keV = ((np.arange(len(c)) - zero_ch) * eV_per_ch) / 1000.0
    mask = (keV >= 0) & np.isfinite(c) & (c >= 0)
    return _out(keV[mask], c[mask], "emsa")


def _parse_bruker_spx(path: str) -> dict:
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    el   = root.find(".//TrueSpectrum") or root.find(".//Channels")
    if el is None or not el.text:
        raise ValueError("Spectrum element not found in SPX")
    counts    = np.array([float(v) for v in el.text.split()], dtype=np.float64)
    calib_el  = root.find(".//CalibLin")
    eV_per_ch = float(calib_el.text) if calib_el is not None and calib_el.text else 10.0
    keV       = np.arange(len(counts)) * eV_per_ch / 1000.0
    mask      = (keV >= 0) & (counts >= 0)
    return _out(keV[mask], counts[mask], "bruker_spx")


def _parse_edax_spc(path: str) -> dict | None:
    with open(path, "rb") as f:
        content = f.read()
    if len(content) <= 4096:
        return None
    raw    = content[4096:]
    trim   = (len(raw) // 4) * 4
    counts = np.frombuffer(raw[:trim], dtype=np.uint32).astype(np.float64)
    counts = counts[counts < 1e9]
    if len(counts) == 0:
        return None
    keV = np.arange(len(counts)) * 0.01
    return _out(keV, counts, "edax_spc")


def _parse_ascii(path: str) -> dict:
    for dlm in (None, ",", "\t"):
        for skip in range(50):
            try:
                data = np.loadtxt(path, delimiter=dlm, skiprows=skip,
                                  comments=["#", "!", ";", "$"])
                if data.ndim == 2 and data.shape[1] >= 2 and data.shape[0] > 1:
                    keV    = data[:, 0]
                    counts = data[:, 1]
                    mask   = np.isfinite(keV) & np.isfinite(counts) & (counts >= 0)
                    return _out(keV[mask], counts[mask], "ascii")
            except Exception:
                continue
    raise ValueError(f"Could not parse {path} as SEM/EDS data")


def _out(keV: np.ndarray, counts: np.ndarray, source: str) -> dict:
    return {
        "technique": "SEM_EDX",
        "axis_name": "Energy_keV",
        "axis":      keV.tolist(),
        "intensity": counts.tolist(),
        "metadata":  {"source": source, "units": "keV"},
    }
