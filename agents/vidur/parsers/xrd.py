# parsers/xrd.py
#
# X-Ray Diffraction (XRD) parser.
# Handles: PANalytical .xrdml, Bruker .raw v3, generic ASCII two-column files.

import os
import struct
import numpy as np


# ── scoring keywords ──────────────────────────────────────────────────────────

_STRONG_KEYWORDS = [
    "2theta", "2 theta", "two theta", "diffraction", "bragg",
    "xrd", "x-ray diffraction", "d-spacing", "crystallite",
]
_WEAK_KEYWORDS = [
    "intensity", "peak", "lattice", "miller", "reflection",
    "powder", "pattern", "scan", "cps", "counts",
]

# XRD 2θ axis is typically 5–90°
_AXIS_MIN = 5.0
_AXIS_MAX = 90.0


def can_parse(data: dict) -> tuple[float, list]:
    """
    Score how likely this file is XRD data.

    Args:
        data: output of extractor.extract()

    Returns:
        (score: float 0–1, signals: list of matched signals)
    """
    signals = []
    score   = 0.0
    text    = data.get("text", "")
    ext     = data.get("extension", "")
    magic   = data.get("magic_bytes", b"")
    numeric = data.get("numeric_data")

    # --- Extension / magic checks (high confidence) ---
    if ext == ".xrdml":
        signals.append("extension:.xrdml")
        score += 0.5
    if ext == ".raw" and magic[:7] == b"RAW1.01":
        signals.append("magic:RAW1.01")
        score += 0.6
    if ext in (".xy", ".dat", ".asc"):
        signals.append(f"extension:{ext} (possible XRD ASCII)")
        score += 0.05

    # --- Keyword scoring ---
    for kw in _STRONG_KEYWORDS:
        if kw in text:
            signals.append(f"keyword:{kw}")
            score += 0.2
    for kw in _WEAK_KEYWORDS:
        if kw in text:
            signals.append(f"weak_keyword:{kw}")
            score += 0.05

    # --- Numeric axis range check ---
    if numeric is not None and numeric.shape[1] >= 2:
        x = numeric[:, 0]
        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        if _AXIS_MIN <= x_min and x_max <= _AXIS_MAX:
            signals.append(f"axis_range:[{x_min:.1f}, {x_max:.1f}] matches 2θ")
            score += 0.25
        # Typical XRD: short-to-mid range (10–80°), not UV-Vis wavelengths
        if 10 < x_max < 100:
            score += 0.1

    return (min(score, 1.0), signals)


def parse(data: dict) -> dict:
    """
    Parse XRD data from the extracted file data.

    Returns:
        {
            "technique": "XRD",
            "axis_name": "2Theta",
            "axis": [...],
            "intensity": [...],
            "metadata": {...},
        }
    """
    path = data["file_path"]
    ext  = data["extension"]
    magic = data["magic_bytes"]

    if ext == ".xrdml":
        return _parse_xrdml(path)

    if ext == ".raw" and magic[:7] == b"RAW1.01":
        return _parse_bruker_raw(path)

    return _parse_ascii(path)


# ── sub-parsers ───────────────────────────────────────────────────────────────

def _parse_xrdml(path: str) -> dict:
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    ns   = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
    tag  = lambda t: f"{{{ns}}}{t}" if ns else t

    counts_el = root.find(f".//{tag('counts')}")
    if counts_el is None or not counts_el.text:
        raise ValueError("<counts> element not found in XRDML")

    intensity = np.array([float(v) for v in counts_el.text.split()], dtype=np.float64)
    start_el  = root.find(f".//{tag('startPosition')}")
    end_el    = root.find(f".//{tag('endPosition')}")
    if start_el is not None and end_el is not None:
        two_theta = np.linspace(float(start_el.text), float(end_el.text), len(intensity))
    else:
        two_theta = np.arange(len(intensity), dtype=np.float64)

    return _out(two_theta, intensity, "xrdml")


def _parse_bruker_raw(path: str) -> dict:
    with open(path, "rb") as f:
        content = f.read()
    raw  = content[712:]  # v3 header = 712 bytes
    trim = (len(raw) // 4) * 4
    data = np.frombuffer(raw[:trim], dtype=np.float32).copy().astype(np.float64)
    data = data[np.isfinite(data) & (data >= 0) & (data < 1e9)]
    two_theta = np.linspace(10.0, 80.0, len(data))
    return _out(two_theta, data, "bruker_raw")


def _parse_ascii(path: str) -> dict:
    data = _load_ascii(path)
    two_theta = data[:, 0]
    intensity = data[:, 1]
    mask = np.isfinite(intensity) & (intensity >= 0)
    return _out(two_theta[mask], intensity[mask], "ascii")


def _out(axis: np.ndarray, intensity: np.ndarray, source: str) -> dict:
    mask = np.isfinite(axis) & np.isfinite(intensity) & (intensity >= 0)
    return {
        "technique": "XRD",
        "axis_name": "2Theta",
        "axis":      axis[mask].tolist(),
        "intensity": intensity[mask].tolist(),
        "metadata":  {"source": source, "units": "degrees"},
    }


# ── shared ASCII loader ────────────────────────────────────────────────────────

def _load_ascii(path: str) -> np.ndarray:
    for dlm in (None, ",", "\t"):
        for skip in range(50):
            try:
                data = np.loadtxt(path, delimiter=dlm, skiprows=skip,
                                  comments=["#", "!", ";", "$"])
                if data.ndim == 2 and data.shape[1] >= 2 and data.shape[0] > 1:
                    return data
            except Exception:
                continue
    raise ValueError(f"Could not parse {path} as XRD ASCII data")
