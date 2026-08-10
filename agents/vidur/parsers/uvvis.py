# parsers/uvvis.py
#
# UV-Vis Spectroscopy parser.
# Handles: PerkinElmer .sp, Galactic .spc, generic ASCII.

import os
import struct
import numpy as np


# ── scoring keywords ──────────────────────────────────────────────────────────

_STRONG_KEYWORDS = [
    "absorbance", "wavelength", "uv-vis", "uv vis", "uvvis",
    "transmittance", "optical density", "absorption spectrum",
    "nm", "extinction",
]
_WEAK_KEYWORDS = [
    "spectrum", "bandgap", "tauc", "beer-lambert",
    "photon", "energy gap", "optical",
]

# UV-Vis wavelength axis: typically 200–1100 nm
_AXIS_MIN = 200.0
_AXIS_MAX = 1100.0


def can_parse(data: dict) -> tuple[float, list]:
    """
    Score how likely this file is UV-Vis data.

    Returns:
        (score: float 0–1, signals: list of matched signals)
    """
    signals = []
    score   = 0.0
    text    = data.get("text", "")
    ext     = data.get("extension", "")
    magic   = data.get("magic_bytes", b"")
    numeric = data.get("numeric_data")

    # --- Extension / magic ---
    if ext in (".sp", ".abs", ".dsp"):
        signals.append(f"extension:{ext}")
        score += 0.4
    if magic[:9] == b"UV WinLab":
        signals.append("magic:UV WinLab")
        score += 0.55

    # --- Keyword scoring ---
    for kw in _STRONG_KEYWORDS:
        if kw in text:
            signals.append(f"keyword:{kw}")
            score += 0.15
    for kw in _WEAK_KEYWORDS:
        if kw in text:
            signals.append(f"weak_keyword:{kw}")
            score += 0.05

    # --- Numeric axis range: UV-Vis is 200–1100 nm ---
    if numeric is not None and numeric.shape[1] >= 2:
        x = numeric[:, 0]
        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        if _AXIS_MIN <= x_min and x_max <= _AXIS_MAX:
            signals.append(f"axis_range:[{x_min:.0f}, {x_max:.0f}] nm (UV-Vis)")
            score += 0.25
        # Two-column data with wavelength-like axis strongly suggests UV-Vis
        if 100 < x_max < 3000 and numeric.shape[1] == 2:
            score += 0.05

    return (min(score, 1.0), signals)


def parse(data: dict) -> dict:
    """
    Parse UV-Vis data.

    Returns:
        {
            "technique": "UV-Vis",
            "axis_name": "Wavelength_nm",
            "axis": [...],
            "intensity": [...],
            "metadata": {...},
        }
    """
    path  = data["file_path"]
    ext   = data["extension"]
    magic = data["magic_bytes"]

    if magic[:9] == b"UV WinLab" or ext == ".sp":
        result = _parse_pe_sp(path)
        if result:
            return result

    if ext == ".spc":
        result = _parse_spc(path)
        if result:
            return result

    return _parse_ascii(path)


# ── sub-parsers ───────────────────────────────────────────────────────────────

def _parse_pe_sp(path: str) -> dict | None:
    with open(path, "rb") as f:
        content = f.read()
    raw  = content[0x1000:]
    trim = (len(raw) // 4) * 4
    y    = np.frombuffer(raw[:trim], dtype=np.float32).copy().astype(np.float64)
    y    = y[np.isfinite(y) & (y > -10) & (y < 10)]
    if len(y) == 0:
        return None
    wl = np.linspace(800.0, 200.0, len(y))
    return _out(wl, y, "pe_sp")


def _parse_spc(path: str) -> dict | None:
    with open(path, "rb") as f:
        content = f.read()
    try:
        fexp  = content[3]
        fnpts = struct.unpack("<I", content[4:8])[0]
        ff    = struct.unpack("<d", content[8:16])[0]
        fl    = struct.unpack("<d", content[16:24])[0]
        y_raw = content[512: 512 + fnpts * 4]
        y     = np.frombuffer(y_raw, dtype=np.float32).copy().astype(np.float64)
        if fexp != 0x80:
            y = y * (2.0 ** (fexp - 128))
        wl = np.linspace(ff, fl, fnpts)
        return _out(wl, y, "spc")
    except Exception:
        return None


def _parse_ascii(path: str) -> dict:
    data = _load_ascii(path)
    wl   = data[:, 0]
    y    = data[:, 1]
    mask = (wl > 100) & (wl < 3000) & np.isfinite(y)
    wl, y = wl[mask], y[mask]
    idx   = np.argsort(wl)
    return _out(wl[idx], y[idx], "ascii")


def _out(wl: np.ndarray, intensity: np.ndarray, source: str) -> dict:
    mask = np.isfinite(wl) & np.isfinite(intensity)
    return {
        "technique": "UV-Vis",
        "axis_name": "Wavelength_nm",
        "axis":      wl[mask].tolist(),
        "intensity": intensity[mask].tolist(),
        "metadata":  {"source": source, "units": "nm"},
    }


def _load_ascii(path: str) -> np.ndarray:
    for dlm in (None, ",", "\t"):
        for skip in range(50):
            try:
                data = np.loadtxt(path, delimiter=dlm, skiprows=skip,
                                  comments=["#", "!", ";"])
                if data.ndim == 2 and data.shape[1] >= 2 and data.shape[0] > 1:
                    return data
            except Exception:
                continue
    raise ValueError(f"Could not parse {path} as UV-Vis ASCII data")
