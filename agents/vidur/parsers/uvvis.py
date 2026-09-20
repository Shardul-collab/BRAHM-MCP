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
    if len(y) == 0:
        return None
    # The wavelength axis lives in the PerkinElmer .sp header, which this parser
    # does not read. Until 2026-09-19 it emitted linspace(800, 200) - a
    # fabricated wavelength axis - after filtering `y`. Channel indices instead.
    return _out(np.arange(len(y), dtype=np.float64), y, "pe_sp",
                calibrated=False)


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
    # 2026-09-19: was (wl > 100) & (wl < 3000) - a hardcoded plausibility window
    # that silently deleted anything outside it. Measured: it removed the first
    # 30 points of a real LabSpec file (1024 rows in, 994 out) because they start
    # at 46.6. No range filtering; _out drops only non-finite values.
    idx = np.argsort(wl)
    return _out(wl[idx], y[idx], "ascii")


def _out(wl: np.ndarray, intensity: np.ndarray, source: str,
         calibrated: bool = True) -> dict:
    # 2026-09-19: VIDUR's job is plot-ready data, so it must not decide which
    # points survive. Value-range predicates used to sit here (intensity >= 0,
    # counts >= 0) and silently removed real points - baseline-corrected XRD and
    # Raman go negative on purpose. Only non-finite values are dropped, because
    # they cannot be written to a CSV or plotted, and the count is reported.
    mask = np.isfinite(wl) & np.isfinite(intensity)
    dropped = int((~mask).sum())
    return {
        "technique": "UV-Vis",
        "axis_name": "Wavelength_nm" if calibrated else "channel",
        "axis":      wl[mask].tolist(),
        "intensity": intensity[mask].tolist(),
        "metadata":  {"source": source,
                      "units": "nm" if calibrated else "index",
                      "axis_calibrated": calibrated,
                      "dropped_nonfinite": dropped},
    }


def _load_ascii(path: str) -> np.ndarray:
    # 2026-09-19: was a loadtxt(skiprows=0..50) sweep, which cannot read a
    # file with BOTH a text header and a trailing metadata block - real
    # JASCO and LabSpec exports have both.
    from parsers._ascii import load_xy
    return load_xy(path, "UV-Vis")
