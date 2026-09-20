# parsers/raman.py
#
# Raman Spectroscopy parser.
# Handles: Renishaw .wdf (binary), Galactic .spc, generic ASCII.

import os
import struct
import numpy as np


# ── scoring keywords ──────────────────────────────────────────────────────────

_STRONG_KEYWORDS = [
    "raman", "raman shift", "raman scattering", "cm-1", "cm−1",
    "wavenumber", "stokes", "anti-stokes", "raman spectrum",
    "renishaw", "horiba", "witec",
]
_WEAK_KEYWORDS = [
    "laser", "intensity", "peak", "band", "vibrational",
    "phonon", "mode", "spectroscopy", "excitation",
]

# Raman shift axis: typically 100–3500 cm⁻¹
_AXIS_MIN = 100.0
_AXIS_MAX = 3500.0


def can_parse(data: dict) -> tuple[float, list]:
    """
    Score how likely this file is Raman data.

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
    if ext == ".wdf":
        signals.append("extension:.wdf (Renishaw WDF)")
        score += 0.5
    if magic[:4] == b"WDF1":
        signals.append("magic:WDF1 (Renishaw binary)")
        score += 0.6

    # --- Keyword scoring ---
    for kw in _STRONG_KEYWORDS:
        if kw in text:
            signals.append(f"keyword:{kw}")
            score += 0.2
    for kw in _WEAK_KEYWORDS:
        if kw in text:
            signals.append(f"weak_keyword:{kw}")
            score += 0.05

    # --- Numeric axis range: Raman is ~100–3500 cm⁻¹ ---
    if numeric is not None and numeric.shape[1] >= 2:
        x = numeric[:, 0]
        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        if _AXIS_MIN <= x_min and x_max <= _AXIS_MAX:
            signals.append(f"axis_range:[{x_min:.0f}, {x_max:.0f}] cm⁻¹ (Raman)")
            score += 0.2
        # Raman shift can go negative (anti-Stokes); check spread
        if (x_max - x_min) > 500 and x_max < 4000:
            score += 0.05

    return (min(score, 1.0), signals)


def parse(data: dict) -> dict:
    """
    Parse Raman spectroscopy data.

    Returns:
        {
            "technique": "Raman",
            "axis_name": "RamanShift_cm-1",
            "axis": [...],
            "intensity": [...],
            "metadata": {...},
        }
    """
    path  = data["file_path"]
    ext   = data["extension"]
    magic = data["magic_bytes"]

    if magic[:4] == b"WDF1":
        result = _parse_wdf(path)
        if result:
            return result

    if ext == ".spc":
        result = _parse_spc(path)
        if result:
            return result

    return _parse_ascii(path)


# ── sub-parsers ───────────────────────────────────────────────────────────────

def _parse_wdf(path: str) -> dict | None:
    with open(path, "rb") as f:
        content = f.read()
    raw  = content[512:]
    trim = (len(raw) // 4) * 4
    data = np.frombuffer(raw[:trim], dtype=np.float32).copy().astype(np.float64)
    if len(data) == 0:
        return None
    # The Raman shift axis lives in the WDF header, which this parser does not
    # read. Until 2026-09-19 it emitted linspace(100, 3500) - a fabricated shift
    # axis, wrong for any spectrum not covering exactly that range - after
    # filtering `data`, which shifted every surviving point. Channel indices and
    # an explicit uncalibrated flag instead.
    return _out(np.arange(len(data), dtype=np.float64), data, "wdf",
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
        shift = np.linspace(ff, fl, fnpts)
        return _out(shift, y, "spc")
    except Exception:
        return None


def _parse_ascii(path: str) -> dict:
    # 2026-09-19: was a loadtxt(skiprows=0..50) sweep. A JASCO export has an
    # 18-line keyword header AND a trailing metadata block, so no single
    # skiprows value parses it - LS4.txt was detected as Raman at confidence
    # 1.0 and then failed to load.
    from parsers._ascii import load_xy
    data  = load_xy(path, "Raman")
    shift, y = data[:, 0], data[:, 1]
    idx   = np.argsort(shift)
    return _out(shift[idx], y[idx], "ascii")


def _out(shift: np.ndarray, intensity: np.ndarray, source: str,
         calibrated: bool = True) -> dict:
    # 2026-09-19: VIDUR's job is plot-ready data, so it must not decide which
    # points survive. Value-range predicates used to sit here (intensity >= 0,
    # counts >= 0) and silently removed real points - baseline-corrected XRD and
    # Raman go negative on purpose. Only non-finite values are dropped, because
    # they cannot be written to a CSV or plotted, and the count is reported.
    mask = np.isfinite(shift) & np.isfinite(intensity)
    dropped = int((~mask).sum())
    return {
        "technique": "Raman",
        "axis_name": "RamanShift_cm-1" if calibrated else "channel",
        "axis":      shift[mask].tolist(),
        "intensity": intensity[mask].tolist(),
        "metadata":  {"source": source,
                      "units": "cm-1" if calibrated else "index",
                      "axis_calibrated": calibrated,
                      "dropped_nonfinite": dropped},
    }
