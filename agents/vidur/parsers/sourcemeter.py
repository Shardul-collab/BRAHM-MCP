"""Source-meter tables: Keithley 2400/2450 CSV exports and similar.

These are not spectra. A metadata block is followed by a labelled table where
each measured column is followed by its own unit column, so the file states what
it measured -- current, voltage or resistance -- and VIDUR reads that rather than
assuming. Shardul's photodetector folder holds both kinds under one extension:

    16znse03iv150.csv   Reading = "Amp DC",  Value = "Volt DC"   -> I-V sweep
    16znse02rt.csv      Reading = "Ohm",     Value = "Volt DC"   -> R vs time

The x axis is chosen from the data, not the filename: if the source column
actually sweeps it is a sweep; if it sits flat while time advances it is a time
trace. The filename usually agrees ("iv" / "rt") but the file is the evidence.
"""
from __future__ import annotations

import csv
import re

import numpy as np

_UNIT_QUANTITY = {
    "amp": ("Current", "A"), "volt": ("Voltage", "V"), "ohm": ("Resistance", "Ohm"),
    "watt": ("Power", "W"), "coul": ("Charge", "C"), "farad": ("Capacitance", "F"),
}
_TIME_LABEL = re.compile(r"relative\s*time|^time", re.I)


def _quantity(unit: str):
    u = (unit or "").strip().lower()
    for key, val in _UNIT_QUANTITY.items():
        if u.startswith(key):
            return val
    return None, (unit or "").strip()


def _table(path: str):
    """(labels, rows) for the labelled table inside the file."""
    labels, rows = None, []
    with open(path, newline="", errors="ignore") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        # EC-Lab writes tab-separated; Keithley writes commas. Sniffing beats
        # assuming: Pt_poly_cv.mpt came back "no labelled data table found".
        delim = "\t" if sample.count("\t") > sample.count(",") else ","
        for fields in csv.reader(fh, delimiter=delim):
            if len(fields) < 3:
                continue
            try:
                float(fields[0])
            except ValueError:
                labels = [f.strip() for f in fields]
                rows = []
                continue
            if labels and len(fields) >= len(labels) - 1:
                rows.append(fields)
    return labels, rows


def can_parse(data: dict) -> tuple[float, list]:
    score, signals = 0.0, []
    path = data.get("file_path", "")
    text = (data.get("text") or "").lower()
    if data.get("extension") in (".csv", ".dat", ".txt"):
        score += 0.05
    for marker in ("amp dc", "volt dc", "source limit", "relative time",
                   "base time seconds"):
        if marker in text:
            signals.append(f"instrument_column:{marker}")
            score += 0.25
    # A vendor banner identifies the file on its own.
    for banner in ("ec-lab ascii", "cyclic voltammetry", "keithley"):
        if banner in text:
            signals.append(f"instrument:{banner}")
            score += 0.6
            break
    try:
        labels, rows = _table(path)
    except Exception:
        return 0.0, signals
    if labels and rows and len(labels) >= 4:
        units = [l for l in labels if l.strip().lower() in ("unit", "units")]
        if units:
            signals.append(f"labelled table, {len(labels)} columns with unit columns")
            score += 0.35
    return min(score, 1.0), signals


def parse(data: dict) -> dict:
    labels, rows = _table(data["file_path"])
    if not labels or not rows:
        raise ValueError("no labelled data table found")

    def col(i):
        return np.array([float(r[i]) if _isnum(r[i]) else np.nan for r in rows])

    # Measured columns: a numeric column whose neighbour names its unit.
    measured = []
    for i, lab in enumerate(labels):
        if i + 1 < len(labels) and labels[i + 1].strip().lower() in ("unit", "units"):
            unit = rows[0][i + 1] if i + 1 < len(rows[0]) else ""
            name, sym = _quantity(unit)
            measured.append({"index": i, "label": lab, "unit": unit,
                             "quantity": name or lab, "symbol": sym})
    if not measured:
        raise ValueError("no measured column with a unit column beside it")

    time_i = next((i for i, l in enumerate(labels) if _TIME_LABEL.search(l)), None)
    source = next((m for m in measured if m["quantity"] == "Voltage"), None)
    signal = next((m for m in measured if m["quantity"] != "Voltage"), measured[0])

    # Does the source actually sweep, or is it held while time runs?
    swept = False
    if source is not None:
        v = col(source["index"])
        finite = v[np.isfinite(v)]
        if finite.size > 2:
            span = float(np.nanmax(finite) - np.nanmin(finite))
            swept = span > 0.05 * max(1e-12, float(np.nanmax(np.abs(finite))))

    if swept and source is not None:
        axis, axis_name, units, mode = col(source["index"]), "Voltage_V", "V", "sweep"
    elif time_i is not None:
        axis, axis_name, units, mode = col(time_i), "Time_s", "s", "time_trace"
    else:
        axis = np.arange(len(rows), dtype=float)
        axis_name, units, mode = "point_index", "index", "unknown"

    y = col(signal["index"])
    mask = np.isfinite(axis) & np.isfinite(y)
    return {
        "technique": "IV" if mode == "sweep" else ("IT" if mode == "time_trace" else "sourcemeter"),
        "axis_name": axis_name,
        "axis": axis[mask].tolist(),
        "intensity": y[mask].tolist(),
        "metadata": {
            "source": "sourcemeter", "units": units, "axis_calibrated": True,
            "dropped_nonfinite": int((~mask).sum()),
            "measured_quantity": signal["quantity"],
            "measured_unit": signal["unit"],
            "mode": mode,
            "columns_available": [m["label"] + f" ({m['unit']})" for m in measured],
        },
    }


def _isnum(x: str) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False
