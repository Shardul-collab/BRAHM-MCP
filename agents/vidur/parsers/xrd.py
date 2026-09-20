# parsers/xrd.py
#
# X-Ray Diffraction (XRD) parser.
# Handles: PANalytical .xrdml, Bruker .raw v3, generic ASCII two-column files.

import os
import re
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
    # Any Bruker RAW stamp identifies the file as XRD, even where the data block
    # layout of that version is not parsed here. Only RAW1.01 was recognised
    # until 2026-09-19, so a real RAW4.00 scored 0.2 and came back "Uncertain".
    if ext == ".raw" and magic[:3] == b"RAW":
        signals.append(f"magic:{magic[:7].decode('ascii', 'replace')}")
        score += 0.6
    # 2026-09-19: the scorer needed the literal string "2theta" and identified
    # 3 of 15 real .xy files. These are label variants for the same quantity,
    # not a weight nudge -- checked against ixdat's labelled fixture set.
    #
    # 2026-09-20: the four patterns below were matched against `text` alone,
    # with re.M for the two that are line-anchored ("^\s*angle", "^\s*q").
    # extractor.extract() flattens newlines out of `text` -- it is
    # " ".join(...) of the lines -- so re.M had nothing to anchor to and `^`
    # only ever matched the very start of the whole file. Measured on the
    # labelled fixture set: bare_label_q.xy and comma_data.xy (label on line 1)
    # scored XRD 0.60, while comment_then_bare_label_q.xy and
    # empty_lines_in_header.xy -- byte-identical data under two "# ..." comment
    # lines -- scored XRD 0.05 and lost to SEM_EDX at 0.35, which had claimed
    # them on nothing but the x range 0.5-1.5 looking like keV. Two files with
    # the same columns came back as two different techniques because one of
    # them had a header comment.
    #
    # The label was never actually missing: extractor.extract() had already
    # parsed it into data["table_headers"] == ['q', 'i(q)'] for all four. The
    # scorer was re-deriving from raw text a fact the extractor hands it
    # cleanly. Match the headers first and fall back to the text scan, so a
    # file whose label the extractor did not isolate still scores as before.
    headers = [str(h).strip().lower() for h in (data.get("table_headers") or [])]
    for pat, name in ((r"2\s*[-_]?\s*th(eta)?\b", "2theta variant"),
                      (r"\btth\b", "tth"),
                      (r"^\s*angle\b|[,;\t]\s*angle\b", "angle label"),
                      (r"^\s*q\s*[,;\t(\[]|^\s*q\s*$", "q label")):
        # A header cell IS a label, so anchor the pattern to the whole cell
        # rather than to a line that no longer exists.
        if any(re.search(pat, h, re.I) for h in headers) or re.search(pat, text, re.I | re.M):
            # An explicit axis label naming a diffraction quantity is the
            # strongest evidence a text file offers -- stronger than any
            # free-text keyword (0.2), because the instrument wrote it to say
            # what the column IS. Weighted accordingly rather than tuned to
            # clear a particular file.
            signals.append(f"axis_label:{name}")
            score += 0.55
            break
    if ext in (".xy", ".xye", ".dat", ".asc"):
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

    if ext == ".raw" and magic[:3] == b"RAW":
        # Unsupported RAW versions raise from here with the version named,
        # instead of falling through to the ASCII loader and reporting the
        # misleading "Could not parse ... as XRD ASCII data".
        return _parse_bruker_raw(path)

    return _parse_ascii(path)


# ── sub-parsers ───────────────────────────────────────────────────────────────

def _parse_xrdml(path: str) -> dict:
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    ns   = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
    tag  = lambda t: f"{{{ns}}}{t}" if ns else t

    # PANalytical writes the data as <intensities unit="counts"> on most scans and
    # <counts> on others. Only <counts> was looked for until 2026-09-19, so three
    # of four real .xrdml files failed with "<counts> element not found".
    # PANalytical writes the wavelength it used. Asking the operator for a
    # constant the file already states is the same mistake as assuming one:
    # found 2026-09-19 when a scan reported parameters_needed: wavelength_a for
    # a file containing <kAlpha1 unit="Angstrom">1.5405980</kAlpha1>.
    wavelength = None
    k1 = root.find(f".//{tag('kAlpha1')}")
    if k1 is not None and k1.text:
        try:
            wavelength = float(k1.text)
        except ValueError:
            pass

    counts_el = None
    for name in ("intensities", "counts"):
        el = root.find(f".//{tag(name)}")
        if el is not None and el.text and el.text.strip():
            counts_el = el
            break
    if counts_el is None:
        raise ValueError("neither <intensities> nor <counts> found in XRDML")

    intensity = np.array([float(v) for v in counts_el.text.split()], dtype=np.float64)
    start_el  = root.find(f".//{tag('startPosition')}")
    end_el    = root.find(f".//{tag('endPosition')}")
    if start_el is not None and end_el is not None:
        two_theta = np.linspace(float(start_el.text), float(end_el.text), len(intensity))
        out = _out(two_theta, intensity, "xrdml")
        out["metadata"]["wavelength_a"] = wavelength
        return out

    # No <startPosition>/<endPosition>: index axis, not degrees.
    out = _out(np.arange(len(intensity), dtype=np.float64), intensity, "xrdml",
               calibrated=False)
    out["metadata"]["wavelength_a"] = wavelength
    return out


def _parse_bruker_raw(path: str) -> dict:
    with open(path, "rb") as f:
        content = f.read()
    # Bruker stamps the version in the first 8 bytes. Only the v3 layout is
    # understood here; a real RAW4.00 file was being handed to this v3 offset
    # until 2026-09-19 and produced whatever those bytes happened to be.
    magic = content[:7]
    if magic != b"RAW1.01":
        raise ValueError(
            f"Bruker RAW version {content[:7].decode('ascii', 'replace')!r} is not "
            "supported; only RAW1.01 (v3) is. The data block offset and the "
            "2theta range differ per version and are not guessed.")
    raw  = content[712:]  # v3 header = 712 bytes
    trim = (len(raw) // 4) * 4
    data = np.frombuffer(raw[:trim], dtype=np.float32).copy().astype(np.float64)
    # The 2theta start/step live in the RAW range header, which this parser does
    # not read. Until 2026-09-19 it emitted linspace(10, 80), i.e. a fabricated
    # angular axis that was wrong for every scan not run over exactly 10-80 deg,
    # and it filtered `data` first so dropped points silently shifted the rest.
    # Report channel indices and mark the axis uncalibrated instead of inventing
    # degrees. _out() masks axis and intensity together.
    return _out(np.arange(len(data), dtype=np.float64), data, "bruker_raw",
                calibrated=False)


_Q_LABEL = re.compile(r"^q\b|^q[\(\[_]|q\s*\(", re.I)
_TT_LABEL = re.compile(r"2\s*[-_]?\s*(th|theta)|^angle|^tth$", re.I)


def _parse_ascii(path: str) -> dict:
    from parsers._ascii import x_label
    data = _load_ascii(path)
    axis, intensity = data[:, 0], data[:, 1]
    label = (x_label(path) or "").strip()

    # 2026-09-19: the axis was always called "2Theta". Real .xy files also carry
    # q, and ixdat's fixture set has several -- labelling a q axis as degrees is
    # the same fabrication as inventing one. Trust what the file says.
    if _Q_LABEL.match(label):
        out = _out(axis, intensity, "ascii")
        out["axis_name"] = "q"
        out["metadata"]["units"] = "A-1 or nm-1 as written in the file"
        out["metadata"]["axis_label_in_file"] = label
        out["metadata"]["axis_is_two_theta"] = False
        return out

    out = _out(axis, intensity, "ascii")
    out["metadata"]["axis_label_in_file"] = label or None
    out["metadata"]["axis_is_two_theta"] = bool(_TT_LABEL.search(label)) or not label
    return out


def _out(axis: np.ndarray, intensity: np.ndarray, source: str,
         calibrated: bool = True) -> dict:
    """Mask axis and intensity TOGETHER. Filtering one before pairing (which the
    binary paths used to do) drops points from the intensity array only and
    silently shifts every remaining point against its axis value."""
    # 2026-09-19: VIDUR's job is plot-ready data, so it must not decide which
    # points survive. Value-range predicates used to sit here (intensity >= 0,
    # counts >= 0) and silently removed real points - baseline-corrected XRD and
    # Raman go negative on purpose. Only non-finite values are dropped, because
    # they cannot be written to a CSV or plotted, and the count is reported.
    mask = np.isfinite(axis) & np.isfinite(intensity)
    dropped = int((~mask).sum())
    return {
        "technique": "XRD",
        "axis_name": "2Theta" if calibrated else "channel",
        "axis":      axis[mask].tolist(),
        "intensity": intensity[mask].tolist(),
        "metadata":  {"source": source,
                      "units": "degrees" if calibrated else "index",
                      "axis_calibrated": calibrated,
                      "dropped_nonfinite": dropped},
    }


# ── shared ASCII loader ────────────────────────────────────────────────────────

def _load_ascii(path: str) -> np.ndarray:
    # 2026-09-19: was a loadtxt(skiprows=0..50) sweep, which cannot read a
    # file with BOTH a text header and a trailing metadata block - real
    # JASCO and LabSpec exports have both.
    from parsers._ascii import load_xy
    return load_xy(path, "XRD")
