"""Shared two-column ASCII loader.

Every parser had its own `np.loadtxt(skiprows=0..50)` sweep. That works on a
bare XY file and fails on real instrument exports, which wrap the data in text:
a JASCO .txt carries an 18-line keyword header AND a trailing metadata block
("CCD temperature", "Shift"), so no single `skiprows` value makes loadtxt parse
the file. Measured 2026-09-19 on real files: LS4.txt was detected as Raman with
confidence 1.0 and then failed to parse.

This scans line by line and keeps the lines that are numeric, which handles
leading headers, trailing blocks, and interruptions in the middle.
"""
import re

import numpy as np

_SEPS = (None, ",", "\t", ";")
_COMMENT = "#!;$'"


def x_label(path: str) -> str | None:
    """The x label the file declares, if any: the first field of the last
    non-numeric line before the data. Files say what their axis is and VIDUR
    should not overrule them -- an .xy holding q was being labelled 2Theta."""
    last = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            head = s.lstrip("".join(_COMMENT)).strip()
            first = re.split(r"[,;\t ]+", head)[0] if head else ""
            try:
                float(first.replace(",", "."))
                return last                      # data started
            except ValueError:
                last = first or last
    return last


def load_xy(path: str, what: str = "data") -> np.ndarray:
    """Return an (n, 2) float array of the first two numeric columns."""
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#!;$":
                continue
            for sep in _SEPS:
                parts = line.split(sep) if sep else line.split()
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0].replace(",", "."))
                    y = float(parts[1].replace(",", "."))
                except ValueError:
                    continue
                rows.append((x, y))
                break
    if len(rows) < 2:
        raise ValueError(f"Could not parse {path} as {what} ASCII data")
    return np.array(rows, dtype=np.float64)
