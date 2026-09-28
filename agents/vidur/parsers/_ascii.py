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


def _is_number(tok: str) -> bool:
    """The same test load_xy applies to a field (decimal comma accepted)."""
    try:
        float(tok.strip().replace(",", "."))
        return True
    except ValueError:
        return False


def _data_fields(line: str) -> tuple[str | None, list[str]] | None:
    """(separator, fields) if load_xy would read `line` as data, else None.
    Kept identical to load_xy's rule so the header and the data are judged
    the same way: not a comment, and the first two fields under one separator
    are numbers."""
    if not line or line[0] in "#!;$":
        return None
    for sep in _SEPS:
        parts = line.split(sep) if sep else line.split()
        if len(parts) >= 2 and _is_number(parts[0]) and _is_number(parts[1]):
            return sep, parts
    return None


def header_label(path: str) -> tuple[str | None, bool]:
    """(label, is_column_header) for the line directly above the first data line.

    is_column_header is True when that line splits into as many fields as the
    data line -- under the data's own separator first, then the others -- and
    none of them is a number; the label is its first field. Otherwise the
    label is only the line's first word, which the caller may use to recognise
    a q or 2theta spelling but must not treat as a declaration. A leading
    comment character is stripped, because exports often write the header as
    "# 2theta intensity".

    2026-09-27: x_label used to return the first word of the last non-numeric
    line before the first line whose first word parsed as a number. That took
    titles ("Industrial TiO2 - Su"), instrument metadata ("'Id: ... Anode: Cu")
    and beamline comments ("# icurr 249.9 ...") as axis labels, and it stopped
    at a real header whose first word is a number ("2 Theta,Intensity" -> "2")
    and at a comment holding a point count ("#      7983"). 11 of the 37 RRD-A
    patterns were then refused as q."""
    prev = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            found = _data_fields(s)
            if found is not None:
                return _label_of(prev, *found)
            prev = s
    return None, False


def _label_of(line: str | None, data_sep, data_fields) -> tuple[str | None, bool]:
    if line is None:
        return None, False
    head = line.lstrip(_COMMENT).strip()
    if not head:
        return None, False
    n = len(data_fields)
    for sep in (data_sep,) + tuple(x for x in _SEPS if x != data_sep):
        parts = head.split(sep) if sep else head.split()
        if len(parts) != n or not parts[0].strip():
            continue
        if any(_is_number(p) for p in parts if p.strip()):
            continue
        return parts[0].strip(), True
    first = re.split(r"[,;\t ]+", head)[0]
    return (None, False) if not first or _is_number(first) else (first, False)


def x_label(path: str) -> str | None:
    """The x label the file declares, if any. See header_label()."""
    return header_label(path)[0]


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
