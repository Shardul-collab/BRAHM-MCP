"""
Reference lists that S4 left inside a body section (decision D9, 2026-09-11).

Noise removal works on section NAMES, so a bibliography that S4 appended to a
section called "conclusion" survives it. Measured on workflow 1: paper 15's
"conclusion" was 24,036 chars, of which ~1.4 KB was the conclusion and ~22.6 KB
the reference list starting at "[12] J. B. D. Soole ...". S5 spent ~7 LLM
chunks on it and its pattern rules harvested formulas from cited titles.

A run of reference-entry starts that is long (at least MIN_ENTRIES entries)
and dense (no gap between consecutive entries over MAX_GAP chars) marks the
start of back matter, and everything from there to the end of the section is
cut. Paper 15 shows why "to the end" and not "to the last entry": its
reference list is followed by ~6.4 KB of IEEE author biographies. In-text
citations ("... devices [15].") do not match an entry start, which needs a
capitalised author right after the marker.
"""
import re

MIN_ENTRIES = 8
MAX_GAP = 700

_ENTRY_START = re.compile(
    r"\[\d{1,3}\]\s+[A-Z][\w.\-']*[\s,]"                            # [12] J. Smith / [17] D.-S. Wuu
    r"|\(\d{1,3}\)\s+[A-Z][a-zA-Z\-']+,\s+[A-Z]\."                 # (1) Ajayan, P.
    r"|(?:^|\n)\s*\d{1,3}\.\s+[A-Z][a-zA-Z\-']+,\s+[A-Z]\."        # 1. Smith, J.
)


def find_trailing_references(text: str):
    """Index where an embedded reference list starts, or None."""
    if not text:
        return None
    starts = [m.start() for m in _ENTRY_START.finditer(text)]
    i = 0
    while i < len(starts):
        j = i
        while j + 1 < len(starts) and starts[j + 1] - starts[j] <= MAX_GAP:
            j += 1
        if j - i + 1 >= MIN_ENTRIES:
            return starts[i]
        i = j + 1
    return None


def strip_trailing_references(text: str):
    """(kept_text, removed_chars)."""
    i = find_trailing_references(text)
    if i is None:
        return text, 0
    return text[:i].rstrip(), len(text) - i
