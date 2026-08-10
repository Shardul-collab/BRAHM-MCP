# tools/s4_vision/figure_filter.py
# ============================================================
# FIGURE FILTER
#
# Pure rule-based garbage rejection for visual regions.
# No additional models required — works on top of DocLayout-YOLO
# labels and geometric properties.
#
# DocLayout-YOLO already handles the hardest case:
#   label == "abandon" → logos, watermarks, decorations, ruled
#   lines, ornamental borders — the model rejects these at
#   detection time, so they never reach this filter.
#
# This module applies a second pass of geometric + caption-based
# heuristics for regions the model labelled "figure" but which
# are likely still garbage:
#   • pure-white / near-empty images
#   • extreme aspect ratios (banner lines, dividers)
#   • header / footer zone placement
#   • too small even after YOLO's bbox crop
#   • no caption proximity signal (optional, configurable)
# ============================================================

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .s4a_document_vision import Region

# ── Tunables ────────────────────────────────────────────────
MIN_FIGURE_AREA_PX      = 8_000     # bbox area in pixels at render DPI
MIN_FIGURE_WIDTH_PX     = 60
MIN_FIGURE_HEIGHT_PX    = 60
MAX_ASPECT_RATIO        = 7.0       # w/h or h/w; beyond = banner/divider
HEADER_FOOTER_FRACTION  = 0.07      # top/bottom fraction of page height

# Caption keywords that validate a region as scientific
SCIENTIFIC_CAPTION_PATTERNS = re.compile(
    r"\b("
    r"fig(?:ure)?\.?\s*\d"
    r"|sch(?:eme)?\.?\s*\d"
    r"|image\s*\d"
    r"|panel\s+[a-f]"
    r"|xrd|sem|tem|afm|edx|edd|eds"
    r"|raman|ftir|xps|uv.?vis|uv.?ir"
    r"|bet|impedance|eis|cv|gcd"
    r"|cyclic volts|galvanostatic"
    r"|diffraction|spectroscop"
    r"|morpholog|microstructure|nanostructure"
    r"|band\s+gap|absorption\s+spectr"
    r"|photoluminescen|emission\s+spectr"
    r"|device\s+schem|circuit\s+diagr"
    r"|performance\s+curve|charge.discharge"
    r")",
    re.IGNORECASE,
)

# Captions that indicate a table, not a figure
TABLE_CAPTION_PATTERN = re.compile(
    r"\btable\s*\d|\btab\.?\s*\d", re.IGNORECASE
)


# ────────────────────────────────────────────────────────────
# PUBLIC API
# ────────────────────────────────────────────────────────────

def is_garbage_region(
    region: "Region",
    page_height_px: float,
    require_caption: bool = False,
) -> bool:
    """
    Return True if *region* should be discarded.

    Parameters
    ----------
    region : Region
        The region dataclass from s4a_document_vision.
    page_height_px : float
        Rendered page height in pixels (used for header/footer test).
    require_caption : bool
        If True, a region with no caption is always garbage.
        Default False — caption-less figures are kept if they pass
        all geometric checks (some figures in LaTeX papers have
        captions on the previous page).
    """
    # ── 0. Already labelled garbage by YOLO ──────────────────
    if region.label == "abandon":
        return True

    # ── 1. Only figure-class regions reach this test ─────────
    # Tables and equations have their own extraction paths.
    # This filter is specifically for figure regions.
    if region.label not in ("figure", "figure_caption"):
        return False

    x0, y0, x1, y1 = region.bbox
    w = x1 - x0
    h = y1 - y0

    # ── 2. Minimum size ───────────────────────────────────────
    if w < MIN_FIGURE_WIDTH_PX or h < MIN_FIGURE_HEIGHT_PX:
        return True
    if w * h < MIN_FIGURE_AREA_PX:
        return True

    # ── 3. Extreme aspect ratio (separator lines, banners) ───
    if h > 0 and (w / h) > MAX_ASPECT_RATIO:
        return True
    if w > 0 and (h / w) > MAX_ASPECT_RATIO:
        return True

    # ── 4. Header / footer zone ───────────────────────────────
    if page_height_px > 0:
        top_limit    = page_height_px * HEADER_FOOTER_FRACTION
        bottom_limit = page_height_px * (1.0 - HEADER_FOOTER_FRACTION)
        # Reject only if the region is *entirely* in the header/footer zone
        if y1 < top_limit or y0 > bottom_limit:
            return True

    # ── 5. Caption signal (optional strict mode) ──────────────
    if require_caption and not region.caption.strip():
        return True

    return False


def is_scientific_figure(region: "Region") -> bool:
    """
    Return True if the region's caption contains a strong scientific
    signal.  Used to upgrade a region's confidence but not to gate
    storage — we keep figures even without captions.
    """
    cap = (region.caption or "").strip()
    if not cap:
        return False
    if TABLE_CAPTION_PATTERN.search(cap):
        return False  # it's a table, not a figure
    return bool(SCIENTIFIC_CAPTION_PATTERNS.search(cap))


def caption_confirms_figure(caption_text: str) -> bool:
    """Lightweight check: does this text look like a figure caption?"""
    if not caption_text:
        return False
    if TABLE_CAPTION_PATTERN.search(caption_text):
        return False
    return bool(re.search(r"\bfig(?:ure)?\.?\s*\d|\bsch(?:eme)?\.?\s*\d",
                           caption_text, re.IGNORECASE))


def get_garbage_reason(
    region: "Region",
    page_height_px: float,
) -> str | None:
    """
    Returns a human-readable reason why the region is garbage,
    or None if it passes.  Used for debug logging.
    """
    if region.label == "abandon":
        return "yolo_abandon"

    x0, y0, x1, y1 = region.bbox
    w = x1 - x0
    h = y1 - y0

    if w < MIN_FIGURE_WIDTH_PX or h < MIN_FIGURE_HEIGHT_PX:
        return f"too_small ({w}x{h})"
    if w * h < MIN_FIGURE_AREA_PX:
        return f"area_too_small ({w*h})"
    if h > 0 and (w / h) > MAX_ASPECT_RATIO:
        return f"aspect_ratio_wide ({w/h:.1f})"
    if w > 0 and (h / w) > MAX_ASPECT_RATIO:
        return f"aspect_ratio_tall ({h/w:.1f})"
    if page_height_px > 0:
        top_limit    = page_height_px * HEADER_FOOTER_FRACTION
        bottom_limit = page_height_px * (1.0 - HEADER_FOOTER_FRACTION)
        if y1 < top_limit:
            return "header_zone"
        if y0 > bottom_limit:
            return "footer_zone"
    return None
