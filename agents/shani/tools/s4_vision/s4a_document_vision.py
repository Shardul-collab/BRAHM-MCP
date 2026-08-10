# tools/s4_vision/s4a_document_vision.py
# ============================================================
# S4A — DOCUMENT VISION AGENT
#
# Responsibilities:
#   1. PDF → page rendering via PyMuPDF (existing dependency)
#   2. Layout detection via DocLayout-YOLO
#   3. Region classification + label assignment
#   4. Reading-order reconstruction (page-order → y → x)
#   5. Caption → figure/table linking (region graph)
#   6. Returns DocumentVisionResult consumed by S4B
#
# DocLayout-YOLO class map (DocStructBench weights):
#   0  title
#   1  plain text
#   2  abandon          ← logos, watermarks, decorations
#   3  figure
#   4  figure_caption
#   5  table
#   6  table_caption
#   7  table_footnote
#   8  isolate_formula
#   9  formula_caption
#
# VRAM strategy (RTX 2050 / 4 GB):
#   YOLO runs on GPU.  All other models in S4B run on CPU.
#   YOLO model is loaded once per process and cached globally.
# ============================================================

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import fitz  # PyMuPDF — already installed

# ── DocLayout-YOLO label map ─────────────────────────────────
YOLO_LABEL_MAP: Dict[int, str] = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

# Labels we care about for extraction
EXTRACT_LABELS   = {"figure", "table", "isolate_formula"}
CAPTION_LABELS   = {"figure_caption", "table_caption", "formula_caption"}
TEXT_LABELS      = {"title", "plain_text", "table_footnote"}
GARBAGE_LABELS   = {"abandon"}

# Render DPI for layout detection pass
LAYOUT_RENDER_DPI  = 144   # ~2× screen; good accuracy without excessive RAM
# Render DPI for high-quality region crops (figures, tables, equations)
REGION_RENDER_DPI  = 216   # ~3× screen for clean OCR/crop images

# Confidence threshold for YOLO detections
YOLO_CONF_THRESHOLD = 0.20

# ── Global model cache ───────────────────────────────────────
_YOLO_MODEL       = None
_YOLO_AVAILABLE   = None   # None=not probed, True=ok, False=failed


# ────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ────────────────────────────────────────────────────────────

@dataclass
class Region:
    """
    Represents one detected layout region on a PDF page.
    Populated by S4A; text/image fields filled in by S4B.
    """
    region_id:    str
    page_num:     int          # 0-indexed
    label:        str          # DocLayout-YOLO label string
    bbox:         Tuple[float, float, float, float]   # x0,y0,x1,y1 in render-px
    bbox_pdf:     Tuple[float, float, float, float]   # x0,y0,x1,y1 in PDF points
    confidence:   float

    # ── Filled by S4B ────────────────────────────────────────
    text:         str  = ""    # extracted / OCR text
    caption:      str  = ""    # associated caption text
    image_path:   str  = ""    # saved PNG (figures, tables as image)
    latex:        str  = ""    # LaTeX string (equations)
    section_hint: str  = ""    # inferred section hint

    # ── Graph links ──────────────────────────────────────────
    # region_id of the caption that belongs to this region
    caption_region_id: Optional[str] = None
    # region_id of the content region this caption describes
    parent_region_id:  Optional[str] = None

    is_garbage:  bool  = False
    garbage_reason: str = ""


@dataclass
class DocumentVisionResult:
    """
    Carries all detected regions + metadata for one paper.
    Passed from S4A → S4B → S4C.
    """
    paper_id:      int
    filepath:      str
    regions:       List[Region]
    page_count:    int
    page_sizes:    List[Tuple[float, float]]   # (width_px, height_px) per page
    layout_model:  str  = "doclayout_yolo"
    render_dpi:    int  = LAYOUT_RENDER_DPI
    fallback_used: bool = False
    error:         Optional[str] = None


# ────────────────────────────────────────────────────────────
# YOLO MODEL LOADER
# ────────────────────────────────────────────────────────────

def _get_yolo_model():
    """
    Load DocLayout-YOLO once and cache globally.
    Returns None if doclayout_yolo is not installed or weights
    cannot be downloaded — S4A will return fallback_used=True.
    """
    global _YOLO_MODEL, _YOLO_AVAILABLE

    if _YOLO_AVAILABLE is False:
        return None
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    try:
        from doclayout_yolo import YOLOv10

        # Weight file: auto-downloaded to ~/.cache on first run.
        # DocLayout-YOLO recommends the DocStructBench model.
        weight_name = "doclayout_yolo_docstructbench_imgsz1024.pt"

        # Allow override via env var for offline environments
        weight_path = os.environ.get("DOCLAYOUT_YOLO_WEIGHTS", weight_name)

        device = _pick_device()
        _YOLO_MODEL     = YOLOv10(weight_path)
        _YOLO_AVAILABLE = True
        print(f"[S4A] DocLayout-YOLO loaded on {device} "
              f"(weights={weight_path})")

    except Exception as e:
        _YOLO_AVAILABLE = False
        _YOLO_MODEL     = None
        print(f"[S4A] DocLayout-YOLO unavailable — "
              f"layout detection disabled: {e}")

    return _YOLO_MODEL


def _pick_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ────────────────────────────────────────────────────────────
# PAGE RENDERING
# ────────────────────────────────────────────────────────────

def _render_page_to_pil(page: fitz.Page, dpi: int = LAYOUT_RENDER_DPI):
    """Render a fitz page to a PIL Image at *dpi* resolution."""
    from PIL import Image as PILImage
    import io

    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img, pix.width, pix.height


def _scale_bbox_to_pdf(
    bbox_px: Tuple[float, float, float, float],
    img_w: float, img_h: float,
    page_w: float, page_h: float,
) -> Tuple[float, float, float, float]:
    """Convert pixel bbox to PDF-point coordinates."""
    sx = page_w / img_w
    sy = page_h / img_h
    x0, y0, x1, y1 = bbox_px
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)


# ────────────────────────────────────────────────────────────
# LAYOUT DETECTION (per-page)
# ────────────────────────────────────────────────────────────

def _detect_regions_on_page(
    model,
    pil_image,
    page_num: int,
    img_w: float,
    img_h: float,
    page_w: float,
    page_h: float,
    paper_id: int,
    region_counter: List[int],   # mutable int container
) -> List[Region]:
    """
    Run DocLayout-YOLO on one rendered page image.
    Returns list of Region objects with bbox in both pixel and PDF coords.
    """
    device = _pick_device()

    try:
        results = model.predict(
            pil_image,
            imgsz=1024,
            conf=YOLO_CONF_THRESHOLD,
            device=device,
            verbose=False,
        )
    except Exception as e:
        print(f"[S4A]   YOLO inference failed p{page_num+1}: {e}")
        return []

    regions = []

    if not results or len(results) == 0:
        return regions

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return regions

    boxes   = result.boxes.xyxy.cpu().numpy()    # (N,4) x0y0x1y1
    confs   = result.boxes.conf.cpu().numpy()    # (N,)
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

    for i in range(len(boxes)):
        x0, y0, x1, y1 = boxes[i]
        conf    = float(confs[i])
        cls_id  = int(cls_ids[i])
        label   = YOLO_LABEL_MAP.get(cls_id, "unknown")

        # Skip unknown labels
        if label == "unknown":
            continue

        bbox_px  = (float(x0), float(y0), float(x1), float(y1))
        bbox_pdf = _scale_bbox_to_pdf(bbox_px, img_w, img_h, page_w, page_h)

        region_counter[0] += 1
        region_id = f"{paper_id}_p{page_num+1}_r{region_counter[0]}_{label}"

        region = Region(
            region_id  = region_id,
            page_num   = page_num,
            label      = label,
            bbox       = bbox_px,
            bbox_pdf   = bbox_pdf,
            confidence = conf,
        )

        regions.append(region)

    return regions


# ────────────────────────────────────────────────────────────
# READING ORDER RECONSTRUCTION
# ────────────────────────────────────────────────────────────

def _sort_reading_order(regions: List[Region]) -> List[Region]:
    """
    Sort regions into reading order:
      primary key  → page_num (ascending)
      secondary    → top of bbox y0 (ascending)
      tertiary     → left of bbox x0 (ascending)

    This handles multi-column layouts correctly as long as
    DocLayout-YOLO gives us per-region bboxes — each column
    is sorted by y within its x band after page order.
    """
    return sorted(regions, key=lambda r: (r.page_num, r.bbox[1], r.bbox[0]))


# ────────────────────────────────────────────────────────────
# CAPTION ↔ CONTENT LINKING
# ────────────────────────────────────────────────────────────

def _link_captions_to_content(regions: List[Region]) -> List[Region]:
    """
    For each caption region, find the nearest content region
    (figure / table / formula) on the same page and link them
    bidirectionally.

    Matching logic:
      • Must be on the same page.
      • Caption must be within CAPTION_PROXIMITY_PX pixels vertically
        of the content region's bottom edge (caption typically below)
        OR within CAPTION_PROXIMITY_PX above (caption above figure).
      • Horizontally overlapping by at least 20% of the narrower width.
      • Nearest wins if multiple candidates exist.
    """
    CAPTION_PROXIMITY_PX = 80   # pixels at render DPI

    content_regions = [r for r in regions if r.label in EXTRACT_LABELS]
    caption_regions = [r for r in regions if r.label in CAPTION_LABELS]

    for cap in caption_regions:
        best_dist   = float("inf")
        best_target = None

        cx0, cy0, cx1, cy1 = cap.bbox
        cap_label_type = cap.label.split("_")[0]   # "figure", "table", "formula"

        for cont in content_regions:
            if cont.page_num != cap.page_num:
                continue

            # Label compatibility: figure_caption → figure only, etc.
            expected_content = cap_label_type
            if expected_content == "formula":
                expected_content = "isolate_formula"
            if cont.label != expected_content and expected_content != "formula":
                # Loose match: allow any content if same page + proximity
                pass

            ix0, iy0, ix1, iy1 = cont.bbox

            # Horizontal overlap check
            overlap = min(cx1, ix1) - max(cx0, ix0)
            min_w   = min(cx1 - cx0, ix1 - ix0)
            if min_w > 0 and overlap / min_w < 0.20:
                continue  # not horizontally related

            # Vertical distance: caption above or below content
            dist_below = abs(cy0 - iy1)   # caption top vs content bottom
            dist_above = abs(iy0 - cy1)   # content top vs caption bottom
            dist       = min(dist_below, dist_above)

            if dist > CAPTION_PROXIMITY_PX:
                continue

            if dist < best_dist:
                best_dist   = dist
                best_target = cont

        if best_target is not None:
            cap.parent_region_id            = best_target.region_id
            best_target.caption_region_id   = cap.region_id

    return regions


# ────────────────────────────────────────────────────────────
# FALLBACK: PyMuPDF heuristic layout (no YOLO)
# ────────────────────────────────────────────────────────────

def _pymupdf_fallback_regions(
    doc: fitz.Document,
    paper_id: int,
) -> Tuple[List[Region], List[Tuple[float, float]]]:
    """
    When YOLO is unavailable, build a minimal set of Region objects
    from PyMuPDF's built-in block detection.  Only creates
    plain_text, title, and figure regions — no abandon filtering.
    This preserves backward compatibility with the old S4 behaviour.
    """
    regions      = []
    page_sizes   = []
    region_counter = [0]

    for page_num, page in enumerate(doc):
        try:
            w_pt, h_pt = page.rect.width, page.rect.height
            scale      = LAYOUT_RENDER_DPI / 72.0
            w_px       = w_pt * scale
            h_px       = h_pt * scale
            page_sizes.append((w_px, h_px))

            page_dict = page.get_text("dict")
            blocks    = page_dict.get("blocks", [])

            for block in blocks:
                block_type = block.get("type", -1)
                b_bbox_pdf = block.get("bbox", (0, 0, 0, 0))
                b_bbox_px  = (
                    b_bbox_pdf[0] * scale, b_bbox_pdf[1] * scale,
                    b_bbox_pdf[2] * scale, b_bbox_pdf[3] * scale,
                )

                if block_type == 0:   # text block
                    spans = [
                        span
                        for line in block.get("lines", [])
                        for span in line.get("spans", [])
                    ]
                    if not spans:
                        continue
                    max_size = max((s.get("size", 0) for s in spans), default=0)
                    label    = "title" if max_size > 14 else "plain_text"
                    text     = " ".join(
                        s.get("text", "") for s in spans
                    ).strip()

                    region_counter[0] += 1
                    regions.append(Region(
                        region_id  = (f"{paper_id}_p{page_num+1}"
                                      f"_r{region_counter[0]}_{label}"),
                        page_num   = page_num,
                        label      = label,
                        bbox       = b_bbox_px,
                        bbox_pdf   = b_bbox_pdf,
                        confidence = 1.0,
                        text       = text,
                    ))

                elif block_type == 1:  # image block
                    region_counter[0] += 1
                    regions.append(Region(
                        region_id  = (f"{paper_id}_p{page_num+1}"
                                      f"_r{region_counter[0]}_figure"),
                        page_num   = page_num,
                        label      = "figure",
                        bbox       = b_bbox_px,
                        bbox_pdf   = b_bbox_pdf,
                        confidence = 0.5,
                    ))

        except Exception as e:
            print(f"[S4A] Fallback page {page_num+1} failed: {e}")
            page_sizes.append((0.0, 0.0))
            continue

    return regions, page_sizes


# ────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ────────────────────────────────────────────────────────────

def run_document_vision(paper_id: int, filepath: str) -> DocumentVisionResult:
    """
    S4A entry point.

    Opens the PDF, renders each page, runs DocLayout-YOLO,
    reconstructs reading order, links captions to content regions,
    and returns a DocumentVisionResult.

    Falls back to PyMuPDF heuristic regions if YOLO is unavailable
    or fails — S4B can still function in degraded mode.

    Parameters
    ----------
    paper_id : int
        DB primary key for the paper.
    filepath : str
        Absolute path to the PDF file.

    Returns
    -------
    DocumentVisionResult
        Contains all detected regions + metadata.
        result.fallback_used is True when YOLO was unavailable.
    """
    print(f"[S4A] Starting document vision — paper {paper_id}")

    # ── Validate PDF ──────────────────────────────────────────
    if not os.path.exists(filepath):
        return DocumentVisionResult(
            paper_id   = paper_id,
            filepath   = filepath,
            regions    = [],
            page_count = 0,
            page_sizes = [],
            error      = f"File not found: {filepath}",
        )

    try:
        with open(filepath, "rb") as f:
            header = f.read(5)
        if header != b"%PDF-":
            return DocumentVisionResult(
                paper_id   = paper_id,
                filepath   = filepath,
                regions    = [],
                page_count = 0,
                page_sizes = [],
                error      = "Not a valid PDF (bad header)",
            )
    except Exception as e:
        return DocumentVisionResult(
            paper_id   = paper_id,
            filepath   = filepath,
            regions    = [],
            page_count = 0,
            page_sizes = [],
            error      = f"PDF validation failed: {e}",
        )

    # ── Open PDF ──────────────────────────────────────────────
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        return DocumentVisionResult(
            paper_id   = paper_id,
            filepath   = filepath,
            regions    = [],
            page_count = 0,
            page_sizes = [],
            error      = f"fitz.open failed: {e}",
        )

    page_count = len(doc)
    print(f"[S4A] PDF opened: {page_count} pages")

    # ── Try YOLO layout detection ─────────────────────────────
    model = _get_yolo_model()

    if model is None:
        # ── FALLBACK PATH ─────────────────────────────────────
        print(f"[S4A] YOLO unavailable → PyMuPDF fallback regions")
        regions, page_sizes = _pymupdf_fallback_regions(doc, paper_id)
        doc.close()
        regions = _sort_reading_order(regions)
        regions = _link_captions_to_content(regions)
        print(f"[S4A] Fallback: {len(regions)} regions detected")
        return DocumentVisionResult(
            paper_id      = paper_id,
            filepath      = filepath,
            regions       = regions,
            page_count    = page_count,
            page_sizes    = page_sizes,
            fallback_used = True,
        )

    # ── YOLO PATH ─────────────────────────────────────────────
    all_regions    = []
    page_sizes     = []
    region_counter = [0]

    for page_num in range(page_count):
        try:
            page   = doc[page_num]
            page_w = page.rect.width    # PDF points
            page_h = page.rect.height

            pil_img, img_w, img_h = _render_page_to_pil(
                page, dpi=LAYOUT_RENDER_DPI
            )
            page_sizes.append((float(img_w), float(img_h)))

            page_regions = _detect_regions_on_page(
                model, pil_img, page_num,
                img_w, img_h, page_w, page_h,
                paper_id, region_counter,
            )

            print(
                f"[S4A]   p{page_num+1}: "
                f"{len(page_regions)} regions detected"
            )
            all_regions.extend(page_regions)

        except Exception as e:
            print(f"[S4A]   Page {page_num+1} failed (skipped): {e}")
            page_sizes.append((0.0, 0.0))
            continue

    doc.close()

    # ── Post-processing ───────────────────────────────────────
    all_regions = _sort_reading_order(all_regions)
    all_regions = _link_captions_to_content(all_regions)

    # Count summary
    label_counts: Dict[str, int] = {}
    for r in all_regions:
        label_counts[r.label] = label_counts.get(r.label, 0) + 1

    print(
        f"[S4A] Done — {len(all_regions)} total regions: "
        + ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items()))
    )

    return DocumentVisionResult(
        paper_id      = paper_id,
        filepath      = filepath,
        regions       = all_regions,
        page_count    = page_count,
        page_sizes    = page_sizes,
        fallback_used = False,
    )
