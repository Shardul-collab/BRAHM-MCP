# tools/s4_vision/s4b_region_extraction.py
# ============================================================
# S4B — REGION EXTRACTION AGENT
#
# Responsibilities (per region from S4A):
#
#   plain_text / title regions:
#     → Try PyMuPDF text extraction from bbox (fast, exact)
#     → Fallback: PaddleOCR on rendered crop (CPU)
#
#   figure regions:
#     → Apply garbage filter (figure_filter.py)
#     → Render region crop at REGION_RENDER_DPI
#     → Save PNG to FIGURES_DIR
#     → Link nearby figure_caption text
#
#   table regions:
#     → Try TableTransformer (HuggingFace, CPU)
#     → Fallback: PyMuPDF page.find_tables()
#     → Fallback: save as image (complex table)
#
#   isolate_formula / formula regions:
#     → Try pix2tex LaTeX OCR (reuses existing S4 logic)
#     → Fallback: normalize_equation() regex
#
#   abandon / garbage regions:
#     → Skipped entirely
#
# VRAM note (RTX 2050 / 4GB):
#   S4A (YOLO) runs on GPU.
#   S4B runs everything on CPU to leave VRAM for YOLO.
#   pix2tex still uses CUDA if available (it manages its own
#   model weight which is small enough to coexist).
# ============================================================

from __future__ import annotations

import os
import re
import json
import tempfile
from typing import List, Optional, Dict, Any, Tuple

import fitz  # PyMuPDF

from .s4a_document_vision import (
    Region, DocumentVisionResult,
    EXTRACT_LABELS, CAPTION_LABELS, TEXT_LABELS, GARBAGE_LABELS,
    REGION_RENDER_DPI,
)
from .figure_filter import (
    is_garbage_region, get_garbage_reason, is_scientific_figure,
)

# ── Output directory (mirrors current S4 constant) ───────────
_THIS_FILE    = os.path.abspath(__file__)
_THIS_DIR     = os.path.dirname(_THIS_FILE)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # tools/s4_vision → root
FIGURES_DIR   = os.path.join(_PROJECT_ROOT, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Limits (mirror current S4 constants) ─────────────────────
MAX_FIGURES_PER_PAPER   = 8
MAX_TABLES_PER_PAPER    = 5
MAX_EQUATIONS_PER_PAPER = 15

# ── CAPTION_SECTION_HINTS (carry forward from current S4) ────
CAPTION_SECTION_HINTS = {
    "xrd": "characterization", "x-ray diffraction": "characterization",
    "sem": "characterization", "scanning electron": "characterization",
    "tem": "characterization", "transmission electron": "characterization",
    "ftir": "characterization", "infrared": "characterization",
    "raman": "characterization", "xps": "characterization",
    "bet": "characterization", "brunauer": "characterization",
    "eis": "characterization", "impedance": "characterization",
    "uv-vis": "characterization", "uv\u2013vis": "characterization",
    "cyclic voltamm": "characterization", "morpholog": "characterization",
    "diffraction": "characterization",
    "synthesis": "synthesis_method", "preparation": "synthesis_method",
    "fabrication": "synthesis_method", "hydrothermal": "synthesis_method",
    "solvothermal": "synthesis_method",
    "capacitance": "application", "supercapacitor": "application",
    "performance": "application", "cycling": "application",
    "charge-discharge": "application", "energy density": "application",
    "power density": "application", "photocatalys": "application",
    "sensor": "application",
    "dft": "computational_method", "simulation": "computational_method",
    "theoretical": "computational_method",
    "nanoparticle": "material", "nanocomposite": "material",
    "crystal structure": "material", "morphology": "material",
}

def _get_section_hint(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    for kw, hint in CAPTION_SECTION_HINTS.items():
        if kw in t:
            return hint
    return ""


# ────────────────────────────────────────────────────────────
# LAZY MODEL LOADERS
# ────────────────────────────────────────────────────────────

_PADDLE_OCR      = None
_PADDLE_AVAILABLE = None

_TABLE_MODEL     = None
_TABLE_PROCESSOR = None
_TABLE_AVAILABLE  = None

_PIX2TEX_MODEL   = None
_PIX2TEX_AVAILABLE = None


def _get_paddle_ocr():
    global _PADDLE_OCR, _PADDLE_AVAILABLE
    if _PADDLE_AVAILABLE is False:
        return None
    if _PADDLE_OCR is not None:
        return _PADDLE_OCR
    try:
        from paddleocr import PaddleOCR
        # use_gpu=False: preserve VRAM for YOLO in S4A
        _PADDLE_OCR      = PaddleOCR(use_angle_cls=True, lang="en",
                                      use_gpu=False, show_log=False)
        _PADDLE_AVAILABLE = True
        print("[S4B] PaddleOCR loaded (CPU)")
    except Exception as e:
        _PADDLE_AVAILABLE = False
        print(f"[S4B] PaddleOCR unavailable: {e}")
    return _PADDLE_OCR


def _get_table_model():
    global _TABLE_MODEL, _TABLE_PROCESSOR, _TABLE_AVAILABLE
    if _TABLE_AVAILABLE is False:
        return None, None
    if _TABLE_MODEL is not None:
        return _TABLE_MODEL, _TABLE_PROCESSOR
    try:
        from transformers import (
            TableTransformerForObjectDetection,
            DetrImageProcessor,
        )
        model_name = "microsoft/table-transformer-structure-recognition"
        _TABLE_PROCESSOR = DetrImageProcessor.from_pretrained(model_name)
        _TABLE_MODEL     = TableTransformerForObjectDetection.from_pretrained(
            model_name
        )
        _TABLE_MODEL.eval()
        _TABLE_AVAILABLE = True
        print("[S4B] TableTransformer loaded (CPU)")
    except Exception as e:
        _TABLE_AVAILABLE = False
        print(f"[S4B] TableTransformer unavailable: {e}")
    return _TABLE_MODEL, _TABLE_PROCESSOR


def _get_pix2tex():
    global _PIX2TEX_MODEL, _PIX2TEX_AVAILABLE
    if _PIX2TEX_AVAILABLE is False:
        return None
    if _PIX2TEX_MODEL is not None:
        return _PIX2TEX_MODEL
    try:
        from pix2tex.cli import LatexOCR
        _PIX2TEX_MODEL     = LatexOCR()
        _PIX2TEX_AVAILABLE = True
        print("[S4B] pix2tex LatexOCR loaded")
    except Exception as e:
        _PIX2TEX_AVAILABLE = False
        print(f"[S4B] pix2tex unavailable: {e}")
    return _PIX2TEX_MODEL


# ────────────────────────────────────────────────────────────
# RENDERING HELPERS
# ────────────────────────────────────────────────────────────

def _render_region_to_pil(page: fitz.Page, bbox_pdf: Tuple, dpi: int = REGION_RENDER_DPI):
    """Render a sub-region of a PDF page to PIL Image."""
    from PIL import Image as PILImage
    rect = fitz.Rect(*bbox_pdf)
    mat  = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix  = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _render_region_to_png(page: fitz.Page, bbox_pdf: Tuple,
                           out_path: str, dpi: int = REGION_RENDER_DPI) -> bool:
    """Render sub-region of page and save as PNG. Returns success."""
    try:
        img = _render_region_to_pil(page, bbox_pdf, dpi)
        img.save(out_path, "PNG")
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception as e:
        print(f"[S4B]   render_region_to_png failed: {e}")
        return False


# ────────────────────────────────────────────────────────────
# TEXT EXTRACTION (plain_text / title regions)
# ────────────────────────────────────────────────────────────

def _extract_text_from_region(
    page: fitz.Page,
    region: Region,
) -> str:
    """
    Extract text from a text/title region.
    Tier 1: PyMuPDF clip-rect text extraction (fast, exact).
    Tier 2: PaddleOCR on rendered crop (for image-rendered text).
    """
    # Tier 1: PyMuPDF
    try:
        rect = fitz.Rect(*region.bbox_pdf)
        text = page.get_textbox(rect).strip()
        if text and len(text) > 3:
            return text
    except Exception:
        pass

    # Tier 2: PaddleOCR fallback
    ocr = _get_paddle_ocr()
    if ocr is None:
        return region.text or ""

    try:
        img = _render_region_to_pil(page, region.bbox_pdf)
        import numpy as np
        result = ocr.ocr(np.array(img), cls=True)
        if result and result[0]:
            lines = [
                item[1][0]
                for item in result[0]
                if item and len(item) > 1
            ]
            return " ".join(lines).strip()
    except Exception as e:
        print(f"[S4B]   PaddleOCR failed on region {region.region_id}: {e}")

    return region.text or ""


# ────────────────────────────────────────────────────────────
# FIGURE EXTRACTION
# ────────────────────────────────────────────────────────────

def _extract_figure_region(
    page: fitz.Page,
    region: Region,
    caption_text: str,
    paper_id: int,
    fig_count: int,
    page_height_px: float,
) -> Optional[Dict[str, Any]]:
    """
    Save figure region as PNG after garbage filtering.
    Returns dict compatible with current S4 figure schema, or None.
    """
    # Attach caption for filtering
    region.caption = caption_text

    # Garbage check
    reason = get_garbage_reason(region, page_height_px)
    if reason:
        region.is_garbage    = True
        region.garbage_reason = reason
        print(
            f"[S4B]   SKIP figure {region.region_id} "
            f"({reason})"
        )
        return None

    # Save PNG
    figure_id  = f"{paper_id}_fig_{fig_count}"
    image_path = os.path.join(FIGURES_DIR, f"{figure_id}.png")

    ok = _render_region_to_png(page, region.bbox_pdf, image_path)
    if not ok:
        print(f"[S4B]   SKIP figure {region.region_id}: render failed")
        return None

    region.image_path = image_path
    section_hint      = _get_section_hint(caption_text)
    region.section_hint = section_hint

    scientific = is_scientific_figure(region)
    print(
        f"[S4B]   FIGURE {figure_id} p{region.page_num+1} "
        f"scientific={scientific} hint={section_hint or 'none'}"
    )

    return {
        "figure_id":    figure_id,
        "image_path":   image_path,
        "caption":      caption_text[:300],
        "section_hint": section_hint,
        "page_number":  region.page_num + 1,
    }


# ────────────────────────────────────────────────────────────
# TABLE EXTRACTION
# ────────────────────────────────────────────────────────────

def _parse_table_transformer_output(
    model, processor, pil_img
) -> Optional[Dict[str, Any]]:
    """
    Run TableTransformer structure recognition on a cropped table image.
    Returns {"headers": [...], "rows": [[...], ...]} or None.
    """
    try:
        import torch
        inputs  = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process: get detected cells with labels
        target_sizes = torch.tensor([pil_img.size[::-1]])
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=target_sizes,
        )[0]

        # Build a simple row/col grid from detected cells
        # Labels: 0=table, 1=table column, 2=table row,
        #         3=table column header, 4=table projected row header,
        #         5=table spanning cell
        labels = results["labels"].tolist()
        boxes  = results["boxes"].tolist()

        col_header_boxes = [
            boxes[i] for i, l in enumerate(labels) if l == 3
        ]
        row_boxes = [
            boxes[i] for i, l in enumerate(labels) if l == 2
        ]

        if not col_header_boxes and not row_boxes:
            return None

        # Simple grid reconstruction:
        # Sort column headers left-to-right → header names
        col_header_boxes.sort(key=lambda b: b[0])
        headers = [f"Col_{i+1}" for i in range(len(col_header_boxes))]

        # Sort rows top-to-bottom
        row_boxes.sort(key=lambda b: b[1])
        rows = [[""] * len(headers) for _ in row_boxes]

        return {"headers": headers, "rows": rows}

    except Exception as e:
        print(f"[S4B]   TableTransformer parse failed: {e}")
        return None


def _extract_table_region(
    page: fitz.Page,
    region: Region,
    caption_text: str,
    paper_id: int,
    tbl_count: int,
) -> Optional[Dict[str, Any]]:
    """
    Extract table from a YOLO-detected table region.
    Tier 1: TableTransformer structure recognition
    Tier 2: PyMuPDF page.find_tables() on clipped area
    Tier 3: Save as image (complex table)
    """
    table_id = f"{paper_id}_tbl_{tbl_count}"
    section_hint = _get_section_hint(caption_text)

    # ── Tier 1: TableTransformer ──────────────────────────────
    tt_model, tt_proc = _get_table_model()
    if tt_model is not None:
        try:
            pil_img  = _render_region_to_pil(page, region.bbox_pdf)
            parsed   = _parse_table_transformer_output(tt_model, tt_proc, pil_img)
            if parsed and parsed.get("headers"):
                print(
                    f"[S4B]   TABLE {table_id} via TableTransformer "
                    f"({len(parsed['headers'])} cols, "
                    f"{len(parsed['rows'])} rows)"
                )
                return {
                    "table_id":    table_id,
                    "table_type":  "data",
                    "headers":     parsed["headers"],
                    "rows":        parsed["rows"],
                    "image_path":  None,
                    "caption":     caption_text[:300],
                    "section_hint": section_hint,
                    "page_number": region.page_num + 1,
                }
        except Exception as e:
            print(f"[S4B]   TableTransformer tier failed: {e}")

    # ── Tier 2: PyMuPDF find_tables() ────────────────────────
    try:
        rect   = fitz.Rect(*region.bbox_pdf)
        clip   = page.get_textpage(clip=rect)
        tables = page.find_tables(clip=rect)
        if tables and len(tables.tables) > 0:
            tbl     = tables.tables[0]
            df      = tbl.to_pandas()
            headers = list(df.columns.astype(str))
            rows    = df.values.tolist()
            if headers:
                print(
                    f"[S4B]   TABLE {table_id} via PyMuPDF "
                    f"({len(headers)} cols, {len(rows)} rows)"
                )
                return {
                    "table_id":    table_id,
                    "table_type":  "data",
                    "headers":     headers,
                    "rows":        rows,
                    "image_path":  None,
                    "caption":     caption_text[:300],
                    "section_hint": section_hint,
                    "page_number": region.page_num + 1,
                }
    except Exception as e:
        print(f"[S4B]   PyMuPDF find_tables tier failed: {e}")

    # ── Tier 3: Save as image ─────────────────────────────────
    image_path = os.path.join(FIGURES_DIR, f"{table_id}.png")
    ok = _render_region_to_png(page, region.bbox_pdf, image_path)
    if not ok:
        image_path = None

    print(f"[S4B]   TABLE {table_id} saved as image (complex)")
    return {
        "table_id":    table_id,
        "table_type":  "complex",
        "headers":     [],
        "rows":        [],
        "image_path":  image_path,
        "caption":     caption_text[:300],
        "section_hint": section_hint,
        "page_number": region.page_num + 1,
    }


# ────────────────────────────────────────────────────────────
# EQUATION EXTRACTION
# ────────────────────────────────────────────────────────────

# Carry forward Greek map + normalize_equation from current S4
GREEK_MAP = {
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
    "delta": r"\delta", "epsilon": r"\epsilon", "eta": r"\eta",
    "theta": r"\theta", "lambda": r"\lambda", "mu": r"\mu",
    "nu": r"\nu", "pi": r"\pi", "rho": r"\rho",
    "sigma": r"\sigma", "tau": r"\tau", "phi": r"\phi",
    "\u03b7": r"\eta", "\u03b1": r"\alpha", "\u03b2": r"\beta",
    "\u03b3": r"\gamma", "\u03b4": r"\delta", "\u03b8": r"\theta",
    "\u03bb": r"\lambda", "\u03bc": r"\mu", "\u03c1": r"\rho",
    "\u03c3": r"\sigma", "\u03c4": r"\tau", "\u03c9": r"\omega",
    "\u0394": r"\Delta", "\u03a3": r"\Sigma", "\u03a9": r"\Omega",
    "\u03c0": r"\pi", "\u03c6": r"\phi", "\u03b5": r"\epsilon",
    "\u00b1": r"\pm", "\u00d7": r"\times", "\u00f7": r"\div",
    "\u2265": r"\geq", "\u2264": r"\leq", "\u2260": r"\neq",
    "\u221e": r"\infty", "\u2211": r"\sum", "\u222b": r"\int",
    "\u221a": r"\sqrt", "\u00b0": r"^{\circ}",
}


def _normalize_equation(raw: str) -> str:
    result = raw.strip()
    for char, latex in GREEK_MAP.items():
        if len(char) == 1:
            result = result.replace(char, latex)
    for word, latex in GREEK_MAP.items():
        if len(word) > 1:
            result = re.sub(
                rf"\b{re.escape(word)}\b", latex, result, flags=re.IGNORECASE
            )
    result = re.sub(
        r"([A-Za-z])(\d+)(?![\}])",
        lambda m: f"{m.group(1)}_{{{m.group(2)}}}",
        result,
    )
    return f"\\begin{{equation}}\n{result}\n\\end{{equation}}"


def _extract_equation_region(
    page: fitz.Page,
    region: Region,
    paper_id: int,
    eq_count: int,
    context_before: str,
    context_after: str,
) -> Optional[Dict[str, Any]]:
    """
    Extract equation from a formula region.
    Tier 1: pix2tex LaTeX OCR
    Tier 2: normalize_equation() regex
    """
    equation_id = f"{paper_id}_eq_{eq_count}"

    # Get raw text from region (PyMuPDF)
    raw_text = ""
    try:
        rect     = fitz.Rect(*region.bbox_pdf)
        raw_text = page.get_textbox(rect).strip()
    except Exception:
        pass

    # ── Tier 1: pix2tex ──────────────────────────────────────
    normalized = None
    model = _get_pix2tex()
    if model is not None:
        try:
            from PIL import Image as PILImage
            pil_img = _render_region_to_pil(page, region.bbox_pdf)
            result  = model(pil_img)
            if result and len(result.strip()) >= 3:
                normalized = result.strip()
                if not normalized.startswith("\\begin{equation}"):
                    normalized = (
                        f"\\begin{{equation}}\n{normalized}\n\\end{{equation}}"
                    )
                print(f"[S4B]   EQ {equation_id}: pix2tex OK")
        except Exception as e:
            print(f"[S4B]   pix2tex failed for {equation_id}: {e}")

    # ── Tier 2: normalize_equation ───────────────────────────
    if normalized is None:
        src      = raw_text or region.text or "?"
        normalized = _normalize_equation(src)
        print(f"[S4B]   EQ {equation_id}: normalize fallback")

    return {
        "equation_id":      equation_id,
        "raw_text":         raw_text or region.text or "",
        "normalized_latex": normalized,
        "context_before":   context_before[:200],
        "context_after":    context_after[:200],
        "section_source":   "",   # filled by S4C
        "page_number":      region.page_num + 1,
        "position_index":   eq_count,
    }


# ────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ────────────────────────────────────────────────────────────

def run_region_extraction(
    vision_result: DocumentVisionResult,
) -> DocumentVisionResult:
    """
    S4B entry point.

    Iterates over all regions in *vision_result*, extracts content
    per region type, and populates each Region's fields in-place.
    Also builds the extracted entities lists (figures, tables,
    equations) attached to the result.

    Returns the same DocumentVisionResult with regions populated
    and three extra attributes:
        vision_result.extracted_figures   : list[dict]
        vision_result.extracted_tables    : list[dict]
        vision_result.extracted_equations : list[dict]
    """
    paper_id  = vision_result.paper_id
    filepath  = vision_result.filepath
    regions   = vision_result.regions
    page_sizes = vision_result.page_sizes

    print(f"[S4B] Region extraction start — {len(regions)} regions")

    # Build region_id → Region lookup for caption linking
    reg_map: Dict[str, Region] = {r.region_id: r for r in regions}

    # Open PDF for rendering
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        print(f"[S4B] Cannot open PDF: {e}")
        vision_result.extracted_figures   = []
        vision_result.extracted_tables    = []
        vision_result.extracted_equations = []
        return vision_result

    figures:   List[Dict] = []
    tables:    List[Dict] = []
    equations: List[Dict] = []

    fig_count = 0
    tbl_count = 0
    eq_count  = 0

    # Process regions in reading order (already sorted by S4A)
    for idx, region in enumerate(regions):

        if region.label in GARBAGE_LABELS:
            region.is_garbage = True
            continue

        page_num = region.page_num
        page_h_px = page_sizes[page_num][1] if page_num < len(page_sizes) else 0.0

        try:
            page = doc[page_num]
        except Exception as e:
            print(f"[S4B]   Cannot open page {page_num+1}: {e}")
            continue

        # ── Resolve caption text for this region ─────────────
        caption_text = ""
        if region.caption_region_id and region.caption_region_id in reg_map:
            cap_region   = reg_map[region.caption_region_id]
            caption_text = cap_region.text or ""
            # If caption text not yet filled, extract it now
            if not caption_text and cap_region.label in CAPTION_LABELS:
                caption_text = _extract_text_from_region(page, cap_region)
                cap_region.text = caption_text

        # ── TEXT / TITLE ──────────────────────────────────────
        if region.label in TEXT_LABELS:
            if not region.text:
                region.text = _extract_text_from_region(page, region)
            continue   # text is consumed by S4C

        # ── CAPTION (standalone, if not already filled) ───────
        if region.label in CAPTION_LABELS:
            if not region.text:
                region.text = _extract_text_from_region(page, region)
            continue

        # ── FIGURE ───────────────────────────────────────────
        if region.label == "figure":
            if fig_count >= MAX_FIGURES_PER_PAPER:
                continue
            fig_count += 1
            result = _extract_figure_region(
                page, region, caption_text,
                paper_id, fig_count, page_h_px,
            )
            if result is not None:
                figures.append(result)
            else:
                fig_count -= 1   # didn't save, don't count
            continue

        # ── TABLE ─────────────────────────────────────────────
        if region.label == "table":
            if tbl_count >= MAX_TABLES_PER_PAPER:
                continue
            tbl_count += 1
            result = _extract_table_region(
                page, region, caption_text, paper_id, tbl_count,
            )
            if result is not None:
                tables.append(result)
            else:
                tbl_count -= 1
            continue

        # ── EQUATION ─────────────────────────────────────────
        if region.label == "isolate_formula":
            if eq_count >= MAX_EQUATIONS_PER_PAPER:
                continue

            # Context: previous and next plain_text region on same page
            prev_text = ""
            next_text = ""
            for j in range(idx - 1, max(idx - 4, -1), -1):
                if regions[j].page_num == page_num and regions[j].text:
                    prev_text = regions[j].text[:200]
                    break
            for j in range(idx + 1, min(idx + 4, len(regions))):
                if regions[j].page_num == page_num and regions[j].text:
                    next_text = regions[j].text[:200]
                    break

            eq_count += 1
            result = _extract_equation_region(
                page, region, paper_id, eq_count,
                prev_text, next_text,
            )
            if result is not None:
                equations.append(result)
            continue

    doc.close()

    print(
        f"[S4B] Done — "
        f"figures={len(figures)} "
        f"tables={len(tables)} "
        f"equations={len(equations)}"
    )

    vision_result.extracted_figures   = figures
    vision_result.extracted_tables    = tables
    vision_result.extracted_equations = equations
    return vision_result
