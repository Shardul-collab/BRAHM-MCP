# tools/extract_paper_content.py
# ============================================================
# EXTRACT PAPER CONTENT — S4  (Vision-upgraded)
#
# UPGRADE: S4 Vision Pipeline (S4A → S4B → S4C)
#
#   S4A — Document Vision Agent
#         DocLayout-YOLO layout detection + region graph
#
#   S4B — Region Extraction Agent
#         Per-region OCR / table / equation extraction
#
#   S4C — Semantic Structuring Agent
#         Section tree reconstruction + S5 chunk prep
#
# MIGRATION STRATEGY — minimally destructive:
#   1. Vision pipeline runs first (S4A → S4B → S4C).
#   2. If ANY unrecoverable error occurs in the vision path,
#      the code falls back to the original heuristic pipeline
#      (extract_sections_by_blocks → extract_figures →
#       extract_tables_advanced → extract_equations_improved).
#   3. The orchestrator contract is UNCHANGED:
#         extract_paper_content(repo, workflow_id, **kwargs)
#         → {"status": "success", "data": [...], "error": None}
#   4. All DB writes use the exact same repo calls.
#   5. No schema changes.
#
# PREVIOUS FIXES (all preserved):
#   BUG-1 through BUG-7 — see git history / legacy comments.
#
# GARBAGE FILTERING (new):
#   DocLayout-YOLO "abandon" class rejects logos, watermarks,
#   decorations, separators at detection time.
#   figure_filter.py adds geometric + caption-based second pass.
# ============================================================

from __future__ import annotations

import os
import re
import json
import tempfile

import fitz  # PyMuPDF

import repositories.paper_repo      as pr
import repositories.paper_content_repo as pc_repo
import repositories.failure_repo    as failure_repo

# ── Vision pipeline ──────────────────────────────────────────
# Imported lazily inside _run_vision_pipeline() so the file
# still loads cleanly if s4_vision deps are not installed.

# ────────────────────────────────────────────────────────────
# CONSTANTS (unchanged from previous S4)
# ────────────────────────────────────────────────────────────

_THIS_FILE    = os.path.abspath(__file__)
_THIS_DIR     = os.path.dirname(_THIS_FILE)
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
FIGURES_DIR   = os.path.join(_PROJECT_ROOT, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MIN_IMAGE_AREA          = 10_000
MAX_FIGURES_PER_PAPER   = 8
MAX_TABLES_PER_PAPER    = 5
MAX_EQUATIONS_PER_PAPER = 15
HEADING_SIZE_RATIO      = 1.15

# ────────────────────────────────────────────────────────────
# CAPTION → SECTION HINT  (unchanged)
# ────────────────────────────────────────────────────────────

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

def get_section_hint(text: str) -> str:
    if not text:
        return None
    t = text.lower()
    for keyword, hint in CAPTION_SECTION_HINTS.items():
        if keyword in t:
            return hint
    return None

def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return FIGURES_DIR


# ============================================================
# LEGACY HEURISTIC PIPELINE
# (preserved verbatim — serves as fallback when vision fails)
# ============================================================

# ── GREEK MAP + normalize_equation ───────────────────────────
GREEK_MAP = {
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
    "delta": r"\delta", "epsilon": r"\epsilon", "eta": r"\eta",
    "theta": r"\theta", "lambda": r"\lambda", "mu": r"\mu",
    "nu": r"\nu", "pi": r"\pi", "rho": r"\rho",
    "sigma": r"\sigma", "tau": r"\tau", "phi": r"\phi",
    "chi": r"\chi", "psi": r"\psi", "omega": r"\omega",
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

def normalize_equation(raw: str) -> str:
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
    result = re.sub(
        r"\^([A-Za-z0-9]+)(?!\{)",
        lambda m: f"^{{{m.group(1)}}}",
        result,
    )
    result = re.sub(
        r"(\b[\w\{\}\\]+)\s*/\s*([\w\{\}\\]+\b)",
        lambda m: rf"\frac{{{m.group(1)}}}{{{m.group(2)}}}",
        result,
    )
    return f"\\begin{{equation}}\n{result}\n\\end{{equation}}"


# ── pix2tex lazy loader (legacy — reused by S4B too) ─────────
_PIX2TEX_MODEL     = None
_PIX2TEX_AVAILABLE = None

def _get_pix2tex_model():
    global _PIX2TEX_MODEL, _PIX2TEX_AVAILABLE
    if _PIX2TEX_AVAILABLE is False:
        return None
    if _PIX2TEX_MODEL is not None:
        return _PIX2TEX_MODEL
    try:
        from pix2tex.cli import LatexOCR
        import torch
        _PIX2TEX_MODEL     = LatexOCR()
        _PIX2TEX_AVAILABLE = True
        print(f"[S4] pix2tex LatexOCR loaded")
    except Exception as e:
        _PIX2TEX_AVAILABLE = False
        _PIX2TEX_MODEL     = None
        print(f"[S4] pix2tex unavailable: {e}")
    return _PIX2TEX_MODEL

def _ocr_equation_bbox(page, bbox) -> str | None:
    model = _get_pix2tex_model()
    if model is None:
        return None
    try:
        from PIL import Image as PILImage
        rect = fitz.Rect(bbox)
        mat  = fitz.Matrix(2.0, 2.0)
        pix  = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_path = tf.name
        pix.save(tmp_path)
        img    = PILImage.open(tmp_path).convert("RGB")
        result = model(img)
        os.unlink(tmp_path)
        if result and len(result.strip()) >= 3:
            return result.strip()
        return None
    except Exception as e:
        print(f"[S4] pix2tex OCR failed: {e}")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None


# ── Equation detection patterns (legacy, unchanged) ──────────
MATH_CHARS = set(
    "=+-*/^\u222b\u2211\u2265\u2264\u2260\u00b1\u00d7\u00f7\u221a"
    "\u03b1\u03b2\u03b3\u03b4\u03b5\u03b7\u03b8\u03bb\u03bc\u03bd"
    "\u03c0\u03c1\u03c3\u03c4\u03c6\u03c7\u03c8\u03c9\u221e\u0394\u03a3\u03a9"
)
MATH_PATTERNS = re.compile(
    r"[=\^]|\\frac|\\int|\\sum|\\alpha|\\beta|\\eta|"
    r"[\u03b1\u03b2\u03b3\u03b4\u03b5\u03b7\u03b8\u03bb\u03bc\u03bd"
    r"\u03c0\u03c1\u03c3\u03c4\u03c6\u03c7\u03c8\u03c9]|"
    r"\b[A-Z][a-z]?\d+\b"
)
_EQ_EXCLUDE_PATTERNS = re.compile(
    r"""
    (?:doi|https?|www\.|dx\.doi|10\.\d{4,})[:/]
    |(?:©|copyright|\bcc\s+by\b|creative\s+commons|all\s+rights|open\s+access)
    |(?:issn|isbn|e-issn|p-issn)\s*[=:]?\s*\d
    |\s*\S{1,25}\s*=\s*\d[\d.,\s/A-Za-z\u00b0\u03bc\u03a9%\-]*\s*$
    |^\s*\[\d+\]|\b(?:et\s+al|ibid|loc\s+cit)\b
    |(?:elsevier|springer|wiley|nature|science|mdpi|acs\s+publications|rsc\s+publishing)
    |^\s*[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

def is_equation_block(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 200:
        return False
    if _EQ_EXCLUDE_PATTERNS.search(t):
        return False
    tl = t.lower()
    if any(w in tl for w in ["fig.", "figure", "table", "tab.", "scheme"]):
        return False
    if re.match(r"^\[\d+\]", t):
        return False
    has_math = bool(MATH_PATTERNS.search(t))
    if not has_math:
        has_math = any(c in MATH_CHARS for c in t)
    if not has_math:
        return False
    non_ascii    = sum(1 for c in t if ord(c) > 127)
    known_math   = sum(1 for c in t if c in MATH_CHARS)
    unknown_high = non_ascii - known_math
    if len(t) > 0 and unknown_high / len(t) > 0.3:
        return False
    eq_match_tokens = []
    if re.search(r"[+\-*/^]|\\frac|\\int|\\sum|_\{|\^\{", t):
        eq_match_tokens.append("operator")
    if any(c in MATH_CHARS - {"="} for c in t):
        eq_match_tokens.append("math_char")
    if re.search(
        r"[\u03b1-\u03c9\u0391-\u03a9]"
        r"|\\(?:alpha|beta|gamma|delta|epsilon|eta|theta|lambda"
        r"|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega|Delta|Sigma|Omega)",
        t
    ):
        eq_match_tokens.append("greek")
    if re.search(r"\b[A-Z][a-z]?\d+\b", t) and re.search(r"[=+\-*/^]", t):
        eq_match_tokens.append("subscript_with_op")
    if not eq_match_tokens:
        return False
    return True


def extract_equations_improved(doc, paper_id: int) -> list:
    """Legacy equation extractor — used by heuristic fallback."""
    equations = []
    eq_count  = 0
    for page_num, page in enumerate(doc):
        if eq_count >= MAX_EQUATIONS_PER_PAPER:
            break
        try:
            page_dict = page.get_text("dict")
        except Exception as e:
            print(f"[S4] Equation page {page_num+1} failed: {e}")
            continue
        blocks           = page_dict.get("blocks", [])
        text_blocks      = []
        text_blocks_bbox = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            block_text = " ".join(
                span.get("text", "")
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ).strip()
            if block_text:
                text_blocks.append(block_text)
                text_blocks_bbox.append(block.get("bbox", None))
        for idx, block_text in enumerate(text_blocks):
            if eq_count >= MAX_EQUATIONS_PER_PAPER:
                break
            if not is_equation_block(block_text):
                continue
            context_before = text_blocks[idx - 1][:200] if idx > 0 else ""
            context_after  = (
                text_blocks[idx + 1][:200] if idx + 1 < len(text_blocks) else ""
            )
            normalized  = None
            ocr_source  = "pix2tex"
            bbox        = text_blocks_bbox[idx]
            if bbox is not None:
                normalized = _ocr_equation_bbox(page, bbox)
            if normalized:
                if not normalized.startswith("\\begin{equation}"):
                    normalized = f"\\begin{{equation}}\n{normalized}\n\\end{{equation}}"
            else:
                ocr_source = "normalize"
                normalized = normalize_equation(block_text)
            eq_count += 1
            equations.append({
                "equation_id":      f"{paper_id}_eq_{eq_count}",
                "raw_text":         block_text,
                "normalized_latex": normalized,
                "context_before":   context_before,
                "context_after":    context_after,
                "section_source":   "",
                "page_number":      page_num + 1,
                "position_index":   eq_count,
                "_ocr_source":      ocr_source,
            })
    return equations

def extract_equations(doc, paper_id: int) -> list:
    return extract_equations_improved(doc, paper_id)


def extract_figures(doc, paper_id: int) -> list:
    """Legacy figure extractor — used by heuristic fallback."""
    figs_dir  = ensure_figures_dir()
    figures   = []
    fig_count = 0
    skipped   = 0
    for page_num, page in enumerate(doc):
        if fig_count >= MAX_FIGURES_PER_PAPER:
            break
        try:
            page_dict   = page.get_text("dict")
            text_blocks = []
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:
                    block_text = " ".join(
                        span.get("text", "")
                        for line in block.get("lines", [])
                        for span in line.get("spans", [])
                    ).strip()
                    if block_text:
                        bbox = block.get("bbox", (0, 0, 0, 0))
                        text_blocks.append((bbox, block_text))
            image_list = page.get_images(full=True)
        except Exception as e:
            print(f"[S4] Figure page {page_num+1} setup failed: {e}")
            skipped += 1
            continue
        for img_info in image_list:
            if fig_count >= MAX_FIGURES_PER_PAPER:
                break
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.colorspace and pix.colorspace.n > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                width  = pix.width
                height = pix.height
                if width * height < MIN_IMAGE_AREA:
                    skipped += 1
                    continue
                fig_count  += 1
                figure_id   = f"{paper_id}_fig_{fig_count}"
                image_path  = os.path.join(figs_dir, f"{figure_id}.png")
                pix.save(image_path)
                if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                    fig_count -= 1
                    skipped   += 1
                    continue
                img_rects  = page.get_image_rects(xref)
                img_bottom = img_rects[0][3] if img_rects else 0
                caption    = ""
                best_dist  = float("inf")
                PROXIMITY  = 60
                for (bx0, by0, bx1, by1), block_text in text_blocks:
                    if by0 < img_bottom:
                        continue
                    dist = by0 - img_bottom
                    if dist < PROXIMITY and dist < best_dist:
                        tl = block_text.lower()
                        if "fig" in tl or len(block_text) < 200:
                            best_dist = dist
                            caption   = block_text
                section_hint = get_section_hint(caption)
                figures.append({
                    "figure_id":    figure_id,
                    "image_path":   image_path,
                    "caption":      caption[:300] if caption else "",
                    "section_hint": section_hint,
                    "page_number":  page_num + 1,
                })
            except Exception as e:
                skipped += 1
                print(f"[S4] SKIP fig xref={xref} p{page_num+1}: {e}")
                continue
    return figures


def _pymupdf_quality_ok(table) -> bool:
    try:
        cells = table.cells
        if not cells:
            return False
        non_empty = sum(1 for c in cells if c is not None and str(c).strip())
        return non_empty / max(len(cells), 1) >= 0.3
    except Exception:
        return False


def _classify_table(headers, rows) -> str:
    if not headers and not rows:
        return "complex"
    total = sum(len(r) for r in rows)
    empty = sum(1 for r in rows for c in r if not str(c).strip())
    if total > 0 and empty / total > 0.6:
        return "complex"
    return "data"


def extract_tables_advanced(doc, paper_id: int, filepath: str = None) -> list:
    """Legacy table extractor — used by heuristic fallback."""
    tables    = []
    tbl_count = 0
    figs_dir  = ensure_figures_dir()
    for page_num, page in enumerate(doc):
        if tbl_count >= MAX_TABLES_PER_PAPER:
            break
        try:
            page_dict   = page.get_text("dict")
            text_blocks = []
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:
                    t = " ".join(
                        s.get("text", "")
                        for l in block.get("lines", [])
                        for s in l.get("spans", [])
                    ).strip()
                    if t:
                        text_blocks.append((block.get("bbox", (0,0,0,0)), t))
        except Exception:
            text_blocks = []

        caption     = ""
        section_hint = None

        # ── Tier 1: PyMuPDF find_tables ──────────────────────
        try:
            found = page.find_tables()
            if found and found.tables:
                for tbl in found.tables:
                    if tbl_count >= MAX_TABLES_PER_PAPER:
                        break
                    if not _pymupdf_quality_ok(tbl):
                        continue
                    try:
                        df      = tbl.to_pandas()
                        headers = list(df.columns.astype(str))
                        rows    = df.values.tolist()
                    except Exception:
                        continue
                    ttype = _classify_table(headers, rows)
                    if ttype == "complex":
                        continue
                    tbl_count += 1
                    table_id   = f"{paper_id}_tbl_{tbl_count}"
                    tables.append({
                        "table_id":    table_id,
                        "table_type":  ttype,
                        "headers":     headers,
                        "rows":        rows,
                        "image_path":  None,
                        "caption":     caption,
                        "section_hint": section_hint,
                        "page_number": page_num + 1,
                    })
                if tables:
                    continue
        except Exception as e:
            print(f"[S4] PyMuPDF find_tables p{page_num+1}: {e}")

        # ── Tier 2: Camelot ──────────────────────────────────
        if filepath:
            try:
                import camelot
                for mode in ("lattice", "stream"):
                    tbls = camelot.read_pdf(
                        filepath, pages=str(page_num + 1), flavor=mode
                    )
                    for tbl in tbls:
                        if tbl_count >= MAX_TABLES_PER_PAPER:
                            break
                        if tbl.accuracy < 50:
                            continue
                        df      = tbl.df
                        headers = list(df.iloc[0].astype(str))
                        rows    = df.iloc[1:].values.tolist()
                        tbl_count += 1
                        table_id   = f"{paper_id}_tbl_{tbl_count}"
                        tables.append({
                            "table_id":    table_id,
                            "table_type":  "data",
                            "headers":     headers,
                            "rows":        rows,
                            "image_path":  None,
                            "caption":     caption,
                            "section_hint": section_hint,
                            "page_number": page_num + 1,
                        })
                    if tables:
                        break
            except Exception as e:
                print(f"[S4] Camelot p{page_num+1}: {e}")

        # ── Tier 3: PNG fallback ──────────────────────────────
        # (only if nothing found above and page has table-like structure)

    return tables


def extract_sections_by_blocks(doc) -> tuple:
    """Legacy section extractor — used by heuristic fallback."""
    sections        = {}
    latex_sections  = {}
    current_section = "preamble"
    current_text    = []

    SECTION_KEYWORDS = [
        "abstract", "introduction", "methods", "methodology",
        "experimental", "materials", "results", "discussion",
        "conclusion", "conclusions", "acknowledgement",
        "acknowledgements", "references", "supplementary",
        "synthesis", "characterization", "electrochemical",
        "background", "theory",
    ]
    kw_pattern = re.compile(
        r"^(?:\d+\.?\s+)?(" +
        "|".join(re.escape(k) for k in SECTION_KEYWORDS) +
        r")",
        re.IGNORECASE,
    )

    def flush():
        nonlocal current_section, current_text
        body = " ".join(current_text).strip()
        if body:
            if current_section in sections:
                sections[current_section] += " " + body
            else:
                sections[current_section] = body
            latex_sections[current_section] = sections[current_section]
        current_text = []

    for page_num, page in enumerate(doc):
        try:
            page_dict = page.get_text("dict")
        except Exception as e:
            print(f"[S4] Sections page {page_num+1} failed: {e}")
            continue
        blocks = page_dict.get("blocks", [])
        for block in blocks:
            if block.get("type") != 0:
                continue
            spans = [
                span
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ]
            if not spans:
                continue
            block_text = " ".join(s.get("text", "") for s in spans).strip()
            if not block_text:
                continue
            sizes    = [s.get("size", 0) for s in spans]
            max_size = max(sizes) if sizes else 0
            avg_size = sum(sizes) / len(sizes) if sizes else 0
            is_heading = (
                max_size > avg_size * HEADING_SIZE_RATIO
                and len(block_text) < 120
                and bool(kw_pattern.match(block_text))
            )
            if is_heading:
                flush()
                raw_name = re.sub(r"^\d+\.?\s*", "", block_text.strip())
                name_key = re.sub(r"[\s\-/&]+", "_", raw_name.lower())
                name_key = re.sub(r"[^a-z0-9_]", "", name_key).strip("_")
                current_section = name_key or "section"
            else:
                current_text.append(block_text)

    flush()
    return sections, latex_sections


def split_sections_keyword(text: str) -> dict:
    """Keyword-based section splitter — last resort fallback."""
    HEADINGS = [
        "Abstract", "Introduction", "Methods", "Methodology",
        "Experimental", "Materials", "Results", "Discussion",
        "Conclusion", "Conclusions", "Acknowledgements",
        "References", "Supplementary",
    ]
    pattern = re.compile(
        r"(?:^|\n)(" + "|".join(re.escape(h) for h in HEADINGS) + r")\b",
        re.IGNORECASE,
    )
    parts   = pattern.split(text)
    sections = {}
    for i in range(1, len(parts), 2):
        heading = parts[i].strip().lower().replace(" ", "_")
        body    = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            sections[heading] = body
    return sections


# ============================================================
# VISION PIPELINE WRAPPER
# ============================================================

def _run_vision_pipeline(paper_id: int, filepath: str) -> dict | None:
    """
    Run S4A → S4B → S4C.  Returns structured output dict on
    success, or None if the vision pipeline fails at any point
    (triggers heuristic fallback in caller).
    """
    try:
        from tools.s4_vision import (
            run_document_vision,
            run_region_extraction,
            run_semantic_structuring,
        )
    except ImportError as e:
        print(f"[S4] s4_vision package not available: {e} — using heuristic")
        return None

    try:
        # S4A — Layout detection
        vision_result = run_document_vision(paper_id, filepath)

        if vision_result.error:
            print(f"[S4] S4A error: {vision_result.error} — using heuristic")
            return None

        if not vision_result.regions:
            print("[S4] S4A returned 0 regions — using heuristic")
            return None

        # S4B — Region extraction
        vision_result = run_region_extraction(vision_result)

        # S4C — Semantic structuring
        structured = run_semantic_structuring(vision_result)
        return structured

    except Exception as e:
        print(f"[S4] Vision pipeline exception: {e} — using heuristic")
        return None


# ============================================================
# MAIN ENTRY POINT  (orchestrator contract — unchanged)
# ============================================================

def extract_paper_content(repo, workflow_id: int, **kwargs) -> dict:
    """
    S4 entry point called by ToolExecutor.

    Signature and return contract are UNCHANGED from previous S4.
    Internally tries the vision pipeline first; falls back to
    heuristic extraction on any failure.

    Returns
    -------
    {"status": "success", "data": [paper_id, ...], "error": None}
    """
    papers = pr.get_papers_for_extraction(repo, workflow_id)

    if not papers:
        print(f"[S4] No papers to extract for workflow {workflow_id}")
        return {"status": "success", "data": [], "error": None}

    processed = []

    for paper in papers:
        paper_id = paper["id"]
        filepath = paper.get("file_path")
        title    = paper.get("title", "")

        print(f"\n[S4] Processing paper {paper_id}: {title[:60]}")

        if not filepath or not os.path.exists(filepath):
            print(f"[S4] No valid file path for paper {paper_id} — skipping")
            pr.update_paper_status(repo, paper_id, "extraction_failed")
            failure_repo.log_failure(
                repo, workflow_id, "EXTRACTION_WARNING",
                "No valid PDF path", paper_id=paper_id,
            )
            processed.append(paper_id)
            continue

        # ── PDF header validation (BUG-7 fix preserved) ──────
        try:
            with open(filepath, "rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError("Not a valid PDF")
        except Exception as e:
            pr.update_paper_status(repo, paper_id, "extraction_failed")
            failure_repo.log_failure(
                repo, workflow_id, "EXTRACTION_WARNING",
                str(e), paper_id=paper_id,
            )
            processed.append(paper_id)
            continue

        try:
            # ============================================
            # ATTEMPT 1: Vision pipeline (S4A → S4B → S4C)
            # ============================================
            structured = _run_vision_pipeline(paper_id, filepath)

            if structured is not None:
                # ── Vision path succeeded ─────────────────
                sections       = structured["sections"]
                latex_sections = structured["latex_sections"]
                raw_text       = structured["raw_text"]
                figures        = structured["figures"]
                tables         = structured["tables"]
                equations      = structured["equations"]
                used_method    = structured["used_method"]
                print(f"[S4] Vision pipeline succeeded ({used_method})")

            else:
                # ============================================
                # ATTEMPT 2: Heuristic fallback (legacy S4)
                # ============================================
                print(f"[S4] Running heuristic fallback for paper {paper_id}")
                doc = fitz.open(filepath)

                raw_text       = ""
                sections       = {}
                latex_sections = {}
                used_method    = "heuristic"

                try:
                    sections, latex_sections = extract_sections_by_blocks(doc)
                    used_method = "heuristic_block"
                except Exception as e:
                    print(f"[S4] Block extraction error: {e}")

                if not sections:
                    try:
                        fallback_text = ""
                        for page in doc:
                            try:
                                fallback_text += page.get_text() + "\n"
                            except Exception:
                                continue
                        fallback_text = fallback_text.strip()
                        if fallback_text:
                            sections    = split_sections_keyword(fallback_text)
                            raw_text    = fallback_text
                            used_method = "heuristic_keyword"
                    except Exception as e:
                        print(f"[S4] Keyword fallback failed: {e}")

                if sections and not raw_text:
                    raw_text = " ".join(v for v in sections.values() if v)

                figures   = []
                tables    = []
                equations = []

                try:
                    equations = extract_equations_improved(doc, paper_id)
                except Exception as e:
                    print(f"[S4] Equation extraction failed: {e}")

                try:
                    figures = extract_figures(doc, paper_id)
                except Exception as e:
                    print(f"[S4] Figure extraction failed: {e}")

                try:
                    tables = extract_tables_advanced(
                        doc, paper_id, filepath=filepath
                    )
                except Exception as e:
                    print(f"[S4] Table extraction failed: {e}")

                doc.close()

            # ── Write sections to DB ──────────────────────
            has_any = False
            for section_name, content in sections.items():
                if content and len(content) > 50:
                    has_any = True
                    latex_text = latex_sections.get(section_name)
                    pc_repo.create_paper_content(
                        repo, paper_id, section_name, content,
                        latex_text=latex_text,
                    )

            if not has_any and raw_text:
                pc_repo.create_paper_content(
                    repo, paper_id, "raw_text", raw_text[:5000]
                )

            if not raw_text or len(raw_text) < 50:
                raw_text = title

            pr.store_paper_text(repo, paper_id, raw_text)

            # ── Write equations to DB ─────────────────────
            try:
                # Map equations to sections (if not already done)
                for eq in equations:
                    if not eq.get("section_source"):
                        context = (
                            eq.get("context_before", "") + " " +
                            eq.get("context_after", "")
                        ).lower()
                        for sec_name in sections:
                            if sec_name.lower()[:6] in context:
                                eq["section_source"] = sec_name
                                break

                for eq in equations:
                    pc_repo.create_paper_equation(
                        repo,
                        paper_id        = paper_id,
                        equation_id     = eq["equation_id"],
                        raw_text        = eq["raw_text"],
                        normalized_latex = eq["normalized_latex"],
                        context_before  = eq.get("context_before", ""),
                        context_after   = eq.get("context_after", ""),
                        section_source  = eq.get("section_source", ""),
                        page_number     = eq.get("page_number", 0),
                        position_index  = eq.get("position_index", 0),
                    )
                print(f"[S4] Equations stored: {len(equations)}")
            except Exception as e:
                print(f"[S4] Equation DB write failed (non-fatal): {e}")

            # ── Write figures to DB ───────────────────────
            try:
                for fig in figures:
                    pc_repo.create_paper_figure(
                        repo,
                        paper_id     = paper_id,
                        figure_id    = fig["figure_id"],
                        image_path   = fig["image_path"],
                        caption      = fig.get("caption", ""),
                        section_hint = fig.get("section_hint", ""),
                        page_number  = fig.get("page_number", 0),
                    )
                print(f"[S4] Figures stored: {len(figures)}")
            except Exception as e:
                print(f"[S4] Figure DB write failed (non-fatal): {e}")

            # ── Write tables to DB ────────────────────────
            try:
                for tbl in tables:
                    pc_repo.create_paper_table(
                        repo,
                        paper_id     = paper_id,
                        table_id     = tbl["table_id"],
                        table_type   = tbl["table_type"],
                        headers      = tbl.get("headers", []),
                        rows         = tbl.get("rows", []),
                        image_path   = tbl.get("image_path"),
                        caption      = tbl.get("caption", ""),
                        section_hint = tbl.get("section_hint", ""),
                        page_number  = tbl.get("page_number", 0),
                    )
                print(f"[S4] Tables stored: {len(tables)}")
            except Exception as e:
                print(f"[S4] Table DB write failed (non-fatal): {e}")

            # ── Advance paper status (BUG-5 fix preserved) ─
            pr.update_paper_status(repo, paper_id, "extracted")

            print(
                f"[S4] Done paper {paper_id} [{used_method}] — "
                f"sections={len(sections)} "
                f"equations={len(equations)} "
                f"figures={len(figures)} "
                f"tables={len(tables)}"
            )
            processed.append(paper_id)

        except Exception as e:
            error_msg = str(e)
            print(f"[S4] Soft-fail {paper_id}: {error_msg}")
            try:
                pr.store_paper_text(repo, paper_id, title)
            except Exception:
                pass
            pr.update_paper_status(repo, paper_id, "extraction_failed")
            failure_repo.log_failure(
                repo, workflow_id, "EXTRACTION_WARNING",
                error_msg, paper_id=paper_id,
            )
            processed.append(paper_id)

    return {"status": "success", "data": processed, "error": None}
