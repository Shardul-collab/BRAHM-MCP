# tools/s4_vision/__init__.py
# S4 Vision sub-package.
# Exposes the three agent entry-points used by extract_paper_content.py.

from .s4a_document_vision   import run_document_vision,   DocumentVisionResult
from .s4b_region_extraction import run_region_extraction
from .s4c_semantic_structuring import run_semantic_structuring

__all__ = [
    "run_document_vision",
    "DocumentVisionResult",
    "run_region_extraction",
    "run_semantic_structuring",
]
