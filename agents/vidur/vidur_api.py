"""
vidur_api.py  —  VIDUR Characterisation Classifier API
=======================================================
Standard BRAHM agent API contract.

Endpoints:
  GET  /health        → agent status
  GET  /techniques    → list of supported techniques
  POST /classify      → classify + parse an instrument file

Run:
  cd /mnt/d/brahm/agents/vidur
  /mnt/d/brahm/agents/vidur/.venv/bin/python -m uvicorn vidur_api:app --host 0.0.0.0 --port 8002
"""

import sys
import logging
from pathlib import Path

# ── ensure vidur modules are importable ──────────────────────────────────────
VIDUR_ROOT = Path(__file__).parent
sys.path.insert(0, str(VIDUR_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
)
log = logging.getLogger("vidur.api")

# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VIDUR — Characterisation Classifier",
    version="1.0.0",
    description=(
        "Classifies scientific instrument output files "
        "(XRD, UV-Vis, SEM/EDX, Raman) and parses them into structured data."
    ),
)

# ── schemas ───────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    file_path: str  # absolute path on the server filesystem

# ── helpers ───────────────────────────────────────────────────────────────────

TECHNIQUES = [
    {
        "technique":       "XRD",
        "description":     "X-Ray Diffraction — powder/single-crystal patterns",
        "extensions":      [".xrdml", ".raw", ".xy", ".dat", ".asc"],
        "axis":            "2Theta (degrees, 5–90°)",
        "strong_keywords": ["2theta", "xrd", "diffraction", "bragg", "d-spacing"],
    },
    {
        "technique":       "UV-Vis",
        "description":     "UV-Visible Spectroscopy — absorbance/transmittance",
        "extensions":      [".sp", ".abs", ".dsp", ".spc", ".csv", ".txt"],
        "axis":            "Wavelength_nm (200–1100 nm)",
        "strong_keywords": ["absorbance", "wavelength", "uv-vis", "transmittance", "nm"],
    },
    {
        "technique":       "SEM_EDX",
        "description":     "Scanning Electron Microscopy / Energy Dispersive X-ray",
        "extensions":      [".emsa", ".msa", ".spx", ".eds", ".spc"],
        "axis":            "Energy_keV (0–20 keV)",
        "strong_keywords": ["keV", "eds", "edx", "sem", "weight %", "atomic %"],
    },
    {
        "technique":       "Raman",
        "description":     "Raman Spectroscopy — vibrational/rotational modes",
        "extensions":      [".wdf", ".spc", ".txt", ".csv", ".dat"],
        "axis":            "RamanShift_cm-1 (100–3500 cm⁻¹)",
        "strong_keywords": ["raman", "cm-1", "wavenumber", "raman shift", "stokes"],
    },
]


def _run_pipeline(file_path: str) -> dict:
    """Run extractor → auto_detector → router. Returns structured result."""
    from extractor import extract
    from auto_detector import detect
    from router import route

    data      = extract(file_path)
    detection = detect(data)
    result    = route(detection, data)

    # Sanitise numpy types so FastAPI can serialise
    parsed = result.get("parsed_data")
    if parsed:
        for key in ("axis", "intensity"):
            if key in parsed:
                parsed[key] = [
                    float(v) for v in parsed[key]
                    if v is not None
                ]
    return result


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Standard BRAHM agent health check."""
    # Verify core modules importable
    try:
        import extractor    # noqa: F401
        import auto_detector # noqa: F401
        import router        # noqa: F401
        modules_ok = True
    except Exception as e:
        modules_ok = False
        log.warning("Module import failed: %s", e)

    return {
        "status":      "ok" if modules_ok else "degraded",
        "agent":       "VIDUR",
        "version":     "1.0.0",
        "port":        8002,
        "modules_ok":  modules_ok,
        "techniques":  [t["technique"] for t in TECHNIQUES],
    }


@app.get("/techniques")
def list_techniques():
    """List all supported characterisation techniques."""
    return {
        "count":      len(TECHNIQUES),
        "techniques": TECHNIQUES,
    }


@app.post("/classify")
def classify(req: ClassifyRequest):
    """
    Classify and parse an instrument file.

    Returns technique, confidence, detection signals, and parsed data arrays.
    """
    path = req.file_path.strip()

    if not path:
        raise HTTPException(status_code=400, detail="file_path is required")

    from pathlib import Path as _Path
    if not _Path(path).is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        result = _run_pipeline(path)
        return {
            "status":      "success",
            "file_path":   path,
            "technique":   result.get("technique", "Unknown"),
            "confidence":  round(result.get("confidence", 0.0), 4),
            "signals":     result.get("signals", []),
            "parsed_data": result.get("parsed_data"),
            "error":       result.get("error"),
        }
    except Exception as exc:
        log.exception("Pipeline failed for %s", path)
        raise HTTPException(status_code=500, detail=str(exc))


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
