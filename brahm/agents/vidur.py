"""
brahm/agents/vidur.py
======================
Group G — VIDUR instrument file classifier tools.
Runs fully locally — no HTTP, no cloud.

Auto-save: after every successful vidur_classify call, result is persisted
to brahm.db via POST /v1/results/instrument (CHITRAGUPTA API on :8003).
Pass project_id in args to enable. Silently skipped if omitted or API is down.
"""

import asyncio
from brahm.brahm_registry import brahm_tool
from brahm.shared.helpers import _ok, _err

CHITRAGUPTA_BASE    = "http://localhost:8003"
CHITRAGUPTA_TIMEOUT = 5


# =========================================================
# CHITRAGUPTA AUTO-SAVE HELPER
# =========================================================

def _chit_save_instrument(
    project_id: int,
    file_path: str,
    technique: str,
    confidence: float,
    signals: list,
    parsed_data: dict,
    cycle_id: int | None,
) -> int | None:
    """
    POST /v1/results/instrument — persist a VIDUR result to brahm.db.
    Returns result_id or None. Never raises.
    """
    try:
        import requests
        r = requests.post(
            f"{CHITRAGUPTA_BASE}/v1/results/instrument",
            json={
                "project_id":  project_id,
                "file_path":   file_path,
                "technique":   technique,
                "confidence":  confidence,
                "signals":     signals,
                "parsed_data": parsed_data or {},
                "cycle_id":    cycle_id,
            },
            timeout=CHITRAGUPTA_TIMEOUT,
        )
        if r.status_code == 200:
            rid = r.json().get("result_id")
            print(f"[CHITRAGUPTA] Instrument result saved: result_id={rid}")
            return rid
    except Exception as e:
        print(f"[CHITRAGUPTA] Auto-save skipped: {e}")
    return None


# =========================================================
# TOOLS
# =========================================================

@brahm_tool(
    name        = "vidur_classify",
    group       = "vidur",
    description = (
        "Classify a scientific instrument file using VIDUR. "
        "Auto-detects the characterization technique (XRD, UV-Vis, SEM_EDX, Raman) "
        "and parses the data into a structured format. "
        "Runs fully locally — no cloud, no HTTP. "
        "Returns technique, confidence score, detection signals, and parsed data. "
        "Pass project_id to auto-save the result to brahm.db."
    ),
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "Absolute path to the instrument data file. "
                    "Supported: .xrdml, .raw, .xy, .dat, .asc (XRD); "
                    ".sp, .abs, .spc (UV-Vis); "
                    ".emsa, .msa, .spx (SEM/EDS); "
                    ".wdf, .spc (Raman); "
                    ".pdf, .docx, .csv, .txt (generic)."
                ),
            },
            "project_id": {
                "type": "integer",
                "description": "Link result to a CHITRAGUPTA project (enables auto-save to brahm.db)",
            },
            "cycle_id": {"type": "integer"},
        },
        "required": ["file_path"],
    },
)
async def vidur_classify(args: dict) -> dict:
    file_path  = args.get("file_path", "").strip()
    project_id = args.get("project_id")
    cycle_id   = args.get("cycle_id")

    if not file_path:
        return _err("Missing required argument: file_path")

    def _run() -> dict:
        try:
            import os as _os
            if not _os.path.isfile(file_path):
                return _err(f"File not found: {file_path}")

            from extractor import extract
            data = extract(file_path)

            from auto_detector import detect
            detection = detect(data)

            from router import route
            result = route(detection, data)

            parsed = result.get("parsed_data")
            if parsed:
                for key in ("axis", "intensity"):
                    if key in parsed:
                        parsed[key] = [
                            float(v) for v in parsed[key]
                            if v != "..." and v is not None
                        ]

            technique  = result.get("technique", "Unknown")
            confidence = round(result.get("confidence", 0.0), 4)
            signals    = result.get("signals", [])

            # ── Auto-save to brahm.db ─────────────────────────
            saved_result_id = None
            if project_id and technique != "Unknown":
                saved_result_id = _chit_save_instrument(
                    project_id=project_id,
                    file_path=file_path,
                    technique=technique,
                    confidence=confidence,
                    signals=signals,
                    parsed_data=parsed or {},
                    cycle_id=cycle_id,
                )

            return _ok({
                "technique":       technique,
                "confidence":      confidence,
                "signals":         signals,
                "parsed_data":     parsed,
                "error":           result.get("error"),
                "saved_result_id": saved_result_id,
            })

        except ImportError as exc:
            return _err("VIDUR import failed", str(exc))
        except Exception as exc:
            return _err("VIDUR pipeline error", str(exc))

    result = await asyncio.to_thread(_run)
    if result.get('status') == 'success':
        import asyncio as _aio
        from brahm.shared.http import _chit_store_async
        _aio.ensure_future(_chit_store_async('/v1/store/vidur', {
            'file_path':  args.get('file_path', ''),
            'technique':  result.get('technique', ''),
            'confidence': result.get('confidence', 0.0),
            'signals':    result.get('signals', []),
            'parsed_data': result.get('parsed_data'),
        }))
    return result


@brahm_tool(
    name        = "vidur_list_techniques",
    group       = "vidur",
    description = (
        "List all characterization techniques that VIDUR can detect and parse. "
        "Returns technique names, supported file extensions, and key detection keywords."
    ),
    input_schema = {"type": "object", "properties": {}, "required": []},
)
async def vidur_list_techniques(args: dict) -> dict:
    techniques = [
        {
            "technique":       "XRD",
            "description":     "X-Ray Diffraction — powder/single-crystal patterns",
            "extensions":      [".xrdml", ".raw", ".xy", ".dat", ".asc"],
            "axis":            "2Theta (degrees, 5-90)",
            "strong_keywords": ["2theta", "xrd", "diffraction", "bragg", "d-spacing"],
        },
        {
            "technique":       "UV-Vis",
            "description":     "UV-Visible Spectroscopy — absorbance/transmittance",
            "extensions":      [".sp", ".abs", ".dsp", ".spc", ".csv", ".txt"],
            "axis":            "Wavelength_nm (200-1100 nm)",
            "strong_keywords": ["absorbance", "wavelength", "uv-vis", "transmittance", "nm"],
        },
        {
            "technique":       "SEM_EDX",
            "description":     "Scanning Electron Microscopy / Energy Dispersive X-ray",
            "extensions":      [".emsa", ".msa", ".spx", ".eds", ".spc"],
            "axis":            "Energy_keV (0-20 keV)",
            "strong_keywords": ["keV", "eds", "edx", "sem", "weight %", "atomic %"],
        },
        {
            "technique":       "Raman",
            "description":     "Raman Spectroscopy — vibrational/rotational modes",
            "extensions":      [".wdf", ".spc", ".txt", ".csv", ".dat"],
            "axis":            "RamanShift_cm-1 (100-3500 cm-1)",
            "strong_keywords": ["raman", "cm-1", "wavenumber", "raman shift", "stokes"],
        },
    ]
    return _ok({"count": len(techniques), "techniques": techniques})


@brahm_tool(
    name        = "vidur_health",
    group       = "vidur",
    description = (
        "Check VIDUR health: verify all parser modules load correctly "
        "and core imports (extractor, auto_detector, router) are available. "
        "Returns per-parser status and overall readiness."
    ),
    input_schema = {"type": "object", "properties": {}, "required": []},
)
async def vidur_health(args: dict) -> dict:
    def _check() -> dict:
        results = {}
        overall = True
        for module_name in ("extractor", "auto_detector", "router"):
            try:
                __import__(module_name)
                results[module_name] = "ok"
            except Exception as exc:
                results[module_name] = f"FAILED: {exc}"
                overall = False
        for parser in ("parsers.xrd", "parsers.uvvis", "parsers.sem_eds", "parsers.raman"):
            short = parser.split(".")[-1]
            try:
                __import__(parser)
                results[f"parser:{short}"] = "ok"
            except Exception as exc:
                results[f"parser:{short}"] = f"FAILED: {exc}"
                overall = False
        return _ok({
            "ready":   overall,
            "modules": results,
            "note": (
                "All modules healthy — VIDUR is ready." if overall else
                "One or more modules failed. Check VIDUR path in sys.path."
            ),
        })
    return await asyncio.to_thread(_check)
