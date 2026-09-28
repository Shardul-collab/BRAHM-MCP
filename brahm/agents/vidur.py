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
import logging

from brahm.brahm_registry import brahm_tool
from brahm.shared.helpers import _ok, _err

# mcp_server.py speaks JSON-RPC over stdio. Anything print()ed from inside a
# tool goes into that same stream and corrupts the protocol frame, so every
# diagnostic in this module logs to stderr instead. (2026-09-20)
log = logging.getLogger("brahm.vidur")

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
) -> tuple[int | None, str | None]:
    """
    POST /v1/results/instrument — persist a VIDUR result to brahm.db.
    Returns (result_id, error). Never raises.

    2026-09-19: used to return just an id, so a caller could not tell a failed
    save from one that was never attempted. Both stores held 0 rows and nothing
    in the result said so.
    """
    try:
        import requests
        from brahm.shared.http import _chit_headers
        # 2026-09-20: this call sent no X-API-Key. It worked only because
        # Chitragupta mounted brahm_db_router without an auth dependency --
        # the whole Projects/Papers/Results/Documents surface was reachable
        # unauthenticated on a server bound to 0.0.0.0. That gap is now closed
        # (agents/chitragupta/api/app.py), so the header is required.
        r = requests.post(
            f"{CHITRAGUPTA_BASE}/v1/results/instrument",
            headers=_chit_headers(),
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
            log.info("Instrument result saved: result_id=%s", rid)
            return rid, None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        log.warning("Instrument auto-save skipped: %s", e)
        return None, f"{type(e).__name__}: {e}"


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
            "include_data": {
                "type": "boolean",
                "description": ("Return the full axis/intensity arrays. Off by "
                                "default: a real scan is tens of thousands of "
                                "numbers. Use vidur_process for the data itself."),
            },
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
            persist_error = None
            if not project_id:
                persist_error = ("not attempted: no project_id was passed, so the "
                                 "result was not written to brahm.db")
            elif technique == "Unknown":
                persist_error = "not attempted: technique is Unknown"
            else:
                saved_result_id, persist_error = _chit_save_instrument(
                    project_id=project_id,
                    file_path=file_path,
                    technique=technique,
                    confidence=confidence,
                    signals=signals,
                    parsed_data=parsed or {},
                    cycle_id=cycle_id,
                )

            # 2026-09-19: this returned the full axis and intensity arrays. A
            # 1,800-point scan is ~53 KB of JSON, and VIDUR's caller is a model,
            # so the tool was unusable on a real file -- measured through the
            # live connector. Summarise by default; the arrays are available on
            # request and the CSV is the real delivery anyway.
            summary = None
            if parsed:
                ax, iy = parsed.get("axis") or [], parsed.get("intensity") or []
                summary = {
                    "axis_name": parsed.get("axis_name"),
                    "points": len(ax),
                    "axis_range": [ax[0], ax[-1]] if ax else None,
                    "intensity_range": [min(iy), max(iy)] if iy else None,
                    "metadata": parsed.get("metadata"),
                }
                if not args.get("include_data"):
                    parsed = summary

            return _ok({
                "technique":       technique,
                "confidence":      confidence,
                "signals":         signals,
                "parsed_data":     parsed,
                "summary":         summary,
                "error":           result.get("error"),
                "saved_result_id": saved_result_id,
                "persisted":       saved_result_id is not None,
                "persist_error":   persist_error,
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
        {
            "technique":       "IV / IT",
            "description":     ("Source-meter tables (Keithley, EC-Lab) — an I-V sweep "
                                "or a time trace. Which one is decided from the data: a "
                                "sweeping source column is IV, a flat one while time "
                                "advances is IT."),
            "extensions":      [".csv", ".dat", ".txt", ".mpt"],
            "axis":            "Voltage_V or Time_s (read from the file's own unit columns)",
            "strong_keywords": ["amp dc", "volt dc", "relative time", "ec-lab", "keithley"],
        },
    ]
    # 2026-09-20: this list was hardcoded and still said 4 techniques the day
    # after `sourcemeter` shipped -- the same drift that made vidur_health report
    # "all modules healthy" without checking it. Assert it covers what the router
    # actually registers, so the next parser cannot be silently omitted.
    try:
        from router import _get_parser_map
        registered = {m.__name__.split(".")[-1] for m in _get_parser_map().values()}
        listed = {"xrd", "uvvis", "sem_eds", "raman", "sourcemeter"}
        missing = registered - listed
    except Exception as exc:
        missing, registered = set(), f"unavailable ({type(exc).__name__})"
    return _ok({"count": len(techniques), "techniques": techniques,
                "parsers_registered": sorted(registered) if isinstance(registered, set) else registered,
                "undocumented_parsers": sorted(missing) or None})


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
        # 2026-09-19: this list was hardcoded and silently omitted `sourcemeter`
        # the day it was added, so health reported "all modules healthy" while
        # not checking a parser that was live. Ask the router what it registers.
        from router import _get_parser_map
        for short in sorted({m.__name__.split(".")[-1]
                             for m in _get_parser_map().values()}):
            parser = f"parsers.{short}"
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


# =========================================================
# BATCH: folder -> plan -> plot-ready CSVs   (2026-09-19)
# =========================================================

@brahm_tool(
    name        = "vidur_scan",
    group       = "vidur",
    description = (
        "Walk an input folder and return a PLAN: which file is which technique, "
        "how the filenames group into series and samples, and an explicit list of "
        "what VIDUR could not resolve. Writes nothing. Answer the questions it "
        "returns, then call vidur_process. Runs fully locally."
    ),
    input_schema = {
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "Absolute path to the input folder"},
        },
        "required": ["folder"],
    },
)
async def vidur_scan(args: dict) -> dict:
    folder = (args.get("folder") or "").strip()
    if not folder:
        return _err("Missing required argument: folder")

    def _run() -> dict:
        try:
            import batch
            return _ok(batch.scan(folder))
        except NotADirectoryError:
            return _err(f"Not a directory: {folder}")
        except ImportError as exc:
            return _err("VIDUR import failed", str(exc))
        except Exception as exc:
            return _err("VIDUR scan failed", f"{type(exc).__name__}: {exc}")

    return await asyncio.to_thread(_run)


@brahm_tool(
    name        = "vidur_process",
    group       = "vidur",
    description = (
        "Write plot-ready CSVs for every file in the folder VIDUR could resolve. "
        "One CSV per file (raw column plus derived columns), and a wide CSV per "
        "series where the sample x-axes actually match. The raw data is never "
        "filtered or altered; every transformation is an extra column. Call "
        "vidur_scan first and pass its questions back as answers."
    ),
    input_schema = {
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "Absolute path to the input folder"},
            "out_dir": {"type": "string", "description": "Defaults to <folder>/vidur_out"},
            "techniques": {
                "type": "object",
                "description": "Override a guess: {\"file.csv\": \"XRD\"}",
            },
            "params": {
                "type": "object",
                "description": (
                    "Constants VIDUR will not assume: wavelength_a (XRD d/q columns), "
                    "thickness_cm and y_kind (UV-Vis alpha/Tauc), reference_band_cm1 (Raman)."
                ),
            },
            "file_params": {
                "type": "object",
                "description": "Per-file overrides of params, keyed by filename",
            },
        },
        "required": ["folder"],
    },
)
async def vidur_process(args: dict) -> dict:
    folder = (args.get("folder") or "").strip()
    if not folder:
        return _err("Missing required argument: folder")
    answers = {
        "techniques":  args.get("techniques") or {},
        "params":      args.get("params") or {},
        "file_params": args.get("file_params") or {},
    }

    def _run() -> dict:
        try:
            import batch
            return _ok(batch.process(folder, answers, args.get("out_dir")))
        except NotADirectoryError:
            return _err(f"Not a directory: {folder}")
        except ImportError as exc:
            return _err("VIDUR import failed", str(exc))
        except Exception as exc:
            return _err("VIDUR processing failed", f"{type(exc).__name__}: {exc}")

    return await asyncio.to_thread(_run)


# =========================================================
# XRD LINE-PROFILE ANALYSIS   (2026-09-25)
# Formulas and limits: agents/vidur/XRD-LINE-PROFILE-METHODS.md
# =========================================================

def _compact_peak(p: dict) -> dict:
    """One reflection, only what a model needs to read. The full table is the
    CSV; a 30-peak report with every field is too large to hand back."""
    keep = ("two_theta", "status", "reason", "d_A", "meas_fwhm_deg",
            "sample_fwhm_deg", "D_scherrer_nm", "D_scherrer_err",
            "D_intbreadth_nm", "D_voigt_nm", "D_voigt_err", "strain_voigt",
            "strain_voigt_err", "delta_scherrer_m2", "chi2_red_poisson", "flags")
    out = {}
    for k in keep:
        v = p.get(k)
        if v is None or v == []:
            continue
        out[k] = float(f"{v:.5g}") if isinstance(v, float) else v
    return out


@brahm_tool(
    name        = "vidur_xrd_profile",
    group       = "vidur",
    description = (
        "XRD line-profile analysis of one scan: peak finding, Voigt fitting "
        "(Kalpha2 doublet modelled when the file or caller says the beam has it), "
        "instrumental-broadening correction, Scherrer and single-line Voigt "
        "crystallite size and microstrain per peak, Williamson-Hall and "
        "size-strain plot across peaks, and dislocation density delta = 1/D^2. "
        "Nothing is assumed: wavelength, Kalpha2 ratio and instrument function come "
        "from the file or params, and what is missing comes back as questions. "
        "Formulas and limits: agents/vidur/XRD-LINE-PROFILE-METHODS.md. "
        "Runs fully locally."
    ),
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string",
                          "description": "XRD scan (.xrdml, .xy, .xye, .dat, Bruker RAW v3)"},
            "out_dir": {"type": "string",
                        "description": "Write <stem>_peaks.csv, _fit.csv, _microstructure.json here"},
            "params": {
                "type": "object",
                "description": (
                    "wavelength_a, ka2_ratio (0 = Ka1 only), wavelength_ka2_a -- only "
                    "when the file does not state them; instrument: {standard_file: "
                    "LaB6/Si scan on the same optics} or {fwhm_gauss_deg, "
                    "fwhm_lorentz_deg}; scherrer_k (0.9); ssp_k (1.0); burgers_nm; "
                    "two_theta_range [lo, hi]; peaks [2theta,...] to fit only these; "
                    "wh_peaks [2theta,...] to restrict the W-H/SSP fits; min_snr; "
                    "false_alarm (0.01)."),
            },
            "include_all_peaks": {"type": "boolean",
                                  "description": "Also list rejected candidates (noise, Ka2 partners)"},
        },
        "required": ["file_path"],
    },
)
async def vidur_xrd_profile(args: dict) -> dict:
    file_path = (args.get("file_path") or "").strip()
    if not file_path:
        return _err("Missing required argument: file_path")

    def _run() -> dict:
        try:
            import os as _os
            if not _os.path.isfile(file_path):
                return _err(f"File not found: {file_path}")
            import line_profile
            rep = line_profile.run(file_path, args.get("params") or {}, args.get("out_dir"))
            if rep.get("status") != "ok":
                return _ok(rep)            # needs_input: questions travel back
            from collections import Counter
            fitted_states = ("ok", "suspect", "unresolved")
            peaks = rep["peaks"] if args.get("include_all_peaks") else \
                [p for p in rep["peaks"] if p.get("status") in fitted_states]
            return _ok({
                "file": file_path,
                "status": "ok",
                "method_reference": rep["method_reference"],
                "radiation": rep["radiation"],
                "instrument": rep["instrument"],
                "instrument_corrected": rep["instrument_corrected"],
                "noise": rep["noise"],
                "candidates_by_status": dict(Counter(p["status"] for p in rep["peaks"])),
                "peaks": [_compact_peak(p) for p in peaks],
                "williamson_hall": {k: v for k, v in rep["williamson_hall"].items()
                                    if k != "points"},
                "williamson_hall_integral": {k: v for k, v in rep["williamson_hall_integral"].items()
                                             if k != "points"},
                "size_strain_plot": rep["size_strain_plot"],
                "summary": rep["summary"],
                "caveats": rep["caveats"],
                "questions": rep.get("questions"),
                "written": rep.get("written"),
            })
        except ImportError as exc:
            return _err("VIDUR import failed", str(exc))
        except ValueError as exc:
            return _err("VIDUR line-profile analysis refused", str(exc))
        except Exception as exc:
            return _err("VIDUR line-profile analysis failed", f"{type(exc).__name__}: {exc}")

    return await asyncio.to_thread(_run)
