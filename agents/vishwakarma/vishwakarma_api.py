"""
vishwakarma_api.py — Vishwakarma FastAPI server
================================================
Run with:
    /mnt/d/brahm/agents/vishwakarma/.venv/bin/python -m uvicorn vishwakarma_api:app --host 0.0.0.0 --port 8004

All /calculate/* endpoints are non-blocking:
  - job is launched in a background thread
  - job_id returned immediately
  - caller polls GET /jobs/{job_id} for status
"""

import sys
import os
import logging
import asyncio
from datetime import datetime
from typing import Optional

sys.path.insert(0, "/mnt/d/brahm/agents/vishwakarma")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vishwakarma import runner, workflow, input_generator, output_parser, pseudo_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vishwakarma.api")

_WORKDIR  = os.environ.get("VISHWAKARMA_WORKDIR",  "/mnt/d/brahm/agents/vishwakarma/jobs")
_BIN_DIR  = os.environ.get("QE_BIN_DIR",           "/usr/bin")
_PSEUDO   = os.environ.get("QE_PSEUDO_DIR",         "/mnt/d/brahm/agents/vishwakarma/pseudo")

app = FastAPI(
    title="Vishwakarma API",
    description="Quantum ESPRESSO DFT calculation engine for BRAHM",
    version="1.0.0",
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class StructureParams(BaseModel):
    structure:   dict
    calc_params: dict
    label:       Optional[str] = "calc"
    mpi_np:      Optional[int] = 1
    timeout:     Optional[int] = 3600

class RelaxParams(BaseModel):
    structure:   dict
    calc_params: dict
    vc_relax:    Optional[bool] = False
    label:       Optional[str] = "relax"
    mpi_np:      Optional[int] = 1
    timeout:     Optional[int] = 7200

class BandsParams(BaseModel):
    structure:   dict
    calc_params: dict
    kpath:       Optional[list] = None
    label:       Optional[str] = "bands"
    mpi_np:      Optional[int] = 1
    timeout:     Optional[int] = 3600

class DosParams(BaseModel):
    structure:   dict
    calc_params: dict
    dense_kmesh: Optional[list] = None
    emin:        Optional[float] = -20.0
    emax:        Optional[float] =  20.0
    label:       Optional[str] = "dos"
    mpi_np:      Optional[int] = 1
    timeout:     Optional[int] = 7200

class PhononParams(BaseModel):
    structure:   dict
    calc_params: dict
    ldisp:       Optional[bool] = True
    nq:          Optional[list] = [4, 4, 4]
    qpoints:     Optional[list] = None
    epsil:       Optional[bool] = True
    lraman:      Optional[bool] = False
    label:       Optional[str] = "phonon"
    mpi_np:      Optional[int] = 1
    timeout:     Optional[int] = 14400

class InputGenParams(BaseModel):
    calc_type:    str
    structure:    dict
    calc_params:  dict
    phonon_params: Optional[dict] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(data: dict) -> dict:
    return {"status": "success", **data}

def _err(msg: str, detail: str = "") -> dict:
    return {"status": "error", "error": msg, "detail": detail}

def _run_in_background(fn, *args, **kwargs):
    """Fire-and-forget: run blocking fn in thread, errors logged not raised."""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, lambda: _safe_run(fn, *args, **kwargs))

def _safe_run(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        log.error("Background job error: %s", exc)

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health():
    binaries = runner.check_binaries(_BIN_DIR)
    found    = sum(1 for v in binaries.values() if v)
    return _ok({
        "agent":      "vishwakarma",
        "bin_dir":    _BIN_DIR,
        "workdir":    _WORKDIR,
        "pseudo_dir": _PSEUDO,
        "binaries_found": f"{found}/{len(binaries)}",
        "binaries":   binaries,
        "timestamp":  datetime.utcnow().isoformat(),
    })

# ── Calculate endpoints ───────────────────────────────────────────────────────

@app.post("/calculate/scf", tags=["Calculate"])
async def calculate_scf(params: StructureParams, background_tasks: BackgroundTasks):
    """Launch SCF calculation. Returns job_id immediately. Poll /jobs/{job_id}."""
    try:
        # Create the real job first so the ID returned to the caller
        # matches the job that background_tasks actually runs.
        job_id = runner.create_job(
            params.label, "pw",
            input_generator.scf(params.structure, params.calc_params),
            _WORKDIR, params.mpi_np,
        )
        background_tasks.add_task(
            workflow.scf_only,
            structure=params.structure,
            calc_params=params.calc_params,
            label=params.label,
            workdir=_WORKDIR,
            bin_dir=_BIN_DIR,
            timeout=params.timeout,
            mpi_np=params.mpi_np,
            job_id=job_id,
        )
        return _ok({"job_id": job_id, "calc_type": "scf", "message": "SCF launched. Poll /jobs/{job_id}."})
    except Exception as exc:
        return _err("Failed to launch SCF", str(exc))


@app.post("/calculate/relax", tags=["Calculate"])
async def calculate_relax(params: RelaxParams, background_tasks: BackgroundTasks):
    """Launch relax or vc-relax. Returns job_id immediately."""
    try:
        job_id = runner.create_job(
            f"{params.label}_relax", "pw",
            input_generator.relax(params.structure, params.calc_params, vc=params.vc_relax),
            _WORKDIR, params.mpi_np,
        )
        background_tasks.add_task(
            workflow.relax_then_scf,
            structure=params.structure,
            calc_params=params.calc_params,
            vc=params.vc_relax,
            label=params.label,
            workdir=_WORKDIR,
            bin_dir=_BIN_DIR,
            timeout=params.timeout,
            mpi_np=params.mpi_np,
            job_id=job_id,
        )
        return _ok({"job_id": job_id, "calc_type": "vc-relax" if params.vc_relax else "relax",
                    "message": "Relax launched. Poll /jobs/{job_id}."})
    except Exception as exc:
        return _err("Failed to launch relax", str(exc))


@app.post("/calculate/bands", tags=["Calculate"])
async def calculate_bands(params: BandsParams, background_tasks: BackgroundTasks):
    """Launch band structure calculation. Returns job_id immediately."""
    try:
        job_id = runner.create_job(
            f"{params.label}_scf", "pw",
            input_generator.scf(params.structure, params.calc_params),
            _WORKDIR, params.mpi_np,
        )
        background_tasks.add_task(
            workflow.band_structure,
            structure=params.structure,
            calc_params=params.calc_params,
            kpath=params.kpath,
            label=params.label,
            workdir=_WORKDIR,
            bin_dir=_BIN_DIR,
            timeout=params.timeout,
            mpi_np=params.mpi_np,
            job_id=job_id,
        )
        return _ok({"job_id": job_id, "calc_type": "bands",
                    "message": "Bands launched. Poll /jobs/{job_id}."})
    except Exception as exc:
        return _err("Failed to launch bands", str(exc))


@app.post("/calculate/dos", tags=["Calculate"])
async def calculate_dos(params: DosParams, background_tasks: BackgroundTasks):
    """Launch DOS calculation. Returns job_id immediately."""
    try:
        job_id = runner.create_job(
            f"{params.label}_scf", "pw",
            input_generator.scf(params.structure, params.calc_params),
            _WORKDIR, params.mpi_np,
        )
        background_tasks.add_task(
            workflow.dos_workflow,
            structure=params.structure,
            calc_params=params.calc_params,
            dense_kmesh=params.dense_kmesh,
            emin=params.emin,
            emax=params.emax,
            label=params.label,
            workdir=_WORKDIR,
            bin_dir=_BIN_DIR,
            timeout=params.timeout,
            mpi_np=params.mpi_np,
            job_id=job_id,
        )
        return _ok({"job_id": job_id, "calc_type": "dos",
                    "message": "DOS launched. Poll /jobs/{job_id}."})
    except Exception as exc:
        return _err("Failed to launch DOS", str(exc))


@app.post("/calculate/phonon", tags=["Calculate"])
async def calculate_phonon(params: PhononParams, background_tasks: BackgroundTasks):
    """Launch phonon calculation. Returns job_id immediately."""
    try:
        # First real job in phonon_workflow is an SCF (type "pw"), not "ph" -
        # fixed the recorded type to match what actually runs first.
        job_id = runner.create_job(
            f"{params.label}_scf", "pw",
            input_generator.scf(params.structure, params.calc_params),
            _WORKDIR, params.mpi_np,
        )
        background_tasks.add_task(
            workflow.phonon_workflow,
            structure=params.structure,
            calc_params=params.calc_params,
            qpoints=params.qpoints,
            ldisp=params.ldisp,
            nq=tuple(params.nq),
            epsil=params.epsil,
            label=params.label,
            workdir=_WORKDIR,
            bin_dir=_BIN_DIR,
            timeout=params.timeout,
            mpi_np=params.mpi_np,
            job_id=job_id,
        )
        return _ok({"job_id": job_id, "calc_type": "phonon",
                    "message": "Phonon launched. Poll /jobs/{job_id}."})
    except Exception as exc:
        return _err("Failed to launch phonon", str(exc))

# ── Input generation ──────────────────────────────────────────────────────────

@app.post("/generate/input", tags=["Generate"])
async def generate_input(params: InputGenParams):
    """Generate QE input file without running. Returns input text for review."""
    ct  = params.calc_type
    s   = params.structure
    cp  = params.calc_params
    ph  = params.phonon_params or {}
    try:
        if ct == "scf":
            text = input_generator.scf(s, cp)
        elif ct == "nscf":
            text = input_generator.nscf(s, cp)
        elif ct == "relax":
            text = input_generator.relax(s, cp, vc=False)
        elif ct == "vc-relax":
            text = input_generator.relax(s, cp, vc=True)
        elif ct == "bands":
            text = input_generator.bands(s, cp)
        elif ct == "dos":
            text = input_generator.dos(s.get("prefix","pwscf"), cp.get("outdir","./out"))
        elif ct == "phonon":
            text = input_generator.phonon(
                s.get("prefix","pwscf"), cp.get("outdir","./out"),
                ldisp=ph.get("ldisp", False), nq=tuple(ph.get("nq",[4,4,4])),
                epsil=ph.get("epsil", False), lraman=ph.get("lraman", False),
            )
        else:
            return _err(f"Unknown calc_type: {ct}")
        return _ok({"calc_type": ct, "input_text": text, "line_count": text.count("\n")})
    except Exception as exc:
        return _err(f"Input generation failed for {ct}", str(exc))

# ── Jobs ──────────────────────────────────────────────────────────────────────

@app.get("/jobs", tags=["Jobs"])
async def list_jobs(status: Optional[str] = None, limit: int = 20):
    """List all jobs. Optional ?status=running|completed|failed|created|timeout"""
    try:
        jobs = runner.list_jobs(workdir=_WORKDIR, limit=limit,
                                status_filter=status if status != "all" else None)
        return _ok({"count": len(jobs), "jobs": jobs})
    except Exception as exc:
        return _err("Failed to list jobs", str(exc))


@app.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job(job_id: str):
    """Get status of a specific job."""
    try:
        status = runner.get_job_status(job_id, _WORKDIR)
        return _ok(status)
    except Exception as exc:
        return _err(f"Job {job_id} not found", str(exc))


@app.get("/jobs/{job_id}/output", tags=["Jobs"])
async def get_job_output(job_id: str, code: str = "pw"):
    """Get parsed output of a completed job."""
    try:
        raw    = runner.get_output(job_id, _WORKDIR)
        if not raw:
            return _err("Output file empty or not found")
        parsed = output_parser.parse(raw, code)
        return _ok({"job_id": job_id, "code": code, "parsed": parsed})
    except Exception as exc:
        return _err("Failed to parse output", str(exc))

# ── Pseudopotentials ──────────────────────────────────────────────────────────

@app.get("/pseudopotentials", tags=["Pseudopotentials"])
async def list_pseudopotentials(functional: str = "pbe", type: str = "us"):
    """Discover UPF files in QE_PSEUDO_DIR."""
    try:
        pseudos = pseudo_manager.discover([_PSEUDO])
        return _ok({"pseudo_dir": _PSEUDO, "total": len(pseudos),
                    "pseudopotentials": pseudos[:100]})
    except Exception as exc:
        return _err("Failed to list pseudopotentials", str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8004))
    log.info("Vishwakarma API starting on port %d", port)
    uvicorn.run("vishwakarma_api:app", host="0.0.0.0", port=port, log_level="info")
