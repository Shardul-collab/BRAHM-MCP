"""
brahm/agents/vishwakarma.py
============================
Group H — Vishwakarma Quantum ESPRESSO DFT tools.
All calculations run locally via subprocess — no internet.

Auto-save: every run_* call routes through _persist_run(), which writes the
result to brahm.db via POST /v1/results/dft (CHITRAGUPTA API on :8003) with
its REAL status, and then fires the /v1/store/vishwakarma call for runs that
actually completed. project_id is optional — pass it in args to link the
result to a project. If CHITRAGUPTA is down, the save is silently skipped —
never blocks calculation.

Before 2026-09-09 each tool duplicated this block inline and all seven were
broken: they read job_id/converged off the top level of a workflow result
(where those keys do not exist), hardcoded status="completed" regardless of
outcome, and guarded the store call on a marker that _ok() had already
overwritten. See _persist_run's docstring for the full account.
"""

import asyncio
import logging
import os
import time

from brahm.brahm_registry import brahm_tool
from brahm.shared.helpers import _ok, _err
from brahm.shared.constants import QE_WORKDIR, QE_BIN_DIR, QE_PSEUDO

CALC_TYPES = ["scf","nscf","relax","vc-relax","bands","phonon","dos","projwfc","pp","neb","hp","cp"]
STRUCTURE_DESC = (
    "Crystal structure dict: prefix, ibrav, cell_parameters (3x3 A), "
    "nat, ntyp, atomic_species [{symbol,mass,pseudo}], "
    "atomic_positions [{symbol,x,y,z}], kpoints {mode,mesh,shift}."
)
CALC_PARAMS_DESC = (
    "Calculation parameters: ecutwfc, ecutrho, occupations, smearing, "
    "degauss, conv_thr, pseudo_dir, outdir, nspin, nbnd, hubbard_u, etc."
)

CHITRAGUPTA_BASE    = "http://localhost:8003"

# mcp_server.py speaks JSON-RPC over stdio; a print() from inside a tool lands
# in that stream and corrupts the frame. Diagnostics go to stderr. (2026-09-20)
log = logging.getLogger("brahm.vishwakarma")
CHITRAGUPTA_TIMEOUT = 5   # never block a calculation on this


# =========================================================
# CHITRAGUPTA AUTO-SAVE HELPER
# =========================================================

def _chit_save_dft(
    project_id: int | None,
    job_id: str,
    calc_type: str,
    structure: dict | None,
    calc_params: dict | None,
    output_parsed: dict | None,
    status: str,
    wall_time_seconds: float | None,
    cycle_id: int | None,
) -> tuple[int | None, str | None]:
    """
    POST /v1/results/dft — persist a completed QE result to brahm.db.
    Returns (result_id, error). Never raises.

    2026-09-20: this returned a bare id and swallowed every failure mode into
    None -- no project_id, Chitragupta down, and a 500 from the write were
    indistinguishable, and the only trace was a print() that went into the
    stdio MCP transport. The caller of these tools is a model with nothing
    else to go on, so the reason now comes back with the id. VIDUR's
    _chit_save_instrument has reported this way since 2026-09-19; this is the
    same contract.
    """
    if project_id is None:
        return None, ("not attempted: no project_id was passed, so the result "
                      "was not linked to a project and nothing was stored")
    try:
        import requests
        from brahm.shared.http import _chit_headers
        # 2026-09-20: this call sent no X-API-Key. It worked only because
        # Chitragupta mounted brahm_db_router without an auth dependency --
        # the whole Projects/Papers/Results/Documents surface was reachable
        # unauthenticated on a server bound to 0.0.0.0. That gap is now closed
        # (agents/chitragupta/api/app.py), so the header is required.
        r = requests.post(
            f"{CHITRAGUPTA_BASE}/v1/results/dft",
            headers=_chit_headers(),
            json={
                "project_id":        project_id,
                "job_id":            job_id,
                "calc_type":         calc_type,
                "structure":         structure,
                "input_params":      calc_params,
                "output_parsed":     output_parsed,
                "status":            status,
                "wall_time_seconds": wall_time_seconds,
                "cycle_id":          cycle_id,
            },
            timeout=CHITRAGUPTA_TIMEOUT,
        )
        if r.status_code == 200:
            rid = r.json().get("result_id")
            log.info("DFT result saved: result_id=%s", rid)
            return rid, None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        log.warning("DFT auto-save skipped: %s", e)
        return None, f"{type(e).__name__}: {e}"


def _run_summary(result: dict) -> dict:
    """
    Pull the flat summary fields out of a workflow.py result.

    workflow._workflow_result() nests everything under "steps" and (since
    2026-09-09) also exposes a precomputed "summary". Older callers read
    job_id/converged/scf_iterations straight off the top level, where those
    keys have never existed — so every record this module wrote to brahm.db
    carried an empty job_id and null convergence. The fallback branch below
    handles the ad-hoc single-job dicts that the NEB/HP handlers used to
    build by hand, so both shapes flatten the same way.
    """
    if not isinstance(result, dict):
        return {"job_id": "", "converged": None,
                "scf_iterations": None, "total_energy_ev": None}

    summary = result.get("summary")
    if isinstance(summary, dict):
        return summary

    parsed = result.get("parsed")
    parsed = parsed if isinstance(parsed, dict) else {}
    return {
        "job_id":          result.get("job_id", ""),
        "converged":       parsed.get("converged"),
        "scf_iterations":  parsed.get("scf_iterations"),
        "total_energy_ev": parsed.get("total_energy_ev"),
    }


def _run_status(result: dict) -> str:
    """
    Derive the persisted status from what actually happened.

    Every run_* tool used to pass a hardcoded status="completed" to
    _chit_save_dft, so a workflow that returned success=False with
    failed_at="scf" was still recorded in brahm.db as a completed
    calculation — failed and successful runs were indistinguishable once
    stored. Read the real flag instead.
    """
    if not isinstance(result, dict):
        return "failed"
    if "success" in result:
        return "completed" if result["success"] else "failed"
    # Ad-hoc single-job shape: status is runner.run_job()'s dict, not a str.
    status = result.get("status")
    if isinstance(status, dict):
        return "completed" if status.get("status") == "completed" else "failed"
    return "completed" if status == "completed" else "failed"


async def _persist_run(
    calc_type: str,
    args: dict,
    result: dict,
    wall_time_seconds: float | None,
    material_name: str = "",
) -> None:
    """
    Single persistence path for every run_* tool: brahm.db via
    _chit_save_dft, plus the fire-and-forget /v1/store/vishwakarma call.

    Consolidated 2026-09-09. This logic was previously duplicated across
    all seven run_* tools with three independent defects — see _run_summary
    and _run_status above, plus the guard bug: the tail check was
    `if result.get('status') == 'success'`, but _ok() merges the payload
    over its own {"status": "success"} marker. The five workflow tools
    happened to pass a payload with no "status" key so the marker survived
    and the guard fired; run_neb and run_hp passed runner.run_job()'s status
    DICT under that same key, which overwrote the marker with a dict, so
    their guard compared a dict to "success" and never fired at all. Their
    store call was dead code for as long as it existed.

    Never raises — a persistence failure must not fail a calculation that
    already ran.
    """
    summary = _run_summary(result)
    status  = _run_status(result)

    saved_id: int | None = None
    persist_error: str | None = None
    try:
        saved_id, persist_error = _chit_save_dft(
            project_id=args.get("project_id"),
            job_id=summary.get("job_id", ""),
            calc_type=calc_type,
            structure=args.get("structure") or args.get("initial_structure"),
            calc_params=args.get("calc_params"),
            output_parsed=result,
            status=status,
            wall_time_seconds=wall_time_seconds,
            cycle_id=args.get("cycle_id"),
        )
    except Exception as exc:
        log.warning("DFT save skipped: %s", exc)
        persist_error = f"{type(exc).__name__}: {exc}"

    # 2026-09-20: _persist_run returned None and told the caller nothing, so a
    # run_* tool reported a successful calculation whether or not a single row
    # had been written. brahm.db's DFTResult table held 0 rows and no tool
    # output had ever said so. Say it on the result itself.
    if isinstance(result, dict):
        result["custody"] = {
            "stored":     saved_id is not None,
            "result_id":  saved_id,
            "project_id": args.get("project_id"),
            "error":      persist_error,
        }

    if status != "completed":
        return

    try:
        from brahm.shared.http import _chit_store_async
        asyncio.ensure_future(_chit_store_async('/v1/store/vishwakarma', {
            'calculation_type': calc_type,
            'material_name':    material_name,
            'output_file_path': summary.get('job_id', ''),
            'scf_iterations':   summary.get('scf_iterations'),
            'converged':        summary.get('converged'),
            'job_id':           summary.get('job_id', ''),
        }))
    except Exception as exc:
        log.warning("store skipped: %s", exc)


def _material_of(args: dict, key: str = "structure") -> str:
    src = args.get(key)
    return src.get("prefix", "") if isinstance(src, dict) else ""


# =========================================================
# TOOLS
# =========================================================

@brahm_tool(
    name="vishwakarma_health", group="vishwakarma",
    description=(
        "Check Vishwakarma health: verify QE binaries (pw.x, ph.x, pp.x, "
        "dos.x, bands.x, neb.x) are reachable and all Python modules import correctly."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
)
async def vishwakarma_health(args: dict) -> dict:
    def _check() -> dict:
        try:
            from vishwakarma import runner as _r
            from vishwakarma import input_generator  # noqa: F401
            from vishwakarma import output_parser    # noqa: F401
            from vishwakarma import pseudo_manager   # noqa: F401
            from vishwakarma import workflow         # noqa: F401
        except ImportError as exc:
            return _err("Vishwakarma modules failed to import", str(exc))
        binaries  = _r.check_binaries(QE_BIN_DIR)
        any_found = any(v is not None for v in binaries.values())
        return _ok({
            "ready":      any_found,
            "bin_dir":    QE_BIN_DIR,
            "workdir":    QE_WORKDIR,
            "pseudo_dir": QE_PSEUDO,
            "binaries":   binaries,
            "note": (
                f"{sum(1 for v in binaries.values() if v)}/{len(binaries)} QE binaries found."
                if any_found else
                "Set QE_BIN_DIR env var if binaries are in a non-standard location."
            ),
        })
    return await asyncio.to_thread(_check)


@brahm_tool(
    name="vishwakarma_generate_input", group="vishwakarma",
    description=(
        "Generate a Quantum ESPRESSO input file without running it. "
        "Returns the input file as a string for review before execution."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "calc_type":     {"type": "string", "enum": CALC_TYPES, "description": "Calculation to generate input for: scf, relax, bands, dos, phonon, neb or hp."},
            "structure":     {"type": "object", "description": STRUCTURE_DESC},
            "calc_params":   {"type": "object", "description": CALC_PARAMS_DESC},
            "phonon_params": {"type": "object", "description": "Extra params for phonon/dos/pp/hp"},
        },
        "required": ["calc_type", "structure", "calc_params"],
    },
)
async def vishwakarma_generate_input(args: dict) -> dict:
    def _gen() -> dict:
        try:
            from vishwakarma import input_generator as ig
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        calc_type   = args.get("calc_type", "scf")
        structure   = args.get("structure", {})
        calc_params = args.get("calc_params", {})
        ph_params   = args.get("phonon_params", {})
        try:
            if calc_type == "scf":
                text = ig.scf(structure, calc_params)
            elif calc_type == "nscf":
                text = ig.nscf(structure, calc_params)
            elif calc_type == "relax":
                text = ig.relax(structure, calc_params, vc=False)
            elif calc_type == "vc-relax":
                text = ig.relax(structure, calc_params, vc=True)
            elif calc_type == "bands":
                text = ig.bands(structure, calc_params)
            elif calc_type == "dos":
                text = ig.dos(structure.get("prefix","pwscf"),
                              calc_params.get("outdir","./out"),
                              **{k: ph_params[k] for k in
                                 ("emin","emax","deltaE","fildos") if k in ph_params})
            elif calc_type == "projwfc":
                text = ig.projwfc(structure.get("prefix","pwscf"),
                                  calc_params.get("outdir","./out"))
            elif calc_type == "pp":
                text = ig.pp(structure.get("prefix","pwscf"),
                             calc_params.get("outdir","./out"),
                             plot_num=ph_params.get("plot_num", 0),
                             fileout=ph_params.get("fileout", "charge.xsf"))
            elif calc_type == "phonon":
                text = ig.phonon(structure.get("prefix","pwscf"),
                                 calc_params.get("outdir","./out"),
                                 qpoints=ph_params.get("qpoints"),
                                 ldisp=ph_params.get("ldisp", False),
                                 nq=tuple(ph_params.get("nq", [4,4,4])),
                                 epsil=ph_params.get("epsil", False),
                                 lraman=ph_params.get("lraman", False),
                                 recover=ph_params.get("recover", False),
                                 fildyn=ph_params.get("fildyn", "dyn"),
                                 qplot=ph_params.get("qplot"))
            elif calc_type == "hp":
                text = ig.hp(structure.get("prefix","pwscf"),
                             calc_params.get("outdir","./out"),
                             nq=tuple(ph_params.get("nq", [2,2,2])))
            elif calc_type == "cp":
                text = ig.cp(structure, calc_params)
            else:
                return _err(f"Unknown calc_type: {calc_type}")
            return _ok({"calc_type": calc_type, "input_text": text,
                        "line_count": text.count("\n")})
        except Exception as exc:
            return _err(f"Input generation failed for {calc_type}", str(exc))
    return await asyncio.to_thread(_gen)


@brahm_tool(
    name="vishwakarma_run_scf", group="vishwakarma",
    description=(
        "Run a pw.x SCF calculation. "
        "Returns job_id, convergence status, total energy, Fermi energy, band gap. "
        "Pass project_id to auto-save result to brahm.db."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "structure":   {"type": "object", "description": "Atomic structure: cell vectors, species and positions, in the form generate_input expects."},
            "calc_params": {"type": "object", "description": "Quantum ESPRESSO parameters — cutoffs, k-points, smearing, pseudopotentials. See vishwakarma_list_pseudopotentials for what is installed."},
            "label":       {"type": "string", "default": "scf"},
            "mpi_np":      {"type": "integer", "default": 1},
            "timeout":     {"type": "integer", "default": 3600},
            "project_id":  {"type": "integer", "description": "Link result to a CHITRAGUPTA project"},
            "cycle_id":    {"type": "integer"},
        },
        "required": ["structure", "calc_params"],
    },
)
async def vishwakarma_run_scf(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import workflow as wf
            result = wf.scf_only(
                structure=args["structure"], calc_params=args["calc_params"],
                label=args.get("label","scf"), workdir=QE_WORKDIR,
                bin_dir=QE_BIN_DIR, timeout=args.get("timeout",3600),
                mpi_np=args.get("mpi_np",1),
            )
            return _ok(result)
        except Exception as exc:
            return _err("SCF calculation failed", str(exc))
    _t0 = time.time()
    result = await asyncio.to_thread(_run)
    await _persist_run('scf', args, result,
                       round(time.time() - _t0, 1),
                       material_name=_material_of(args))
    return result


@brahm_tool(
    name="vishwakarma_run_relax", group="vishwakarma",
    description=(
        "Run ionic relaxation (relax or vc-relax) followed by final SCF. "
        "Pass project_id to auto-save result to brahm.db."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "structure":   {"type": "object", "description": "Atomic structure: cell vectors, species and positions, in the form generate_input expects."},
            "calc_params": {"type": "object", "description": "Quantum ESPRESSO parameters — cutoffs, k-points, smearing, pseudopotentials. See vishwakarma_list_pseudopotentials for what is installed."},
            "vc_relax":    {"type": "boolean", "default": False},
            "label":       {"type": "string", "default": "relax"},
            "mpi_np":      {"type": "integer", "default": 1},
            "timeout":     {"type": "integer", "default": 7200},
            "project_id":  {"type": "integer"},
            "cycle_id":    {"type": "integer"},
        },
        "required": ["structure", "calc_params"],
    },
)
async def vishwakarma_run_relax(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import workflow as wf
            result = wf.relax_then_scf(
                structure=args["structure"], calc_params=args["calc_params"],
                vc=args.get("vc_relax",False), label=args.get("label","relax"),
                workdir=QE_WORKDIR, bin_dir=QE_BIN_DIR,
                timeout=args.get("timeout",7200), mpi_np=args.get("mpi_np",1),
            )
            calc_type = "vc-relax" if args.get("vc_relax") else "relax"
            return _ok(result)
        except Exception as exc:
            return _err("Relaxation failed", str(exc))
    _t0 = time.time()
    result = await asyncio.to_thread(_run)
    await _persist_run('relax', args, result,
                       round(time.time() - _t0, 1),
                       material_name=_material_of(args))
    return result


@brahm_tool(
    name="vishwakarma_run_bands", group="vishwakarma",
    description=(
        "Run band structure: SCF -> NSCF on k-path -> bands.x post-processing. "
        "Pass project_id to auto-save result to brahm.db."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "structure":   {"type": "object", "description": "Atomic structure: cell vectors, species and positions, in the form generate_input expects."},
            "calc_params": {"type": "object", "description": "Quantum ESPRESSO parameters — cutoffs, k-points, smearing, pseudopotentials. See vishwakarma_list_pseudopotentials for what is installed."},
            "kpath":       {"type": "array", "items": {"type": "array"}},
            "label":       {"type": "string", "default": "bands"},
            "mpi_np":      {"type": "integer", "default": 1},
            "timeout":     {"type": "integer", "default": 3600},
            "project_id":  {"type": "integer"},
            "cycle_id":    {"type": "integer"},
        },
        "required": ["structure", "calc_params"],
    },
)
async def vishwakarma_run_bands(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import workflow as wf
            result = wf.band_structure(
                structure=args["structure"], calc_params=args["calc_params"],
                kpath=args.get("kpath"), label=args.get("label","bands"),
                workdir=QE_WORKDIR, bin_dir=QE_BIN_DIR,
                timeout=args.get("timeout",3600), mpi_np=args.get("mpi_np",1),
            )
            return _ok(result)
        except Exception as exc:
            return _err("Band structure calculation failed", str(exc))
    _t0 = time.time()
    result = await asyncio.to_thread(_run)
    await _persist_run('bands', args, result,
                       round(time.time() - _t0, 1),
                       material_name=_material_of(args))
    return result


@brahm_tool(
    name="vishwakarma_run_dos", group="vishwakarma",
    description=(
        "Run density of states: SCF -> dense NSCF -> dos.x. "
        "Pass project_id to auto-save result to brahm.db."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "structure":   {"type": "object", "description": "Atomic structure: cell vectors, species and positions, in the form generate_input expects."},
            "calc_params": {"type": "object", "description": "Quantum ESPRESSO parameters — cutoffs, k-points, smearing, pseudopotentials. See vishwakarma_list_pseudopotentials for what is installed."},
            "dense_kmesh": {"type": "array"},
            "emin":        {"type": "number", "default": -20.0},
            "emax":        {"type": "number", "default":  20.0},
            "label":       {"type": "string", "default": "dos"},
            "mpi_np":      {"type": "integer", "default": 1},
            "timeout":     {"type": "integer", "default": 7200},
            "project_id":  {"type": "integer"},
            "cycle_id":    {"type": "integer"},
        },
        "required": ["structure", "calc_params"],
    },
)
async def vishwakarma_run_dos(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import workflow as wf
            result = wf.dos_workflow(
                structure=args["structure"], calc_params=args["calc_params"],
                dense_kmesh=args.get("dense_kmesh"), emin=args.get("emin",-20.0),
                emax=args.get("emax",20.0), label=args.get("label","dos"),
                workdir=QE_WORKDIR, bin_dir=QE_BIN_DIR,
                timeout=args.get("timeout",7200), mpi_np=args.get("mpi_np",1),
            )
            return _ok(result)
        except Exception as exc:
            return _err("DOS calculation failed", str(exc))
    _t0 = time.time()
    result = await asyncio.to_thread(_run)
    await _persist_run('dos', args, result,
                       round(time.time() - _t0, 1),
                       material_name=_material_of(args))
    return result


@brahm_tool(
    name="vishwakarma_run_phonon", group="vishwakarma",
    description=(
        "Run DFPT phonon calculation: SCF -> ph.x. "
        "Can compute dielectric tensor + Born charges (epsil=true). "
        "Pass project_id to auto-save result to brahm.db. "
        "To resume a previously interrupted/timed-out run instead of restarting "
        "from scratch, pass recover=true plus existing_scf_job_id (the job_id "
        "of the SCF step from the original run) -- ph.x will pick up from its "
        "own .recover scratch file instead of redoing already-converged q-points."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "structure":   {"type": "object", "description": "Atomic structure: cell vectors, species and positions, in the form generate_input expects."},
            "calc_params": {"type": "object", "description": "Quantum ESPRESSO parameters — cutoffs, k-points, smearing, pseudopotentials. See vishwakarma_list_pseudopotentials for what is installed."},
            "ldisp":   {"type": "boolean", "default": True},
            "nq":      {"type": "array", "default": [4,4,4]},
            "qpoints": {"type": "array"},
            "epsil":   {"type": "boolean", "default": True},
            "lraman":  {"type": "boolean", "default": False},
            "fildyn":  {"type": "string", "default": "dyn", "description": "Custom dynamical-matrix output filename passed to ph.x"},
            "qplot":   {"type": "boolean", "description": "Force the qplot-style multi-q-point card format; auto-selected from qpoints length if omitted"},
            "label":   {"type": "string", "default": "phonon"},
            "mpi_np":  {"type": "integer", "default": 1},
            "timeout": {"type": "integer", "default": 14400},
            "project_id": {"type": "integer"},
            "cycle_id":   {"type": "integer"},
            "recover": {"type": "boolean", "default": False, "description": "Resume from a prior interrupted run's .recover scratch file instead of restarting"},
            "existing_scf_job_id": {"type": "string", "description": "job_id of the SCF step from the original run (required when recover=true)"},
        },
        "required": ["structure", "calc_params"],
    },
)
async def vishwakarma_run_phonon(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import workflow as wf
            result = wf.phonon_workflow(
                structure=args["structure"], calc_params=args["calc_params"],
                qpoints=args.get("qpoints"), ldisp=args.get("ldisp",True),
                nq=tuple(args.get("nq",[4,4,4])), epsil=args.get("epsil",True),
                label=args.get("label","phonon"), workdir=QE_WORKDIR,
                bin_dir=QE_BIN_DIR, timeout=args.get("timeout",14400),
                mpi_np=args.get("mpi_np",1),
                recover=args.get("recover", False),
                existing_scf_job_id=args.get("existing_scf_job_id"),
                fildyn=args.get("fildyn", "dyn"),
                lraman=args.get("lraman", False),
                qplot=args.get("qplot"),
            )
            return _ok(result)
        except Exception as exc:
            return _err("Phonon calculation failed", str(exc))
    _t0 = time.time()
    result = await asyncio.to_thread(_run)
    await _persist_run('phonon', args, result,
                       round(time.time() - _t0, 1),
                       material_name=_material_of(args))
    return result


@brahm_tool(
    name="vishwakarma_run_neb", group="vishwakarma",
    description=(
        "Run nudged elastic band (NEB) to find transition states between two structures. "
        "Pass project_id to auto-save result to brahm.db."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "initial_structure": {"type": "object", "description": "Starting structure of the NEB path."},
            "final_structure":   {"type": "object", "description": "End structure of the NEB path. Must have the same species and count as initial_structure."},
            "calc_params":       {"type": "object", "description": "Quantum ESPRESSO parameters — cutoffs, k-points, smearing, pseudopotentials. See vishwakarma_list_pseudopotentials for what is installed."},
            "num_images":  {"type": "integer", "default": 7},
            "ci_scheme":   {"type": "string", "enum": ["no-CI","auto","manual"], "default": "auto"},
            "opt_scheme":  {"type": "string", "enum": ["broyden","sd","lbfgs"], "default": "broyden"},
            "nstep_path":  {"type": "integer", "default": 200},
            "label":       {"type": "string", "default": "neb"},
            "mpi_np":      {"type": "integer", "default": 1},
            "timeout":     {"type": "integer", "default": 28800},
            "project_id":  {"type": "integer"},
            "cycle_id":    {"type": "integer"},
        },
        "required": ["initial_structure", "final_structure", "calc_params"],
    },
)
async def vishwakarma_run_neb(args: dict) -> dict:
    def _run() -> dict:
        try:
            # Was an inline ig→runner→parse sequence assembled here, which is
            # why NEB was unreachable from vishwakarma_api.py and the planner.
            # Now composed like every other calculation type.
            from vishwakarma import workflow as wf
            return _ok(wf.neb_workflow(
                initial_structure=args["initial_structure"],
                final_structure=args["final_structure"],
                calc_params=args["calc_params"],
                num_images=args.get("num_images", 7),
                ci_scheme=args.get("ci_scheme", "auto"),
                opt_scheme=args.get("opt_scheme", "broyden"),
                nstep_path=args.get("nstep_path", 200),
                label=args.get("label", "neb"), workdir=QE_WORKDIR,
                bin_dir=QE_BIN_DIR, timeout=args.get("timeout", 28800),
                mpi_np=args.get("mpi_np", 1),
            ))
        except Exception as exc:
            return _err("NEB calculation failed", str(exc))
    _t0 = time.time()
    result = await asyncio.to_thread(_run)
    await _persist_run('neb', args, result,
                       round(time.time() - _t0, 1),
                       material_name=_material_of(args, 'initial_structure'))
    return result


@brahm_tool(
    name="vishwakarma_run_hp", group="vishwakarma",
    description=(
        "Compute Hubbard U parameters from linear response theory using hp.x. "
        "Two modes: pass structure+calc_params to run SCF then hp.x, or pass "
        "prefix+outdir to attach to an SCF that already ran. "
        "Pass project_id to auto-save result to brahm.db."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "structure":   {"type": "object", "description": STRUCTURE_DESC},
            "calc_params": {"type": "object", "description": CALC_PARAMS_DESC},
            "prefix":    {"type": "string", "description": "Attach mode: prefix of an existing SCF."},
            "outdir":    {"type": "string", "description": "Attach mode: outdir of an existing SCF."},
            "existing_scf_job_id": {"type": "string", "description": "Attach to a prior SCF job by id."},
            "nq":        {"type": "array", "default": [2,2,2]},
            "job_label": {"type": "string", "default": "hp"},
            "mpi_np":    {"type": "integer", "default": 1},
            "timeout":   {"type": "integer", "default": 7200},
            "project_id": {"type": "integer"},
            "cycle_id":   {"type": "integer"},
        },
        # prefix+outdir were required before 2026-09-09, which forced every
        # caller into attach mode and made hp the only calculation type that
        # could not be run from a structure alone.
        "required": [],
    },
)
async def vishwakarma_run_hp(args: dict) -> dict:
    def _run() -> dict:
        try:
            from vishwakarma import workflow as wf
            if not (args.get("structure") or (args.get("prefix") and args.get("outdir"))):
                return _err(
                    "HP calculation failed",
                    "Provide either structure (+calc_params) to run SCF then hp.x, "
                    "or both prefix and outdir to attach to an existing SCF.",
                )
            return _ok(wf.hp_workflow(
                structure=args.get("structure") or {},
                calc_params=args.get("calc_params") or {},
                nq=tuple(args.get("nq", [2, 2, 2])),
                label=args.get("job_label", "hp"), workdir=QE_WORKDIR,
                bin_dir=QE_BIN_DIR, timeout=args.get("timeout", 7200),
                mpi_np=args.get("mpi_np", 1),
                existing_scf_job_id=args.get("existing_scf_job_id"),
                prefix=args.get("prefix"), outdir=args.get("outdir"),
            ))
        except Exception as exc:
            return _err("HP calculation failed", str(exc))
    _t0 = time.time()
    result = await asyncio.to_thread(_run)
    await _persist_run('hp', args, result,
                       round(time.time() - _t0, 1),
                       material_name=_material_of(args) or args.get('prefix', ''))
    return result


@brahm_tool(
    name="vishwakarma_parse_output", group="vishwakarma",
    description="Parse a Quantum ESPRESSO output file from a job_id or file path.",
    input_schema={
        "type": "object",
        "properties": {
            "source":    {"type": "string", "enum": ["job_id","file_path"], "description": "Path to the QE output file to parse, or the job id that produced it."},
            "job_id":    {"type": "string", "description": "Job id returned by any vishwakarma_run_* tool. vishwakarma_list_jobs shows the known ids."},
            "file_path": {"type": "string"},
            "code":      {"type": "string", "enum": ["pw","ph","dos","bands","neb"], "default": "pw", "description": "Which parser to use, matching the calculation that produced the output, e.g. 'pw', 'ph', 'neb', 'hp'."},
        },
        "required": ["source", "code"],
    },
)
async def vishwakarma_parse_output(args: dict) -> dict:
    def _parse() -> dict:
        try:
            from vishwakarma import output_parser as op
            from vishwakarma import runner as r
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        source = args.get("source","job_id")
        code   = args.get("code","pw")
        if source == "job_id":
            job_id = args.get("job_id","")
            if not job_id:
                return _err("job_id required when source=job_id")
            text = r.get_output(job_id, QE_WORKDIR)
        else:
            fp = args.get("file_path","")
            if not fp or not os.path.isfile(fp):
                return _err(f"File not found: {fp}")
            with open(fp, errors="replace") as f:
                text = f.read()
        if not text:
            return _err("Output file is empty or not found")
        try:
            return _ok({"code": code, "parsed": op.parse(text, code)})
        except Exception as exc:
            return _err("Parse failed", str(exc))
    return await asyncio.to_thread(_parse)


@brahm_tool(
    name="vishwakarma_list_pseudopotentials", group="vishwakarma",
    description=(
        "Discover and list all UPF pseudopotential files in the configured pseudo_dir. "
        "Optionally cross-check against a structure to flag missing pseudopotentials."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pseudo_dirs":          {"type": "array", "items": {"type": "string"}},
            "structure":            {"type": "object", "description": "Atomic structure: cell vectors, species and positions, in the form generate_input expects."},
            "preferred_functional": {"type": "string", "default": "pbe"},
            "preferred_type":       {"type": "string", "default": "us"},
        },
        "required": [],
    },
)
async def vishwakarma_list_pseudopotentials(args: dict) -> dict:
    def _list() -> dict:
        try:
            from vishwakarma import pseudo_manager as pm
        except ImportError as exc:
            return _err("Vishwakarma import failed", str(exc))
        dirs    = args.get("pseudo_dirs") or [QE_PSEUDO]
        pseudos = pm.discover(dirs)
        result  = _ok({"pseudo_dirs": dirs, "total_found": len(pseudos),
                        "pseudopotentials": pseudos[:100]})
        if args.get("structure"):
            result["structure_check"] = pm.list_for_structure(
                args["structure"], dirs,
                preferred_functional=args.get("preferred_functional","pbe"),
                preferred_type=args.get("preferred_type","us"),
            )
        return result
    return await asyncio.to_thread(_list)


@brahm_tool(
    name="vishwakarma_get_job_status", group="vishwakarma",
    description="Get the status of a specific Vishwakarma job by job_id.",
    input_schema={
        "type": "object",
        "properties": {"job_id": {"type": "string", "description": "Job id returned by any vishwakarma_run_* tool. vishwakarma_list_jobs shows the known ids."}},
        "required": ["job_id"],
    },
)
async def vishwakarma_get_job_status(args: dict) -> dict:
    def _get() -> dict:
        try:
            from vishwakarma import runner as r
            return _ok(r.get_job_status(args.get("job_id",""), QE_WORKDIR))
        except Exception as exc:
            return _err("Job status failed", str(exc))
    return await asyncio.to_thread(_get)


@brahm_tool(
    name="vishwakarma_list_jobs", group="vishwakarma",
    description="List all Vishwakarma calculation jobs, newest first.",
    input_schema={
        "type": "object",
        "properties": {
            "status_filter": {
                "type": "string",
                "enum": ["all","created","running","completed","failed","timeout"],
                "default": "all",
            },
            "limit": {"type": "integer", "default": 20},
        },
        "required": [],
    },
)
async def vishwakarma_list_jobs(args: dict) -> dict:
    def _list() -> dict:
        try:
            from vishwakarma import runner as r
            sf   = args.get("status_filter","all")
            jobs = r.list_jobs(workdir=QE_WORKDIR, limit=args.get("limit",20),
                               status_filter=None if sf == "all" else sf)
            return _ok({"count": len(jobs), "jobs": jobs})
        except Exception as exc:
            return _err("List jobs failed", str(exc))
    return await asyncio.to_thread(_list)
