"""
brahm/agents/vishwakarma.py
============================
Group H — Vishwakarma Quantum ESPRESSO DFT tools.
All calculations run locally via subprocess — no internet.

Auto-save: after every successful run_* call, result is persisted
to brahm.db via POST /v1/results/dft (CHITRAGUPTA API on :8003).
project_id is optional — pass it in args to link the result to a project.
If CHITRAGUPTA is down, the save is silently skipped — never blocks calculation.
"""

import asyncio
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
) -> int | None:
    """
    POST /v1/results/dft — persist a completed QE result to brahm.db.
    Returns result_id or None. Never raises.
    Skipped silently if project_id is None or CHITRAGUPTA is unreachable.
    """
    if project_id is None:
        return None
    try:
        import requests
        r = requests.post(
            f"{CHITRAGUPTA_BASE}/v1/results/dft",
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
            print(f"[CHITRAGUPTA] DFT result saved: result_id={rid}")
            return rid
    except Exception as e:
        print(f"[CHITRAGUPTA] Auto-save skipped: {e}")
    return None


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
            "calc_type":     {"type": "string", "enum": CALC_TYPES},
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
            "structure":   {"type": "object"},
            "calc_params": {"type": "object"},
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
            t0 = time.time()
            result = wf.scf_only(
                structure=args["structure"], calc_params=args["calc_params"],
                label=args.get("label","scf"), workdir=QE_WORKDIR,
                bin_dir=QE_BIN_DIR, timeout=args.get("timeout",3600),
                mpi_np=args.get("mpi_np",1),
            )
            wall = round(time.time() - t0, 1)
            _chit_save_dft(
                project_id=args.get("project_id"),
                job_id=result.get("job_id",""),
                calc_type="scf",
                structure=args.get("structure"),
                calc_params=args.get("calc_params"),
                output_parsed=result,
                status="completed",
                wall_time_seconds=wall,
                cycle_id=args.get("cycle_id"),
            )
            return _ok(result)
        except Exception as exc:
            return _err("SCF calculation failed", str(exc))
    result = await asyncio.to_thread(_run)
    if result.get('status') == 'success':
        import asyncio as _aio
        from brahm.shared.http import _chit_store_async
        _aio.ensure_future(_chit_store_async('/v1/store/vishwakarma', {
            'calculation_type': 'scf',
            'material_name':    args.get('structure',{}).get('prefix',''),
            'output_file_path': result.get('output_file', ''),
            'scf_iterations':   result.get('scf_iterations'),
            'converged':        result.get('converged'),
            'job_id':           result.get('job_id', ''),
        }))
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
            "structure":   {"type": "object"},
            "calc_params": {"type": "object"},
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
            t0 = time.time()
            result = wf.relax_then_scf(
                structure=args["structure"], calc_params=args["calc_params"],
                vc=args.get("vc_relax",False), label=args.get("label","relax"),
                workdir=QE_WORKDIR, bin_dir=QE_BIN_DIR,
                timeout=args.get("timeout",7200), mpi_np=args.get("mpi_np",1),
            )
            wall = round(time.time() - t0, 1)
            calc_type = "vc-relax" if args.get("vc_relax") else "relax"
            _chit_save_dft(
                project_id=args.get("project_id"),
                job_id=result.get("job_id",""),
                calc_type=calc_type,
                structure=args.get("structure"),
                calc_params=args.get("calc_params"),
                output_parsed=result,
                status="completed",
                wall_time_seconds=wall,
                cycle_id=args.get("cycle_id"),
            )
            return _ok(result)
        except Exception as exc:
            return _err("Relaxation failed", str(exc))
    result = await asyncio.to_thread(_run)
    if result.get('status') == 'success':
        import asyncio as _aio
        from brahm.shared.http import _chit_store_async
        _aio.ensure_future(_chit_store_async('/v1/store/vishwakarma', {
            'calculation_type': 'relax',
            'material_name':    args.get('structure',{}).get('prefix',''),
            'output_file_path': result.get('output_file', ''),
            'scf_iterations':   result.get('scf_iterations'),
            'converged':        result.get('converged'),
            'job_id':           result.get('job_id', ''),
        }))
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
            "structure":   {"type": "object"},
            "calc_params": {"type": "object"},
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
            t0 = time.time()
            result = wf.band_structure(
                structure=args["structure"], calc_params=args["calc_params"],
                kpath=args.get("kpath"), label=args.get("label","bands"),
                workdir=QE_WORKDIR, bin_dir=QE_BIN_DIR,
                timeout=args.get("timeout",3600), mpi_np=args.get("mpi_np",1),
            )
            wall = round(time.time() - t0, 1)
            _chit_save_dft(
                project_id=args.get("project_id"),
                job_id=result.get("job_id",""),
                calc_type="bands",
                structure=args.get("structure"),
                calc_params=args.get("calc_params"),
                output_parsed=result,
                status="completed",
                wall_time_seconds=wall,
                cycle_id=args.get("cycle_id"),
            )
            return _ok(result)
        except Exception as exc:
            return _err("Band structure calculation failed", str(exc))
    result = await asyncio.to_thread(_run)
    if result.get('status') == 'success':
        import asyncio as _aio
        from brahm.shared.http import _chit_store_async
        _aio.ensure_future(_chit_store_async('/v1/store/vishwakarma', {
            'calculation_type': 'bands',
            'material_name':    args.get('structure',{}).get('prefix',''),
            'output_file_path': result.get('output_file', ''),
            'scf_iterations':   result.get('scf_iterations'),
            'converged':        result.get('converged'),
            'job_id':           result.get('job_id', ''),
        }))
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
            "structure":   {"type": "object"},
            "calc_params": {"type": "object"},
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
            t0 = time.time()
            result = wf.dos_workflow(
                structure=args["structure"], calc_params=args["calc_params"],
                dense_kmesh=args.get("dense_kmesh"), emin=args.get("emin",-20.0),
                emax=args.get("emax",20.0), label=args.get("label","dos"),
                workdir=QE_WORKDIR, bin_dir=QE_BIN_DIR,
                timeout=args.get("timeout",7200), mpi_np=args.get("mpi_np",1),
            )
            wall = round(time.time() - t0, 1)
            _chit_save_dft(
                project_id=args.get("project_id"),
                job_id=result.get("job_id",""),
                calc_type="dos",
                structure=args.get("structure"),
                calc_params=args.get("calc_params"),
                output_parsed=result,
                status="completed",
                wall_time_seconds=wall,
                cycle_id=args.get("cycle_id"),
            )
            return _ok(result)
        except Exception as exc:
            return _err("DOS calculation failed", str(exc))
    result = await asyncio.to_thread(_run)
    if result.get('status') == 'success':
        import asyncio as _aio
        from brahm.shared.http import _chit_store_async
        _aio.ensure_future(_chit_store_async('/v1/store/vishwakarma', {
            'calculation_type': 'dos',
            'material_name':    args.get('structure',{}).get('prefix',''),
            'output_file_path': result.get('output_file', ''),
            'scf_iterations':   result.get('scf_iterations'),
            'converged':        result.get('converged'),
            'job_id':           result.get('job_id', ''),
        }))
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
            "structure":   {"type": "object"},
            "calc_params": {"type": "object"},
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
            t0 = time.time()
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
            wall = round(time.time() - t0, 1)
            _chit_save_dft(
                project_id=args.get("project_id"),
                job_id=result.get("job_id",""),
                calc_type="phonon",
                structure=args.get("structure"),
                calc_params=args.get("calc_params"),
                output_parsed=result,
                status="completed",
                wall_time_seconds=wall,
                cycle_id=args.get("cycle_id"),
            )
            return _ok(result)
        except Exception as exc:
            return _err("Phonon calculation failed", str(exc))
    result = await asyncio.to_thread(_run)
    if result.get('status') == 'success':
        import asyncio as _aio
        from brahm.shared.http import _chit_store_async
        _aio.ensure_future(_chit_store_async('/v1/store/vishwakarma', {
            'calculation_type': 'phonon',
            'material_name':    args.get('structure',{}).get('prefix',''),
            'output_file_path': result.get('output_file', ''),
            'scf_iterations':   result.get('scf_iterations'),
            'converged':        result.get('converged'),
            'job_id':           result.get('job_id', ''),
        }))
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
            "initial_structure": {"type": "object"},
            "final_structure":   {"type": "object"},
            "calc_params":       {"type": "object"},
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
            from vishwakarma import input_generator as ig
            from vishwakarma import runner as r
            from vishwakarma import output_parser as op
            t0 = time.time()
            neb_input = ig.neb(
                images=[args["initial_structure"], args["final_structure"]],
                calc_params=args["calc_params"],
                num_of_images=args.get("num_images",7),
                ci_scheme=args.get("ci_scheme","auto"),
                opt_scheme=args.get("opt_scheme","broyden"),
                nstep_path=args.get("nstep_path",200),
            )
            jid    = r.create_job(args.get("label","neb"), "neb", neb_input,
                                  QE_WORKDIR, args.get("mpi_np",1))
            status = r.run_job(jid, QE_WORKDIR, args.get("timeout",28800), QE_BIN_DIR)
            parsed = op.parse_neb(r.get_output(jid, QE_WORKDIR))
            wall   = round(time.time() - t0, 1)
            result = {"job_id": jid, "status": status, "parsed": parsed}
            _chit_save_dft(
                project_id=args.get("project_id"),
                job_id=jid,
                calc_type="neb",
                structure=args.get("initial_structure"),
                calc_params=args.get("calc_params"),
                output_parsed=parsed,
                status="completed" if status == "completed" else "failed",
                wall_time_seconds=wall,
                cycle_id=args.get("cycle_id"),
            )
            return _ok(result)
        except Exception as exc:
            return _err("NEB calculation failed", str(exc))
    result = await asyncio.to_thread(_run)
    if result.get('status') == 'success':
        import asyncio as _aio
        from brahm.shared.http import _chit_store_async
        _aio.ensure_future(_chit_store_async('/v1/store/vishwakarma', {
            'calculation_type': 'neb',
            'material_name':    args.get('initial_structure',{}).get('prefix',''),
            'output_file_path': result.get('output_file', ''),
            'scf_iterations':   result.get('scf_iterations'),
            'converged':        result.get('converged'),
            'job_id':           result.get('job_id', ''),
        }))
    return result


@brahm_tool(
    name="vishwakarma_run_hp", group="vishwakarma",
    description=(
        "Compute Hubbard U parameters from linear response theory using hp.x. "
        "Pass project_id to auto-save result to brahm.db."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prefix":    {"type": "string"},
            "outdir":    {"type": "string"},
            "nq":        {"type": "array", "default": [2,2,2]},
            "job_label": {"type": "string", "default": "hp"},
            "mpi_np":    {"type": "integer", "default": 1},
            "timeout":   {"type": "integer", "default": 7200},
            "project_id": {"type": "integer"},
            "cycle_id":   {"type": "integer"},
        },
        "required": ["prefix", "outdir"],
    },
)
async def vishwakarma_run_hp(args: dict) -> dict:
    def _run() -> dict:
        try:
            import re as _re
            from vishwakarma import input_generator as ig
            from vishwakarma import runner as r
            t0       = time.time()
            hp_input = ig.hp(args["prefix"], args["outdir"],
                             nq=tuple(args.get("nq",[2,2,2])))
            jid    = r.create_job(args.get("job_label","hp"), "hp", hp_input,
                                  QE_WORKDIR, args.get("mpi_np",1))
            status = r.run_job(jid, QE_WORKDIR, args.get("timeout",7200), QE_BIN_DIR)
            out    = r.get_output(jid, QE_WORKDIR)
            u_vals = _re.findall(r"Hubbard U\s*\(\w+\)\s*=\s*([-\d.]+)", out)
            wall   = round(time.time() - t0, 1)
            parsed = {"u_values_ev": [float(u) for u in u_vals]}
            _chit_save_dft(
                project_id=args.get("project_id"),
                job_id=jid,
                calc_type="hp",
                structure=None,
                calc_params={"prefix": args["prefix"], "outdir": args["outdir"],
                             "nq": args.get("nq",[2,2,2])},
                output_parsed=parsed,
                status="completed" if status == "completed" else "failed",
                wall_time_seconds=wall,
                cycle_id=args.get("cycle_id"),
            )
            return _ok({"job_id": jid, "status": status,
                        "u_values_ev": [float(u) for u in u_vals],
                        "note": "U values in eV. Use via hubbard_u in calc_params."})
        except Exception as exc:
            return _err("HP calculation failed", str(exc))
    result = await asyncio.to_thread(_run)
    if result.get('status') == 'success':
        import asyncio as _aio
        from brahm.shared.http import _chit_store_async
        _aio.ensure_future(_chit_store_async('/v1/store/vishwakarma', {
            'calculation_type': 'hp',
            'material_name':    args.get('prefix',''),
            'output_file_path': result.get('output_file', ''),
            'scf_iterations':   result.get('scf_iterations'),
            'converged':        result.get('converged'),
            'job_id':           result.get('job_id', ''),
        }))
    return result


@brahm_tool(
    name="vishwakarma_parse_output", group="vishwakarma",
    description="Parse a Quantum ESPRESSO output file from a job_id or file path.",
    input_schema={
        "type": "object",
        "properties": {
            "source":    {"type": "string", "enum": ["job_id","file_path"]},
            "job_id":    {"type": "string"},
            "file_path": {"type": "string"},
            "code":      {"type": "string", "enum": ["pw","ph","dos","bands","neb"], "default": "pw"},
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
            "structure":            {"type": "object"},
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
        "properties": {"job_id": {"type": "string"}},
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
