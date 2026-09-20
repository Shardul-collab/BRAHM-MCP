# vishwakarma/workflow.py
#
# Orchestrate multi-step Quantum ESPRESSO calculation sequences.
# Each workflow creates one job per step, chains outputs, and returns
# a structured summary of all steps.
#
# Standard workflows:
#   scf_only          — single SCF
#   relax_then_scf    — ionic relax → final SCF on relaxed geometry
#   band_structure    — SCF → NSCF → bands.x post-processing
#   dos_workflow      — SCF → NSCF (dense k) → dos.x
#   phonon_workflow   — SCF → ph.x (DFPT)
#   neb_workflow      — neb.x (minimum energy path / transition state)
#   hp_workflow       — SCF → hp.x (Hubbard U from linear response)
#   full_characterization — SCF → relax → bands + DOS + phonons
#
# neb_workflow and hp_workflow were added 2026-09-09. Before that, NEB and
# HP existed only as ad-hoc input_generator→runner→parse sequences assembled
# inline inside brahm/agents/vishwakarma.py's MCP handlers, so they were
# unreachable from vishwakarma_api.py and from the coordinator's planner —
# the only two of the seven calculation types that could not be composed.

import logging
from pathlib import Path
from typing import Optional

from vishwakarma import input_generator as ig
from vishwakarma import runner
from vishwakarma import output_parser as op

logger = logging.getLogger("vishwakarma.workflow")


# ─── Workflow definitions ─────────────────────────────────────────────────────

def scf_only(structure: dict, calc_params: dict,
             label: str = "scf",
             workdir: str = runner._DEFAULT_WORKDIR,
             bin_dir: str = runner._DEFAULT_BIN_DIR,
             timeout: Optional[int] = 3600,
             mpi_np: int = 1,
             job_id: Optional[str] = None) -> dict:
    """Single SCF calculation."""
    input_text = ig.scf(structure, calc_params)
    jid = job_id or runner.create_job(label, "pw", input_text, workdir, mpi_np)
    status = runner.run_job(jid, workdir, timeout, bin_dir)
    out    = runner.get_output(jid, workdir)
    parsed = op.parse_pw(out)
    return _workflow_result("scf_only", [{"step": "scf", "job_id": jid, "status": status, "parsed": parsed}])


def relax_then_scf(structure: dict, calc_params: dict,
                   vc: bool = False,
                   label: str = "relax",
                   workdir: str = runner._DEFAULT_WORKDIR,
                   bin_dir: str = runner._DEFAULT_BIN_DIR,
                   timeout: Optional[int] = 7200,
                   mpi_np: int = 1,
                   job_id: Optional[str] = None) -> dict:
    """Ionic (or variable-cell) relaxation followed by a final SCF."""
    steps = []

    # Step 1: relax
    relax_input = ig.relax(structure, calc_params, vc=vc)
    jid = job_id or runner.create_job(f"{label}_relax", "pw", relax_input, workdir, mpi_np)
    status = runner.run_job(jid, workdir, timeout, bin_dir)
    out    = runner.get_output(jid, workdir)
    parsed = op.parse_pw(out)
    steps.append({"step": "relax", "job_id": jid, "status": status, "parsed": parsed})

    if status["status"] != "completed":
        return _workflow_result("relax_then_scf", steps, failed_at="relax")

    # Use relaxed structure if positions were extracted
    relaxed = dict(structure)
    if parsed.get("final_positions"):
        relaxed["atomic_positions"] = parsed["final_positions"]
    if parsed.get("final_cell_ang") and vc:
        relaxed["cell_parameters"] = parsed["final_cell_ang"]

    # Step 2: final SCF
    scf_input = ig.scf(relaxed, calc_params)
    jid2 = runner.create_job(f"{label}_scf", "pw", scf_input, workdir, mpi_np)
    status2 = runner.run_job(jid2, workdir, timeout, bin_dir)
    out2    = runner.get_output(jid2, workdir)
    parsed2 = op.parse_pw(out2)
    steps.append({"step": "scf", "job_id": jid2, "status": status2, "parsed": parsed2})

    return _workflow_result("relax_then_scf", steps)


def band_structure(structure: dict, calc_params: dict,
                   kpath: list | None = None,
                   label: str = "bands",
                   workdir: str = runner._DEFAULT_WORKDIR,
                   bin_dir: str = runner._DEFAULT_BIN_DIR,
                   timeout: Optional[int] = 3600,
                   mpi_np: int = 1,
                   job_id: Optional[str] = None) -> dict:
    """
    SCF → NSCF (k-path) → bands.x post-processing.

    kpath: list of k-points with weights, format [[kx,ky,kz, npt], ...]
           where npt is number of interpolation points to next segment.
           If None, uses a default Gamma-X-M-Gamma path.
    """
    steps = []

    # Step 1: SCF
    scf_input = ig.scf(structure, calc_params)
    jid = job_id or runner.create_job(f"{label}_scf", "pw", scf_input, workdir, mpi_np)
    status = runner.run_job(jid, workdir, timeout, bin_dir)
    steps.append({"step": "scf", "job_id": jid, "status": status,
                  "parsed": op.parse_pw(runner.get_output(jid, workdir))})
    if status["status"] != "completed":
        return _workflow_result("band_structure", steps, failed_at="scf")

    # Shared outdir: NSCF and bands.x must read the SCF step's charge density,
    # but each step runs in its own isolated job directory (runner.run_job
    # uses cwd=job_dir). Resolve outdir against the SCF job's own directory
    # so later steps point at the same absolute location instead of a fresh
    # empty one. Jobs remain fully separate/monitorable - only this path
    # value is shared.
    shared_outdir = _resolve_shared_outdir(workdir, jid, calc_params.get("outdir", "./out"))

    # Step 2: NSCF on k-path
    bands_structure = dict(structure)
    if kpath is None:
        kpath = _default_kpath()
    bands_structure["kpoints"] = {"mode": "crystal_b", "points": kpath}
    bands_p = dict(calc_params)
    bands_p["outdir"] = shared_outdir
    bands_p.setdefault("nbnd", _estimate_nbnd(structure, calc_params))
    nscf_input = ig.bands(bands_structure, bands_p)
    jid2 = runner.create_job(f"{label}_nscf", "pw", nscf_input, workdir, mpi_np)
    status2 = runner.run_job(jid2, workdir, timeout, bin_dir)
    steps.append({"step": "nscf_bands", "job_id": jid2, "status": status2,
                  "parsed": op.parse_pw(runner.get_output(jid2, workdir))})
    if status2["status"] != "completed":
        return _workflow_result("band_structure", steps, failed_at="nscf_bands")

    # Step 3: bands.x
    prefix  = structure.get("prefix", "pwscf")
    pp_input = ig.bands_pp(prefix, shared_outdir)
    jid3 = runner.create_job(f"{label}_pp", "bands", pp_input, workdir, mpi_np)
    status3 = runner.run_job(jid3, workdir, timeout, bin_dir)
    steps.append({"step": "bands_pp", "job_id": jid3, "status": status3})

    return _workflow_result("band_structure", steps)


def dos_workflow(structure: dict, calc_params: dict,
                 dense_kmesh: list | None = None,
                 emin: float = -20.0,
                 emax: float = 20.0,
                 label: str = "dos",
                 workdir: str = runner._DEFAULT_WORKDIR,
                 bin_dir: str = runner._DEFAULT_BIN_DIR,
                 timeout: Optional[int] = 7200,
                 mpi_np: int = 1,
                 job_id: Optional[str] = None) -> dict:
    """SCF → dense NSCF → dos.x."""
    steps = []

    # SCF
    scf_input = ig.scf(structure, calc_params)
    jid = job_id or runner.create_job(f"{label}_scf", "pw", scf_input, workdir, mpi_np)
    status = runner.run_job(jid, workdir, timeout, bin_dir)
    steps.append({"step": "scf", "job_id": jid, "status": status,
                  "parsed": op.parse_pw(runner.get_output(jid, workdir))})
    if status["status"] != "completed":
        return _workflow_result("dos_workflow", steps, failed_at="scf")

    # Shared outdir: NSCF and dos.x must read the SCF step's charge density,
    # but each step runs in its own isolated job directory (runner.run_job
    # uses cwd=job_dir). Resolve outdir against the SCF job's own directory
    # so later steps point at the same absolute location instead of a fresh
    # empty one. Jobs remain fully separate/monitorable - only this path
    # value is shared.
    shared_outdir = _resolve_shared_outdir(workdir, jid, calc_params.get("outdir", "./out"))

    # Dense NSCF
    nscf_s = dict(structure)
    base_mesh = structure.get("kpoints", {}).get("mesh", [4, 4, 4])
    dense     = dense_kmesh or [m * 2 for m in base_mesh]
    nscf_s["kpoints"] = {"mode": "automatic", "mesh": dense, "shift": [0, 0, 0]}
    nscf_p = dict(calc_params)
    nscf_p["outdir"] = shared_outdir
    nscf_p.setdefault("occupations", "tetrahedra")
    nscf_p.setdefault("nbnd", _estimate_nbnd(structure, calc_params))
    nscf_input = ig.nscf(nscf_s, nscf_p)
    jid2 = runner.create_job(f"{label}_nscf", "pw", nscf_input, workdir, mpi_np)
    status2 = runner.run_job(jid2, workdir, timeout, bin_dir)
    steps.append({"step": "nscf", "job_id": jid2, "status": status2,
                  "parsed": op.parse_pw(runner.get_output(jid2, workdir))})
    if status2["status"] != "completed":
        return _workflow_result("dos_workflow", steps, failed_at="nscf")

    # dos.x
    prefix = structure.get("prefix", "pwscf")
    dos_input = ig.dos(prefix, shared_outdir, emin=emin, emax=emax)
    jid3 = runner.create_job(f"{label}_dos", "dos", dos_input, workdir, mpi_np)
    status3 = runner.run_job(jid3, workdir, timeout, bin_dir)
    steps.append({"step": "dos", "job_id": jid3, "status": status3,
                  "parsed": op.parse_dos(runner.get_output(jid3, workdir))})

    return _workflow_result("dos_workflow", steps)


def phonon_workflow(structure: dict, calc_params: dict,
                    qpoints: list | None = None,
                    ldisp: bool = True,
                    nq: tuple = (4, 4, 4),
                    epsil: bool = True,
                    label: str = "phonon",
                    workdir: str = runner._DEFAULT_WORKDIR,
                    bin_dir: str = runner._DEFAULT_BIN_DIR,
                    timeout: Optional[int] = 14400,
                    mpi_np: int = 1,
                    job_id: Optional[str] = None,
                    recover: bool = False,
                    existing_scf_job_id: Optional[str] = None,
                    fildyn: str = "dyn",
                    lraman: bool = False,
                    qplot: bool | None = None) -> dict:
    """SCF → ph.x (DFPT).

    recover: resume a previously interrupted ph.x run instead of starting
             from scratch. Requires existing_scf_job_id — the job_id of the
             SCF step from the ORIGINAL (killed) run, so ph.x is pointed at
             the exact same outdir where its .recover scratch file lives.
             When set, the SCF step is skipped entirely (no re-run needed —
             ph.x only needs the charge density, which is unchanged).
    fildyn:  custom dynamical-matrix output filename, forwarded to ph.x.
             Previously accepted here but silently dropped before reaching
             ig.phonon() — every run used the 'dyn' default regardless of
             what was requested.
    lraman:  compute Raman tensors (requires epsil=True). Same drop-on-the-
             floor issue as fildyn — now forwarded.
    qplot:   forwarded to ig.phonon() for explicit multi-q-point runs; see
             ig.phonon() docstring for the qplot card-format rules.
    """
    steps = []

    if recover and existing_scf_job_id:
        # Reuse the prior run's SCF output directly — do not re-run SCF.
        shared_outdir = _resolve_shared_outdir(workdir, existing_scf_job_id, calc_params.get("outdir", "./out"))
        steps.append({"step": "scf", "job_id": existing_scf_job_id,
                     "status": {"status": "reused"}, "parsed": None})
    else:
        scf_input = ig.scf(structure, calc_params)
        jid = job_id or runner.create_job(f"{label}_scf", "pw", scf_input, workdir, mpi_np)
        status = runner.run_job(jid, workdir, timeout, bin_dir)
        steps.append({"step": "scf", "job_id": jid, "status": status,
                      "parsed": op.parse_pw(runner.get_output(jid, workdir))})
        if status["status"] != "completed":
            return _workflow_result("phonon_workflow", steps, failed_at="scf")

        # Shared outdir: ph.x must read the SCF step's charge density/wavefunctions,
        # but each step runs in its own isolated job directory (runner.run_job
        # uses cwd=job_dir). Resolve outdir against the SCF job's own directory
        # so ph.x points at the same absolute location instead of a fresh
        # empty one. Jobs remain fully separate/monitorable - only this path
        # value is shared. (Mirrors the same fix already in dos_workflow/band_structure.)
        shared_outdir = _resolve_shared_outdir(workdir, jid, calc_params.get("outdir", "./out"))

    prefix = structure.get("prefix", "pwscf")
    ph_input = ig.phonon(prefix, shared_outdir, qpoints=qpoints,
                         ldisp=ldisp, nq=nq, epsil=epsil, recover=recover,
                         fildyn=fildyn, lraman=lraman, qplot=qplot)
    jid2 = runner.create_job(f"{label}_ph", "ph", ph_input, workdir, mpi_np)
    status2 = runner.run_job(jid2, workdir, timeout * 3, bin_dir)
    steps.append({"step": "phonon", "job_id": jid2, "status": status2,
                  "parsed": op.parse_ph(runner.get_output(jid2, workdir))})

    return _workflow_result("phonon_workflow", steps)


def neb_workflow(initial_structure: dict, final_structure: dict,
                 calc_params: dict,
                 num_images: int = 7,
                 ci_scheme: str = "auto",
                 opt_scheme: str = "broyden",
                 nstep_path: int = 200,
                 label: str = "neb",
                 workdir: str = runner._DEFAULT_WORKDIR,
                 bin_dir: str = runner._DEFAULT_BIN_DIR,
                 timeout: Optional[int] = 28800,
                 mpi_np: int = 1,
                 job_id: Optional[str] = None) -> dict:
    """
    neb.x — minimum energy path / transition state between two structures.

    Single-step: neb.x drives its own internal SCF cycle per image, so there
    is no separate SCF step to chain (unlike dos/bands/phonon). Only the
    endpoints are supplied; neb.x linearly interpolates the intermediate
    images itself according to num_of_images.
    """
    neb_input = ig.neb(
        images=[initial_structure, final_structure],
        calc_params=calc_params,
        num_of_images=num_images,
        ci_scheme=ci_scheme,
        opt_scheme=opt_scheme,
        nstep_path=nstep_path,
    )
    jid    = job_id or runner.create_job(label, "neb", neb_input, workdir, mpi_np)
    status = runner.run_job(jid, workdir, timeout, bin_dir)
    parsed = op.parse_neb(runner.get_output(jid, workdir))
    return _workflow_result("neb_workflow",
                            [{"step": "neb", "job_id": jid,
                              "status": status, "parsed": parsed}])


def hp_workflow(structure: dict, calc_params: dict,
                nq: tuple = (2, 2, 2),
                label: str = "hp",
                workdir: str = runner._DEFAULT_WORKDIR,
                bin_dir: str = runner._DEFAULT_BIN_DIR,
                timeout: Optional[int] = 7200,
                mpi_np: int = 1,
                job_id: Optional[str] = None,
                hp_job_id: Optional[str] = None,
                existing_scf_job_id: Optional[str] = None,
                prefix: Optional[str] = None,
                outdir: Optional[str] = None) -> dict:
    """
    SCF → hp.x (Hubbard U from linear response).

    hp.x reads the charge density of a preceding SCF run, so this composes
    the two the same way phonon_workflow does — including the shared-outdir
    resolution, without which hp.x reads a fresh empty directory.

    existing_scf_job_id: attach to an SCF that already ran instead of
        repeating it (mirrors phonon_workflow's recover path). This is the
        mode the vishwakarma_run_hp MCP tool used exclusively before this
        workflow existed — it took prefix/outdir directly and left the
        caller responsible for having run a compatible SCF first.
    prefix/outdir: explicit override for the attach mode, for an SCF that
        this runner did not launch (so no job_id exists to resolve against).
    job_id:    reuse a pre-created job for the SCF step.
    hp_job_id: reuse a pre-created job for the hp.x step. Needed by the
        non-blocking API, which must return a job_id to poll before the
        work starts — in attach mode there is no SCF step, so hp.x is the
        only job and job_id alone would leave the returned id orphaned.
    """
    steps = []

    if prefix and outdir:
        # Fully explicit attach — caller knows exactly where the density is.
        hp_prefix, shared_outdir = prefix, outdir
        steps.append({"step": "scf", "job_id": existing_scf_job_id or "",
                      "status": {"status": "reused"}, "parsed": None})
    elif existing_scf_job_id:
        hp_prefix = structure.get("prefix", "pwscf")
        shared_outdir = _resolve_shared_outdir(
            workdir, existing_scf_job_id, calc_params.get("outdir", "./out"))
        steps.append({"step": "scf", "job_id": existing_scf_job_id,
                      "status": {"status": "reused"}, "parsed": None})
    else:
        scf_input = ig.scf(structure, calc_params)
        jid = job_id or runner.create_job(f"{label}_scf", "pw", scf_input, workdir, mpi_np)
        status = runner.run_job(jid, workdir, timeout, bin_dir)
        steps.append({"step": "scf", "job_id": jid, "status": status,
                      "parsed": op.parse_pw(runner.get_output(jid, workdir))})
        if status["status"] != "completed":
            return _workflow_result("hp_workflow", steps, failed_at="scf")
        hp_prefix = structure.get("prefix", "pwscf")
        shared_outdir = _resolve_shared_outdir(
            workdir, jid, calc_params.get("outdir", "./out"))

    hp_input = ig.hp(hp_prefix, shared_outdir, nq=tuple(nq))
    jid2 = hp_job_id or runner.create_job(f"{label}_hp", "hp", hp_input, workdir, mpi_np)
    status2 = runner.run_job(jid2, workdir, timeout, bin_dir)
    steps.append({"step": "hp", "job_id": jid2, "status": status2,
                  "parsed": op.parse_hp(runner.get_output(jid2, workdir))})

    return _workflow_result("hp_workflow", steps)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_shared_outdir(workdir: str, scf_job_id: str, outdir: str) -> str:
    """
    Resolve an outdir string to an absolute path pointing at the SCF step's
    own job directory, so later steps (NSCF, dos.x, bands.x) read the charge
    density QE actually wrote instead of a fresh, empty isolated directory.

    If outdir is already absolute, it is returned unchanged (caller supplied
    an explicit shared location and knows what they are doing).
    """
    p = Path(outdir)
    if p.is_absolute():
        return str(p)
    return str((Path(workdir) / scf_job_id / outdir).resolve())


# A step's status["status"] value that counts as "this step is fine".
# "reused" is set by the paths that deliberately skip re-running an SCF
# (phonon_workflow's recover=True, hp_workflow's attach modes). Before
# 2026-09-09 only "completed" was accepted here, so a phonon recover run
# that finished perfectly still reported success=False — the skipped SCF
# step dragged the whole result down. That in turn fed the persistence
# layer, which derives its saved status from this flag.
_OK_STEP_STATUSES = ("completed", "reused")


def _workflow_result(name: str, steps: list, failed_at: str | None = None) -> dict:
    all_ok = all(
        s.get("status", {}).get("status") in _OK_STEP_STATUSES for s in steps
    )
    return {
        "workflow":   name,
        "success":    all_ok and failed_at is None,
        "failed_at":  failed_at,
        "step_count": len(steps),
        "steps":      steps,
        # Flattened summary of the scientifically-relevant step, so callers
        # (notably the MCP persistence layer) do not have to reach into
        # steps[] and silently get None when they forget to.
        "summary":    _summarize_steps(steps),
    }


def _summarize_steps(steps: list) -> dict:
    """
    Flatten the step list into the fields downstream consumers actually want.

    job_id is the LAST step's job — the one whose output is the result of
    the workflow (the dos.x job for a DOS run, not its SCF). Convergence and
    iteration counts come from the last step that produced a parse, since
    post-processing steps like bands.x parse to almost nothing.

    This exists because brahm/agents/vishwakarma.py was reading job_id,
    converged and scf_iterations straight off the top level of the workflow
    result, where they have never existed — so every DFT record written to
    brahm.db carried an empty job_id and a null convergence flag.
    """
    if not steps:
        return {"job_id": "", "converged": None,
                "scf_iterations": None, "total_energy_ev": None}

    last = steps[-1]
    parsed_steps = [s for s in steps if isinstance(s.get("parsed"), dict)]
    parsed = parsed_steps[-1]["parsed"] if parsed_steps else {}
    energy_steps = [
        s for s in parsed_steps
        if s["parsed"].get("total_energy_ev") is not None
    ]
    iter_steps = [
        s for s in parsed_steps
        if s["parsed"].get("scf_iterations") is not None
    ]

    return {
        "job_id":          last.get("job_id", ""),
        "converged":       parsed.get("converged"),
        "scf_iterations":  (iter_steps[-1]["parsed"]["scf_iterations"]
                            if iter_steps else None),
        "total_energy_ev": (energy_steps[-1]["parsed"]["total_energy_ev"]
                            if energy_steps else None),
    }


def _default_kpath() -> list:
    """Simple Γ-X-M-Γ-R path for cubic systems."""
    return [
        [0.0, 0.0, 0.0, 20],   # Γ
        [0.5, 0.0, 0.0, 20],   # X
        [0.5, 0.5, 0.0, 20],   # M
        [0.0, 0.0, 0.0, 20],   # Γ
        [0.5, 0.5, 0.5,  1],   # R
    ]


def _estimate_nbnd(structure: dict, calc_params: dict) -> int:
    """
    Rough estimate: number of occupied bands + 20% extra.
    Uses sum of valence electrons if known, else falls back to 4 per atom.
    """
    nat = len(structure.get("atomic_positions", []))
    return max(8, int(nat * 4 * 1.2))
