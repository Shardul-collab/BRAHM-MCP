# vishwakarma/runner.py
#
# Execute Quantum ESPRESSO binaries as local subprocesses.
# Each calculation is isolated in its own job directory.
# Status is persisted in status.json so jobs survive server restarts.

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vishwakarma.runner")

# ── QE binary locations ───────────────────────────────────────────────────────
# Override via QE_BIN_DIR environment variable or set here.
_DEFAULT_BIN_DIR = os.environ.get("QE_BIN_DIR", "/tmp/miniforge/bin")

QE_BINARIES = {
    "pw":       "pw.x",
    "ph":       "ph.x",
    "pp":       "pp.x",
    "dos":      "dos.x",
    "bands":    "bands.x",
    "projwfc":  "projwfc.x",
    "neb":      "neb.x",
    "cp":       "cp.x",
    "hp":       "hp.x",
    "dynmat":   "dynmat.x",
    "matdyn":   "matdyn.x",
    "q2r":      "q2r.x",
    "plotband": "plotband.x",
    "wannier90":"wannier90.x",
}

# Work directory — each job gets a subdirectory here
_DEFAULT_WORKDIR = os.environ.get("VISHWAKARMA_WORKDIR", "/tmp/vishwakarma_jobs")


# ─── Job lifecycle ────────────────────────────────────────────────────────────

def create_job(label: str, code: str, input_text: str,
               workdir: str = _DEFAULT_WORKDIR,
               mpi_np: int = 1,
               extra_args: list | None = None) -> str:
    """
    Initialise a job directory and write the input file.

    Returns:
        job_id (str) — UUID4, directory is {workdir}/{job_id}/
    """
    job_id  = str(uuid.uuid4())
    job_dir = Path(workdir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    (job_dir / "input.in").write_text(input_text, encoding="utf-8")

    status = {
        "job_id":     job_id,
        "label":      label,
        "code":       code,
        "status":     "created",
        "created_at": _now(),
        "started_at": None,
        "ended_at":   None,
        "exit_code":  None,
        "error":      None,
        "mpi_np":     mpi_np,
        "extra_args": extra_args or [],
        "workdir":    str(job_dir),
    }
    _write_status(job_dir, status)
    logger.info("Job created: %s  code=%s  label=%s", job_id, code, label)
    return job_id


def run_job(job_id: str,
            workdir: str = _DEFAULT_WORKDIR,
            timeout: Optional[int] = None,
            bin_dir: str = _DEFAULT_BIN_DIR) -> dict:
    """
    Execute a created job synchronously.
    Writes stdout → output.out, stderr → error.err.

    Returns the final status dict.
    """
    job_dir = Path(workdir) / job_id
    status  = _read_status(job_dir)

    if status["status"] not in ("created", "failed"):
        return status  # already running or completed

    binary = _resolve_binary(status["code"], bin_dir)
    if not binary:
        status["status"] = "failed"
        status["error"]  = f"Binary not found for code '{status['code']}' in {bin_dir}"
        _write_status(job_dir, status)
        return status


    # Pre-create QE save directory to avoid 'cannot open' errors
    input_text = (job_dir / "input.in").read_text(encoding="utf-8")
    import re as _re2
    prefix_match = _re2.search(r"prefix\s*=\s*['\"](.+?)['\"]", input_text)
    if prefix_match:
        save_dir = job_dir / f"{prefix_match.group(1)}.save"
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Pre-created save dir: %s", save_dir)

    # Build command
    cmd = []
    if status["mpi_np"] > 1:
        mpi = shutil.which("mpirun") or shutil.which("mpiexec")
        if mpi:
            # WSL2's hwloc/PRRTE slot detection routinely under-reports
            # available slots (e.g. reporting 1 slot on a 4-core machine),
            # so mpirun refuses to launch with a "not enough slots" error
            # even when the core count is genuinely sufficient. --map-by
            # :OVERSUBSCRIBE tells PRRTE to skip that slot check entirely.
            # Confirmed via direct reproduction: `mpirun -np 4 pw.x` fails
            # with PRRTE's slot error on this environment; the tool's own
            # error message recommends this exact flag.
            cmd = [mpi, "--map-by", ":OVERSUBSCRIBE", "-np", str(status["mpi_np"])]
    stdbuf = shutil.which("stdbuf")
    if stdbuf:
        # QE's Fortran I/O is block-buffered when stdout isn't a TTY (i.e.
        # always, since we redirect to a file). Without this, output.out can
        # sit empty or stale for long stretches during a real run, causing
        # false "is it stuck?" alarms that require scratch-file timestamp
        # checks or strace to resolve. -oL/-eL force line buffering so the
        # file reflects what QE has actually printed so far.
        cmd += [stdbuf, "-oL", "-eL"]
    cmd += [binary] + (status.get("extra_args") or [])
    cmd += ["-input", "input.in"]

    status["status"]     = "running"
    status["started_at"] = _now()
    _write_status(job_dir, status)
    logger.info("Running job %s: %s", job_id, " ".join(cmd))

    out_path = job_dir / "output.out"
    err_path = job_dir / "error.err"

    try:
        with open(out_path, "w") as fout, open(err_path, "w") as ferr:
            result = subprocess.run(
                cmd,
                cwd=str(job_dir),
                stdin=open(job_dir / "input.in"),
                stdout=fout,
                stderr=ferr,
                timeout=timeout,
            )
        status["exit_code"] = result.returncode
        if result.returncode == 0:
            status["status"] = "completed"
            logger.info("Job %s completed (exit=0)", job_id)
        else:
            status["status"] = "failed"
            status["error"]  = f"Non-zero exit code: {result.returncode}"
            # Capture tail of stderr for quick diagnosis
            try:
                err_tail = err_path.read_text(errors="replace")[-2000:]
                status["error"] += f"\n---\n{err_tail}"
            except Exception:
                pass
            logger.warning("Job %s failed (exit=%s)", job_id, result.returncode)

    except subprocess.TimeoutExpired:
        status["status"] = "timeout"
        status["error"]  = f"Timed out after {timeout}s"
        logger.error("Job %s timed out", job_id)
    except Exception as exc:
        status["status"] = "failed"
        status["error"]  = str(exc)
        logger.error("Job %s exception: %s", job_id, exc)

    status["ended_at"] = _now()
    _write_status(job_dir, status)
    return status


def get_job_status(job_id: str, workdir: str = _DEFAULT_WORKDIR) -> dict:
    """Return the status dict for a job. Returns error dict if not found."""
    job_dir = Path(workdir) / job_id
    if not job_dir.exists():
        return {"status": "not_found", "job_id": job_id}
    return _read_status(job_dir)


def list_jobs(workdir: str = _DEFAULT_WORKDIR,
              limit: int = 50,
              status_filter: str | None = None) -> list[dict]:
    """
    List all jobs in workdir, newest first.
    Optionally filter by status: created|running|completed|failed|timeout
    """
    work = Path(workdir)
    if not work.exists():
        return []
    jobs = []
    for d in sorted(work.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        try:
            s = _read_status(d)
            if status_filter is None or s.get("status") == status_filter:
                jobs.append(s)
        except Exception:
            continue
        if len(jobs) >= limit:
            break
    return jobs


def get_output(job_id: str, workdir: str = _DEFAULT_WORKDIR) -> str:
    """Read raw stdout of a completed job."""
    path = Path(workdir) / job_id / "output.out"
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def get_input(job_id: str, workdir: str = _DEFAULT_WORKDIR) -> str:
    """Read the input file of a job."""
    path = Path(workdir) / job_id / "input.in"
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def check_binaries(bin_dir: str = _DEFAULT_BIN_DIR) -> dict:
    """
    Check which QE binaries are available on PATH or in bin_dir.
    Returns {code: path_or_None} for each known binary.
    """
    result = {}
    for code, exe in QE_BINARIES.items():
        found = shutil.which(exe) or (
            str(Path(bin_dir) / exe) if (Path(bin_dir) / exe).exists() else None
        )
        result[code] = found
    return result


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_binary(code: str, bin_dir: str) -> Optional[str]:
    exe = QE_BINARIES.get(code, f"{code}.x")
    full = Path(bin_dir) / exe
    if full.exists():
        return str(full)
    on_path = shutil.which(exe)
    return on_path


def _write_status(job_dir: Path, status: dict):
    (job_dir / "status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )


def _read_status(job_dir: Path) -> dict:
    p = job_dir / "status.json"
    if not p.exists():
        return {"status": "unknown", "job_id": job_dir.name}
    return json.loads(p.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
