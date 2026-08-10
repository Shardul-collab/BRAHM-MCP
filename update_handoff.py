#!/usr/bin/env python3
"""
Append a dated session summary to SESSION_HANDOFF.md.
File is top-level under /mnt/d/brahm/ (NTFS) -> CRLF-safe write required.
"""
import shutil

TARGET = "/mnt/d/brahm/SESSION_HANDOFF.md"
BACKUP = TARGET + ".bak_20260713"

ENTRY = """
## Session: 2026-07-13 (Vishwakarma bug fixes)

**Fixed & verified end-to-end:**

- **Dual-job-ID bug** (`vishwakarma_api.py`, all `/calculate/*` endpoints): the
  job returned to the caller and the job actually run were two different
  UUIDs. Fixed by creating the job once, synchronously, before queuing the
  background task, and passing that same `job_id` into `workflow.py`'s
  functions (each now accepts an optional `job_id` param and reuses it
  instead of creating its own). Verified with a real Si SCF run: single
  matching job_id from request -> poll -> server log -> `completed`, exit 0.
  Backup: `vishwakarma_api.py.bak_dualjobid`, `workflow.py.bak_dualjobid`.

- **SCF -> NSCF directory isolation bug** (`workflow.py`, `dos_workflow()` and
  `band_structure()`): each QE step runs in its own isolated job directory
  (`runner.run_job` uses `cwd=job_dir`), so a relative `outdir` in the NSCF
  step resolved against its own empty directory instead of the SCF step's
  charge density. Fixed by resolving `outdir` to an absolute path pointing at
  the SCF job's own directory (`_resolve_shared_outdir()`, new helper) and
  passing that same absolute path into NSCF and dos.x/bands.x. Job isolation
  and independent monitoring are fully preserved - only the QE-level outdir
  value is shared. Verified with a real Si DOS run: SCF -> NSCF -> dos.x all
  completed (exit 0 each), where NSCF previously failed instantly with
  MPI_ABORT (same failure signature confirmed in yesterday's
  MoS2_3x3_S_vacancy_dos_nscf job in the job history).
  Backup: `workflow.py.bak_outdirlink` (applied on top of the dual-job-ID fix).

**Not started this session:** Priority 3 (critic JSON-escape fix verification)
and Priority 4 (G3 result-aggregation bug) - both need GANESH up and Groq TPD
headroom; GANESH was left down since it wasn't needed for today's Vishwakarma
work.

**New direction identified for next session:** BRAHM has no orchestration
layer or user-facing interface - no way for someone to type a research prompt
and have it converted into a concrete multi-agent workflow and executed.
Next session's focus shifts from bug-fixing to designing this: a
prompt -> workflow interpretation layer sitting above SHANI/GANESH/Vishwakarma/
Chitragupta/VIDUR, plus a usable (non-technical) interface, aimed at making
BRAHM something other people can actually run. Shardul has interface
reference material (reports/mockups) to share as a starting point.
"""

def main():
    shutil.copy2(TARGET, BACKUP)
    print(f"Backup written: {BACKUP}")

    with open(TARGET, "r", newline='') as f:
        content = f.read()

    # Normalize the new entry to CRLF to match file convention
    entry_crlf = ENTRY.replace("\r\n", "\n").replace("\n", "\r\n")

    with open(TARGET, "a", newline='') as f:
        f.write(entry_crlf)

    print(f"Appended session summary to {TARGET}")

if __name__ == "__main__":
    main()
