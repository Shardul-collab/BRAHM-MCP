"""
Regression tests for the workflow result contract.

These exist because of a defect that ran silently for months: every run_*
tool in brahm/agents/vishwakarma.py read job_id, converged and
scf_iterations off the TOP LEVEL of a workflow result, where those keys have
never existed. dict.get() returns None rather than raising, so every DFT
record written to brahm.db carried an empty job_id and a null convergence
flag, and nothing ever failed loudly enough to notice.

The contract these tests lock down:
  1. _workflow_result exposes a "summary" with the flat fields consumers want.
  2. summary.job_id is the FINAL step's job, not the first.
  3. "reused" steps (phonon recover, hp attach) do not count as failures.
  4. A failed step yields success=False, which the persistence layer maps to
     a "failed" status rather than the hardcoded "completed" it used to send.
"""

import pytest

from vishwakarma import workflow as wf


def _step(name, jid, status="completed", parsed=None):
    return {"step": name, "job_id": jid,
            "status": {"status": status}, "parsed": parsed}


# ─── summary shape ────────────────────────────────────────────────────────────

def test_result_exposes_summary():
    r = wf._workflow_result("t", [_step("scf", "j1", parsed={"converged": True})])
    assert "summary" in r, "consumers depend on a flat summary being present"
    assert set(r["summary"]) >= {
        "job_id", "converged", "scf_iterations", "total_energy_ev"
    }


def test_summary_job_id_is_the_last_step():
    """
    For DOS/bands the scientifically relevant job is the post-processing
    step, not the SCF that seeded it.
    """
    r = wf._workflow_result("dos_workflow", [
        _step("scf",  "job-scf",  parsed={"converged": True}),
        _step("nscf", "job-nscf", parsed={"converged": True}),
        _step("dos",  "job-dos",  parsed={"converged": True}),
    ])
    assert r["summary"]["job_id"] == "job-dos"


def test_summary_pulls_fields_from_the_last_step_that_parsed_them():
    """bands.x parses to almost nothing; the SCF numbers must still survive."""
    r = wf._workflow_result("band_structure", [
        _step("scf", "j1", parsed={"converged": True, "scf_iterations": 9,
                                   "total_energy_ev": -215.5}),
        _step("bands_pp", "j2", parsed={"converged": True}),
    ])
    assert r["summary"]["scf_iterations"] == 9
    assert r["summary"]["total_energy_ev"] == pytest.approx(-215.5)


def test_summary_is_safe_on_empty_steps():
    r = wf._workflow_result("t", [])
    assert r["summary"]["job_id"] == ""
    assert r["summary"]["converged"] is None


# ─── success semantics ────────────────────────────────────────────────────────

def test_reused_step_does_not_fail_the_workflow():
    """
    phonon_workflow(recover=True) and hp_workflow's attach modes mark the
    skipped SCF as "reused". Only "completed" used to count as OK, so a
    perfectly good recover run reported success=False.
    """
    r = wf._workflow_result("phonon_workflow", [
        _step("scf", "old-job", status="reused"),
        _step("phonon", "ph-job", status="completed", parsed={"converged": True}),
    ])
    assert r["success"] is True


def test_failed_step_makes_the_workflow_unsuccessful():
    r = wf._workflow_result("scf_only", [_step("scf", "j1", status="failed")])
    assert r["success"] is False


def test_failed_at_overrides_otherwise_ok_steps():
    r = wf._workflow_result("dos_workflow",
                            [_step("scf", "j1")], failed_at="scf")
    assert r["success"] is False


# ─── the persistence contract that consumes the above ─────────────────────────

def test_persistence_layer_reads_the_summary_not_the_top_level():
    """
    The exact bug, pinned: reading job_id off the top level yields "", while
    the helper must find the real one.
    """
    from brahm.agents.vishwakarma import _run_summary, _run_status

    result = wf._workflow_result("dos_workflow", [
        _step("scf", "job-scf", parsed={"converged": True, "scf_iterations": 7}),
        _step("dos", "job-dos", parsed={"converged": True}),
    ])
    result["status"] = "success"          # what _ok() merges in

    assert result.get("job_id", "") == "", "top-level job_id must not exist"
    assert _run_summary(result)["job_id"] == "job-dos"
    assert _run_status(result) == "completed"


def test_persistence_status_reflects_failure():
    from brahm.agents.vishwakarma import _run_status

    failed = wf._workflow_result("scf_only", [_step("scf", "j1", status="failed")])
    assert _run_status(failed) == "failed", (
        "a failed calculation must not be stored as completed"
    )
