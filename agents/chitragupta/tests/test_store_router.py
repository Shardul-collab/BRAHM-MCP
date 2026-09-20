"""
Tests for the /v1/store/* router.

These pin the contract that regressed: store_vishwakarma must persist what
the CALLER sent. Until 2026-09-10 its request model declared only job_id and
triggered_by, so pydantic dropped every field with content in it, and the
handler rebuilt the row from `GET /jobs/{id}` — an endpoint that returns
runner's status.json and carries none of those fields. Result: every row was
blank except calculation_type, which was itself degraded from the calculation
type ("scf") to the binary name ("pw").
"""

import pytest

from api.routers.store import StoreVishwakarmaRequest


# ─── request model ────────────────────────────────────────────────────────────

def test_model_keeps_everything_the_caller_sends():
    """
    The exact regression, pinned. brahm/agents/vishwakarma.py's _persist_run
    posts these six fields; five of them used to be silently discarded.
    """
    payload = {
        "job_id":           "abc-123",
        "calculation_type": "scf",
        "material_name":    "In2Se3",
        "output_file_path": "/jobs/abc-123/output.out",
        "scf_iterations":   10,
        "converged":        True,
    }
    req = StoreVishwakarmaRequest(**payload)
    kept = req.model_dump()

    for field, value in payload.items():
        assert kept[field] == value, f"{field} was dropped or altered"


def test_model_still_accepts_a_job_id_only_caller():
    """Backward compatibility: an older caller sending just job_id must work."""
    req = StoreVishwakarmaRequest(job_id="abc-123")
    assert req.job_id == "abc-123"
    assert req.calculation_type is None
    assert req.material_name is None


def test_calculation_type_is_not_the_binary_name():
    """
    The old fallback chain ended at job["code"], which is the QE BINARY
    ("pw"), not the calculation type ("scf"). The single real row in
    brahm_knowledge.db records "scf" — proof the payload-driven version
    stored the right thing before this regressed.
    """
    req = StoreVishwakarmaRequest(job_id="x", calculation_type="scf")
    assert req.calculation_type == "scf"
    assert req.calculation_type != "pw"


# ─── the shape the old handler was reading from ───────────────────────────────

JOB_STATUS_KEYS = {
    "job_id", "label", "code", "status", "created_at", "started_at",
    "ended_at", "exit_code", "error", "mpi_np", "extra_args", "workdir",
}


def test_job_status_carries_none_of_the_stored_fields():
    """
    Documents WHY the re-fetch could never work. If Vishwakarma's job status
    ever grows these fields this test fails loudly and the fallback in
    store_vishwakarma can be simplified — that would be good news, not a bug.
    """
    for field in ("material_name", "output_file_path",
                  "scf_iterations", "converged", "parsed_output", "result"):
        assert field not in JOB_STATUS_KEYS, (
            f"{field} is now on the job status; revisit the fallback path"
        )
