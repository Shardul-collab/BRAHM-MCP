"""
Tests for the orchestrator's resume decision.

The pipeline is deliberately one-directional. The bug these pin was in the
RECOVERY path, which violated that invariant harder than anything else could:
a Stage row left at 'running' by an interrupted process matched neither the
'completed' nor the 'failed' branch, fell through to `else`, and restarted the
entire pipeline from S1.

Measured consequences in workflow 2: S1 ran 5x, S2 5x, S4 3x, S5 4x with a
single completion — 31% of all recorded pipeline time — and the repeated S2
runs pushed a max_papers=20 workflow to 40 ingested papers.

Two subtler faults in the same three lines, also pinned below:
  * ORDER BY id DESC reads the LAST-INSERTED row, not the furthest-along
    stage, so after a retry the newest row can be an earlier stage.
  * A workflow with no Stage rows and one with a stale 'running' row landed
    in the same branch — "never started" and "interrupted at S5" were
    indistinguishable.
"""

import os
import sqlite3
import sys

import pytest

# The orchestrator imports its siblings as top-level modules ("from
# repositories.repository import ..."), so both the agent root and core/ have
# to be importable. Without this the suite only passes when pytest happens to
# be invoked from a directory that already put them on sys.path.
_SHANI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SHANI, os.path.join(_SHANI, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


SEQUENCE = ("S1", "S2", "S2_75", "S2_5", "S3", "S4", "S5", "S5_5")


class FakeRepo:
    """Minimal stand-in: the resume logic only reads Stage and config."""

    def __init__(self, stages, max_papers=None, paper_count=0, use_local=0):
        self._stages = stages          # list of (stage_name, status, ended_at)
        self._max_papers = max_papers
        self._paper_count = paper_count
        self._use_local = use_local
        self.executed = []

    def fetch_all(self, sql, params=()):
        if "FROM Stage" in sql and "ended_at IS NULL" in sql:
            return [{"id": i, "stage_name": s[0]}
                    for i, s in enumerate(self._stages)
                    if s[2] is None and s[1] not in
                    ("completed", "failed", "interrupted")]
        if "FROM Stage" in sql:
            return [{"stage_name": s[0], "status": s[1], "ended_at": s[2]}
                    for s in self._stages]
        return []

    def fetch_one(self, sql, params=()):
        if "use_local" in sql:
            return {"use_local": self._use_local}
        if "max_papers" in sql:
            return {"max_papers": self._max_papers}
        if "COUNT(*) AS n FROM Paper" in sql:
            return {"n": self._paper_count}
        return None

    def execute(self, sql, params=()):
        self.executed.append((sql, params))


def resume_point(repo, use_local=0):
    """Invoke the real logic against the fake repo."""
    from orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)   # bypass ToolExecutor
    orch.repo = repo
    config = {"use_local": use_local}
    return orch._resume_point(1, config)


# ─── the exact regression ─────────────────────────────────────────────────────

def test_stale_running_stage_resumes_instead_of_restarting():
    """
    S1-S4 completed, S5 interrupted and left at 'running'. The old code
    restarted at S1. It must resume at S5.
    """
    repo = FakeRepo([
        ("S1", "completed", "t"), ("S2", "completed", "t"),
        ("S2_75", "completed", "t"), ("S2_5", "completed", "t"),
        ("S3", "completed", "t"), ("S4", "completed", "t"),
        ("S5", "running", None),
    ])
    stage, reason = resume_point(repo)
    assert stage == "S5", f"restarted instead of resuming ({reason})"


def test_furthest_completed_wins_not_last_inserted():
    """
    ORDER BY id DESC took the newest ROW. Here S2 was retried after S4
    completed, so the newest row is S2 — the old code would resume at S2_75
    and redo S2_5, S3, S4.
    """
    repo = FakeRepo([
        ("S1", "completed", "t"), ("S2", "completed", "t"),
        ("S2_75", "completed", "t"), ("S2_5", "completed", "t"),
        ("S3", "completed", "t"), ("S4", "completed", "t"),
        ("S2", "completed", "t"),          # retried later — newest row
    ])
    stage, _ = resume_point(repo)
    assert stage == "S5", "resume point came from the newest row, not the furthest stage"


def test_no_history_is_distinguishable_from_stale_running():
    fresh = FakeRepo([])
    interrupted = FakeRepo([("S1", "completed", "t"), ("S2", "running", None)])

    assert resume_point(fresh)[0] == "S1"
    # S1 completed, S2 was interrupted and never completed -> re-enter S2.
    # That is correct and now cheap: the S2 guard skips the search when the
    # workflow already holds its papers, so re-entry no longer duplicates.
    assert resume_point(interrupted)[0] == "S2", (
        "a stale running row must not look like a fresh workflow"
    )


# ─── ordinary paths still behave ──────────────────────────────────────────────

def test_failed_stage_is_retried():
    """Furthest completed is S1, so the failed S2 is re-entered."""
    repo = FakeRepo([("S1", "completed", "t"), ("S2", "failed", "t")])
    assert resume_point(repo)[0] == "S2"


def test_completed_workflow_returns_none():
    repo = FakeRepo([(s, "completed", "t") for s in SEQUENCE])
    stage, reason = resume_point(repo)
    assert stage is None
    assert "completed" in reason


def test_fresh_use_local_starts_at_s4():
    repo = FakeRepo([])
    from orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch.repo = repo
    orch.ingest_local_papers = lambda wid: None    # side effect not under test
    stage, _ = orch._resume_point(1, {"use_local": 1})
    assert stage == "S4"


def test_completed_without_ended_at_is_not_trusted():
    """
    'completed' with a NULL ended_at is not a finished stage — it is a row
    that was never closed out. Treating it as done would skip real work.
    """
    repo = FakeRepo([("S1", "completed", "t"), ("S2", "completed", None)])
    stage, _ = resume_point(repo)
    assert stage == "S2", "an unclosed row was treated as a completed stage"


# ─── S2 idempotency: the guard that stops corpus duplication ──────────────────

def _s2_satisfied(max_papers, paper_count):
    from orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch.repo = FakeRepo([], max_papers=max_papers, paper_count=paper_count)
    return orch._s2_satisfied(1)


def test_s2_skipped_once_target_papers_exist():
    assert _s2_satisfied(20, 20) is True
    assert _s2_satisfied(20, 21) is True


def test_s2_runs_when_short_of_target():
    assert _s2_satisfied(20, 19) is False
    assert _s2_satisfied(20, 0) is False


def test_s2_without_max_papers_skips_once_any_paper_exists():
    """
    search_papers falls back to FINAL_PAPER_LIMIT=500 when max_papers is
    unset. Counting toward 500 would never trip, so every resume would
    re-ingest — which is exactly how workflow 2 reached 40 papers.
    """
    assert _s2_satisfied(None, 1) is True
    assert _s2_satisfied(None, 0) is False
