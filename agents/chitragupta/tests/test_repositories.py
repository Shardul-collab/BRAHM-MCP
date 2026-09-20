"""
Repository tests for brahm_db.

The id-recovery contract is the point here. Every repository used to recover
a freshly-inserted row's id by re-querying after commit:

    SELECT id FROM <table> [WHERE project_id=?] ORDER BY id DESC LIMIT 1

which returns whatever row is newest at SELECT time, not the row the call
inserted. `lastrowid` was used nowhere. These tests assert the returned id
belongs to the row the caller actually created.

Honest note: the race was NOT reproducible (40 trials x 16 concurrent threads
through register_paper produced zero mis-attributed ids — SQLite serialises
writers and the window is tiny). These tests therefore verify correctness of
the returned id, not the absence of a race; the interleaving test below is a
best-effort regression net, not a proof.
"""

import threading

import pytest


# ─── id recovery returns the caller's own row ─────────────────────────────────

def test_create_project_returns_its_own_id(scratch_db):
    from brahm_db.repositories import ProjectRepo

    with ProjectRepo() as r:
        pid_a = r.create_project("Project A", "objective A")
        pid_b = r.create_project("Project B", "objective B")

        assert pid_a != pid_b
        assert r.get_project(pid_a)["name"] == "Project A"
        assert r.get_project(pid_b)["name"] == "Project B"


def test_same_named_projects_get_distinct_ids(scratch_db):
    """
    create_project recovered its id with `WHERE name=? ORDER BY id DESC`.
    Two projects sharing a name is legal — nothing enforces uniqueness — so
    that lookup was ambiguous by construction.
    """
    from brahm_db.repositories import ProjectRepo

    with ProjectRepo() as r:
        first = r.create_project("Duplicate Name", "first")
        second = r.create_project("Duplicate Name", "second")

        assert first != second
        assert r.get_project(first)["objective"] == "first"
        assert r.get_project(second)["objective"] == "second"


def test_register_paper_returns_its_own_id(scratch_db):
    """
    register_paper's recovery query had no WHERE clause at all —
    `SELECT id FROM GlobalPaper ORDER BY id DESC LIMIT 1` — so it returned
    the newest paper globally, regardless of who inserted it.
    """
    from brahm_db.repositories import PaperRepo

    with PaperRepo() as r:
        a = r.register_paper(doi="10.1/aaa", title="Paper A", abstract="")
        b = r.register_paper(doi="10.1/bbb", title="Paper B", abstract="")

        assert a != b
        assert r.get_paper(a)["title"] == "Paper A"
        assert r.get_paper(b)["title"] == "Paper B"


def test_dft_result_returns_its_own_id(scratch_db):
    from brahm_db.repositories import ProjectRepo, DFTResultRepo

    with ProjectRepo() as p:
        pid = p.create_project("DFT proj", "obj")

    with DFTResultRepo() as r:
        first = r.save(project_id=pid, job_id="job-1", calc_type="scf")
        second = r.save(project_id=pid, job_id="job-2", calc_type="relax")

        assert first != second
        assert r.get(first)["job_id"] == "job-1"
        assert r.get(second)["job_id"] == "job-2"


def test_failed_dft_status_is_stored_as_given(scratch_db):
    """
    Vishwakarma now sends a real status. Confirm the repo round-trips it
    rather than defaulting everything to completed.
    """
    from brahm_db.repositories import ProjectRepo, DFTResultRepo

    with ProjectRepo() as p:
        pid = p.create_project("Status proj", "obj")

    with DFTResultRepo() as r:
        rid = r.save(project_id=pid, job_id="job-fail",
                     calc_type="scf", status="failed")
        assert r.get(rid)["status"] == "failed"


# ─── best-effort concurrency net ──────────────────────────────────────────────

def test_concurrent_inserts_each_get_their_own_id(scratch_db):
    """
    Not a proof — see the module docstring. With lastrowid this cannot fail;
    with the old re-query it was merely unlikely to.
    """
    from brahm_db.repositories import PaperRepo

    got, lock = [], threading.Lock()
    barrier = threading.Barrier(8)

    def worker(i):
        with PaperRepo() as r:
            barrier.wait()
            pid = r.register_paper(doi=f"10.2/{i}", title=f"Concurrent {i}",
                                   abstract="")
        with lock:
            got.append((i, pid))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with PaperRepo() as r:
        for i, pid in got:
            assert r.get_paper(pid)["title"] == f"Concurrent {i}"
