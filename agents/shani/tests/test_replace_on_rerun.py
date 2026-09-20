"""
Re-running a stage on a paper replaces its output instead of appending
(2026-09-11), and the vector index is a pure function of the DB (D4).
"""
import os
import sqlite3
import sys
import tempfile

SHANI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SHANI not in sys.path:
    sys.path.insert(0, SHANI)

from tools.vector_index_maintenance import knowledge_records, rebuild_vector_index  # noqa: E402


class FakeVS:
    def __init__(self):
        self.records = None

    def rebuild(self, records):
        self.records = records


def _db():
    path = tempfile.mktemp(suffix=".db")
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE Paper (id INTEGER PRIMARY KEY, workflow_id INTEGER, doi TEXT, title TEXT, year INTEGER);
        CREATE TABLE ResearchKnowledge (id INTEGER PRIMARY KEY, paper_id INTEGER, category TEXT,
            value TEXT, sentence TEXT, source_type TEXT);
        INSERT INTO Paper VALUES (1, 1, '10.1/x', 'Paper one', 2020), (2, 1, NULL, 'Paper two', NULL);
        INSERT INTO ResearchKnowledge VALUES
            (1, 1, 'material', 'In2Se3', 'Films of In2Se3.', 'llm'),
            (2, 2, 'characterization', 'XRD', NULL, 'abstract');
    """)
    return c


def test_index_is_rebuilt_from_live_rows_only():
    c = _db()
    vs = FakeVS()
    assert rebuild_vector_index(c, vs) == 2
    c.execute("DELETE FROM ResearchKnowledge WHERE id = 1")
    c.execute("INSERT INTO ResearchKnowledge VALUES (1, 2, 'material', 'GaSe', 'x', 'llm')")  # id reused
    rebuild_vector_index(c, vs)
    got = {(r["knowledge_id"], r["paper_id"], r["value"]) for r in vs.records}
    assert got == {(1, 2, "GaSe"), (2, 2, "XRD")}          # no stale In2Se3 vector


def test_abstract_rows_are_embedded_too():
    recs = knowledge_records(_db())
    assert any(r["category"] == "characterization" and r["paper_id"] == 2 for r in recs)
    assert recs[0]["text"] == "material : In2Se3 | Films of In2Se3."
