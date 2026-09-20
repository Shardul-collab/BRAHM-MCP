"""G2 must write GaneshSection rows the live schema accepts (2026-09-12).

The first real G2 run failed with "table GaneshSection has no column named depends_on":
ganesh/schema.py creates `dependencies`, plan_document inserted `depends_on`, and the root
section_graph.py selected `depends_on` - so G2 -> G3 had never completed on this schema.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(os.path.dirname(ROOT), "shani")):
    if p not in sys.path:
        sys.path.insert(0, p)
from repositories.repository import Repository          # noqa: E402  (SHANI's repository)
from ganesh.schema import run_migration                 # noqa: E402
from ganesh.tools.plan_document import plan_document    # noqa: E402
from section_graph import SectionGraph                  # noqa: E402
import ganesh.tools.plan_document as PD                 # noqa: E402


def test_g2_writes_sections_the_schema_accepts_and_the_graph_reads(tmp_path, monkeypatch):
    repo = Repository(str(tmp_path / "t.db"))
    run_migration(repo)
    now = "2026-09-12T00:00:00"
    with repo.transaction() as cur:
        cur.execute("INSERT INTO GaneshDocument (title, document_type, status, source_type, source_ids, "
                    "total_iterations, created_at, updated_at) VALUES ('T','literature_review','planning',"
                    "'shani','[1]',0,?,?)", (now, now))
        doc = cur.lastrowid
        cur.execute("INSERT INTO GaneshContext (document_id, context_type, context_ref, context_json, created_at) "
                    "VALUES (?, 'shani', '[1]', ?, ?)", (doc, json.dumps({"knowledge_summary": {}}), now))
    # no LLM in the test: the template fallback must be enough to produce a valid plan
    monkeypatch.setattr(PD, "call_llm_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    out = plan_document(repo, doc, {"document_type": "literature_review", "source_ids": "[1]"})
    assert out["sections_planned"] == 9
    graph = SectionGraph.from_document(repo, doc)
    nodes = graph._nodes
    assert "Introduction" in nodes and "Conclusion" in nodes and len(nodes) == 9
    assert nodes["Synthesis Methods"].depends_on == ["Materials Overview"]
    repo.close()
