"""
Rebuild the FAISS knowledge index from ResearchKnowledge (D4, 2026-09-11).

Run after any purge, reset or re-extraction; S5 also calls it at the end of
every run. Standalone:  venv/bin/python -m tools.vector_index_maintenance
"""


def knowledge_records(conn) -> list:
    rows = conn.execute(
        """
        SELECT k.id, k.paper_id, p.workflow_id, k.category, k.value,
               COALESCE(k.sentence, ''), COALESCE(p.doi, ''), COALESCE(p.title, ''), p.year
        FROM ResearchKnowledge k JOIN Paper p ON p.id = k.paper_id
        ORDER BY k.id
        """).fetchall()
    out = []
    for kid, pid, wid, cat, val, sent, doi, title, year in rows:
        text = f"{cat} : {val}"
        if sent:
            text += f" | {sent[:200]}"
        out.append({"knowledge_id": kid, "paper_id": pid, "workflow_id": wid,
                    "category": cat, "value": val, "sentence": sent[:300],
                    "doi": doi, "title": title, "year": year, "text": text})
    return out


def rebuild_vector_index(conn, vector_service=None) -> int:
    if vector_service is None:
        from services.vector_db_service import VectorDBService
        vector_service = VectorDBService()
    records = knowledge_records(conn)
    vector_service.rebuild(records)
    return len(records)


if __name__ == "__main__":
    import sqlite3
    from pathlib import Path
    db = Path(__file__).resolve().parents[1] / "database" / "research_workflow.db"
    with sqlite3.connect(db) as c:
        print("rebuilt:", rebuild_vector_index(c))
