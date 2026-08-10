"""
ganesh/schema.py
=================
GANESH database schema migration — idempotent, runs on every startup.
"""
from __future__ import annotations
import logging

log = logging.getLogger("ganesh.schema")

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS GaneshDocument (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        title             TEXT    NOT NULL,
        document_type     TEXT    NOT NULL DEFAULT 'literature_review',
        status            TEXT    NOT NULL DEFAULT 'planning',
        source_type       TEXT    NOT NULL DEFAULT 'shani',
        source_ids        TEXT    NOT NULL DEFAULT '[]',
        outline_json      TEXT,
        final_output      TEXT,
        quality_flag      TEXT,
        total_iterations  INTEGER NOT NULL DEFAULT 0,
        project_id        INTEGER,
        created_at        TEXT    NOT NULL,
        updated_at        TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS GaneshContext (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id     INTEGER NOT NULL REFERENCES GaneshDocument(id),
        context_type    TEXT    NOT NULL DEFAULT 'shani',
        context_ref     TEXT    NOT NULL DEFAULT '[]',
        context_json    TEXT    NOT NULL,
        paper_count     INTEGER NOT NULL DEFAULT 0,
        knowledge_count INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS GaneshSection (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id     INTEGER NOT NULL REFERENCES GaneshDocument(id),
        section_name    TEXT    NOT NULL,
        section_type    TEXT    NOT NULL DEFAULT 'body',
        status          TEXT    NOT NULL DEFAULT 'pending',
        brief_json      TEXT,
        dependencies    TEXT    NOT NULL DEFAULT '[]',
        exec_order      INTEGER NOT NULL DEFAULT 0,
        quality_score   REAL,
        iteration_count INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT    NOT NULL,
        updated_at      TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS GaneshDraft (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        section_id  INTEGER NOT NULL REFERENCES GaneshSection(id),
        version     INTEGER NOT NULL DEFAULT 1,
        content     TEXT    NOT NULL,
        word_count  INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS GaneshCritique (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        section_id    INTEGER NOT NULL REFERENCES GaneshSection(id),
        draft_id      INTEGER NOT NULL REFERENCES GaneshDraft(id),
        scope         TEXT    NOT NULL DEFAULT 'section',
        scores_json   TEXT    NOT NULL DEFAULT '{}',
        issues_json   TEXT    NOT NULL DEFAULT '[]',
        overall_score REAL    NOT NULL DEFAULT 0.0,
        created_at    TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS GaneshRevision (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        section_id       INTEGER NOT NULL REFERENCES GaneshSection(id),
        from_draft_id    INTEGER NOT NULL REFERENCES GaneshDraft(id),
        to_draft_id      INTEGER NOT NULL REFERENCES GaneshDraft(id),
        critique_id      INTEGER REFERENCES GaneshCritique(id),
        changes_summary  TEXT,
        created_at       TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ganesh_section_document ON GaneshSection(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_ganesh_draft_section    ON GaneshDraft(section_id)",
    "CREATE INDEX IF NOT EXISTS idx_ganesh_critique_section ON GaneshCritique(section_id)",
    "CREATE INDEX IF NOT EXISTS idx_ganesh_critique_draft   ON GaneshCritique(draft_id)",
    "CREATE INDEX IF NOT EXISTS idx_ganesh_context_document ON GaneshContext(document_id)",
]

def run_migration(repo) -> dict:
    tables_checked = 0
    errors = []
    for ddl in _DDL:
        stmt = ddl.strip()
        try:
            repo._conn.execute(stmt)
            repo._conn.commit()
            tables_checked += 1
        except Exception as exc:
            log.error("Migration DDL failed: %s\nError: %s", stmt[:80], exc)
            errors.append(str(exc))
    if errors:
        log.warning("Migration completed with %d error(s): %s", len(errors), errors)
    else:
        log.info("GANESH migration complete — %d statements applied.", tables_checked)
    return {"status": "ok" if not errors else "partial", "tables_checked": tables_checked, "errors": errors}
