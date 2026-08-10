"""
migrate_constraints.py
======================
Migrates an existing research_workflow.db to the updated schema.

Changes applied
---------------
1. Paper.status CHECK constraint — adds:
     extracted, knowledge_ready, extraction_failed, knowledge_failed

2. DraftSection.section_status CHECK constraint — adds:
     fallback

3. DraftSection — adds column:
     inline_tables_json TEXT DEFAULT NULL

SQLite limitation
-----------------
SQLite does not support ALTER TABLE ... MODIFY COLUMN or
ALTER TABLE ... DROP CONSTRAINT.  The only way to change a
CHECK constraint on an existing table is:

  1. Rename the old table.
  2. Create the new table with the updated constraint.
  3. Copy all data.
  4. Drop the old table.

This script performs that operation inside a single transaction
so the database is never left in a half-migrated state.
Foreign-key enforcement is disabled during the rename/copy to
avoid constraint ordering issues, then re-enabled at the end.

Usage
-----
  python scripts/migrate_constraints.py

Safe to run multiple times — checks whether migration is
already applied before touching anything.
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from repositories.repository import DB_PATH


# ── helpers ──────────────────────────────────────────────────

def _table_exists(cur, name: str) -> bool:
    row = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)
    ).fetchone()
    return row is not None


def _column_exists(cur, table: str, column: str) -> bool:
    rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def _get_check_sql(cur, table: str) -> str:
    """Return the CREATE TABLE SQL for a table from sqlite_master."""
    row = cur.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    ).fetchone()
    return row[0] if row else ""


def _check_already_has_value(cur, table: str, value: str) -> bool:
    """Return True if the CHECK constraint SQL already contains value."""
    sql = _get_check_sql(cur, table)
    return f"'{value}'" in sql


# ── Paper migration ───────────────────────────────────────────

PAPER_NEW_DDL = """
CREATE TABLE Paper (
    id INTEGER PRIMARY KEY,
    workflow_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    file_path TEXT,
    pdf_url TEXT,
    abstract TEXT,
    pdf_candidates TEXT,
    pdf_status TEXT DEFAULT 'metadata',
    pdf_path TEXT,
    doi TEXT,
    failed_candidates TEXT,
    last_error TEXT,
    status TEXT NOT NULL
        CHECK (status IN (
            'pending',
            'downloading',
            'processing',
            'extracted',
            'knowledge_ready',
            'completed',
            'failed',
            'extraction_failed',
            'knowledge_failed'
        )),
    raw_text TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    UNIQUE(workflow_id, title),
    FOREIGN KEY (workflow_id)
        REFERENCES Workflow(id)
        ON DELETE CASCADE
);
"""

PAPER_COLUMNS = """
    id, workflow_id, title, source, file_path, pdf_url,
    abstract, pdf_candidates, pdf_status, pdf_path, doi,
    failed_candidates, last_error, status, raw_text,
    created_at, updated_at
"""


def migrate_paper(cur):
    if _check_already_has_value(cur, "Paper", "extracted"):
        print("[migrate] Paper.status already has 'extracted' — skipping Paper migration.")
        return

    print("[migrate] Migrating Paper table...")
    cur.execute("ALTER TABLE Paper RENAME TO Paper_old;")
    cur.execute(PAPER_NEW_DDL)
    cur.execute(f"INSERT INTO Paper ({PAPER_COLUMNS}) SELECT {PAPER_COLUMNS} FROM Paper_old;")
    cur.execute("DROP TABLE Paper_old;")
    print("[migrate] Paper table migrated.")


# ── DraftSection migration ────────────────────────────────────

DRAFT_SECTION_NEW_DDL = """
CREATE TABLE DraftSection (
    id INTEGER PRIMARY KEY,
    workflow_id INTEGER NOT NULL,
    section_name TEXT NOT NULL,
    content TEXT NOT NULL,
    section_status TEXT NOT NULL DEFAULT 'completed'
        CHECK (section_status IN (
            'pending',
            'completed',
            'fallback',
            'failed',
            'skipped'
        )),
    figures_json TEXT DEFAULT NULL,
    tables_json TEXT DEFAULT NULL,
    equations_json TEXT DEFAULT NULL,
    inline_tables_json TEXT DEFAULT NULL,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (workflow_id)
        REFERENCES Workflow(id)
        ON DELETE CASCADE
);
"""

DRAFT_SECTION_SHARED_COLUMNS = """
    id, workflow_id, section_name, content, section_status,
    figures_json, tables_json, equations_json, created_at
"""


def migrate_draft_section(cur):
    needs_check = not _check_already_has_value(cur, "DraftSection", "fallback")
    needs_col   = not _column_exists(cur, "DraftSection", "inline_tables_json")

    if not needs_check and not needs_col:
        print("[migrate] DraftSection already up to date — skipping.")
        return

    if needs_col and not needs_check:
        # Only the column is missing — a simple ALTER TABLE suffices.
        print("[migrate] Adding inline_tables_json column to DraftSection...")
        cur.execute(
            "ALTER TABLE DraftSection ADD COLUMN inline_tables_json TEXT DEFAULT NULL;"
        )
        print("[migrate] Column added.")
        return

    # CHECK constraint needs changing — must recreate.
    print("[migrate] Migrating DraftSection table...")
    cur.execute("ALTER TABLE DraftSection RENAME TO DraftSection_old;")
    cur.execute(DRAFT_SECTION_NEW_DDL)

    # Copy shared columns; inline_tables_json gets NULL for existing rows.
    cur.execute(
        f"INSERT INTO DraftSection ({DRAFT_SECTION_SHARED_COLUMNS}, inline_tables_json) "
        f"SELECT {DRAFT_SECTION_SHARED_COLUMNS}, NULL FROM DraftSection_old;"
    )
    cur.execute("DROP TABLE DraftSection_old;")
    print("[migrate] DraftSection table migrated.")


# ── index recreation ──────────────────────────────────────────

def recreate_indexes(cur):
    """Recreate any indexes that were on the renamed tables."""
    # The UNIQUE(workflow_id, title) constraint on Paper is part of the
    # table DDL and is recreated automatically with the new table.
    # Add any explicit CREATE INDEX statements here if they exist.
    pass


# ── main ─────────────────────────────────────────────────────

def run_migration():
    print(f"[migrate] Target database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # Disable FK enforcement during rename/copy to avoid ordering issues.
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute("PRAGMA journal_mode = WAL;")   # safer for multi-step ops

    try:
        cur = conn.cursor()

        # Wrap everything in one transaction — all-or-nothing.
        cur.execute("BEGIN;")

        migrate_paper(cur)
        migrate_draft_section(cur)
        recreate_indexes(cur)

        conn.commit()
        print("[migrate] Migration committed successfully.")

    except Exception as e:
        conn.rollback()
        print(f"[migrate] ERROR — transaction rolled back: {e}")
        raise

    finally:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.close()

    # Verify
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    paper_sql = _get_check_sql(cur, "Paper")
    ds_sql    = _get_check_sql(cur, "DraftSection")
    ds_col    = _column_exists(cur, "DraftSection", "inline_tables_json")
    conn.close()

    ok = True
    for val in ("extracted", "knowledge_ready", "extraction_failed", "knowledge_failed"):
        if f"'{val}'" not in paper_sql:
            print(f"[migrate] VERIFY FAIL: Paper.status missing '{val}'")
            ok = False

    if "'fallback'" not in ds_sql:
        print("[migrate] VERIFY FAIL: DraftSection.section_status missing 'fallback'")
        ok = False

    if not ds_col:
        print("[migrate] VERIFY FAIL: DraftSection.inline_tables_json column missing")
        ok = False

    if ok:
        print("[migrate] Verification passed. Schema is correct.")
    else:
        print("[migrate] Verification FAILED — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    run_migration()
