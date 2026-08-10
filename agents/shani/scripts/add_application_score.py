"""
add_application_score.py
=========================
Adds a nullable application_score column to Paper.

Rationale
---------
paper_ingestor.py already computes compute_application_score() per paper
at ingestion time (S2), scoring whether the paper's stated application
(from _APPLICATION_TERMS) overlaps with the workflow's target application
(from WorkflowResearchConfig.focus/properties). This score is correctly
computed but only used transiently inside the blended _WEIGHTS score
(weight=0.05) and then discarded — never persisted.

Confirmed case: paper_id=24 (Ag/Cr2O3 antibacterial nanoparticle paper,
workflow 1 / MoS2 FET) scores application_score=0.0 correctly (zero
overlap between config's {"transistor"} and paper's {} application sets),
but the 0.05 weight lets material_score (0.40, high due to literal "MoS2"
in title) carry it past MIN_SCORE=0.12 anyway.

This migration adds the column so the already-correct signal can be
persisted (going forward at ingestion, and backfilled for existing rows)
and used as a real filter downstream in Chitragupta's /context/load,
without changing SHANI's ingestion/scoring behavior at all.

SQLite note
-----------
No CHECK constraint is involved, so this is a plain ALTER TABLE ADD
COLUMN — no rename/recreate/copy needed (unlike migrate_constraints.py's
Paper.status migration, which changes a CHECK constraint).

Usage
-----
  python scripts/add_application_score.py

Safe to run multiple times — checks whether the column already exists
before touching anything.
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from repositories.repository import DB_PATH


def _column_exists(cur, table: str, column: str) -> bool:
    rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def run_migration():
    print(f"[migrate] Target database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()

        if _column_exists(cur, "Paper", "application_score"):
            print("[migrate] Paper.application_score already exists — skipping.")
            return

        print("[migrate] Adding application_score column to Paper...")
        cur.execute(
            "ALTER TABLE Paper ADD COLUMN application_score REAL DEFAULT NULL;"
        )
        conn.commit()
        print("[migrate] Column added and committed.")

    except Exception as e:
        conn.rollback()
        print(f"[migrate] ERROR — rolled back: {e}")
        raise
    finally:
        conn.close()

    # Verify
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    ok = _column_exists(cur, "Paper", "application_score")
    conn.close()

    if ok:
        print("[migrate] Verification passed. application_score column exists.")
    else:
        print("[migrate] VERIFY FAIL: application_score column missing after migration.")
        sys.exit(1)


if __name__ == "__main__":
    run_migration()
