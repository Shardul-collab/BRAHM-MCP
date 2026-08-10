"""
backfill_application_score.py
==============================
Computes and persists application_score for every existing Paper row,
using the already-existing, already-correct compute_application_score()
from paper_ingestor.py.

Context
-------
add_application_score.py added the column (NULL for all existing rows).
This script fills it in retroactively by re-running the same scoring
logic SHANI already uses at ingestion time — no new scoring logic is
introduced here, this only persists a signal that was previously
computed and discarded.

Scope: all workflows, not just workflow 1 — application-score
contamination (paper matching material keyword but wrong application
domain) is a structural gap, not specific to MoS2/FET.

Usage
-----
  python scripts/backfill_application_score.py [--dry-run]

Safe to re-run — recomputes and overwrites application_score each time
(idempotent, since compute_application_score is a pure function of
paper title/abstract + workflow config, both immutable post-ingestion
for already-knowledge_ready papers).
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from repositories.repository import DB_PATH
from tools.paper_ingestor import compute_application_score


def run_backfill(dry_run: bool = False):
    print(f"[backfill] Target database: {DB_PATH}")
    print(f"[backfill] Dry run: {dry_run}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # One config per workflow, fetched once, reused for all its papers.
    config_rows = cur.execute(
        "SELECT workflow_id, focus, properties FROM WorkflowResearchConfig"
    ).fetchall()
    configs = {r["workflow_id"]: dict(r) for r in config_rows}
    print(f"[backfill] Loaded config for {len(configs)} workflow(s): {sorted(configs.keys())}")

    papers = cur.execute(
        "SELECT id, workflow_id, title, abstract FROM Paper"
    ).fetchall()
    print(f"[backfill] Found {len(papers)} total Paper rows.")

    updates = []
    skipped_no_config = 0
    score_distribution = {"0.0": 0, "0.0-0.25": 0, "0.25-0.5": 0, "0.5-0.75": 0, "0.75-1.0": 0}

    for p in papers:
        config = configs.get(p["workflow_id"])
        if config is None:
            skipped_no_config += 1
            continue

        paper_dict = {"title": p["title"], "abstract": p["abstract"]}
        score = compute_application_score(paper_dict, config)
        updates.append((score, p["id"]))

        if score == 0.0:
            score_distribution["0.0"] += 1
        elif score < 0.25:
            score_distribution["0.0-0.25"] += 1
        elif score < 0.5:
            score_distribution["0.25-0.5"] += 1
        elif score < 0.75:
            score_distribution["0.5-0.75"] += 1
        else:
            score_distribution["0.75-1.0"] += 1

    print(f"[backfill] Computed scores for {len(updates)} papers "
          f"({skipped_no_config} skipped — no WorkflowResearchConfig for their workflow_id).")
    print(f"[backfill] Score distribution: {score_distribution}")

    if dry_run:
        print("[backfill] Dry run — no writes performed.")
        conn.close()
        return

    try:
        cur.execute("BEGIN;")
        cur.executemany(
            "UPDATE Paper SET application_score = ? WHERE id = ?",
            updates,
        )
        conn.commit()
        print(f"[backfill] Committed {len(updates)} updates.")
    except Exception as e:
        conn.rollback()
        print(f"[backfill] ERROR — rolled back: {e}")
        raise
    finally:
        conn.close()

    # Verify
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    remaining_null = cur.execute(
        "SELECT COUNT(*) FROM Paper WHERE application_score IS NULL"
    ).fetchone()[0]
    conn.close()
    print(f"[backfill] Papers still NULL after backfill: {remaining_null} "
          f"(expected: {skipped_no_config}, from workflows with no config)")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_backfill(dry_run=dry_run)
