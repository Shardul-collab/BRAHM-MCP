import sqlite3
import csv
import os
from datetime import datetime
from pathlib import Path


# ============================================================
# INTEGRITY AUDIT — configs/integrity_audit.py
#
# FIX [14]: Original file had a broken get_connection()
# function — the body was unindented at module level,
# causing a SyntaxError that prevented the file from
# being imported or run at all:
#
#   def get_connection():
#   from pathlib import Path          ← wrong indent
#   DB_PATH = Path(...) / "..."       ← wrong indent
#   conn = sqlite3.connect(DB_PATH)   ← wrong indent
#       conn.execute(...)             ← suddenly indented
#
# Fix: all lines correctly indented inside the function.
#
# Additional fixes:
# - DB_PATH now resolved correctly relative to project root
#   using the same pattern as repository.py
# - Output files (CSV, log, summary) written to reports/
#   directory rather than wherever the script is run from
# - reports/ directory created if it does not exist
# - DB_NAME constant removed (was unused after fix)
# ============================================================

# Resolve project root and reports directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH      = PROJECT_ROOT / "database" / "research_workflow.db"
REPORTS_DIR  = PROJECT_ROOT / "reports"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def timestamp():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def iso_now():
    return datetime.utcnow().isoformat()


def main():

    REPORTS_DIR.mkdir(exist_ok=True)

    run_time      = timestamp()
    readable_time = iso_now()

    report_file  = str(REPORTS_DIR / f"integrity_report_{run_time}.csv")
    log_file     = str(REPORTS_DIR / f"integrity_log_{run_time}.txt")
    summary_file = str(REPORTS_DIR / f"integrity_summary_{run_time}.txt")

    conn   = get_connection()
    cursor = conn.cursor()

    results = []

    def log(message):
        with open(log_file, "a") as f:
            f.write(message + "\n")

    def run_test(test_id, test_name, expected_behavior, func):
        ts = iso_now()
        try:
            func()
            conn.commit()
            result    = "FAIL"
            error_msg = "No error raised"
        except Exception as e:
            result    = "PASS"
            error_msg = str(e)

        results.append([
            test_id,
            test_name,
            expected_behavior,
            result,
            error_msg,
            ts
        ])

        log(f"[{ts}] TEST {test_id}: {test_name}")
        log(f"Expected: {expected_behavior}")
        log(f"Result:   {result}")
        log(f"Error:    {error_msg}")
        log("-" * 50)

    now = iso_now()

    # --------------------------------------------------
    # SETUP — clean slate before tests
    # --------------------------------------------------
    cursor.execute("DELETE FROM FailureLog")
    cursor.execute("DELETE FROM ExecutionAttempt")
    cursor.execute("DELETE FROM Stage")
    cursor.execute("DELETE FROM Paper")
    cursor.execute("DELETE FROM Workflow")
    conn.commit()

    # --------------------------------------------------
    # BASELINE — valid workflow insert (required for FK tests)
    # --------------------------------------------------
    cursor.execute("""
        INSERT INTO Workflow (name, current_stage, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """, ("Test Workflow", "S1", "running", now, now))
    conn.commit()
    workflow_id = cursor.lastrowid

    # --------------------------------------------------
    # TEST 2 — Invalid stage name in current_stage
    # --------------------------------------------------
    run_test(
        2,
        "Invalid Stage Name",
        "Reject stage outside S1-S7",
        lambda: cursor.execute("""
            INSERT INTO Workflow (name, current_stage, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("Bad Stage", "S9", "running", now, now))
    )

    # --------------------------------------------------
    # TEST 3 — Invalid workflow status
    # --------------------------------------------------
    run_test(
        3,
        "Invalid Workflow Status",
        "Reject invalid status value",
        lambda: cursor.execute("""
            INSERT INTO Workflow (name, current_stage, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("Bad Status", "S1", "flying", now, now))
    )

    # --------------------------------------------------
    # TEST 4 — Stage with non-existent workflow_id
    # --------------------------------------------------
    run_test(
        4,
        "Stage with invalid workflow_id",
        "Reject foreign key violation",
        lambda: cursor.execute("""
            INSERT INTO Stage (workflow_id, stage_name, status)
            VALUES (?, ?, ?)
        """, (99999, "S1", "running"))
    )

    # --------------------------------------------------
    # TEST 5 — ExecutionAttempt with attempt_number = 0
    # --------------------------------------------------
    cursor.execute("""
        INSERT INTO Stage (workflow_id, stage_name, status)
        VALUES (?, ?, ?)
    """, (workflow_id, "S1", "running"))
    conn.commit()
    stage_id = cursor.lastrowid

    run_test(
        5,
        "ExecutionAttempt with attempt_number = 0",
        "Reject attempt_number <= 0",
        lambda: cursor.execute("""
            INSERT INTO ExecutionAttempt (stage_id, attempt_number, status)
            VALUES (?, ?, ?)
        """, (stage_id, 0, "running"))
    )

    # --------------------------------------------------
    # TEST 6 — Paper with invalid status
    # --------------------------------------------------
    run_test(
        6,
        "Paper with invalid status",
        "Reject invalid paper status",
        lambda: cursor.execute("""
            INSERT INTO Paper (workflow_id, title, source, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (workflow_id, "Title", "Source", "unknown", now, now))
    )

    # --------------------------------------------------
    # TEST 7 — NULL current_stage (valid after fix [1])
    # --------------------------------------------------
    run_test(
        7,
        "NULL current_stage allowed",
        "Accept NULL current_stage (fix [1])",
        lambda: (
            cursor.execute("""
                INSERT INTO Workflow (name, current_stage, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, ("Reset Workflow", None, "created", now, now)),
            conn.commit()
        )
    )
    # TEST 7 should FAIL (no exception = allowed) — invert logic for this one
    if results[-1][3] == "FAIL":
        results[-1][3] = "PASS"
    elif results[-1][3] == "PASS":
        results[-1][3] = "FAIL"

    # --------------------------------------------------
    # TEST 8 — 'created' status allowed (fix [2])
    # --------------------------------------------------
    run_test(
        8,
        "'created' status accepted",
        "Accept status='created' (fix [2])",
        lambda: (
            cursor.execute("""
                INSERT INTO Workflow (name, current_stage, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, ("Created Workflow", "S1", "created", now, now)),
            conn.commit()
        )
    )
    if results[-1][3] == "FAIL":
        results[-1][3] = "PASS"
    elif results[-1][3] == "PASS":
        results[-1][3] = "FAIL"

    # --------------------------------------------------
    # WRITE CSV REPORT
    # --------------------------------------------------
    with open(report_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_id", "test_name", "expected_behavior",
            "result", "error_message", "timestamp"
        ])
        writer.writerows(results)

    # --------------------------------------------------
    # WRITE SUMMARY
    # --------------------------------------------------
    total  = len(results)
    passed = sum(1 for r in results if r[3] == "PASS")
    failed = total - passed

    with open(summary_file, "w") as f:
        f.write("FORMAL INTEGRITY AUDIT REPORT\n")
        f.write("=" * 40 + "\n")
        f.write(f"Run Time (UTC): {readable_time}\n\n")
        f.write(f"Total Tests: {total}\n")
        f.write(f"Passed:      {passed}\n")
        f.write(f"Failed:      {failed}\n\n")
        f.write("System Integrity Status: ")
        f.write("STABLE\n" if failed == 0 else "UNSTABLE\n")

    print("\nIntegrity Audit Complete")
    print(f"CSV Report: {report_file}")
    print(f"Log File:   {log_file}")
    print(f"Summary:    {summary_file}")

    conn.close()


if __name__ == "__main__":
    main()
