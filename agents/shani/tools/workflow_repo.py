from datetime import datetime
from repositories.repository import Repository


# ============================================================
# WORKFLOW REPOSITORY
#
# FIX [9]: get_workflow() was using repo.transaction() for a
# SELECT. transaction() issues BEGIN ... COMMIT around every
# call. For writes this is correct and required. For reads it
# is wrong — it acquires an unnecessary write lock in SQLite's
# default journal mode, which blocks concurrent writes from
# the orchestrator's main loop.
#
# The orchestrator calls get_workflow() after every stage to
# check if the workflow was reset to 'created' status. This
# happens inside the same run loop that is also writing stage
# and execution attempt records. Wrapping the read in a
# transaction held a lock that could prevent those writes from
# proceeding cleanly.
#
# Fix: get_workflow() now uses repo.fetch_one() — the correct
# read path used by every other read in the codebase.
# All write functions (create, update_status, update_stage)
# correctly retain repo.transaction().
# ============================================================


# ============================================================
# WRITES
# ============================================================

def create_workflow(
    repo: Repository,
    name: str,
    current_stage: str,
    status: str
) -> int:
    """
    Creates a new Workflow row.

    current_stage: initial stage name or None (NULL allowed
                   per updated schema fix [1]).
    status: must be one of created|running|paused|completed|failed
            (fix [2] added 'created' to the schema CHECK).

    Returns:
        int: newly created workflow ID

    Raises:
        sqlite3.IntegrityError if constraints fail.
    """
    timestamp = datetime.utcnow().isoformat()

    with repo.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO Workflow (
                name,
                current_stage,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?);
            """,
            (name, current_stage, status, timestamp, timestamp)
        )

        return cursor.lastrowid


def update_workflow_status(
    repo: Repository,
    workflow_id: int,
    new_status: str
) -> None:
    """
    Updates Workflow.status and refreshes updated_at.

    Valid values: created|running|paused|completed|failed
    'created' is used by the S2 failure reset path in
    orchestrator.py to signal the workflow needs a fresh start.
    """
    timestamp = datetime.utcnow().isoformat()

    with repo.transaction() as cursor:
        cursor.execute(
            """
            UPDATE Workflow
            SET status = ?,
                updated_at = ?
            WHERE id = ?;
            """,
            (new_status, timestamp, workflow_id)
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Workflow ID {workflow_id} does not exist.")


def update_current_stage(
    repo: Repository,
    workflow_id: int,
    new_stage: str | None
) -> None:
    """
    Updates Workflow.current_stage and refreshes updated_at.

    new_stage: stage name string, or None (NULL) to signal
               the workflow has been reset and has no active
               stage. NULL is explicitly allowed per schema
               fix [1]. orchestrator.py passes None on S2
               failure reset.
    """
    timestamp = datetime.utcnow().isoformat()

    with repo.transaction() as cursor:
        cursor.execute(
            """
            UPDATE Workflow
            SET current_stage = ?,
                updated_at = ?
            WHERE id = ?;
            """,
            (new_stage, timestamp, workflow_id)
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Workflow ID {workflow_id} does not exist.")


# ============================================================
# READS
#
# FIX [9]: get_workflow() changed from repo.transaction()
# to repo.fetch_one(). No BEGIN/COMMIT for a SELECT.
# ============================================================

def get_workflow(repo: Repository, workflow_id: int) -> dict | None:
    """
    Retrieves a Workflow row by ID.

    Uses repo.fetch_one() — the correct read path.
    Does NOT use repo.transaction() (that is for writes only).

    Returns:
        dict with keys: id, name, current_stage, status,
                        created_at, updated_at
        None if not found.
    """
    row = repo.fetch_one(
        """
        SELECT id, name, current_stage, status, created_at, updated_at
        FROM Workflow
        WHERE id = ?;
        """,
        (workflow_id,)
    )

    if not row:
        return None

    return {
        "id":            row[0],
        "name":          row[1],
        "current_stage": row[2],
        "status":        row[3],
        "created_at":    row[4],
        "updated_at":    row[5],
    }
