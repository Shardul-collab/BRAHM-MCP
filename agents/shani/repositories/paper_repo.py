from datetime import datetime
import sqlite3
from repositories.repository import Repository


# ============================================================
# PAPER REPOSITORY
#
# Changes vs previous version:
# - create_paper() now accepts and stores abstract.
#   Abstracts from S2 search APIs are persisted so S2_75
#   (extract_lightweight_knowledge) can read them from DB.
# - store_abstract() added for targeted abstract updates.
# All other functions unchanged.
# ============================================================


def create_paper(
    repo: Repository,
    workflow_id: int,
    title: str,
    source: str,
    pdf_url: str,
    status: str,
    abstract: str = None,
    year: int = None
):
    """
    Inserts a new Paper row.

    abstract: optional — stored from search API response so
              S2_75 can extract knowledge without the PDF.
              NULL if source did not return an abstract.

    Returns:
        int:  newly created paper ID
        None: duplicate (UNIQUE on workflow_id + title)
    """
    timestamp = datetime.utcnow().isoformat()

    try:
        with repo.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO Paper (
                    workflow_id,
                    title,
                    source,
                    pdf_url,
                    abstract,
                    status,
                    year,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    workflow_id, title, source,
                    pdf_url, abstract, status,
                    year, timestamp, timestamp
                )
            )
            return cursor.lastrowid

    except sqlite3.IntegrityError:
        return None


def store_abstract(repo: Repository, paper_id: int, abstract: str):
    """
    Updates abstract for a paper that was created without one.
    """
    timestamp = datetime.utcnow().isoformat()

    with repo.transaction() as cursor:
        cursor.execute(
            """
            UPDATE Paper
            SET abstract = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (abstract, timestamp, paper_id)
        )


def update_paper_status(repo: Repository, paper_id: int, new_status: str):

    timestamp = datetime.utcnow().isoformat()

    with repo.transaction() as cursor:
        cursor.execute(
            """
            UPDATE Paper
            SET status = ?,
                updated_at = ?
            WHERE id = ?;
            """,
            (new_status, timestamp, paper_id)
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Paper ID {paper_id} does not exist.")


def update_paper_file_path(repo: Repository, paper_id: int, file_path: str):
    """
    Stores local disk path after S3 download.
    S4 reads this column to open the PDF.
    """
    timestamp = datetime.utcnow().isoformat()

    with repo.transaction() as cursor:
        cursor.execute(
            """
            UPDATE Paper
            SET file_path = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (file_path, timestamp, paper_id)
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Paper ID {paper_id} does not exist.")


def store_paper_text(repo: Repository, paper_id: int, raw_text: str):
    """
    Stores raw extracted PDF text. Called by S4. Required by S5.
    """
    timestamp = datetime.utcnow().isoformat()

    with repo.transaction() as cursor:
        cursor.execute(
            """
            UPDATE Paper
            SET raw_text = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (raw_text, timestamp, paper_id)
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Paper ID {paper_id} does not exist.")


def get_pending_papers(repo: Repository, workflow_id: int):

    rows = repo.fetch_all(
        """
        SELECT id, workflow_id, title, source, pdf_url, abstract, status
        FROM Paper
        WHERE workflow_id = ?
        AND status = 'pending'
        """,
        (workflow_id,)
    )

    return [dict(row) for row in rows]


def get_processing_paper(repo: Repository, workflow_id: int):

    rows = repo.fetch_all(
        """
        SELECT id, workflow_id, title, source, pdf_url, file_path, abstract, status
        FROM Paper
        WHERE workflow_id = ?
        AND status = 'processing'
        LIMIT 1
        """,
        (workflow_id,)
    )

    if not rows:
        return None

    return dict(rows[0])

def get_papers_for_extraction(repo, workflow_id: int):
    """Fetch downloaded papers that need content extraction (S4)."""
    rows = repo.fetch_all(
        """
        SELECT id, title, file_path, doi, abstract
        FROM Paper
        WHERE workflow_id = ?
          AND file_path IS NOT NULL
          AND status NOT IN ('extracted', 'knowledge_ready', 'completed', 'extraction_failed')
        ORDER BY id ASC
        """,
        (workflow_id,)
    )
    return [dict(r) for r in rows]
