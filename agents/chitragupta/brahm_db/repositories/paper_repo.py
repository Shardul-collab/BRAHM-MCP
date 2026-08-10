"""
brahm_db/repositories/paper_repo.py
=====================================
Global paper registry — deduplication across all workflows and projects.
SHANI checks here before downloading any paper.
"""

import json
import logging
from datetime import datetime, timezone

from brahm_db.repositories.base import BaseRepository

log = logging.getLogger("brahm_db.paper_repo")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperRepo(BaseRepository):

    # ── Deduplication check ───────────────────────────────────────────────────

    def check_doi(self, doi: str) -> dict | None:
        """
        Check if a paper with this DOI already exists globally.
        Returns the GlobalPaper row or None.
        SHANI calls this before downloading.
        """
        if not doi:
            return None
        row = self.fetch_one(
            "SELECT * FROM GlobalPaper WHERE doi=?",
            (doi.strip().lower(),),
        )
        return dict(row) if row else None

    def check_title(self, title: str) -> dict | None:
        """
        Fuzzy title check — exact match only (normalised).
        Used when DOI is missing.
        """
        if not title:
            return None
        normalised = title.strip().lower()
        row = self.fetch_one(
            "SELECT * FROM GlobalPaper WHERE LOWER(title)=?",
            (normalised,),
        )
        return dict(row) if row else None

    def check_paper(
        self, doi: str | None, title: str | None
    ) -> dict | None:
        """
        Master deduplication check.
        Tries DOI first (authoritative), falls back to title.
        Returns existing GlobalPaper row or None (paper is new).
        """
        if doi:
            result = self.check_doi(doi)
            if result:
                return result
        if title:
            result = self.check_title(title)
            if result:
                return result
        return None

    # ── Register new paper ────────────────────────────────────────────────────

    def register_paper(
        self,
        title: str,
        doi: str | None = None,
        abstract: str | None = None,
        project_id: int | None = None,
        workflow_id: int | None = None,
        shani_paper_id: int | None = None,
    ) -> int:
        """
        Add a new paper to the global registry.
        Called by SHANI after successfully downloading a new paper.
        Returns global_paper_id.
        """
        doi_clean = doi.strip().lower() if doi else None
        with self.transaction() as c:
            c.execute(
                "INSERT INTO GlobalPaper"
                " (doi, title, abstract, first_seen_project_id,"
                "  first_seen_workflow_id, shani_paper_id,"
                "  vector_indexed, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                (
                    doi_clean, title, abstract,
                    project_id, workflow_id,
                    shani_paper_id, _now(),
                ),
            )
        row = self.fetch_one(
            "SELECT id FROM GlobalPaper ORDER BY id DESC LIMIT 1"
        )
        global_id = row["id"]

        # Record the reference
        if project_id:
            self._add_reference(global_id, project_id, workflow_id)

        log.info(
            "Registered new paper | global_id=%d doi=%s title=%.60s",
            global_id, doi_clean, title,
        )
        return global_id

    def _add_reference(
        self,
        global_paper_id: int,
        project_id: int,
        workflow_id: int | None = None,
    ) -> None:
        """Link an existing GlobalPaper to a project/workflow."""
        try:
            with self.transaction() as c:
                c.execute(
                    "INSERT OR IGNORE INTO GlobalPaperReference"
                    " (global_paper_id, project_id, workflow_id, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (global_paper_id, project_id, workflow_id, _now()),
                )
        except Exception as exc:
            log.warning("Reference insert failed: %s", exc)

    def link_paper_to_project(
        self,
        global_paper_id: int,
        project_id: int,
        workflow_id: int | None = None,
    ) -> None:
        """
        Link an already-registered paper to a new project/workflow.
        Called when deduplication finds an existing paper.
        """
        self._add_reference(global_paper_id, project_id, workflow_id)
        log.info(
            "Linked existing paper | global_id=%d project=%d workflow=%s",
            global_paper_id, project_id, workflow_id,
        )

    def mark_vector_indexed(self, global_paper_id: int) -> None:
        """Called by FAISS indexer after embedding is added."""
        with self.transaction() as c:
            c.execute(
                "UPDATE GlobalPaper SET vector_indexed=1 WHERE id=?",
                (global_paper_id,),
            )

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_paper(self, global_paper_id: int) -> dict | None:
        row = self.fetch_one(
            "SELECT * FROM GlobalPaper WHERE id=?",
            (global_paper_id,),
        )
        return dict(row) if row else None

    def list_papers(
        self,
        project_id: int | None = None,
        workflow_id: int | None = None,
        not_vector_indexed: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        if project_id:
            sql = """
                SELECT gp.* FROM GlobalPaper gp
                JOIN GlobalPaperReference ref ON ref.global_paper_id = gp.id
                WHERE ref.project_id = ?
            """
            params: list = [project_id]
            if workflow_id:
                sql += " AND ref.workflow_id = ?"
                params.append(workflow_id)
            if not_vector_indexed:
                sql += " AND gp.vector_indexed = 0"
            sql += " ORDER BY gp.created_at DESC LIMIT ?"
            params.append(limit)
        else:
            sql = "SELECT * FROM GlobalPaper"
            params = []
            if not_vector_indexed:
                sql += " WHERE vector_indexed = 0"
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

        return [dict(r) for r in self.fetch_all(sql, tuple(params))]

    def stats(self) -> dict:
        row = self.fetch_one(
            """
            SELECT
                COUNT(*)                            AS total_papers,
                COUNT(CASE WHEN doi IS NOT NULL
                           THEN 1 END)              AS with_doi,
                COUNT(CASE WHEN vector_indexed = 1
                           THEN 1 END)              AS vector_indexed,
                COUNT(DISTINCT first_seen_project_id) AS projects_covered
            FROM GlobalPaper
            """
        )
        return dict(row) if row else {}
