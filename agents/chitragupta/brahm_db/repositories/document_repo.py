"""
brahm_db/repositories/document_repo.py
========================================
Repository for GaneshDocument and GaneshSection.
GANESH reads and writes here during G1-G5 and daily reports.
"""

import json
import logging
from datetime import datetime, timezone

from brahm_db.repositories.base import BaseRepository

log = logging.getLogger("brahm_db.document_repo")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentRepo(BaseRepository):

    # ── GaneshDocument ────────────────────────────────────────────────────────

    def create_document(
        self,
        project_id: int,
        document_type: str,
        title: str,
        workflow_ids: list[int] | None = None,
        dft_result_ids: list[int] | None = None,
        instrument_result_ids: list[int] | None = None,
        llm_backend: str | None = None,
        cycle_id: int | None = None,
    ) -> int:
        with self.transaction() as c:
            c.execute(
                """
                INSERT INTO GaneshDocument
                (project_id, cycle_id, document_type, title, status,
                 workflow_ids_json, dft_result_ids_json,
                 instrument_result_ids_json, llm_backend, created_at)
                VALUES (?, ?, ?, ?, 'draft', ?, ?, ?, ?, ?)
                """,
                (
                    project_id, cycle_id, document_type, title,
                    json.dumps(workflow_ids or []),
                    json.dumps(dft_result_ids or []),
                    json.dumps(instrument_result_ids or []),
                    llm_backend,
                    _now(),
                ),
            )
        row = self.fetch_one(
            "SELECT id FROM GaneshDocument"
            " WHERE project_id=? ORDER BY id DESC LIMIT 1",
            (project_id,),
        )
        doc_id = row["id"]
        log.info(
            "GaneshDocument created | id=%d project=%d type=%s title=%.50s",
            doc_id, project_id, document_type, title,
        )
        return doc_id

    def get_document(self, document_id: int) -> dict | None:
        row = self.fetch_one(
            "SELECT * FROM GaneshDocument WHERE id=?", (document_id,)
        )
        if not row:
            return None
        return self._deserialise_doc(dict(row))

    def list_documents(
        self,
        project_id: int,
        document_type: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM GaneshDocument WHERE project_id=?"
        params: list = [project_id]
        if document_type:
            sql += " AND document_type=?"
            params.append(document_type)
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        return [
            self._deserialise_doc(dict(r))
            for r in self.fetch_all(sql, tuple(params))
        ]

    def update_document_status(
        self, document_id: int, status: str
    ) -> None:
        now = _now()
        extra = ", completed_at=?" if status == "completed" else ""
        params = (
            (status, now, document_id)
            if not extra
            else (status, now, now, document_id)
        )
        with self.transaction() as c:
            c.execute(
                f"UPDATE GaneshDocument SET status=?{extra},"
                f" completed_at=? WHERE id=?"
                if extra else
                "UPDATE GaneshDocument SET status=? WHERE id=?",
                params,
            )

    def get_latest_daily_report(self, project_id: int) -> dict | None:
        row = self.fetch_one(
            "SELECT * FROM GaneshDocument"
            " WHERE project_id=? AND document_type='daily_report'"
            " ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        )
        return self._deserialise_doc(dict(row)) if row else None

    # ── GaneshSection ─────────────────────────────────────────────────────────

    def create_section(
        self,
        document_id: int,
        section_name: str,
        order_index: int = 0,
    ) -> int:
        with self.transaction() as c:
            c.execute(
                """
                INSERT INTO GaneshSection
                (document_id, section_name, order_index,
                 status, created_at)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (document_id, section_name, order_index, _now()),
            )
        row = self.fetch_one(
            "SELECT id FROM GaneshSection"
            " WHERE document_id=? AND section_name=?"
            " ORDER BY id DESC LIMIT 1",
            (document_id, section_name),
        )
        return row["id"]

    def save_draft(self, section_id: int, draft: str) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE GaneshSection"
                " SET draft=?, status='drafted' WHERE id=?",
                (draft, section_id),
            )

    def save_critique(self, section_id: int, critique: str) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE GaneshSection"
                " SET critique=?, status='critiqued' WHERE id=?",
                (critique, section_id),
            )

    def save_final(self, section_id: int, final_content: str) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE GaneshSection"
                " SET final_content=?, status='final' WHERE id=?",
                (final_content, section_id),
            )

    def get_sections(
        self, document_id: int, status: str | None = None
    ) -> list[dict]:
        sql = (
            "SELECT * FROM GaneshSection WHERE document_id=?"
        )
        params: list = [document_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY order_index ASC"
        return [dict(r) for r in self.fetch_all(sql, tuple(params))]

    def get_full_document_text(self, document_id: int) -> str:
        """
        Assemble final document text from all completed sections
        in order. Used by GANESH G5 and for export.
        """
        sections = self.get_sections(document_id, status="final")
        if not sections:
            # Fall back to drafted sections if no finals yet
            sections = self.get_sections(document_id, status="drafted")
        parts = []
        for s in sections:
            parts.append(f"## {s['section_name']}\n\n{s['final_content'] or s['draft'] or ''}")
        return "\n\n---\n\n".join(parts)

    def get_document_with_sections(self, document_id: int) -> dict | None:
        doc = self.get_document(document_id)
        if not doc:
            return None
        doc["sections"] = self.get_sections(document_id)
        doc["full_text"] = self.get_full_document_text(document_id)
        return doc

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _deserialise_doc(row: dict) -> dict:
        for field in (
            "workflow_ids_json",
            "dft_result_ids_json",
            "instrument_result_ids_json",
        ):
            if row.get(field):
                try:
                    row[field.replace("_json", "")] = json.loads(row[field])
                except Exception:
                    pass
        return row
