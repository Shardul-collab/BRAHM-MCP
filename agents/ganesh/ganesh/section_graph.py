"""
ganesh/section_graph.py
========================
SectionNode, SectionStatus, SectionGraph — DAG management for G3.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SectionStatus(str, Enum):
    PENDING      = "pending"
    READY        = "ready"
    DRAFTING     = "drafting"
    UNDER_REVIEW = "under_review"
    REVISING     = "revising"
    APPROVED     = "approved"
    INTEGRATED   = "integrated"


@dataclass
class SectionNode:
    id:           int
    document_id:  int
    section_name: str
    section_type: str
    status:       SectionStatus
    dependencies: List[str]
    exec_order:   int
    brief_json:   Optional[str] = None


class SectionGraph:
    """
    Manages the DAG of sections for one document.
    Tracks statuses in memory; writes to DB via repo on transitions.
    """

    def __init__(self, repo, document_id: int, nodes: List[SectionNode]):
        self.repo        = repo
        self.document_id = document_id
        self._nodes      = {n.section_name: n for n in nodes}

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_document(cls, repo, document_id: int) -> "SectionGraph":
        rows = repo.fetch_all(
            """
            SELECT id, document_id, section_name, section_type, status,
                   dependencies, exec_order, brief_json
            FROM GaneshSection WHERE document_id = ?
            ORDER BY exec_order ASC
            """,
            (document_id,),
        )
        nodes = []
        for r in rows:
            deps = json.loads(r["dependencies"]) if r["dependencies"] else []
            nodes.append(SectionNode(
                id=r["id"],
                document_id=r["document_id"],
                section_name=r["section_name"],
                section_type=r["section_type"],
                status=SectionStatus(r["status"]),
                dependencies=deps,
                exec_order=r["exec_order"],
                brief_json=r["brief_json"],
            ))
        return cls(repo, document_id, nodes)

    # ── DAG queries ───────────────────────────────────────────────────────────

    def get_ready_sections(self) -> List[SectionNode]:
        """Return sections whose dependencies are all APPROVED."""
        ready = []
        for node in self._nodes.values():
            if node.status not in (SectionStatus.PENDING, SectionStatus.READY):
                continue
            deps_done = all(
                self._nodes[dep].status in (SectionStatus.APPROVED, SectionStatus.INTEGRATED)
                for dep in node.dependencies
                if dep in self._nodes
            )
            if deps_done:
                ready.append(node)
        return sorted(ready, key=lambda n: n.exec_order)

    def get_approved_sections_ordered(self) -> List[SectionNode]:
        return sorted(
            [n for n in self._nodes.values()
             if n.status in (SectionStatus.APPROVED, SectionStatus.INTEGRATED)],
            key=lambda n: n.exec_order,
        )

    def is_complete(self) -> bool:
        return all(
            n.status in (SectionStatus.APPROVED, SectionStatus.INTEGRATED)
            for n in self._nodes.values()
        )

    def has_deadlock(self) -> bool:
        """True if no section is ready but graph is not complete."""
        if self.is_complete():
            return False
        return len(self.get_ready_sections()) == 0

    def get_status_summary(self) -> dict:
        return {name: node.status.value for name, node in self._nodes.items()}

    # ── Status transitions (memory + DB) ──────────────────────────────────────

    def _set_status(self, section_name: str, status: SectionStatus):
        node = self._nodes[section_name]
        node.status = status
        from datetime import datetime
        with self.repo.transaction() as cursor:
            cursor.execute(
                "UPDATE GaneshSection SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, datetime.utcnow().isoformat(), node.id),
            )

    def mark_drafting(self, section_name: str):
        self._set_status(section_name, SectionStatus.DRAFTING)

    def mark_approved(self, section_name: str):
        self._set_status(section_name, SectionStatus.APPROVED)

    def mark_integrated(self, section_name: str):
        self._set_status(section_name, SectionStatus.INTEGRATED)
