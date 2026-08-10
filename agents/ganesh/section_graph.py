"""
ganesh/section_graph.py
========================
SectionGraph: dependency-aware DAG executor for document sections.

This is the engine that runs INSIDE G3 (execute_section_graph).
It is completely invisible to the top-level Orchestrator.

Responsibilities:
  - Build the dependency graph from the DocumentPlan
  - Track which sections are PENDING, READY, DRAFTING, or APPROVED
  - Advance the graph as sections are approved
  - Detect deadlocks (circular dependencies, unapproved blocking sections)

Usage (inside the G3 tool function):
--------------------------------------
    graph = SectionGraph.from_document(repo, document_id)
    executor = SectionExecutor(repo, document_id, context_bundle)

    while not graph.is_complete():
        ready = graph.get_ready_sections()

        if not ready:
            raise OrchestrationError("Section graph deadlocked — no READY sections")

        # Optional: run ready sections concurrently (threading)
        for section in ready:
            graph.mark_drafting(section.section_name)
            executor.run(section)                          # write/critique/revise loop
            graph.mark_approved(section.section_name)     # unlocks dependents
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# ─────────────────────────────────────────────────────────────────────────────
# Section lifecycle constants
# ─────────────────────────────────────────────────────────────────────────────

class SectionStatus:
    PENDING         = "pending"
    READY           = "ready"
    DRAFTING        = "drafting"
    DRAFT_COMPLETE  = "draft_complete"
    UNDER_REVIEW    = "under_review"
    REVISION_NEEDED = "revision_needed"
    REVISING        = "revising"
    APPROVED        = "approved"
    INTEGRATED      = "integrated"


# ─────────────────────────────────────────────────────────────────────────────
# Section node
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SectionNode:
    """
    In-memory representation of a GaneshSection row.

    Mirrors the DB row but adds dependency resolution helpers.
    The graph works with SectionNode objects; DB writes happen
    via explicit calls to _update_status().
    """

    section_id:   int
    section_name: str
    section_type: str
    brief_json:   Optional[str]
    depends_on:   List[str]      # section_name strings
    exec_order:   Optional[int]
    status:       str            # from SectionStatus

    @classmethod
    def from_db_row(cls, row: dict) -> "SectionNode":
        return cls(
            section_id   = row["id"],
            section_name = row["section_name"],
            section_type = row["section_type"],
            brief_json   = row["brief_json"],
            depends_on   = json.loads(row["depends_on"] or "[]"),
            exec_order   = row["exec_order"],
            status       = row["status"],
        )

    @property
    def brief(self) -> dict:
        """Parsed brief JSON, or empty dict if not yet set."""
        return json.loads(self.brief_json) if self.brief_json else {}


# ─────────────────────────────────────────────────────────────────────────────
# Section graph
# ─────────────────────────────────────────────────────────────────────────────

class SectionGraph:
    """
    In-memory, DB-backed dependency graph for a single GaneshDocument's sections.

    The graph is built once from the DB at the start of G3.
    All status transitions are written back to the DB immediately.

    State transitions driven by this class:
      pending  → ready       (when all dependencies are APPROVED)
      ready    → drafting    (when SectionExecutor picks up the section)
      drafting → approved    (delegated to SectionExecutor's outcome)
      approved → integrated  (when DocumentIntegrator processes the section)
    """

    def __init__(self, repo, document_id: int):
        self.repo        = repo
        self.document_id = document_id
        self._nodes: Dict[str, SectionNode] = {}
        self._load()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load all GaneshSection rows for this document into memory."""
        rows = self.repo.fetch_all(
            """
            SELECT id, section_name, section_type, brief_json,
                   depends_on, exec_order, status
            FROM GaneshSection
            WHERE document_id = ?
            ORDER BY exec_order ASC
            """,
            (self.document_id,),
        )
        self._nodes = {
            row["section_name"]: SectionNode.from_db_row(dict(row))
            for row in rows
        }

    @classmethod
    def from_document(cls, repo, document_id: int) -> "SectionGraph":
        graph = cls(repo, document_id)
        # On load, compute which pending sections are now READY.
        graph._refresh_ready()
        return graph

    # ------------------------------------------------------------------
    # Ready queue
    # ------------------------------------------------------------------

    def _refresh_ready(self) -> None:
        """
        Promote PENDING sections to READY if all their dependencies
        are currently APPROVED (or INTEGRATED).
        Writes status changes to the DB.
        """
        approved_names: Set[str] = {
            name for name, node in self._nodes.items()
            if node.status in (SectionStatus.APPROVED, SectionStatus.INTEGRATED)
        }

        for name, node in self._nodes.items():
            if node.status != SectionStatus.PENDING:
                continue
            deps_met = all(dep in approved_names for dep in node.depends_on)
            if deps_met:
                self._update_status(name, SectionStatus.READY)

    def get_ready_sections(self) -> List[SectionNode]:
        """
        Returns all sections currently in READY status.
        These are safe to dispatch to SectionExecutors.
        Multiple sections may be READY simultaneously (parallel execution).
        """
        return [
            node for node in self._nodes.values()
            if node.status == SectionStatus.READY
        ]

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def mark_drafting(self, section_name: str) -> None:
        """Called when a SectionExecutor picks up a READY section."""
        self._assert_status(section_name, SectionStatus.READY)
        self._update_status(section_name, SectionStatus.DRAFTING)

    def mark_approved(self, section_name: str) -> None:
        """
        Called when a SectionExecutor completes its loop for a section.
        Immediately refreshes the READY queue — this may unlock dependents.
        """
        self._update_status(section_name, SectionStatus.APPROVED)
        self._refresh_ready()

    def mark_integrated(self, section_name: str) -> None:
        """Called by DocumentIntegrator when a section is merged into final doc."""
        self._assert_status(section_name, SectionStatus.APPROVED)
        self._update_status(section_name, SectionStatus.INTEGRATED)

    # ------------------------------------------------------------------
    # Completion and deadlock detection
    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        """True when all sections are APPROVED or INTEGRATED."""
        return all(
            node.status in (SectionStatus.APPROVED, SectionStatus.INTEGRATED)
            for node in self._nodes.values()
        )

    def has_deadlock(self) -> bool:
        """
        Returns True if there are PENDING sections but no READY sections
        and no DRAFTING sections — i.e., the graph cannot advance.

        This indicates a circular dependency or a broken dependency reference.
        """
        statuses = {node.status for node in self._nodes.values()}
        has_blocked = SectionStatus.PENDING in statuses
        can_advance = bool(
            self.get_ready_sections()
            or SectionStatus.DRAFTING in statuses
        )
        return has_blocked and not can_advance

    def get_status_summary(self) -> Dict[str, str]:
        """Returns {section_name: status} for all sections."""
        return {name: node.status for name, node in self._nodes.items()}

    def get_approved_sections_ordered(self) -> List[SectionNode]:
        """
        Returns APPROVED sections sorted by exec_order.
        Used by DocumentIntegrator to assemble in narrative order.
        """
        approved = [
            node for node in self._nodes.values()
            if node.status == SectionStatus.APPROVED
        ]
        return sorted(approved, key=lambda n: (n.exec_order or 0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_status(self, section_name: str, new_status: str) -> None:
        from datetime import datetime

        node = self._nodes[section_name]
        node.status = new_status

        with self.repo.transaction() as cursor:
            cursor.execute(
                "UPDATE GaneshSection SET status = ?, updated_at = ? WHERE id = ?",
                (new_status, datetime.utcnow().isoformat(), node.section_id),
            )

    def _assert_status(self, section_name: str, expected: str) -> None:
        node = self._nodes.get(section_name)
        if node is None:
            raise ValueError(f"Section '{section_name}' not found in graph.")
        if node.status != expected:
            raise ValueError(
                f"Section '{section_name}': expected status '{expected}', "
                f"got '{node.status}'."
            )
