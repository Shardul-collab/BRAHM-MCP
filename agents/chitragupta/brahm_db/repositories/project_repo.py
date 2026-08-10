"""
brahm_db/repositories/project_repo.py
=======================================
CRUD for Project, ProjectEvent, Workload, DecisionPoint, ResearchCycle.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from brahm_db.repositories.base import BaseRepository

log = logging.getLogger("brahm_db.project_repo")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectRepo(BaseRepository):

    # ── Project ───────────────────────────────────────────────────────────────

    def create_project(self, name: str, objective: str) -> int:
        now = _now()
        with self.transaction() as c:
            c.execute(
                "INSERT INTO Project (name, objective, status, created_at, updated_at)"
                " VALUES (?, ?, 'active', ?, ?)",
                (name, objective, now, now),
            )
        row = self.fetch_one(
            "SELECT id FROM Project WHERE name=? ORDER BY id DESC LIMIT 1",
            (name,),
        )
        return row["id"]

    def get_project(self, project_id: int) -> dict | None:
        row = self.fetch_one(
            "SELECT * FROM Project WHERE id=?", (project_id,)
        )
        return dict(row) if row else None

    def list_projects(self, status: str | None = None) -> list[dict]:
        if status:
            rows = self.fetch_all(
                "SELECT * FROM Project WHERE status=? ORDER BY created_at DESC",
                (status,),
            )
        else:
            rows = self.fetch_all(
                "SELECT * FROM Project ORDER BY created_at DESC"
            )
        return [dict(r) for r in rows]

    def update_project_status(self, project_id: int, status: str) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE Project SET status=?, updated_at=? WHERE id=?",
                (status, _now(), project_id),
            )

    # ── ProjectEvent ──────────────────────────────────────────────────────────

    def log_event(
        self,
        project_id: int,
        agent: str,
        event_type: str,
        summary: str,
        payload: dict | None = None,
    ) -> int:
        # Normalize casing — the CHECK constraint is case-sensitive
        # ('VISHWAKARMA', 'HUMAN', ...) but callers (MCP tool schemas, etc.)
        # may pass mixed case (e.g. "Vishwakarma", "Human").
        agent = agent.upper()
        with self.transaction() as c:
            c.execute(
                "INSERT INTO ProjectEvent "
                "(project_id, agent, event_type, summary, payload_json,"
                " started_at, status)"
                " VALUES (?, ?, ?, ?, ?, ?, 'running')",
                (
                    project_id, agent, event_type, summary,
                    json.dumps(payload) if payload else None,
                    _now(),
                ),
            )
        row = self.fetch_one(
            "SELECT id FROM ProjectEvent WHERE project_id=?"
            " ORDER BY id DESC LIMIT 1",
            (project_id,),
        )
        return row["id"]

    def complete_event(
        self,
        event_id: int,
        status: str = "completed",
    ) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE ProjectEvent SET status=?, ended_at=? WHERE id=?",
                (status, _now(), event_id),
            )

    def get_events(
        self,
        project_id: int,
        agent: str | None = None,
        since: str | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM ProjectEvent WHERE project_id=?"
        params: list = [project_id]
        if agent:
            sql += " AND agent=?"
            params.append(agent)
        if since:
            sql += " AND started_at >= ?"
            params.append(since)
        sql += " ORDER BY started_at ASC"
        return [dict(r) for r in self.fetch_all(sql, tuple(params))]

    def get_today_events(self, project_id: int) -> list[dict]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.get_events(project_id, since=today)

    # ── Workload ──────────────────────────────────────────────────────────────

    def create_workload(
        self,
        project_id: int,
        agent: str,
        workload_type: str,
        config: dict,
        priority: int = 5,
    ) -> int:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO Workload "
                "(project_id, agent, workload_type, config_json,"
                " status, priority, created_at)"
                " VALUES (?, ?, ?, ?, 'queued', ?, ?)",
                (
                    project_id, agent, workload_type,
                    json.dumps(config), priority, _now(),
                ),
            )
        row = self.fetch_one(
            "SELECT id FROM Workload WHERE project_id=?"
            " ORDER BY id DESC LIMIT 1",
            (project_id,),
        )
        return row["id"]

    def get_workload(self, workload_id: int) -> dict | None:
        row = self.fetch_one(
            "SELECT * FROM Workload WHERE id=?", (workload_id,)
        )
        return dict(row) if row else None

    def list_workloads(
        self,
        project_id: int,
        status: str | None = None,
        agent: str | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM Workload WHERE project_id=?"
        params: list = [project_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        if agent:
            sql += " AND agent=?"
            params.append(agent)
        sql += " ORDER BY priority DESC, created_at ASC"
        return [dict(r) for r in self.fetch_all(sql, tuple(params))]

    def update_workload_status(
        self,
        workload_id: int,
        status: str,
    ) -> None:
        now = _now()
        extra_sql = ""
        if status == "running":
            extra_sql = ", started_at=?"
            with self.transaction() as c:
                c.execute(
                    f"UPDATE Workload SET status=?{extra_sql} WHERE id=?",
                    (status, now, workload_id),
                )
        elif status in ("completed", "failed", "cancelled"):
            extra_sql = ", completed_at=?"
            with self.transaction() as c:
                c.execute(
                    f"UPDATE Workload SET status=?{extra_sql} WHERE id=?",
                    (status, now, workload_id),
                )
        else:
            with self.transaction() as c:
                c.execute(
                    "UPDATE Workload SET status=? WHERE id=?",
                    (status, workload_id),
                )

    def get_next_queued_workload(
        self, agent: str
    ) -> dict | None:
        row = self.fetch_one(
            "SELECT * FROM Workload WHERE agent=? AND status='queued'"
            " ORDER BY priority DESC, created_at ASC LIMIT 1",
            (agent,),
        )
        return dict(row) if row else None

    # ── DecisionPoint ─────────────────────────────────────────────────────────

    def create_decision(
        self,
        project_id: int,
        question: str,
        options: list | None = None,
        recommendation: str | None = None,
        workload_id: int | None = None,
    ) -> int:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO DecisionPoint "
                "(project_id, workload_id, question, options_json,"
                " brahm_recommendation, status, created_at)"
                " VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (
                    project_id, workload_id, question,
                    json.dumps(options) if options else None,
                    recommendation, _now(),
                ),
            )
        row = self.fetch_one(
            "SELECT id FROM DecisionPoint WHERE project_id=?"
            " ORDER BY id DESC LIMIT 1",
            (project_id,),
        )
        return row["id"]

    def resolve_decision(
        self, decision_id: int, human_response: str
    ) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE DecisionPoint"
                " SET human_response=?, decided_at=?, status='decided'"
                " WHERE id=?",
                (human_response, _now(), decision_id),
            )

    def get_pending_decisions(self, project_id: int) -> list[dict]:
        rows = self.fetch_all(
            "SELECT * FROM DecisionPoint"
            " WHERE project_id=? AND status='pending'"
            " ORDER BY created_at ASC",
            (project_id,),
        )
        return [dict(r) for r in rows]

    def get_decision(self, decision_id: int) -> dict | None:
        row = self.fetch_one(
            "SELECT * FROM DecisionPoint WHERE id=?", (decision_id,)
        )
        return dict(row) if row else None

    # ── ResearchCycle ─────────────────────────────────────────────────────────

    def create_cycle(
        self,
        project_id: int,
        objective: str | None = None,
    ) -> int:
        row = self.fetch_one(
            "SELECT COALESCE(MAX(cycle_number), 0) AS n"
            " FROM ResearchCycle WHERE project_id=?",
            (project_id,),
        )
        next_num = (row["n"] if row else 0) + 1
        with self.transaction() as c:
            c.execute(
                "INSERT INTO ResearchCycle"
                " (project_id, cycle_number, objective, status, started_at)"
                " VALUES (?, ?, ?, 'active', ?)",
                (project_id, next_num, objective, _now()),
            )
        row2 = self.fetch_one(
            "SELECT id FROM ResearchCycle WHERE project_id=?"
            " ORDER BY id DESC LIMIT 1",
            (project_id,),
        )
        return row2["id"]

    def complete_cycle(
        self, cycle_id: int, notes: str | None = None
    ) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE ResearchCycle"
                " SET status='completed', completed_at=?, notes=?"
                " WHERE id=?",
                (_now(), notes, cycle_id),
            )

    def get_active_cycle(self, project_id: int) -> dict | None:
        row = self.fetch_one(
            "SELECT * FROM ResearchCycle"
            " WHERE project_id=? AND status='active'"
            " ORDER BY cycle_number DESC LIMIT 1",
            (project_id,),
        )
        return dict(row) if row else None
