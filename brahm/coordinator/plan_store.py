"""
brahm/coordinator/plan_store.py
=================================
In-memory tracking of Layer 2 plan proposals — mirrors run_store.py's
pattern exactly. Separate dict, separate lifecycle from RunStore. Does
not read or write to any agent's database, and does not touch any
orchestrator.

v1 scope: in-memory only, thread-safe via a simple lock. Resets on
coordinator restart (same limitation RunStore already has — see the
known drift issue documented in SESSION_HANDOFF.md / this session's
planning discussion).
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Optional


class PlanStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._plans: dict[str, dict] = {}

    def create(self, prompt: str, state_check: dict, steps: list[dict]) -> str:
        plan_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._plans[plan_id] = {
                "plan_id":     plan_id,
                "prompt":      prompt,
                "status":      "pending_approval",  # -> approved -> executing -> complete | rejected | failed
                "state_check": state_check,
                "steps":       steps,
                "run_id":      None,  # set once execution starts, if it maps to a RunStore run
                "error":       None,
                "created_at":  datetime.now(timezone.utc).isoformat(),
            }
        return plan_id

    def get(self, plan_id: str) -> Optional[dict]:
        with self._lock:
            p = self._plans.get(plan_id)
            return dict(p) if p else None

    def set_status(self, plan_id: str, status: str) -> None:
        with self._lock:
            if plan_id in self._plans:
                self._plans[plan_id]["status"] = status

    def set_steps(self, plan_id: str, steps: list[dict]) -> None:
        """Used when the researcher edits steps before approving
        (edit-and-resubmit UX, per this session's decision)."""
        with self._lock:
            if plan_id in self._plans:
                self._plans[plan_id]["steps"] = steps

    def set_run_id(self, plan_id: str, run_id: str) -> None:
        with self._lock:
            if plan_id in self._plans:
                self._plans[plan_id]["run_id"] = run_id

    def set_error(self, plan_id: str, error: str) -> None:
        with self._lock:
            if plan_id in self._plans:
                self._plans[plan_id]["status"] = "failed"
                self._plans[plan_id]["error"] = error

    def all_plans(self) -> list[dict]:
        with self._lock:
            return [dict(p) for p in self._plans.values()]
