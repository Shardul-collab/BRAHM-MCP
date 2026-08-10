"""
brahm/coordinator/run_store.py
================================
In-memory tracking of coordinator-initiated runs.

This is NEW state that only the coordinator owns — it does not read or
write to any agent's database, and it does not touch any orchestrator.
It exists purely to remember, for the lifetime of this process, which
SHANI workflow_id and GANESH document_id belong to a given dashboard
"run", and what phase that run is currently in.

v1 scope: in-memory only, thread-safe via a simple lock. Resets on
coordinator restart. If runs need to survive a restart later, this
should move to its own small SQLite file under brahm/coordinator/ —
never into an existing agent's database.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Optional


class RunStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, dict] = {}

    def create(self, workflow_id: int, document_type: str, auto_write: bool) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._runs[run_id] = {
                "run_id":        run_id,
                "workflow_id":   workflow_id,
                "document_id":   None,
                "document_type": document_type,
                "auto_write":    auto_write,
                "phase":         "created",   # created -> shani_running -> shani_done
                                                # -> ganesh_writing -> ganesh_reviewing
                                                # -> complete | failed
                "error":         None,
                "created_at":    datetime.now(timezone.utc).isoformat(),
            }
        return run_id

    def get(self, run_id: str) -> Optional[dict]:
        with self._lock:
            run = self._runs.get(run_id)
            return dict(run) if run else None

    def set_phase(self, run_id: str, phase: str) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id]["phase"] = phase

    def set_document_id(self, run_id: str, document_id: int) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id]["document_id"] = document_id

    def set_error(self, run_id: str, error: str) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id]["phase"] = "failed"
                self._runs[run_id]["error"] = error

    def all_runs(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._runs.values()]
