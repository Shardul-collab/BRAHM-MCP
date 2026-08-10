"""
brahm_db/repositories/results_repo.py
=======================================
Repositories for InstrumentResult (VIDUR) and DFTResult (Vishwakarma).
Both agents call these directly after completing a job — no manual save needed.
"""

import json
import logging
from datetime import datetime, timezone

from brahm_db.repositories.base import BaseRepository

log = logging.getLogger("brahm_db.results_repo")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InstrumentResultRepo(BaseRepository):
    """VIDUR saves here automatically after every classification."""

    def save(
        self,
        project_id: int,
        file_path: str,
        technique: str,
        confidence: float,
        signals: list,
        parsed_data: dict,
        cycle_id: int | None = None,
        comparison_result: dict | None = None,
        match_score: float | None = None,
        gaps_identified: list | None = None,
    ) -> int:
        with self.transaction() as c:
            c.execute(
                """
                INSERT INTO InstrumentResult
                (project_id, cycle_id, file_path, technique,
                 confidence, signals_json, parsed_data_json,
                 comparison_result_json, match_score,
                 gaps_identified_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id, cycle_id, file_path, technique,
                    confidence,
                    json.dumps(signals),
                    json.dumps(parsed_data),
                    json.dumps(comparison_result) if comparison_result else None,
                    match_score,
                    json.dumps(gaps_identified) if gaps_identified else None,
                    _now(),
                ),
            )
        row = self.fetch_one(
            "SELECT id FROM InstrumentResult"
            " WHERE project_id=? ORDER BY id DESC LIMIT 1",
            (project_id,),
        )
        result_id = row["id"]
        log.info(
            "InstrumentResult saved | id=%d project=%d technique=%s"
            " confidence=%.2f match_score=%s",
            result_id, project_id, technique, confidence, match_score,
        )
        return result_id

    def get(self, result_id: int) -> dict | None:
        row = self.fetch_one(
            "SELECT * FROM InstrumentResult WHERE id=?", (result_id,)
        )
        if not row:
            return None
        return self._deserialise(dict(row))

    def list_for_project(
        self,
        project_id: int,
        technique: str | None = None,
        cycle_id: int | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM InstrumentResult WHERE project_id=?"
        params: list = [project_id]
        if technique:
            sql += " AND technique=?"
            params.append(technique)
        if cycle_id:
            sql += " AND cycle_id=?"
            params.append(cycle_id)
        sql += " ORDER BY created_at DESC"
        return [
            self._deserialise(dict(r))
            for r in self.fetch_all(sql, tuple(params))
        ]

    def get_gaps(self, project_id: int) -> list[str]:
        """
        Return all gap strings identified across all instrument results
        for a project. CHITRAGUPTA uses this to trigger new SHANI workloads.
        """
        rows = self.fetch_all(
            "SELECT gaps_identified_json FROM InstrumentResult"
            " WHERE project_id=? AND gaps_identified_json IS NOT NULL",
            (project_id,),
        )
        gaps: list[str] = []
        for r in rows:
            try:
                gaps.extend(json.loads(r["gaps_identified_json"]))
            except Exception:
                pass
        return list(set(gaps))  # deduplicate

    def update_comparison(
        self,
        result_id: int,
        comparison_result: dict,
        match_score: float,
        gaps_identified: list | None = None,
    ) -> None:
        """Called after VISHWAKARMA predictions are available for comparison."""
        with self.transaction() as c:
            c.execute(
                """
                UPDATE InstrumentResult
                SET comparison_result_json=?,
                    match_score=?,
                    gaps_identified_json=?
                WHERE id=?
                """,
                (
                    json.dumps(comparison_result),
                    match_score,
                    json.dumps(gaps_identified) if gaps_identified else None,
                    result_id,
                ),
            )

    @staticmethod
    def _deserialise(row: dict) -> dict:
        for field in (
            "signals_json", "parsed_data_json",
            "comparison_result_json", "gaps_identified_json",
        ):
            if row.get(field):
                try:
                    row[field.replace("_json", "")] = json.loads(row[field])
                except Exception:
                    pass
        return row


class DFTResultRepo(BaseRepository):
    """Vishwakarma saves here automatically after every completed job."""

    def save(
        self,
        project_id: int,
        job_id: str,
        calc_type: str,
        structure: dict | None = None,
        input_params: dict | None = None,
        output_parsed: dict | None = None,
        status: str = "completed",
        wall_time_seconds: float | None = None,
        cycle_id: int | None = None,
    ) -> int:
        with self.transaction() as c:
            c.execute(
                """
                INSERT INTO DFTResult
                (project_id, cycle_id, job_id, calc_type,
                 structure_json, input_params_json, output_parsed_json,
                 status, wall_time_seconds, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id, cycle_id, job_id, calc_type,
                    json.dumps(structure) if structure else None,
                    json.dumps(input_params) if input_params else None,
                    json.dumps(output_parsed) if output_parsed else None,
                    status,
                    wall_time_seconds,
                    _now(),
                ),
            )
        row = self.fetch_one(
            "SELECT id FROM DFTResult"
            " WHERE project_id=? ORDER BY id DESC LIMIT 1",
            (project_id,),
        )
        result_id = row["id"]
        log.info(
            "DFTResult saved | id=%d project=%d job=%s calc=%s status=%s",
            result_id, project_id, job_id, calc_type, status,
        )
        return result_id

    def get(self, result_id: int) -> dict | None:
        row = self.fetch_one(
            "SELECT * FROM DFTResult WHERE id=?", (result_id,)
        )
        if not row:
            return None
        return self._deserialise(dict(row))

    def list_for_project(
        self,
        project_id: int,
        calc_type: str | None = None,
        status: str | None = None,
        cycle_id: int | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM DFTResult WHERE project_id=?"
        params: list = [project_id]
        if calc_type:
            sql += " AND calc_type=?"
            params.append(calc_type)
        if status:
            sql += " AND status=?"
            params.append(status)
        if cycle_id:
            sql += " AND cycle_id=?"
            params.append(cycle_id)
        sql += " ORDER BY created_at DESC"
        return [
            self._deserialise(dict(r))
            for r in self.fetch_all(sql, tuple(params))
        ]

    def get_latest(
        self,
        project_id: int,
        calc_type: str,
    ) -> dict | None:
        """Get the most recent completed DFT result of a given type."""
        row = self.fetch_one(
            "SELECT * FROM DFTResult"
            " WHERE project_id=? AND calc_type=? AND status='completed'"
            " ORDER BY created_at DESC LIMIT 1",
            (project_id, calc_type),
        )
        return self._deserialise(dict(row)) if row else None

    def get_calculation_summary(self, project_id: int) -> dict:
        """
        Summary of all DFT calculations for a project.
        Used by GANESH when writing dft_report sections.
        """
        rows = self.fetch_all(
            """
            SELECT calc_type,
                   COUNT(*)                                  AS total,
                   COUNT(CASE WHEN status='completed' THEN 1 END) AS completed,
                   COUNT(CASE WHEN status='failed'    THEN 1 END) AS failed,
                   AVG(wall_time_seconds)                    AS avg_wall_time
            FROM DFTResult
            WHERE project_id=?
            GROUP BY calc_type
            ORDER BY calc_type
            """,
            (project_id,),
        )
        return {r["calc_type"]: dict(r) for r in rows}

    @staticmethod
    def _deserialise(row: dict) -> dict:
        for field in (
            "structure_json", "input_params_json", "output_parsed_json"
        ):
            if row.get(field):
                try:
                    row[field.replace("_json", "")] = json.loads(row[field])
                except Exception:
                    pass
        return row
