"""
brahm_db/schema.py
==================
Creates and migrates brahm.db — the central BRAHM database.
All agents write here through the repository layer.
Run directly to initialise: python schema.py
"""

import sqlite3
import logging
from pathlib import Path

log = logging.getLogger("brahm_db.schema")

BRAHM_DB_PATH = Path("/mnt/d/brahm/data/brahm.db")


def get_connection() -> sqlite3.Connection:
    BRAHM_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(BRAHM_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate_project_event_human_agent(conn: sqlite3.Connection) -> None:
    """
    One-time migration: the ProjectEvent.agent CHECK constraint originally
    didn't include 'HUMAN', but chitragupta_log_event's MCP tool schema
    allows agent="Human" — any such call 500'd on the CHECK constraint.
    SQLite can't ALTER a CHECK constraint in place, so this rebuilds the
    table (and its indexes, which SQLite would otherwise drop along with
    the renamed-away old table) if the old constraint is detected. Guarded
    and idempotent: only runs once, safe to call on every startup after.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='ProjectEvent'"
    ).fetchone()
    if row is None or "'HUMAN'" in row[0]:
        return  # table doesn't exist yet, or already migrated

    log.info("Migrating ProjectEvent.agent CHECK constraint to add HUMAN...")
    conn.executescript("""
        ALTER TABLE ProjectEvent RENAME TO ProjectEvent_old;

        CREATE TABLE ProjectEvent (
            id          INTEGER PRIMARY KEY,
            project_id  INTEGER NOT NULL,
            agent       TEXT NOT NULL
                        CHECK (agent IN (
                            'BRAHM','SHANI','VISHWAKARMA',
                            'VIDUR','GANESH','CHITRAGUPTA','HUMAN'
                        )),
            event_type  TEXT NOT NULL,
            summary     TEXT NOT NULL,
            payload_json TEXT,
            started_at  DATETIME NOT NULL DEFAULT (datetime('now')),
            ended_at    DATETIME,
            status      TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running','completed','failed')),
            FOREIGN KEY (project_id) REFERENCES Project(id) ON DELETE CASCADE
        );

        INSERT INTO ProjectEvent
            (id, project_id, agent, event_type, summary, payload_json,
             started_at, ended_at, status)
        SELECT
            id, project_id, UPPER(agent), event_type, summary, payload_json,
            started_at, ended_at, status
        FROM ProjectEvent_old;

        DROP TABLE ProjectEvent_old;

        CREATE INDEX IF NOT EXISTS idx_event_project
            ON ProjectEvent(project_id, started_at);
        CREATE INDEX IF NOT EXISTS idx_event_agent
            ON ProjectEvent(agent, status);
    """)
    conn.commit()
    log.info("ProjectEvent migration complete.")


def init_db() -> None:
    conn = get_connection()
    try:
        c = conn.cursor()

        # ── Projects ─────────────────────────────────────────────────────────
        c.executescript("""

        CREATE TABLE IF NOT EXISTS Project (
            id          INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            objective   TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active','paused','completed','abandoned')),
            created_at  DATETIME NOT NULL DEFAULT (datetime('now')),
            updated_at  DATETIME NOT NULL DEFAULT (datetime('now'))
        );

        -- Every agent action, timestamped — the project timeline
        CREATE TABLE IF NOT EXISTS ProjectEvent (
            id          INTEGER PRIMARY KEY,
            project_id  INTEGER NOT NULL,
            agent       TEXT NOT NULL
                        CHECK (agent IN (
                            'BRAHM','SHANI','VISHWAKARMA',
                            'VIDUR','GANESH','CHITRAGUPTA','HUMAN'
                        )),
            event_type  TEXT NOT NULL,
            summary     TEXT NOT NULL,
            payload_json TEXT,
            started_at  DATETIME NOT NULL DEFAULT (datetime('now')),
            ended_at    DATETIME,
            status      TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running','completed','failed')),
            FOREIGN KEY (project_id) REFERENCES Project(id) ON DELETE CASCADE
        );

        -- Workloads assigned to agents by BRAHM
        CREATE TABLE IF NOT EXISTS Workload (
            id              INTEGER PRIMARY KEY,
            project_id      INTEGER NOT NULL,
            agent           TEXT NOT NULL
                            CHECK (agent IN (
                                'SHANI','VISHWAKARMA','VIDUR','GANESH'
                            )),
            workload_type   TEXT NOT NULL,
            config_json     TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'queued'
                            CHECK (status IN (
                                'queued','running','completed','failed','cancelled'
                            )),
            priority        INTEGER NOT NULL DEFAULT 5,
            created_at      DATETIME NOT NULL DEFAULT (datetime('now')),
            started_at      DATETIME,
            completed_at    DATETIME,
            FOREIGN KEY (project_id) REFERENCES Project(id) ON DELETE CASCADE
        );

        -- Human decision points
        CREATE TABLE IF NOT EXISTS DecisionPoint (
            id                   INTEGER PRIMARY KEY,
            project_id           INTEGER NOT NULL,
            workload_id          INTEGER,
            question             TEXT NOT NULL,
            options_json         TEXT,
            brahm_recommendation TEXT,
            human_response       TEXT,
            decided_at           DATETIME,
            status               TEXT NOT NULL DEFAULT 'pending'
                                 CHECK (status IN ('pending','decided','timeout')),
            created_at           DATETIME NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (project_id)  REFERENCES Project(id)  ON DELETE CASCADE,
            FOREIGN KEY (workload_id) REFERENCES Workload(id) ON DELETE SET NULL
        );

        -- Research cycles within a project (each synthesis attempt)
        CREATE TABLE IF NOT EXISTS ResearchCycle (
            id           INTEGER PRIMARY KEY,
            project_id   INTEGER NOT NULL,
            cycle_number INTEGER NOT NULL DEFAULT 1,
            objective    TEXT,
            status       TEXT NOT NULL DEFAULT 'active'
                         CHECK (status IN ('active','completed','abandoned')),
            started_at   DATETIME NOT NULL DEFAULT (datetime('now')),
            completed_at DATETIME,
            notes        TEXT,
            FOREIGN KEY (project_id) REFERENCES Project(id) ON DELETE CASCADE
        );

        -- ── Global Paper Registry (deduplication) ────────────────────────────

        CREATE TABLE IF NOT EXISTS GlobalPaper (
            id                      INTEGER PRIMARY KEY,
            doi                     TEXT UNIQUE,
            title                   TEXT NOT NULL,
            abstract                TEXT,
            first_seen_project_id   INTEGER,
            first_seen_workflow_id  INTEGER,
            shani_paper_id          INTEGER,
            vector_indexed          INTEGER NOT NULL DEFAULT 0,
            created_at              DATETIME NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (first_seen_project_id)
                REFERENCES Project(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS GlobalPaperReference (
            id              INTEGER PRIMARY KEY,
            global_paper_id INTEGER NOT NULL,
            project_id      INTEGER NOT NULL,
            workflow_id     INTEGER,
            created_at      DATETIME NOT NULL DEFAULT (datetime('now')),
            UNIQUE (global_paper_id, project_id, workflow_id),
            FOREIGN KEY (global_paper_id)
                REFERENCES GlobalPaper(id) ON DELETE CASCADE,
            FOREIGN KEY (project_id)
                REFERENCES Project(id) ON DELETE CASCADE
        );

        -- ── Instrument Results (VIDUR) ────────────────────────────────────────

        CREATE TABLE IF NOT EXISTS InstrumentResult (
            id                      INTEGER PRIMARY KEY,
            project_id              INTEGER NOT NULL,
            cycle_id                INTEGER,
            file_path               TEXT NOT NULL,
            technique               TEXT NOT NULL
                                    CHECK (technique IN (
                                        'XRD','UV-Vis','SEM_EDX','Raman','Unknown'
                                    )),
            confidence              REAL,
            signals_json            TEXT,
            parsed_data_json        TEXT,
            comparison_result_json  TEXT,
            match_score             REAL,
            gaps_identified_json    TEXT,
            created_at              DATETIME NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (project_id) REFERENCES Project(id) ON DELETE CASCADE,
            FOREIGN KEY (cycle_id)   REFERENCES ResearchCycle(id) ON DELETE SET NULL
        );

        -- ── DFT Results (VISHWAKARMA) ─────────────────────────────────────────

        CREATE TABLE IF NOT EXISTS DFTResult (
            id                  INTEGER PRIMARY KEY,
            project_id          INTEGER NOT NULL,
            cycle_id            INTEGER,
            job_id              TEXT NOT NULL,
            calc_type           TEXT NOT NULL
                                CHECK (calc_type IN (
                                    'scf','nscf','relax','vc-relax','bands',
                                    'dos','projwfc','pp','phonon','neb','hp','cp'
                                )),
            structure_json      TEXT,
            input_params_json   TEXT,
            output_parsed_json  TEXT,
            status              TEXT NOT NULL DEFAULT 'completed'
                                CHECK (status IN ('completed','failed')),
            wall_time_seconds   REAL,
            created_at          DATETIME NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (project_id) REFERENCES Project(id) ON DELETE CASCADE,
            FOREIGN KEY (cycle_id)   REFERENCES ResearchCycle(id) ON DELETE SET NULL
        );

        -- ── GANESH Documents ──────────────────────────────────────────────────

        CREATE TABLE IF NOT EXISTS GaneshDocument (
            id                          INTEGER PRIMARY KEY,
            project_id                  INTEGER NOT NULL,
            cycle_id                    INTEGER,
            document_type               TEXT NOT NULL
                                        CHECK (document_type IN (
                                            'literature_review','dft_report',
                                            'daily_report','manuscript_draft',
                                            'research_report','technical_summary'
                                        )),
            title                       TEXT NOT NULL,
            status                      TEXT NOT NULL DEFAULT 'draft'
                                        CHECK (status IN (
                                            'draft','reviewing','completed','failed'
                                        )),
            workflow_ids_json           TEXT,
            dft_result_ids_json         TEXT,
            instrument_result_ids_json  TEXT,
            llm_backend                 TEXT,
            created_at                  DATETIME NOT NULL DEFAULT (datetime('now')),
            completed_at                DATETIME,
            FOREIGN KEY (project_id) REFERENCES Project(id) ON DELETE CASCADE,
            FOREIGN KEY (cycle_id)   REFERENCES ResearchCycle(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS GaneshSection (
            id            INTEGER PRIMARY KEY,
            document_id   INTEGER NOT NULL,
            section_name  TEXT NOT NULL,
            order_index   INTEGER NOT NULL DEFAULT 0,
            draft         TEXT,
            critique      TEXT,
            final_content TEXT,
            status        TEXT NOT NULL DEFAULT 'pending'
                          CHECK (status IN (
                              'pending','drafted','critiqued','final','failed'
                          )),
            created_at    DATETIME NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (document_id)
                REFERENCES GaneshDocument(id) ON DELETE CASCADE
        );

        -- ── Indexes ───────────────────────────────────────────────────────────

        CREATE INDEX IF NOT EXISTS idx_event_project
            ON ProjectEvent(project_id, started_at);
        CREATE INDEX IF NOT EXISTS idx_event_agent
            ON ProjectEvent(agent, status);
        CREATE INDEX IF NOT EXISTS idx_workload_project
            ON Workload(project_id, status);
        CREATE INDEX IF NOT EXISTS idx_workload_agent
            ON Workload(agent, status);
        CREATE INDEX IF NOT EXISTS idx_decision_project
            ON DecisionPoint(project_id, status);
        CREATE INDEX IF NOT EXISTS idx_global_paper_doi
            ON GlobalPaper(doi);
        CREATE INDEX IF NOT EXISTS idx_global_paper_title
            ON GlobalPaper(title);
        CREATE INDEX IF NOT EXISTS idx_instrument_project
            ON InstrumentResult(project_id, technique);
        CREATE INDEX IF NOT EXISTS idx_dft_project
            ON DFTResult(project_id, calc_type);
        CREATE INDEX IF NOT EXISTS idx_ganesh_doc_project
            ON GaneshDocument(project_id, document_type);
        CREATE INDEX IF NOT EXISTS idx_ganesh_section_doc
            ON GaneshSection(document_id, order_index);

        """)
        conn.commit()
        _migrate_project_event_human_agent(conn)
        log.info("brahm.db initialised successfully at %s", BRAHM_DB_PATH)
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print(f"brahm.db ready at {BRAHM_DB_PATH}")
