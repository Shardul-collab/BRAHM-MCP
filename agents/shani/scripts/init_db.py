import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from repositories.repository import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _migrate_stage_add_s5_5(conn: sqlite3.Connection) -> None:
    """
    One-time migration: Stage.stage_name's CHECK constraint didn't
    include 'S5_5', even though the orchestrator's STAGE_SEQUENCE and
    Workflow.current_stage's own constraint both already do — so
    creating a Stage row for S5_5 would fail outright. SQLite can't
    ALTER a CHECK constraint in place, so this rebuilds the table.

    Stage is referenced by ExecutionAttempt.stage_id and
    FailureLog.stage_id via FOREIGN KEY. Renaming Stage away (as in a
    naive rename-then-recreate) would make SQLite rewrite those FK
    clauses to point at the temporary name, then dropping that temp
    table would leave them dangling. So this goes the other way:
    build the replacement under a temp name, copy rows, DROP the old
    Stage, THEN rename the replacement into 'Stage' — the children's
    FK clauses (always just "REFERENCES Stage(id)", never touched)
    resolve correctly again the moment a table named Stage exists.

    Guarded and idempotent: only runs once, safe on every startup.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='Stage'"
    ).fetchone()
    if row is None or "'S5_5'" in row[0]:
        return  # table doesn't exist yet, or already migrated

    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.executescript("""
        CREATE TABLE Stage_new (
            id INTEGER PRIMARY KEY,
            workflow_id INTEGER NOT NULL,
            stage_name TEXT NOT NULL
                CHECK (stage_name IN (
                    'S1','S2','S2_75','S2_5','S3',
                    'S4','S5','S5_5'
                )),
            status TEXT NOT NULL
                CHECK (status IN ('running','completed','failed')),
            started_at DATETIME,
            ended_at DATETIME,
            FOREIGN KEY (workflow_id)
                REFERENCES Workflow(id)
                ON DELETE CASCADE
        );

        INSERT INTO Stage_new SELECT * FROM Stage;

        DROP TABLE Stage;

        ALTER TABLE Stage_new RENAME TO Stage;
    """)
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON;")


def create_tables(conn):

    with conn:

        conn.execute("""
        CREATE TABLE IF NOT EXISTS Workflow (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            current_stage TEXT
                CHECK (
                    current_stage IS NULL OR
                    current_stage IN (
                        'S1','S2','S2_75','S2_5','S3',
                        'S4','S5','S5_5'
                    )
                ),
            status TEXT NOT NULL
                CHECK (status IN ('created','running','paused','completed','failed')),
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS Stage (
            id INTEGER PRIMARY KEY,
            workflow_id INTEGER NOT NULL,
            stage_name TEXT NOT NULL
                CHECK (stage_name IN (
                    'S1','S2','S2_75','S2_5','S3',
                    'S4','S5','S5_5'
                )),
            status TEXT NOT NULL
                CHECK (status IN ('running','completed','failed')),
            started_at DATETIME,
            ended_at DATETIME,
            FOREIGN KEY (workflow_id)
                REFERENCES Workflow(id)
                ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS ExecutionAttempt (
            id INTEGER PRIMARY KEY,
            stage_id INTEGER NOT NULL,
            attempt_number INTEGER NOT NULL
                CHECK (attempt_number > 0),
            status TEXT NOT NULL
                CHECK (status IN ('running','failed','completed')),
            started_at DATETIME,
            ended_at DATETIME,
            error_message TEXT,
            FOREIGN KEY (stage_id)
                REFERENCES Stage(id)
                ON DELETE CASCADE
        );
        """)

        # =====================================================
        # PAPER
        #
        # Paper.status tracks the routing state of a paper
        # through the pipeline stages.  The full progressive
        # contract is:
        #
        #   pending          → created by S2, not yet downloaded
        #   downloading      → S3 actively fetching the PDF
        #   processing       → S3 download complete; S4 may begin
        #   extracted        → S4 complete; S5 may begin
        #   knowledge_ready  → S5 complete; S6/S7 may proceed
        #   completed        → entire pipeline finished
        #
        # Terminal failure states (never retried by pipeline):
        #   failed           → generic / download failure (S2/S3)
        #   extraction_failed → S4 hard failure
        #   knowledge_failed  → S5 hard failure
        #
        # Rule: every stage reads exactly one status value and
        # writes exactly one success status value.  The read
        # value of stage N+1 equals the write value of stage N.
        # =====================================================

        conn.execute("""
        CREATE TABLE IF NOT EXISTS Paper (
            id INTEGER PRIMARY KEY,
            workflow_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            file_path TEXT,
            pdf_url TEXT,
            abstract TEXT,
            pdf_candidates TEXT,
            pdf_status TEXT DEFAULT 'metadata',
            pdf_path TEXT,
            doi TEXT,
            failed_candidates TEXT,
            last_error TEXT,
            status TEXT NOT NULL
                CHECK (status IN (
                    'pending',
                    'downloading',
                    'processing',
                    'extracted',
                    'knowledge_ready',
                    'completed',
                    'failed',
                    'extraction_failed',
                    'knowledge_failed'
                )),
            raw_text TEXT,
            year INTEGER,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            UNIQUE(workflow_id, title),
            FOREIGN KEY (workflow_id)
                REFERENCES Workflow(id)
                ON DELETE CASCADE
        );
        """)

        # =====================================================
        # PAPER CONTENT
        #
        # ADDED: latex_text TEXT DEFAULT NULL
        #   Stores the block-reconstructed LaTeX for this
        #   section of the paper. Set by S4's block-based
        #   extractor. NULL for sections extracted by the
        #   keyword fallback. Backward compatible.
        # =====================================================

        conn.execute("""
        CREATE TABLE IF NOT EXISTS PaperContent (
            id INTEGER PRIMARY KEY,
            paper_id INTEGER NOT NULL,
            section_name TEXT NOT NULL,
            content TEXT NOT NULL,
            latex_text TEXT DEFAULT NULL,
            FOREIGN KEY (paper_id)
                REFERENCES Paper(id)
                ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS PaperFigure (
            id INTEGER PRIMARY KEY,
            paper_id INTEGER NOT NULL,
            figure_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            caption TEXT,
            section_hint TEXT,
            page_number INTEGER,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (paper_id)
                REFERENCES Paper(id)
                ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS PaperTable (
            id INTEGER PRIMARY KEY,
            paper_id INTEGER NOT NULL,
            table_id TEXT NOT NULL,
            table_type TEXT NOT NULL
                CHECK (table_type IN ('data','summary','complex')),
            headers_json TEXT,
            rows_json TEXT,
            image_path TEXT,
            caption TEXT,
            section_hint TEXT,
            page_number INTEGER,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (paper_id)
                REFERENCES Paper(id)
                ON DELETE CASCADE
        );
        """)

        # =====================================================
        # PAPER EQUATION  (NEW)
        #
        # Stores equations detected and normalized by S4.
        #
        # raw_text:        as detected from PDF blocks
        #                  (may contain Unicode math chars)
        # normalized_latex: rule-based LaTeX normalization
        #                  e.g. η=(E-E0)/m →
        #                  \eta = \frac{E - E_0}{m}
        #                  Stored as full equation environment:
        #                  \begin{equation}...\end{equation}
        # context_before:  sentence immediately before the
        #                  equation block (defines meaning)
        # context_after:   sentence immediately after
        # section_source:  which paper section it came from
        # position_index:  order of this equation in the paper
        #                  (page * 100 + block_index)
        # =====================================================

        conn.execute("""
        CREATE TABLE IF NOT EXISTS PaperEquation (
            id INTEGER PRIMARY KEY,
            paper_id INTEGER NOT NULL,
            equation_id TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            normalized_latex TEXT NOT NULL,
            context_before TEXT,
            context_after TEXT,
            section_source TEXT,
            page_number INTEGER,
            position_index INTEGER,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (paper_id)
                REFERENCES Paper(id)
                ON DELETE CASCADE
        );
        """)

        # =====================================================
        # RESEARCH KNOWLEDGE
        #
        # ADDED: equation_id INTEGER DEFAULT NULL
        #   Soft reference to PaperEquation.id.
        #   Set when a knowledge entry was derived from
        #   an equation's context (S5 equation-context path).
        #   NULL for all other knowledge entries.
        #   No FK constraint — soft link, NULL safe.
        # =====================================================

        conn.execute("""
        CREATE TABLE IF NOT EXISTS ResearchKnowledge (
            id INTEGER PRIMARY KEY,
            paper_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            value TEXT NOT NULL,
            section_source TEXT,
            sentence TEXT,
            source_type TEXT DEFAULT NULL,
            confidence TEXT DEFAULT NULL,
            equation_id INTEGER DEFAULT NULL,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (paper_id)
                REFERENCES Paper(id)
                ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS ResearchRelation (
            id INTEGER PRIMARY KEY,
            paper_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            relation TEXT NOT NULL,
            object TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (paper_id)
                REFERENCES Paper(id)
                ON DELETE CASCADE
        );
        """)

        # =====================================================
        # RESEARCH FINDING  (S5.5)
        #
        # Reconstructed, paper-level findings built from grouped
        # ResearchKnowledge rows (tools/reconstruct_findings.py).
        # Was previously created ad-hoc outside this script; never
        # part of the tracked base schema, so a from-scratch DB
        # (e.g. after a full reset) was missing it entirely and
        # S5_5 could not run at all. Added here so it's part of
        # the actual schema going forward.
        # =====================================================

        conn.execute("""
        CREATE TABLE IF NOT EXISTS ResearchFinding (
            id INTEGER PRIMARY KEY,
            paper_id INTEGER NOT NULL,
            workflow_id INTEGER NOT NULL,
            material TEXT,
            synthesis_method TEXT,
            characterization TEXT,
            property_name TEXT,
            property_value TEXT,
            condition_text TEXT,
            finding_text TEXT NOT NULL,
            source_sentence TEXT,
            section_source TEXT,
            confidence TEXT,
            source_type TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id)
                REFERENCES Paper(id)
                ON DELETE CASCADE,
            FOREIGN KEY (workflow_id)
                REFERENCES Workflow(id)
                ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_finding_workflow
            ON ResearchFinding(workflow_id);
        """)

        # S55Checkpoint tracks which papers have already been
        # processed by S5.5, so reconstruct_findings.py doesn't
        # reprocess a paper on every call. paper_id as PRIMARY KEY
        # matches the existing "INSERT OR REPLACE" usage (one row
        # per paper; a paper belongs to exactly one workflow via
        # Paper.workflow_id, so paper_id alone is a safe conflict
        # target).

        conn.execute("""
        CREATE TABLE IF NOT EXISTS S55Checkpoint (
            paper_id INTEGER PRIMARY KEY,
            workflow_id INTEGER NOT NULL,
            status TEXT NOT NULL
                CHECK (status IN ('completed','skipped')),
            findings_count INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (paper_id)
                REFERENCES Paper(id)
                ON DELETE CASCADE,
            FOREIGN KEY (workflow_id)
                REFERENCES Workflow(id)
                ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_checkpoint_workflow
            ON S55Checkpoint(workflow_id);
        """)

        # =====================================================
        # DRAFT SECTION
        #
        # section_status values:
        #   completed  → section generated successfully
        #   fallback   → LLM failed; deterministic fallback
        #                text was substituted (still valid
        #                content, pipeline continues)
        #   skipped    → no evidence available for section
        #   failed     → unrecoverable error (should not reach
        #                S7; treated as empty by synthesizer)
        #   pending    → reserved for future async use
        #
        # ADDED: fallback
        #   Distinguishes "LLM failed but we have content"
        #   from "failed" (no content at all).  S7 accepts
        #   both 'completed' and 'fallback' as renderable.
        #
        # ADDED: inline_tables_json TEXT DEFAULT NULL
        #   Structured comparison tables parsed from LLM
        #   output by S6 (COMPARISON_TABLE_START/END blocks).
        #   Stored as JSON array of table dicts.
        #   S7 reads this to render LaTeX tabular environments.
        #   NULL when no inline tables were produced.
        # =====================================================

        conn.execute("""
        CREATE TABLE IF NOT EXISTS DraftSection (
            id INTEGER PRIMARY KEY,
            workflow_id INTEGER NOT NULL,
            section_name TEXT NOT NULL,
            content TEXT NOT NULL,
            section_status TEXT NOT NULL DEFAULT 'completed'
                CHECK (section_status IN (
                    'pending',
                    'completed',
                    'fallback',
                    'failed',
                    'skipped'
                )),
            figures_json TEXT DEFAULT NULL,
            tables_json TEXT DEFAULT NULL,
            equations_json TEXT DEFAULT NULL,
            inline_tables_json TEXT DEFAULT NULL,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (workflow_id)
                REFERENCES Workflow(id)
                ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS FinalPaperSection (
            id INTEGER PRIMARY KEY,
            workflow_id INTEGER NOT NULL,
            section_name TEXT NOT NULL,
            order_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (workflow_id)
                REFERENCES Workflow(id)
                ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS WorkflowResearchConfig (
            id INTEGER PRIMARY KEY,
            workflow_id INTEGER NOT NULL,
            domain TEXT,
            material TEXT,
            structure TEXT,
            focus TEXT,
            method TEXT,
            properties TEXT,
            characterization TEXT,
            use_local INTEGER DEFAULT 0,
            max_papers INTEGER DEFAULT NULL,
            FOREIGN KEY (workflow_id)
                REFERENCES Workflow(id)
                ON DELETE CASCADE
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS FailureLog (
            id INTEGER PRIMARY KEY,
            workflow_id INTEGER NOT NULL,
            stage_id INTEGER,
            execution_attempt_id INTEGER,
            paper_id INTEGER,
            error_type TEXT NOT NULL,
            error_message TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (workflow_id)
                REFERENCES Workflow(id)
                ON DELETE CASCADE,
            FOREIGN KEY (stage_id)
                REFERENCES Stage(id)
                ON DELETE SET NULL,
            FOREIGN KEY (execution_attempt_id)
                REFERENCES ExecutionAttempt(id)
                ON DELETE SET NULL,
            FOREIGN KEY (paper_id)
                REFERENCES Paper(id)
                ON DELETE SET NULL
        );
        """)

    # Outside the `with conn:` block on purpose: this migration needs
    # PRAGMA foreign_keys to actually take effect, which SQLite defers
    # until any open transaction commits.
    _migrate_stage_add_s5_5(conn)


if __name__ == "__main__":

    print("Database path:", DB_PATH)
    conn = get_connection()
    print("SQLite version:", sqlite3.sqlite_version)
    fk_status = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
    print("Foreign keys enabled:", fk_status)
    create_tables(conn)
    print("Initialization complete.")
    conn.close()
