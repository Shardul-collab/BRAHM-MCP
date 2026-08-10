"""
ganesh/schema.py
=================
SQLite migration for all GANESH tables.

Run once against the shared DB to add GANESH's tables.
SHANI's tables are untouched — no schema conflicts.

Migration is idempotent (CREATE TABLE IF NOT EXISTS).
"""

GANESH_SCHEMA = """

-- ─────────────────────────────────────────────────────────────────────────────
-- GaneshDocument
-- Top-level artifact: one document per GANESH run.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS GaneshDocument (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT    NOT NULL,
    document_type    TEXT    NOT NULL,
    -- 'planning' | 'drafting' | 'reviewing' | 'integrating' | 'completed' | 'failed'
    status           TEXT    NOT NULL DEFAULT 'planning',
    -- where knowledge comes from: 'shani' | 'vishwakarma' | 'multi_agent' | 'manual'
    source_type      TEXT,
    -- JSON array of source IDs, e.g. [42, 43] for SHANI workflow IDs
    source_ids       TEXT,
    -- DocumentPlan as JSON: section specs, argument flow map, citation plan
    outline_json     TEXT,
    -- final assembled document text (set on completion)
    final_output     TEXT,
    -- null | 'below_threshold' | 'human_review_requested'
    quality_flag     TEXT,
    total_iterations INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT    NOT NULL,
    updated_at       TEXT    NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────────
-- GaneshSection
-- One row per section in the document.
-- The status column tracks the section's lifecycle through drafting.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS GaneshSection (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES GaneshDocument(id) ON DELETE CASCADE,
    section_name    TEXT    NOT NULL,   -- 'Introduction', 'Methods', etc.
    -- 'intro' | 'body' | 'conclusion' | 'abstract' | 'refs'
    section_type    TEXT    NOT NULL,
    -- SectionSpec as JSON: brief, evidence_refs, target_word_count, quality_criteria
    brief_json      TEXT,
    -- JSON array of section_name strings that must be APPROVED before this can start
    depends_on      TEXT    NOT NULL DEFAULT '[]',
    -- position in the final assembled document (set by Planner)
    exec_order      INTEGER,
    -- State machine:
    -- pending → ready → drafting → draft_complete
    -- → under_review → revision_needed → revising
    -- → approved → integrated
    status          TEXT    NOT NULL DEFAULT 'pending',
    -- Critic's overall score when the section was approved (0.0–10.0)
    quality_score   REAL,
    iteration_count INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    UNIQUE(document_id, section_name)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- GaneshDraft
-- Full version history for a section's content.
-- version=1 is the Writer's initial draft.
-- version=2,3,... are Reviser outputs.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS GaneshDraft (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id  INTEGER NOT NULL REFERENCES GaneshSection(id) ON DELETE CASCADE,
    version     INTEGER NOT NULL,   -- 1=initial, 2+=revisions
    content     TEXT    NOT NULL,
    word_count  INTEGER,
    created_at  TEXT    NOT NULL,
    UNIQUE(section_id, version)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- GaneshCritique
-- Critic's evaluation record for a specific draft.
-- Preserved permanently — used by Reviser, by human reviewers,
-- and for quality trend analysis.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS GaneshCritique (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id    INTEGER NOT NULL REFERENCES GaneshSection(id) ON DELETE CASCADE,
    draft_id      INTEGER NOT NULL REFERENCES GaneshDraft(id)   ON DELETE CASCADE,
    -- 'section' | 'cross_section' | 'document'
    scope         TEXT    NOT NULL DEFAULT 'section',
    -- JSON: {"scientific_accuracy": 8.2, "narrative_coherence": 7.1, ...}
    scores_json   TEXT,
    -- JSON: [{"dimension": "...", "issue": "...", "suggestion": "..."}, ...]
    issues_json   TEXT,
    overall_score REAL    NOT NULL,
    created_at    TEXT    NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────────
-- GaneshRevision
-- Links a before-draft, an after-draft, and the critique that drove the change.
-- Provides full auditability of why the section changed between versions.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS GaneshRevision (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id      INTEGER NOT NULL REFERENCES GaneshSection(id) ON DELETE CASCADE,
    from_draft_id   INTEGER          REFERENCES GaneshDraft(id),
    to_draft_id     INTEGER          REFERENCES GaneshDraft(id),
    critique_id     INTEGER          REFERENCES GaneshCritique(id),
    -- brief human-readable description of what the Reviser changed
    changes_summary TEXT,
    created_at      TEXT NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────────
-- GaneshContext
-- Snapshot of the knowledge loaded into a document at run time.
-- Immutable after creation — even if SHANI reruns, this document
-- is not affected.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS GaneshContext (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES GaneshDocument(id) ON DELETE CASCADE,
    -- 'shani_workflow' | 'vishwakarma_job' | 'manual'
    context_type TEXT    NOT NULL,
    -- ID in the source system (workflow_id, job_id, etc.)
    context_ref  TEXT,
    -- extracted + summarised knowledge bundle as JSON
    context_json TEXT,
    created_at   TEXT    NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Indexes
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS ix_ganesh_section_document
    ON GaneshSection(document_id);

CREATE INDEX IF NOT EXISTS ix_ganesh_section_status
    ON GaneshSection(document_id, status);

CREATE INDEX IF NOT EXISTS ix_ganesh_draft_section
    ON GaneshDraft(section_id);

CREATE INDEX IF NOT EXISTS ix_ganesh_critique_draft
    ON GaneshCritique(draft_id);

CREATE INDEX IF NOT EXISTS ix_ganesh_context_document
    ON GaneshContext(document_id);

"""


def run_migration(repo) -> None:
    """
    Apply the GANESH schema migration against the shared Repository.

    Idempotent — safe to call multiple times.
    All statements use CREATE TABLE IF NOT EXISTS.

    Usage:
        from ganesh.schema import run_migration
        from repositories.repository import Repository

        repo = Repository()
        run_migration(repo)
        repo.close()
    """
    with repo.transaction() as cursor:
        # SQLite doesn't support multiple statements in one execute() call.
        # Split on semicolons and run each statement individually.
        statements = [
            stmt.strip()
            for stmt in GANESH_SCHEMA.split(";")
            if stmt.strip()
        ]
        for stmt in statements:
            cursor.execute(stmt)

    print("✅ GANESH schema migration complete.")
