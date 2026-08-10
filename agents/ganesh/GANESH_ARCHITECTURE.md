# GANESH — Architecture Design Document
# Scientific Writing & Synthesis Agent for BRAHM MCP
# =====================================================

---

## The Core Thesis

SHANI is a deterministic, linear knowledge pipeline.
GANESH is a cognitive, iterative writing agent.

They are architecturally different by design.

SHANI:  structured input  →  structured output  (data transformation)
GANESH: structured input  →  scientific prose   (reasoning + synthesis)

The orchestration framework from SHANI (WorkflowDefinition, StageDefinition) is
reused at the surface, but GANESH's internal execution model is fundamentally different:
it runs a dependency-aware section graph with iterative quality loops per section.

---

## 1. GANESH Architecture — Five Subsystems

```
┌─────────────────────────────────────────────────────────────────────┐
│  GANESH Agent Boundary                                              │
│                                                                     │
│  ┌──────────────────┐      ┌────────────────────────────────────┐  │
│  │  Context Loader  │─────▶│       Document Planner             │  │
│  │                  │      │  outline · section specs ·         │  │
│  │  Reads from DB:  │      │  argument flow · citation plan     │  │
│  │  - ResearchKnow. │      └──────────────┬─────────────────────┘  │
│  │  - PaperContent  │                     │                         │
│  │  - QE results    │           ┌─────────▼──────────┐             │
│  │  - User spec     │           │   Section Graph     │             │
│  └──────────────────┘           │   (Dependency DAG)  │             │
│                                 │   tracks what is    │             │
│                                 │   READY to draft    │             │
│                                 └─────────┬───────────┘             │
│                                           │ next ready section      │
│                                 ┌─────────▼───────────────────────┐ │
│                                 │       Section Executor          │ │
│                                 │  ┌──────────────────────────┐   │ │
│                                 │  │  Writer → draft           │   │ │
│                                 │  │  Critic → evaluate        │   │ │
│                                 │  │  Reviser → improve        │   │ │
│                                 │  │  ↑________________________│   │ │
│                                 │  │  loop until threshold met  │   │ │
│                                 │  └──────────────────────────┘   │ │
│                                 └─────────────────────┬───────────┘ │
│                                                       │ approved     │
│                                 ┌─────────────────────▼───────────┐ │
│                                 │       Document Integrator       │ │
│                                 │  merges approved sections into  │ │
│                                 │  coherent final document        │ │
│                                 └─────────────────────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Working Memory / State Store (SQLite)                      │   │
│  │  drafts · critiques · revisions · evidence map · counters   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Subsystem responsibilities

**Context Loader**
Reads structured knowledge from the shared DB (or other agent outputs).
Translates raw data into a context bundle usable by the Planner:
- extracts key findings, parameters, citations, tables from PaperContent
- extracts structured knowledge records from ResearchKnowledge
- extracts QE results from Vishwakarma job tables
- accepts a user-provided document specification (type, venue, constraints)

**Document Planner**
Takes the context bundle and produces a DocumentPlan:
- the section list and dependency graph
- per-section briefs (what goes in each, what evidence to use, target length)
- the argument flow map (how claims build across sections)
- citation placement plan
The Planner is an LLM call with a structured output schema.

**Section Graph**
An in-memory (and DB-persisted) directed acyclic graph of GaneshSection nodes.
Tracks which sections are PENDING, READY (dependencies met), DRAFTING, APPROVED.
The executor picks from the READY queue — multiple sections can run in parallel
if they have no shared dependencies (e.g., Introduction and Related Work).

**Section Executor**
Runs the write → critic → reviser loop for a single section.
This is GANESH's core cognitive unit.
Each iteration: Writer produces draft → Critic scores it against the section brief
and quality rubric → if below threshold, Reviser applies the critique → repeat.
On convergence (score ≥ threshold, or max iterations reached), the section
is marked APPROVED and its dependencies are unlocked in the Section Graph.

**Document Integrator**
Once all sections are APPROVED, the Integrator:
- merges sections in narrative order (which may differ from execution order)
- checks cross-section coherence (transitions, repeated claims, contradictions)
- applies final formatting for the target venue
- produces the final document artifact (stored as GaneshDocument.final_output)


---

## 2. Internal Workflow Model — The Document Execution Graph (DEG)

This is the single most important architectural difference from SHANI.

SHANI executes stages in a fixed linear sequence:
  S1 → S2 → S2_75 → S2_5 → S3 → S4 → S5 → S5_5
  Each stage waits for the previous one. No parallelism. No loops.

GANESH executes a dependency DAG of sections:
  - sections with no dependencies start immediately
  - sections whose dependencies are all APPROVED become READY
  - READY sections are dispatched to Section Executors
  - each section runs its own iterative quality loop
  - the graph advances as sections are approved

Example DEG for a literature review paper:

```
Level 0 (no deps — can start in parallel):
  ┌─────────────────┐       ┌─────────────────┐
  │   Introduction  │       │  Related Work   │
  │  (context, why) │       │  (prior art)    │
  └────────┬────────┘       └────────┬────────┘
           │                         │
Level 1:   │                         │
  ┌────────▼────────┐                │
  │     Methods     │                │
  │  (approach)     │                │
  └────────┬────────┘                │
           │                         │
Level 2:   │                         │
  ┌────────▼────────┐                │
  │     Results     │                │
  │  (findings)     │                │
  └────────┬────────┘                │
           │                         │
Level 3:   └──────────┬──────────────┘
                      │ (both feed Discussion)
             ┌────────▼────────┐
             │   Discussion    │
             │  (synthesis)    │
             └────────┬────────┘
                      │
Level 4:     ┌────────▼────────┐
             │    Abstract     │
             │  (written last) │
             └─────────────────┘
```

The Section Graph has two important properties:

PROPERTY 1: PARALLELISM
  Introduction and Related Work have no dependencies.
  They are both in the READY queue at t=0.
  They can be drafted concurrently (via threading).

PROPERTY 2: DEPENDENCY-GATED EXECUTION
  Methods cannot start until Introduction is APPROVED.
  Discussion cannot start until BOTH Results AND Related Work are APPROVED.
  Abstract cannot start until Discussion is APPROVED.
  This mirrors the cognitive reality of writing: you cannot discuss results
  you have not yet described.

The DEG is defined per document type (review, report, manuscript, etc.).
Different document types have different graphs.
See Section 4 (Document Planning System) for type-specific graphs.


---

## 3. Memory / State Model

GANESH maintains three categories of state:

### 3a. Context Memory (read-once at startup)
Loaded by Context Loader before planning begins.
Stored as GaneshContext rows in the DB.
Includes: paper abstracts, extracted knowledge records, QE results,
user constraints (document type, target venue, word count, style guide).
Immutable after loading — the Planner and Executors read from it but never modify it.

### 3b. Document State (durable, DB-persisted)
Evolves throughout the run. Stored across: GaneshDocument, GaneshSection,
GaneshDraft, GaneshCritique, GaneshRevision (see Schema, Section 7).

State machine per GaneshSection (see Section 5 for full state diagram):
  PENDING → READY → DRAFTING → DRAFT_COMPLETE
  → UNDER_REVIEW → REVISION_NEEDED → REVISING
  → APPROVED → INTEGRATED

The full draft + critique + revision history is preserved.
This means: if GANESH crashes mid-run, it can resume from the last
APPROVED section without re-running earlier sections.

### 3c. Working Memory (in-process, per executor)
Short-lived, per-section-executor state:
- current draft text
- current critique (scores + actionable feedback)
- iteration counter
- quality score history (to detect improvement stalls)
- evidence fragments from context (already retrieved, don't re-query)

This is Python object state, not DB-persisted. Lost on crash (which is fine —
the Executor restarts from the READY state of the section).

### Evidence Map
Cross-cutting concern: which evidence fragments support which claims.
Stored as part of GaneshContext or as annotations on GaneshDraft.
Used by the Critic to flag unsupported claims.
Used by the Integrator to check citation coverage across the document.

### The Isolation Principle
Each Section Executor is isolated: it sees its own context bundle
(section brief + supporting evidence + prior approved sections for coherence)
but not the internal state of sibling executors.
Cross-section coherence is handled by the Critic's cross-section evaluation
(run after all sections are drafted) and by the Integrator.


---

## 4. Document Planning System

The Planner is invoked once per GANESH run with:
  - document_type: the template to use (see below)
  - context_bundle: structured knowledge from source agents
  - constraints: target length, venue, audience, style

It produces a DocumentPlan with three components:

### 4a. Section Specification
For each section:
  - section_name: str (e.g. "Introduction", "Methods")
  - section_type: enum (INTRO / BODY / CONCLUSION / ABSTRACT / REFS)
  - brief: str — what this section must cover, in plain language
  - evidence_refs: list[str] — which context items to draw on
  - target_word_count: int
  - depends_on: list[str] — other section names that must be APPROVED first
  - quality_criteria: list[str] — checkable criteria for the Critic

### 4b. Argument Flow Map
A narrative arc description: how the argument evolves from section to section.
Used by the Critic to flag sections that break the logical flow.
e.g.:
  "Introduction establishes the problem space and identifies the gap.
   Related Work substantiates the gap by cataloguing prior approaches.
   Methods describes the analysis approach used to address the gap.
   Results presents findings without interpretation.
   Discussion interprets findings in light of the established gap.
   Abstract synthesises the arc in 250 words."

### 4c. Document Type Graphs
Pre-defined section DAGs per document type.
Each type specifies the default section order and dependency graph.
Custom sections can be added; defaults can be overridden.

```python
DOCUMENT_TYPES = {
    "literature_review": LiteratureReviewGraph,    # Intro, RelWork, Methods, Results, Discussion, Abstract
    "research_report":   ResearchReportGraph,       # Intro, Background, Methodology, Findings, Conclusions
    "dft_report":        DFTReportGraph,            # Intro, CompDetails, StructureAnalysis, ElecStruct, Discussion
    "technical_summary": TechnicalSummaryGraph,     # Overview, Methods, KeyFindings, Limitations
    "manuscript_draft":  ManuscriptDraftGraph,      # Title, Abstract, Intro, Methods, Results, Discussion, Refs
}
```

Each graph is a Python dataclass with a `sections` list of SectionSpec
objects and a method `build_dag() -> Dict[str, List[str]]` that returns
the dependency mapping.


---

## 5. Critique / Revision Loop System

This is GANESH's core quality mechanism.
It runs inside the Section Executor, per section.

### The Loop

```
┌──────────────────────────┐
│   Receive section brief  │  ← brief + evidence + prior approved sections
└────────────┬─────────────┘
             │
┌────────────▼─────────────┐
│   WRITER: Draft section  │  ← LLM call: "write this section given this brief"
└────────────┬─────────────┘
             │
┌────────────▼─────────────┐
│   CRITIC: Evaluate draft │  ← LLM call: "score this draft against these criteria"
│   produces CritiqueRecord│    returns: {score, dimension_scores, issues[]}
└────────────┬─────────────┘
             │
    ┌────────▼─────────────┐
    │ score ≥ threshold?   │
    │ OR max_iterations?   │
    └──┬───────────────────┘
       │                  │
      YES                 NO
       │                  │
       ▼                  ▼
  ┌─────────┐    ┌─────────────────────────┐
  │ APPROVE │    │ REVISER: Apply critique │
  └─────────┘    │ produces revised draft  │
                 └────────────┬────────────┘
                              │
                              └──► back to CRITIC
```

### Quality Dimensions
The Critic evaluates each dimension independently and returns a score (0–10):

  SCIENTIFIC_ACCURACY   — are claims supported by the evidence provided?
  NARRATIVE_COHERENCE   — does the section flow logically as prose?
  BRIEF_COMPLIANCE      — does the section cover what the brief specified?
  EVIDENCE_INTEGRATION  — are citations/data woven in naturally?
  LOGICAL_PROGRESSION   — do paragraphs build on each other?
  CROSS_SECTION_FIT     — is this consistent with already-approved sections?

Overall quality score = weighted average of dimension scores.
Default threshold: 7.5 / 10.
Default max_iterations: 4 (1 initial draft + 3 revision cycles).

### Convergence and Fallback
If max_iterations is reached before threshold:
  - The best draft (highest score) is selected from revision history.
  - Section is marked APPROVED with a quality_flag = BELOW_THRESHOLD.
  - The Document Integrator notes the flag and may request a final human review.

### Cross-Section Critique
After all sections are individually APPROVED, a cross-section critique runs:
  - Checks for contradictions between sections
  - Checks for duplicate content
  - Checks that the argument flow from the Planner is actually reflected
  - May trigger targeted revisions on specific sections (not a full re-run)

### Critic Persona Profiles
The Critic can be configured with a domain-specific persona:
  - "peer reviewer for Nature Materials"
  - "internal scientific report reviewer"
  - "PhD thesis examiner"
  - "technical documentation editor"
The persona affects the Critic's weighting of quality dimensions.


---

## 6. Interfaces with SHANI and Other Agents

### GANESH consumes — it never invokes source agents directly

GANESH does not call SHANI. It reads from the shared DB that SHANI has already
populated. The interface is the database schema, not an API call.

This is intentional: GANESH is stateless with respect to how knowledge was
acquired. It doesn't care whether knowledge came from SHANI, from Vishwakarma,
or was manually entered. It only reads context bundles.

### Interface A: SHANI → GANESH (literature review path)

After SHANI completes S5_5:
  - Paper table: papers, abstracts, pdf_urls
  - PaperContent table: extracted full text, tables, equations
  - ResearchKnowledge table: structured knowledge parameters
  - WorkflowResearchConfig table: the research focus/material

GANESH's Context Loader reads all of these given a workflow_id.

BRAHM invokes GANESH via MCP tool:
  ganesh_write_review(
      source_workflow_ids = [42, 43, 44],   # SHANI workflow IDs to draw from
      document_type       = "literature_review",
      title               = "ZnO nanostructures: a review",
      constraints         = { "word_count": 8000, "venue": "Progress in Materials Science" }
  )

### Interface B: Vishwakarma → GANESH (DFT report path)

After Vishwakarma completes QE calculations:
  - VishwakarmaJob table: job metadata, structure, parameters
  - VishwakarmaResult table: SCF energy, band gaps, phonon data, etc.

BRAHM invokes GANESH via MCP tool:
  ganesh_write_dft_report(
      job_ids       = [101, 102],
      document_type = "dft_report",
      title         = "First-principles study of ZnO electronic structure"
  )

### Interface C: Multi-agent synthesis

Multiple source agents have run. BRAHM invokes:
  ganesh_synthesize(
      sources = [
          { "type": "shani_workflow", "id": 42 },
          { "type": "vishwakarma_job", "id": 101 },
      ],
      document_type = "manuscript_draft",
      title         = "Structural and electronic properties of ZnO: combined literature and DFT study"
  )

### Interface D: Section-level tools (exposed to BRAHM Group F replacement)

Individual section tools for human-in-the-loop control:
  ganesh_draft_section(document_id, section_name)
  ganesh_critique_section(document_id, section_name)
  ganesh_revise_section(document_id, section_name, instructions)
  ganesh_approve_section(document_id, section_name)
  ganesh_get_document(document_id)
  ganesh_status(document_id)


---

## 7. Storage Schema

All GANESH tables live in the same SQLite DB as SHANI.
Table prefix: Ganesh to avoid naming collisions.

```sql
-- Top-level document artifact
CREATE TABLE GaneshDocument (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    document_type   TEXT NOT NULL,          -- 'literature_review', 'dft_report', etc.
    status          TEXT NOT NULL,          -- planning|drafting|reviewing|integrating|completed|failed
    source_type     TEXT,                   -- 'shani'|'vishwakarma'|'multi_agent'|'manual'
    source_ids      TEXT,                   -- JSON array of source IDs
    outline_json    TEXT,                   -- DocumentPlan as JSON (section specs, arg flow)
    final_output    TEXT,                   -- completed document text
    quality_flag    TEXT,                   -- null | 'below_threshold' | 'human_review_requested'
    total_iterations INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- One row per section in the document
CREATE TABLE GaneshSection (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES GaneshDocument(id),
    section_name    TEXT NOT NULL,          -- 'Introduction', 'Methods', etc.
    section_type    TEXT NOT NULL,          -- INTRO|BODY|CONCLUSION|ABSTRACT|REFS
    brief_json      TEXT,                   -- SectionSpec as JSON
    depends_on      TEXT,                   -- JSON array of section_names that must be APPROVED first
    exec_order      INTEGER,                -- position in final assembled document
    status          TEXT NOT NULL DEFAULT 'pending',
    -- pending|ready|drafting|draft_complete|under_review|revision_needed|revising|approved|integrated
    quality_score   REAL,                   -- final critic score when approved
    iteration_count INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(document_id, section_name)
);

-- Version history for a section's content
CREATE TABLE GaneshDraft (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id  INTEGER NOT NULL REFERENCES GaneshSection(id),
    version     INTEGER NOT NULL,           -- 1 = initial draft, 2+ = revisions
    content     TEXT NOT NULL,
    word_count  INTEGER,
    created_at  TEXT NOT NULL,
    UNIQUE(section_id, version)
);

-- Critic evaluation record per draft
CREATE TABLE GaneshCritique (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id      INTEGER NOT NULL REFERENCES GaneshSection(id),
    draft_id        INTEGER NOT NULL REFERENCES GaneshDraft(id),
    scope           TEXT NOT NULL,          -- 'section'|'cross_section'|'document'
    scores_json     TEXT,                   -- {"scientific_accuracy": 8.2, "narrative_coherence": 7.1, ...}
    issues_json     TEXT,                   -- [{"dimension": ..., "issue": ..., "suggestion": ...}]
    overall_score   REAL NOT NULL,
    created_at      TEXT NOT NULL
);

-- Revision record: what changed between drafts and why
CREATE TABLE GaneshRevision (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id      INTEGER NOT NULL REFERENCES GaneshSection(id),
    from_draft_id   INTEGER REFERENCES GaneshDraft(id),
    to_draft_id     INTEGER REFERENCES GaneshDraft(id),
    critique_id     INTEGER REFERENCES GaneshCritique(id),
    changes_summary TEXT,                   -- brief description of what changed
    created_at      TEXT NOT NULL
);

-- Context bundle: source knowledge loaded into a document run
CREATE TABLE GaneshContext (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES GaneshDocument(id),
    context_type    TEXT NOT NULL,          -- 'shani_workflow'|'vishwakarma_job'|'manual'
    context_ref     TEXT,                   -- ID in source system (workflow_id, job_id, etc.)
    context_json    TEXT,                   -- extracted/summarised context bundle
    created_at      TEXT NOT NULL
);
```

### Schema notes

- GaneshDocument.outline_json stores the full DocumentPlan including the
  argument flow map and per-section briefs. Kept as JSON for flexibility.
- GaneshSection.depends_on is a JSON array of section_name strings
  (not foreign keys to section IDs) because sections reference each other
  by name in the Planner output.
- The full draft lineage (GaneshDraft → GaneshCritique → GaneshRevision)
  gives complete auditability: you can reconstruct exactly why a section
  changed between version 1 and version 3.
- GaneshContext is a snapshot: it stores the context AS LOADED at planning
  time. If SHANI runs again and adds more papers, the existing document is
  not affected.


---

## 8. Orchestration Strategy inside BRAHM MCP

### 8a. Where GANESH lives in BRAHM's tool groups

Current BRAHM groups:
  Group A — SHANI pipeline control (S1–S5_5)
  Group B — Chitragupta / Notion
  Group C — SQLite read
  Group D — ResearchAnalyzer
  Group E — SQLite write
  Group F — Review generation (S6/S7 — BEING REPLACED BY GANESH)
  Group G — VIDUR
  Group H — Vishwakarma / QE

New group:
  Group I — GANESH (scientific writing tools)

Group F (review_draft_sections, review_synthesize_final) is deprecated.
Its functionality is replaced by Group I tools.

### 8b. GANESH and the WorkflowDefinition framework

GANESH reuses the orchestration backbone (WorkflowDefinition, StageDefinition,
Orchestrator, ToolExecutor) for its TOP-LEVEL lifecycle stages.

GANESH's top-level stages are not document sections — they are the phases
of the writing process:

```python
GANESH_STAGES = [
    StageDefinition("G1", "load_context",        RetryPolicy.none(),          FailurePolicy.HARD_FAIL,
                    "Load and validate source knowledge into context bundle"),
    StageDefinition("G2", "plan_document",        RetryPolicy.transient(2,30), FailurePolicy.HARD_FAIL,
                    "LLM-driven document planning: outline + section specs + arg flow"),
    StageDefinition("G3", "execute_section_graph",RetryPolicy.none(),          FailurePolicy.HARD_FAIL,
                    "Run the Section Graph: draft/critique/revise loop per section"),
    StageDefinition("G4", "cross_section_review", RetryPolicy.transient(1,30), FailurePolicy.SOFT_CONTINUE,
                    "Cross-section coherence check, targeted revisions"),
    StageDefinition("G5", "integrate_document",   RetryPolicy.transient(2,30), FailurePolicy.HARD_FAIL,
                    "Assemble approved sections into final document artifact"),
]
```

The SHANI Orchestrator loop runs G1 → G2 → G3 → G4 → G5.

Inside G3 (execute_section_graph), the SectionGraphExecutor runs its own
internal loop — completely invisible to the top-level Orchestrator.
The top-level Orchestrator only sees G3 succeed or fail as a unit.

This nesting is intentional and clean:
  Top-level Orchestrator  →  manages G1–G5 lifecycle
  SectionGraphExecutor    →  manages section DAG traversal
  SectionExecutor         →  manages write/critique/revise loop per section

Each layer of the orchestration knows nothing about the layers below it.

### 8c. GaneshWorkflowDefinition

```python
# ganesh/workflow_config.py

GANESH_WORKFLOW = WorkflowDefinition(
    agent_name   = "GANESH",
    stages       = GANESH_STAGES,
    tools        = GANESH_TOOLS,        # G1-G5 tool functions
    pre_run_hook = ganesh_pre_run_hook, # validates source IDs exist, sets document_type
)
```

The Orchestrator is instantiated the same way as for SHANI:
  orch = Orchestrator(repo, GANESH_WORKFLOW)
  orch.start_workflow(ganesh_workflow_id)

GANESH gets its own workflow_id in the Workflow table.
Its stages are tracked in Stage table as "G1", "G2", etc.
Its execution attempts are tracked in ExecutionAttempt table.
All retry/failure logic from the shared Orchestrator applies.

### 8d. GANESH API endpoint

GANESH gets its own API (ganesh_api.py) on a separate port (e.g. 8001)
OR its endpoints are added to the existing SHANI api.py under /ganesh/*.

Recommended: separate service, separate port, same SQLite DB.

  POST /ganesh/documents                  Create + plan document
  POST /ganesh/documents/{id}/run         Run writing pipeline
  GET  /ganesh/documents/{id}/status      Full status + section states
  GET  /ganesh/documents/{id}/sections/{name}/draft  Get current draft
  POST /ganesh/documents/{id}/sections/{name}/approve  Manual approval
  GET  /ganesh/documents/{id}/output      Final assembled document

### 8e. BRAHM Group I tools (MCP)

```python
ganesh_write_review(source_workflow_ids, title, constraints)
ganesh_write_dft_report(job_ids, title, constraints)
ganesh_synthesize(sources, document_type, title, constraints)
ganesh_draft_section(document_id, section_name)
ganesh_critique_section(document_id, section_name)
ganesh_revise_section(document_id, section_name, instructions)
ganesh_get_document(document_id)
ganesh_status(document_id)
```

These tools call the GANESH API over HTTP — same pattern as Group A calling SHANI.


---

## Summary of Architectural Decisions

| Decision | Rationale |
|---|---|
| GANESH reuses WorkflowDefinition framework for G1–G5 | Keeps lifecycle tracking (stages, retries, failures) consistent with SHANI |
| Section graph execution is internal to G3 | Orchestrator stays simple; complexity is contained |
| GANESH reads from DB, never calls SHANI | Decouples the two agents completely — SHANI can fail without affecting GANESH |
| Evidence map stored per document | GANESH needs to know WHY it wrote what it wrote for the Critic |
| Full draft lineage preserved | Auditability, resumability, human review support |
| Critic uses configurable quality dimensions | Different venues have different standards |
| DocumentPlan stored as outline_json in DB | GANESH can be paused after planning and resumed later |
| Separate section states (8 states) | Enables partial resume, human intervention, parallel execution |
| Cross-section critique is a separate G4 stage | Section-level critique misses coherence issues that only appear at document level |
| Group F (S6/S7) replaced by Group I (GANESH) | S6/S7 were monolithic; GANESH makes writing iterative and auditable |


---

## Implementation Order (Recommended)

Phase 1 — Core infrastructure (this session)
  1. ganesh/schema.py — DB migration for 5 Ganesh tables
  2. ganesh/workflow_config.py — GANESH_WORKFLOW (G1–G5 stage definitions)
  3. ganesh/context_loader.py — reads SHANI DB, builds context bundle
  4. ganesh/document_planner.py — LLM call → DocumentPlan
  5. ganesh/section_graph.py — SectionGraph class, DAG traversal
  6. ganesh/section_executor.py — write/critique/revise loop
  7. ganesh/document_integrator.py — merge approved sections
  8. ganesh/tools.py — G1–G5 tool functions (wired to above classes)

Phase 2 — API and MCP
  9. ganesh_api.py — FastAPI endpoints
  10. mcp_server.py Group I tools (brahm_ganesh_* tools)

Phase 3 — Document type library
  11. ganesh/document_types/ — LiteratureReviewGraph, DFTReportGraph, etc.
  12. ganesh/critic_personas/ — domain-specific critic configurations

Phase 4 — Group F migration
  13. Deprecate review_draft_sections, review_synthesize_final in mcp_server.py
  14. Replace with ganesh_write_review, ganesh_synthesize
