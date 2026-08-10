"""
ganesh/workflow_config.py
==========================
GANESH's declarative pipeline definition — G1 through G5.

The top-level stages are PHASES of the writing process, not sections.
The section-level iteration (write/critique/revise per section) is
contained entirely inside G3 (execute_section_graph) and is invisible
to the Orchestrator.

Stage map:
  G1 — load_context          Reads SHANI/QE outputs → builds context bundle
  G2 — plan_document         LLM call → DocumentPlan (outline, section specs)
  G3 — execute_section_graph Runs Section DAG: draft/critique/revise per section
  G4 — cross_section_review  Coherence check across all approved sections
  G5 — integrate_document    Assembles final document from approved sections

Retry / failure rationale:
  G1 — no retry: DB reads are deterministic; failure = bad source IDs
  G2 — 2 retries: LLM call may fail transiently (timeout, API error)
  G3 — no retry at stage level: section-level retries are internal to G3
       if G3 fails it means the Section Graph is in a bad state → hard fail
  G4 — 1 retry: LLM call, but soft continue — if cross-section review
       fails, document is still usable (quality_flag set)
  G5 — 2 retries: LLM call for final assembly; must succeed

Note: Unlike SHANI's pre_run_hook (which routes local vs remote paper ingestion),
GANESH's pre_run_hook validates that the source IDs exist and are in a
completed/usable state before the pipeline begins.
"""

from __future__ import annotations

from typing import Optional

from core.stage_definition    import StageDefinition, RetryPolicy, FailurePolicy
from core.workflow_definition  import WorkflowDefinition

# ── Tool imports ──────────────────────────────────────────────────────────────
# These are populated in Phase 1 implementation.
# Stubs are provided here so the WorkflowDefinition can be constructed
# and validated before all tool modules exist.

from ganesh.tools.load_context          import load_context           # G1
from ganesh.tools.plan_document         import plan_document          # G2
from ganesh.tools.execute_section_graph import execute_section_graph  # G3
from ganesh.tools.cross_section_review  import cross_section_review   # G4
from ganesh.tools.integrate_document    import integrate_document     # G5


# ═════════════════════════════════════════════════════════════════════════════
# STAGE DEFINITIONS  (G1 – G5)
# ═════════════════════════════════════════════════════════════════════════════

GANESH_STAGES: list[StageDefinition] = [

    StageDefinition(
        stage_name     = "G1",
        tool_name      = "load_context",
        retry_policy   = RetryPolicy.none(),
        failure_policy = FailurePolicy.HARD_FAIL,
        description    = "Load source knowledge into context bundle",
    ),

    StageDefinition(
        stage_name     = "G2",
        tool_name      = "plan_document",
        # G2 calls an LLM — transient failures possible (timeout, API error).
        retry_policy   = RetryPolicy.transient(max_retries=2, delay_seconds=30),
        failure_policy = FailurePolicy.HARD_FAIL,
        description    = "LLM-driven document planning: outline + section specs + argument flow",
    ),

    StageDefinition(
        stage_name     = "G3",
        tool_name      = "execute_section_graph",
        # Section-level retries are handled internally.
        # If the graph fails entirely, the state is unrecoverable at stage level.
        retry_policy   = RetryPolicy.none(),
        failure_policy = FailurePolicy.HARD_FAIL,
        description    = "Execute section dependency graph: write/critique/revise per section",
    ),

    StageDefinition(
        stage_name     = "G4",
        tool_name      = "cross_section_review",
        # G4 is cross-section coherence. One retry for LLM transients.
        # If it fails after retry, soft-continue — document is still usable.
        retry_policy   = RetryPolicy.transient(max_retries=1, delay_seconds=30),
        failure_policy = FailurePolicy.SOFT_CONTINUE,
        description    = "Cross-section coherence check and targeted revisions",
    ),

    StageDefinition(
        stage_name     = "G5",
        tool_name      = "integrate_document",
        retry_policy   = RetryPolicy.transient(max_retries=2, delay_seconds=30),
        failure_policy = FailurePolicy.HARD_FAIL,
        description    = "Assemble approved sections into final document artifact",
    ),

]


# ═════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

GANESH_TOOLS: dict = {
    "load_context":          load_context,
    "plan_document":         plan_document,
    "execute_section_graph": execute_section_graph,
    "cross_section_review":  cross_section_review,
    "integrate_document":    integrate_document,
}


# ═════════════════════════════════════════════════════════════════════════════
# PRE-RUN HOOK
# ═════════════════════════════════════════════════════════════════════════════

def ganesh_pre_run_hook(repo, workflow_id: int) -> Optional[str]:
    """
    Called by the Orchestrator before the G1–G5 loop begins.

    Reads the GaneshRunConfig table (set at document creation time)
    to validate that source IDs exist and are in usable states.

    Returns None — GANESH always starts at G1.
    (Unlike SHANI, there is no entry-stage override for GANESH.)

    Raises ValueError if sources are missing or in incomplete states,
    which will propagate as an OrchestrationError and halt the run
    before any LLM calls are made.
    """

    config = repo.fetch_one(
        """
        SELECT source_type, source_ids, document_type
        FROM GaneshRunConfig
        WHERE workflow_id = ?
        """,
        (workflow_id,),
    )

    if config is None:
        raise ValueError(
            f"No GaneshRunConfig found for workflow_id={workflow_id}. "
            f"Create a document via the GANESH API before running."
        )

    import json
    source_ids = json.loads(config["source_ids"] or "[]")

    if config["source_type"] == "shani" and source_ids:
        for wf_id in source_ids:
            wf = repo.fetch_one(
                "SELECT status FROM Workflow WHERE id = ?",
                (wf_id,),
            )
            if wf is None:
                raise ValueError(
                    f"SHANI workflow_id={wf_id} does not exist."
                )
            if wf["status"] not in ("completed", "paused"):
                raise ValueError(
                    f"SHANI workflow_id={wf_id} is in status '{wf['status']}'. "
                    f"GANESH requires 'completed' or 'paused' (after S5_5)."
                )

    # Always start at G1 — no entry stage override.
    return None


# ═════════════════════════════════════════════════════════════════════════════
# ASSEMBLED WORKFLOW DEFINITION
# ═════════════════════════════════════════════════════════════════════════════

GANESH_WORKFLOW = WorkflowDefinition(
    agent_name   = "GANESH",
    stages       = GANESH_STAGES,
    tools        = GANESH_TOOLS,
    pre_run_hook = ganesh_pre_run_hook,
)

STAGE_SEQUENCE: tuple  = tuple(GANESH_WORKFLOW.stage_sequence)   # ("G1","G2","G3","G4","G5")
VALID_STAGES:   frozenset = frozenset(STAGE_SEQUENCE)
