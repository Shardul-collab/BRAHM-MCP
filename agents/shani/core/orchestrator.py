from repositories.repository import Repository
import repositories.workflow_repo as workflow_repo
import repositories.stage_repo as stage_repo
import repositories.execution_repo as execution_repo
import repositories.failure_repo as failure_repo

from core.tool_executor import ToolExecutor
from datetime import datetime
import os
import time


class OrchestrationError(Exception):
    pass


class WorkflowNotFoundError(OrchestrationError):
    pass


class InvalidTransitionError(OrchestrationError):
    pass


class StageNotFoundError(OrchestrationError):
    pass


class Orchestrator:

    # =========================================================
    # STAGE SEQUENCE
    #
    # ADDED: 'S2_75' between S2 and S2_5.
    #
    # Full sequence:
    #   S1  — generate_queries
    #   S2  — search_papers
    #   S2_75 — extract_lightweight_knowledge  ← NEW
    #   S2_5  — resolve_pdf
    #   S3  — download_papers
    #   S4  — extract_paper_content
    #   S5  — extract_research_knowledge
    # =========================================================

    STAGE_SEQUENCE = (
        "S1", "S2", "S2_75", "S2_5",
        "S3", "S4", "S5", "S5_5"
    )

    def __init__(self, repo: Repository):
        self.repo = repo
        self.tools = ToolExecutor(repo)
        # Set per start_workflow() call; execute_stage reads it to decide
        # whether idempotency guards apply.
        self._force_restart = False

    def _s2_satisfied(self, workflow_id: int) -> bool:
        """
        Has S2 already produced the papers this workflow asked for?

        max_papers set  -> satisfied once that many papers exist.
        max_papers unset-> satisfied once ANY paper exists. search_papers
                           falls back to FINAL_PAPER_LIMIT=500 in that case,
                           which no run reaches, so counting toward it would
                           never trip and every resume would re-ingest.
        """
        row = self.repo.fetch_one(
            "SELECT max_papers FROM WorkflowResearchConfig WHERE workflow_id = ?",
            (workflow_id,),
        )
        target = row["max_papers"] if row and row["max_papers"] else None
        count = self.repo.fetch_one(
            "SELECT COUNT(*) AS n FROM Paper WHERE workflow_id = ?",
            (workflow_id,),
        )["n"]
        if target:
            return count >= target
        return count > 0

    # =====================================================
    # LOCAL PAPER INGESTION
    # =====================================================

    def ingest_local_papers(self, workflow_id: int):

        papers_dir = "papers"

        if not os.path.exists(papers_dir):
            print("No papers directory found.")
            return

        files = os.listdir(papers_dir)
        count = 0

        for f in files:
            if not f.endswith(".pdf"):
                continue

            title = f.replace(".pdf", "")

            existing = self.repo.fetch_one(
                """
                SELECT id FROM Paper
                WHERE workflow_id = ? AND title = ?
                """,
                (workflow_id, title)
            )

            if existing:
                continue

            timestamp = datetime.utcnow().isoformat()

            with self.repo.transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO Paper (
                        workflow_id,
                        title,
                        source,
                        pdf_url,
                        status,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workflow_id,
                        title,
                        "local",
                        os.path.join("papers", f),
                        "pending",
                        timestamp,
                        timestamp
                    )
                )

            count += 1

        print(f"Ingested {count} local papers.")

    # =====================================================
    # STAGE EXECUTION
    # =====================================================

    def execute_stage(self, stage):

        MAX_RETRIES = 3
        RETRY_DELAY = 60  # seconds

        workflow_id = stage["workflow_id"]
        stage_name  = stage["stage_name"]

        print(f"\nExecuting {stage_name}")

        attempts = 0

        while True:

            try:

                print(f"[{stage_name}] Attempt {attempts + 1}")

                # ---------------------------
                # STAGE DISPATCH
                # ---------------------------

                if stage_name == "S1":
                    result = self.tools.execute("generate_queries", workflow_id)

                elif stage_name == "S2":
                    # S2 is the ONLY stage with no idempotency of its own, and
                    # the only one whose re-run corrupts data rather than just
                    # wasting time. S3 selects pdf_status='enriched', S5
                    # selects papers at status='extracted' and advances each to
                    # 'knowledge_ready' — re-entering those processes only what
                    # is left. search_papers has no such filter: it searches and
                    # inserts unconditionally, so every re-entry appends a fresh
                    # set of papers. Workflow 2 reached 40 papers for a
                    # max_papers=20 run this way.
                    #
                    # The cost of this guard is one COUNT. The cost of not
                    # having it was a duplicated corpus.
                    if not self._force_restart and self._s2_satisfied(workflow_id):
                        print("[S2] Workflow already holds its target papers "
                              "— skipping search (pass force_restart to override)")
                        result = {"status": "success", "skipped": True,
                                  "reason": "papers already ingested"}
                    else:
                        result = self.tools.execute("search_papers", workflow_id)

                elif stage_name == "S2_75":
                    result = self.tools.execute(
                        "extract_lightweight_knowledge", workflow_id
                    )

                elif stage_name == "S2_5":
                    result = self.tools.execute("resolve_pdf", workflow_id)

                elif stage_name == "S3":
                    result = self.tools.execute("download_papers", workflow_id)

                elif stage_name == "S4":
                    result = self.tools.execute("extract_paper_content", workflow_id)

                elif stage_name == "S5":
                    result = self.tools.execute(
                        "extract_research_knowledge", workflow_id
                    )

                elif stage_name == "S5_5":
                    result = self.tools.execute(
                        "reconstruct_findings", workflow_id
                    )




                else:
                    return

                print(f"{stage_name} result:", result)

                # ---------------------------
                # ERROR CHECK
                # ---------------------------

                if result["status"] == "error":
                    raise Exception(result.get("error"))

                # ---------------------------
                # SUCCESS HANDLING
                # ---------------------------

                latest_attempt = execution_repo.get_latest_attempt_for_stage(
                    self.repo,
                    stage["id"]
                )

                if latest_attempt:
                    execution_repo.update_execution_attempt_status(
                        self.repo,
                        latest_attempt["id"],
                        "completed"
                    )
                else:
                    print(f"[WARN] No execution attempt found for {stage_name}")

                stage_repo.update_stage_status(
                    self.repo, stage["id"], "completed"
                )

                return  # SUCCESS

            except Exception as e:

                error_msg = str(e)
                print(f"❌ Stage {stage_name} failed:", error_msg)

                attempts += 1

                # ---------------------------
                # RETRY — S2 and S2_75 only
                # S2_75 retries because LLM calls may
                # fail transiently (Ollama timeout).
                # ---------------------------

                if stage_name in ("S2", "S2_75") and attempts <= MAX_RETRIES:
                    print(
                        f"⏳ Retrying {stage_name} "
                        f"in {RETRY_DELAY} seconds..."
                    )
                    time.sleep(RETRY_DELAY)
                    continue

                # ---------------------------
                # FINAL FAILURE HANDLING
                # ---------------------------

                latest_attempt = execution_repo.get_latest_attempt_for_stage(
                    self.repo,
                    stage["id"]
                )

                if latest_attempt:
                    execution_repo.update_execution_attempt_status(
                        self.repo,
                        latest_attempt["id"],
                        "failed",
                        error_msg
                    )

                    failure_repo.log_failure(
                        self.repo,
                        workflow_id,
                        "SYSTEM_ERROR",
                        error_msg,
                        stage_id=stage["id"],
                        execution_attempt_id=latest_attempt["id"]
                    )
                else:
                    print(
                        f"[WARN] Failure but no execution attempt "
                        f"found for {stage_name}"
                    )

                # S2 failure → reset workflow
                if stage_name == "S2":
                    print(
                        "🚨 S2 failed after retries. "
                        "Resetting workflow to CREATED state."
                    )
                    workflow_repo.update_workflow_status(
                        self.repo, workflow_id, "created"
                    )
                    workflow_repo.update_current_stage(
                        self.repo, workflow_id, None
                    )
                    return

                # S2_75 failure → non-fatal, advance to S2_5
                # Abstract knowledge is optional — if extraction
                # fails entirely, the pipeline continues.
                # PDF-based S5 will cover all papers that have PDFs.
                if stage_name == "S2_75":
                    print(
                        "⚠️ S2_75 failed after retries. "
                        "Continuing pipeline — PDF extraction (S5) "
                        "will cover papers without abstract knowledge."
                    )
                    stage_repo.update_stage_status(
                        self.repo, stage["id"], "failed"
                    )
                    return  # advance to S2_5

                # All other stages → hard fail
                raise OrchestrationError(error_msg)

    # =====================================================
    # START WORKFLOW
    # =====================================================

    def _reconcile_interrupted_stages(self, workflow_id: int) -> int:
        """
        Close out Stage rows left at 'running' with no ended_at.

        A row in that state means the process died mid-stage — nothing is
        running now, because start_workflow is the only entry point and it
        refuses to act on a workflow that is not 'paused'. Leaving the rows
        as 'running' is what made the resume logic misread history.

        They are marked 'failed' and given an ended_at. 'interrupted' would
        be the more honest status — the stage did not fail, it was cut off —
        but Stage carries a CHECK constraint of
        status IN ('running','completed','failed'), and SQLite cannot ALTER a
        CHECK in place; adding a value means rebuilding the table (see
        chitragupta's _migrate_project_event_human_agent for the pattern this
        project uses when that is worth doing). It is not worth it here:
        'failed' already produces the correct resume behaviour, because the
        stage is re-entered either way. Returns how many were closed.
        """
        rows = self.repo.fetch_all(
            """SELECT id, stage_name FROM Stage
               WHERE workflow_id = ? AND ended_at IS NULL
                 AND status NOT IN ('completed', 'failed')""",
            (workflow_id,),
        )
        if not rows:
            return 0
        # Repository exposes fetch_one/fetch_all/transaction — there is no
        # bare execute(); writes go through the transaction context manager.
        with self.repo.transaction() as cursor:
            for r in rows:
                cursor.execute(
                    "UPDATE Stage SET status='failed', ended_at=? WHERE id=?",
                    (datetime.now().isoformat(), r["id"]),
                )
                print(f"[RESUME] Stage {r['stage_name']} (id={r['id']}) was left "
                      f"running by an interrupted process — closing it as "
                      f"'failed' so it is re-entered rather than skipped")
        return len(rows)

    def _resume_point(self, workflow_id: int, config) -> tuple:
        """
        Decide which stage to enter, from the workflow's real history.

        Returns (stage_name, reason). stage_name is None when every stage in
        STAGE_SEQUENCE has already completed.

        THE BUG THIS REPLACES
        ---------------------
        The previous implementation read ONE row:

            SELECT stage_name, status FROM Stage
             WHERE workflow_id=? ORDER BY id DESC LIMIT 1

        and branched: 'completed' -> next stage, 'failed' -> retry it,
        else -> S1. A stage left at 'running' by an interrupted process
        matched neither branch and fell through to S1, restarting the entire
        pipeline from query generation.

        That is not a small mis-step. This pipeline is deliberately
        one-directional, and restarting from S1 is the largest possible
        backward jump — the design's own invariant, broken by its recovery
        path. In workflow 2 it fired three times: S1 ran 5x, S2 5x, S4 3x,
        S5 4x with only one completion, wasting 31% of all recorded pipeline
        time and silently re-ingesting papers until the corpus held 40 rows
        for a 20-paper workflow.

        Two further faults in the same three lines:
          * ORDER BY id DESC takes the LAST-INSERTED row, not the
            furthest-along stage. After any retry the newest row can be an
            earlier stage than one already completed.
          * A workflow with no Stage rows at all and one with a stale
            'running' row both landed in the same else-branch, so "never
            started" and "interrupted at S5" were indistinguishable.

        This version derives the resume point from position in
        STAGE_SEQUENCE, and treats a stage as done only when it is both
        'completed' AND carries an ended_at. Anything else is incomplete and
        gets re-entered — which is safe, because the stages that cost real
        time already skip work that exists: S3 selects only
        pdf_status='enriched' papers, and S5 selects only papers at
        status='extracted', advancing each to 'knowledge_ready' as it goes.
        Re-entering S5 with 8 of 9 papers done processes the 9th, not all 9.
        """
        rows = self.repo.fetch_all(
            "SELECT stage_name, status, ended_at FROM Stage WHERE workflow_id = ?",
            (workflow_id,),
        )

        def pos(name):
            return self.STAGE_SEQUENCE.index(name) if name in self.STAGE_SEQUENCE else -1

        completed = [r["stage_name"] for r in rows
                     if r["status"] == "completed" and r["ended_at"]
                     and pos(r["stage_name"]) >= 0]

        if completed:
            furthest = max(pos(n) for n in completed)
            if furthest + 1 >= len(self.STAGE_SEQUENCE):
                return None, "every stage already completed"
            nxt = self.STAGE_SEQUENCE[furthest + 1]
            return nxt, (f"resuming at {nxt} — furthest completed stage is "
                         f"{self.STAGE_SEQUENCE[furthest]}")

        attempted = [r["stage_name"] for r in rows if pos(r["stage_name"]) >= 0]
        if attempted:
            furthest = max(pos(n) for n in attempted)
            name = self.STAGE_SEQUENCE[furthest]
            return name, (f"re-entering {name} — it was attempted but never "
                          f"completed, and no later stage has completed")

        if config and config["use_local"]:
            self.ingest_local_papers(workflow_id)
            return "S4", "fresh workflow with use_local — starting at S4"
        return "S1", "fresh workflow — starting at S1"

    def start_workflow(self, workflow_id: int, stop_after_stage: str = None,
                       resume_from: str = None, force_restart: bool = False):
        """
        Run a workflow forward from wherever it genuinely left off.

        resume_from:   enter at this stage explicitly, skipping inference.
        force_restart: begin again at S1 even if later stages completed.
                       This is the escape hatch for deliberately redoing work
                       (e.g. re-extracting after an extractor fix) — without
                       it, idempotent resume would be a cage.
        """
        workflow = workflow_repo.get_workflow(self.repo, workflow_id)

        if workflow is None:
            raise WorkflowNotFoundError(
                f"Workflow {workflow_id} not found."
            )

        if workflow["status"] != "paused":
            raise InvalidTransitionError(
                f"Workflow must be paused to start. "
                f"Current status: {workflow['status']}"
            )

        self._force_restart = bool(force_restart)

        # Close out rows left behind by an interrupted process BEFORE reading
        # history, so the resume decision is made against an honest table.
        self._reconcile_interrupted_stages(workflow_id)

        workflow_repo.update_workflow_status(
            self.repo, workflow_id, "running"
        )

        config = self.repo.fetch_one(
            """
            SELECT use_local FROM WorkflowResearchConfig
            WHERE workflow_id = ?
            """,
            (workflow_id,)
        )

        if force_restart:
            current_stage_name = "S1"
            print("[RESUME] force_restart requested — starting at S1")
        elif resume_from:
            if resume_from not in self.STAGE_SEQUENCE:
                workflow_repo.update_workflow_status(self.repo, workflow_id, "paused")
                raise InvalidTransitionError(
                    f"resume_from='{resume_from}' is not a known stage. "
                    f"Valid stages: {', '.join(self.STAGE_SEQUENCE)}"
                )
            current_stage_name = resume_from
            print(f"[RESUME] explicit resume_from={resume_from}")
        else:
            current_stage_name, reason = self._resume_point(workflow_id, config)
            print(f"[RESUME] {reason}")
            if current_stage_name is None:
                workflow_repo.update_workflow_status(
                    self.repo, workflow_id, "completed"
                )
                print("✅ Nothing left to run — workflow already complete.")
                return

        workflow_repo.update_current_stage(
            self.repo, workflow_id, current_stage_name
        )

        # =====================================================
        # MAIN LOOP
        # =====================================================

        while True:

            stage_id = stage_repo.create_stage(
                self.repo,
                workflow_id,
                current_stage_name,
                "running"
            )

            execution_repo.create_execution_attempt(
                self.repo,
                stage_id,
                1,
                "running"
            )

            stage = stage_repo.get_stage_by_id(self.repo, stage_id)

            self.execute_stage(stage)

            # STOP IF WORKFLOW RESET (S2 hard failure)
            workflow = workflow_repo.get_workflow(self.repo, workflow_id)

            if workflow["status"] == "created":
                print("🛑 Workflow terminated and reset to CREATED state.")
                break

            if current_stage_name == "S5_5":
                print("\n✅ Workflow completed.")
                workflow_repo.update_workflow_status(
                    self.repo, workflow_id, "completed"
                )
                break

            if stop_after_stage and current_stage_name == stop_after_stage:
                print(f"\n⏸ Stopping after {stop_after_stage} as requested.")
                workflow_repo.update_workflow_status(
                    self.repo, workflow_id, "paused"
                )
                workflow_repo.update_current_stage(
                    self.repo, workflow_id, current_stage_name
                )
                break

            index = self.STAGE_SEQUENCE.index(current_stage_name)
            current_stage_name = self.STAGE_SEQUENCE[index + 1]

            workflow_repo.update_current_stage(
                self.repo, workflow_id, current_stage_name
            )
