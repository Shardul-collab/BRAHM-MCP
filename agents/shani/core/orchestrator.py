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

    def start_workflow(self, workflow_id: int, stop_after_stage: str = None):

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

        last_stage = self.repo.fetch_one(
            """SELECT stage_name, status FROM Stage
               WHERE workflow_id=? ORDER BY id DESC LIMIT 1""",
            (workflow_id,)
        )

        if last_stage and last_stage["status"] == "completed":
            idx = self.STAGE_SEQUENCE.index(last_stage["stage_name"])
            current_stage_name = self.STAGE_SEQUENCE[idx + 1]
        elif last_stage and last_stage["status"] == "failed":
            current_stage_name = last_stage["stage_name"]
        else:
            if config and config["use_local"]:
                current_stage_name = "S4"
                self.ingest_local_papers(workflow_id)
            else:
                current_stage_name = "S1"

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
