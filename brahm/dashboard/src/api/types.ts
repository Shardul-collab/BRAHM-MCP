// Types derived from real /v2/status, /v2/health, /v2/runs responses (2026-07-19 session)

export interface StageAttempt {
  id: number;
  stage_id: number;
  attempt_number: number;
  status: string; // "completed" | "running" | "failed" | ...
  started_at: string | null;
  ended_at: string | null;
  error_message: string | null;
}

export interface Stage {
  id: number;
  workflow_id: number;
  stage_name: string; // "S1" | "S2" | "S2_75" | "S2_5" | "S3" | "S4" | "S5" | "S5_5"
  status: string;
  started_at: string | null;
  ended_at: string | null;
  latest_attempt: StageAttempt | null;
}

export interface ShaniWorkflow {
  id: number;
  name: string;
  current_stage: string;
  status: string; // "running" | "completed" | "failed" | "paused"
  created_at: string;
  updated_at: string;
}

export interface ShaniBlock {
  status: string; // "success" | "error"
  workflow: ShaniWorkflow;
  stages: Stage[];
}

export interface GaneshSection {
  id: number;
  section_name: string;
  section_type: string;
  status: string; // "approved" | "pending" | "failed" | ...
  quality_score: number | null;
  iteration_count: number;
  exec_order: number;
  latest_draft: string | null;
  draft_version: number | null;
}

export interface GaneshDocument {
  id: number;
  title: string;
  document_type: string;
  status: string; // "reviewing" | "completed" | "failed"
  source_type: string;
  source_ids: string;
  outline_json: string;
  final_output: string | null;
  quality_flag: string | null;
  total_iterations: number;
  created_at: string;
  updated_at: string;
  sections: GaneshSection[];
}

export interface GaneshBlock {
  status: string;
  ok: boolean;
  document: GaneshDocument;
}

export interface RunStatus {
  run_id: string;
  workflow_id: number;
  document_id: number | null;
  phase: string;
  error: string | null;
  created_at: string;
  shani?: ShaniBlock;
  ganesh?: GaneshBlock;
}

export interface RunSummary {
  run_id: string;
  workflow_id: number;
  document_id: number | null;
  document_type: string;
  auto_write: boolean;
  phase: string;
  error: string | null;
  created_at: string;
}

export interface RunsListResponse {
  ok: boolean;
  runs: RunSummary[];
}

export interface HttpAgentHealth {
  agent: string;
  status: string; // "online" | "offline"
  type: string;
  hint: string;
}

export interface LocalComponentHealth {
  component?: string;
  agent?: string;
  status: string;
  path?: string;
  type?: string;
}

export interface HealthResponse {
  status: string;
  overall: string; // "healthy" | "degraded" | ...
  http_agents: HttpAgentHealth[];
  local: LocalComponentHealth[];
}

export interface RunRequest {
  name: string;
  material: string;
  focus: string;
  structure?: string;
  method?: string;
  properties?: string;
  characterization?: string;
  use_local?: boolean;
  max_papers?: number;
  document_type?: string;
  auto_write?: boolean;
}

export interface RunCreateResponse {
  ok: boolean;
  run_id: string;
  workflow_id: number;
  phase: string;
}

// Confirmed exhaustively via grep on orchestration.py/app.py set_phase() calls (2026-07-19).
export const KNOWN_PHASES = [
  "shani_running",
  "shani_done",       // terminal if auto_write=false - WS loop does NOT break on this, by design gap
  "ganesh_writing",
  "ganesh_reviewing",
  "complete",         // terminal - WS loop breaks here
  "failed",           // terminal - WS loop breaks here
] as const;

export const TERMINAL_PHASES = new Set(["complete", "failed", "shani_done"]);
// Note: "shani_done" is only truly terminal when the run had auto_write=false.
// UI treats it as terminal defensively rather than checking auto_write, since the
// WS payload does not currently expose that flag back to the client.

// ─── Layer 2: conversational plan proposals (/v2/plan*) ──────────────────
// Derived directly from plan_store.py (PlanStore.create) and app.py's
// PlanRequest/ApproveRequest models (2026-07-23 session). state_check's
// inner list item shapes are NOT independently confirmed via a live call —
// planning.py's _check_current_state passes through raw items from
// shani_get_all_status()["workflows"] and ganesh_list_documents()["documents"],
// which likely carry more fields than shown here. Treat as partial + open
// index signature until an actual /v2/plan response is captured and diffed.

export interface PlanStep {
  order: number;
  tool_name: string;
  reason: string;
  params: Record<string, unknown>;
  skip: boolean;
}

export interface PlanStateCheck {
  matching_workflows: Array<{ name?: string; [key: string]: unknown }>;
  matching_documents: Array<{ title?: string; [key: string]: unknown }>;
}

export type PlanStatus =
  | "pending_approval"
  | "approved"
  | "executing"
  | "complete"
  | "rejected"
  | "failed";

export interface Plan {
  plan_id: string;
  prompt: string;
  status: PlanStatus;
  state_check: PlanStateCheck;
  steps: PlanStep[];
  run_id: string | null;
  error: string | null;
  created_at: string;
}

export interface PlanRequest {
  prompt: string;
}

export interface ApproveRequest {
  edited_steps?: PlanStep[] | null;
}

export interface ApproveResponse {
  ok: boolean;
  plan_id: string;
  status: "executing";
}

export interface RejectResponse {
  ok: boolean;
  plan_id: string;
  status: "rejected";
}

export const TERMINAL_PLAN_STATUSES = new Set<PlanStatus>([
  "complete",
  "rejected",
  "failed",
]);
