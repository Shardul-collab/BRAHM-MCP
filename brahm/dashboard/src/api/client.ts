import axios from "axios";
import type {
  RunStatus,
  RunsListResponse,
  HealthResponse,
  RunRequest,
  RunCreateResponse,
  Plan,
  PlanRequest,
  PlanStep,
  ApproveResponse,
  RejectResponse,
} from "./types";

// Relative base - Vite proxies /v2 to the coordinator (localhost:8010). See vite.config.ts.
const api = axios.create({ baseURL: "/v2" });

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>("/health");
  return data;
}

export async function getRuns(): Promise<RunsListResponse> {
  const { data } = await api.get<RunsListResponse>("/runs");
  return data;
}

export async function getStatus(runId: string): Promise<RunStatus> {
  const { data } = await api.get<RunStatus>(`/status/${runId}`);
  return data;
}

export async function createRun(req: RunRequest): Promise<RunCreateResponse> {
  const { data } = await api.post<RunCreateResponse>("/run", req);
  return data;
}

export async function createPlan(req: PlanRequest): Promise<Plan> {
  const { data } = await api.post<Plan>("/plan", req);
  return data;
}

export async function getPlan(planId: string): Promise<Plan> {
  const { data } = await api.get<Plan>(`/plan/${planId}`);
  return data;
}

export async function approvePlan(
  planId: string,
  editedSteps?: PlanStep[]
): Promise<ApproveResponse> {
  const { data } = await api.post<ApproveResponse>(`/plan/${planId}/approve`, {
    edited_steps: editedSteps ?? null,
  });
  return data;
}

export async function rejectPlan(planId: string): Promise<RejectResponse> {
  const { data } = await api.post<RejectResponse>(`/plan/${planId}/reject`);
  return data;
}
