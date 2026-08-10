import { useEffect, useRef, useState } from "react";
import { createPlan, getPlan, approvePlan, rejectPlan } from "../api/client";
import type { Plan, PlanStep } from "../api/types";

interface PlanReviewProps {
  onRunStarted: (runId: string) => void;
}

const inputStyle: React.CSSProperties = {
  padding: "6px 8px",
  fontSize: 13,
  border: "1px solid #444",
  borderRadius: 4,
  background: "transparent",
  color: "inherit",
};

const labelStyle: React.CSSProperties = {
  fontSize: 12,
  opacity: 0.7,
  marginBottom: 4,
  display: "block",
};

const POLL_MS = 1500;
const TERMINAL = new Set(["complete", "rejected", "failed"]);

const ACTIVE_PLAN_KEY = "brahm_active_plan_id";

export function PlanReview({ onRunStarted }: PlanReviewProps) {
  const [prompt, setPrompt] = useState("");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [editedSteps, setEditedSteps] = useState<PlanStep[] | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const notifiedRunRef = useRef(false);

  function stopPolling() {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  useEffect(() => stopPolling, []); // cleanup on unmount

  // Restore-on-mount: browser refresh wipes React state, but the backend's
  // plan_store still has the real plan/status in server memory. Without
  // this, a refresh mid-execution silently looked like "nothing ever
  // happened" even though the plan (and any workflow it created) kept
  // running server-side the whole time.
  useEffect(() => {
    const savedPlanId = localStorage.getItem(ACTIVE_PLAN_KEY);
    if (!savedPlanId) return;

    (async () => {
      try {
        const restored = await getPlan(savedPlanId);
        setPlan(restored);
        setEditedSteps(restored.steps);

        if (restored.run_id) {
          notifiedRunRef.current = true;
          onRunStarted(restored.run_id);
        }

        if (!TERMINAL.has(restored.status) && restored.status !== "pending_approval") {
          startPollingPlan(savedPlanId);
        }
      } catch {
        // Plan no longer exists (coordinator restarted, plan_store reset) --
        // clear the stale pointer rather than repeatedly failing to fetch it.
        localStorage.removeItem(ACTIVE_PLAN_KEY);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleGeneratePlan(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim() || submitting) return;

    setSubmitting(true);
    setError(null);
    setPlan(null);
    setEditedSteps(null);
    notifiedRunRef.current = false;
    stopPolling();

    try {
      const result = await createPlan({ prompt: prompt.trim() });
      setPlan(result);
      setEditedSteps(result.steps);
      try {
        localStorage.setItem(ACTIVE_PLAN_KEY, result.plan_id);
      } catch {
        // not fatal, just means refresh-persistence won't work this session
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate plan");
    } finally {
      setSubmitting(false);
    }
  }

  function updateStepParam(index: number, key: string, value: string) {
    setEditedSteps((prev) => {
      if (!prev) return prev;
      const next = [...prev];
      next[index] = { ...next[index], params: { ...next[index].params, [key]: value } };
      return next;
    });
  }

  function toggleSkip(index: number) {
    setEditedSteps((prev) => {
      if (!prev) return prev;
      const next = [...prev];
      next[index] = { ...next[index], skip: !next[index].skip };
      return next;
    });
  }

  function startPollingPlan(planId: string) {
    stopPolling();
    pollRef.current = window.setInterval(async () => {
      try {
        const latest = await getPlan(planId);
        setPlan(latest);

        if (latest.run_id && !notifiedRunRef.current) {
          notifiedRunRef.current = true;
          onRunStarted(latest.run_id);
        }

        if (TERMINAL.has(latest.status)) {
          stopPolling();
        }
      } catch {
        // transient poll failure - keep trying, don't surface as a hard error
      }
    }, POLL_MS);
  }

  async function handleApprove() {
    if (!plan) return;
    setSubmitting(true);
    setError(null);

    // Only send edited_steps if something actually differs from what the
    // backend already has stored, otherwise omit it so approve_plan keeps
    // the originally generated steps untouched.
    const changed = JSON.stringify(editedSteps) !== JSON.stringify(plan.steps);

    try {
      await approvePlan(plan.plan_id, changed ? editedSteps ?? undefined : undefined);
      startPollingPlan(plan.plan_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to approve plan");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleReject() {
    if (!plan) return;
    setSubmitting(true);
    setError(null);
    try {
      const result = await rejectPlan(plan.plan_id);
      setPlan((prev) => (prev ? { ...prev, status: result.status } : prev));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reject plan");
    } finally {
      setSubmitting(false);
    }
  }

  const isPending = plan?.status === "pending_approval";
  const isTerminal = plan ? TERMINAL.has(plan.status) : false;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 12, border: "1px solid #333", borderRadius: 8 }}>
      <strong style={{ fontSize: 14 }}>New Research Plan</strong>

      <form onSubmit={handleGeneratePlan} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <label style={labelStyle}>Describe what you want to investigate</label>
        <textarea
          style={{ ...inputStyle, width: "100%", minHeight: 60, resize: "vertical", fontFamily: "inherit" }}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g. look into MoS2 synthesis gaps"
        />
        <button
          type="submit"
          disabled={!prompt.trim() || submitting}
          style={{
            alignSelf: "flex-start",
            padding: "8px 16px",
            fontSize: 13,
            borderRadius: 4,
            border: "none",
            background: prompt.trim() && !submitting ? "#2ecc71" : "#555",
            color: "#111",
            cursor: prompt.trim() && !submitting ? "pointer" : "not-allowed",
            fontWeight: 600,
          }}
        >
          {submitting && !plan ? "Generating..." : "Generate Plan"}
        </button>
      </form>

      {error && <div style={{ color: "#e74c3c", fontSize: 13 }}>{error}</div>}

      {plan && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12, borderTop: "1px solid #333", paddingTop: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontSize: 13, opacity: 0.7 }}>Status</span>
            <span style={{ fontSize: 13, fontWeight: 600 }}>{plan.status}</span>
          </div>

          {(plan.state_check.matching_workflows.length > 0 || plan.state_check.matching_documents.length > 0) && (
            <div style={{ fontSize: 12, opacity: 0.7 }}>
              Existing work found: {plan.state_check.matching_workflows.length} workflow(s),{" "}
              {plan.state_check.matching_documents.length} document(s)
            </div>
          )}

          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {(editedSteps ?? plan.steps).map((step, i) => (
              <div key={i} style={{ border: "1px solid #444", borderRadius: 6, padding: 8, opacity: step.skip ? 0.5 : 1 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontFamily: "monospace", fontSize: 13 }}>
                    #{step.order} {step.tool_name}
                  </span>
                  {isPending && (
                    <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}>
                      <input type="checkbox" checked={step.skip} onChange={() => toggleSkip(i)} />
                      Skip
                    </label>
                  )}
                </div>
                <div style={{ fontSize: 12, opacity: 0.7, margin: "4px 0" }}>{step.reason}</div>
                {Object.entries(step.params).map(([k, v]) => (
                  <div key={k} style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 12, marginTop: 2 }}>
                    <span style={{ opacity: 0.6, minWidth: 100 }}>{k}</span>
                    {isPending ? (
                      <input
                        style={{ ...inputStyle, flex: 1 }}
                        value={String(v)}
                        onChange={(e) => updateStepParam(i, k, e.target.value)}
                      />
                    ) : (
                      <span>{String(v)}</span>
                    )}
                  </div>
                ))}
              </div>
            ))}
          </div>

          {plan.error && (
            <div style={{ color: "#e74c3c", fontSize: 13, padding: 8, background: "rgba(231,76,60,0.1)", borderRadius: 4 }}>
              {plan.error}
            </div>
          )}

          {isPending && (
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={handleApprove}
                disabled={submitting}
                style={{ padding: "8px 16px", fontSize: 13, borderRadius: 4, border: "none", background: "#2ecc71", color: "#111", fontWeight: 600, cursor: "pointer" }}
              >
                Approve
              </button>
              <button
                onClick={handleReject}
                disabled={submitting}
                style={{ padding: "8px 16px", fontSize: 13, borderRadius: 4, border: "1px solid #e74c3c", background: "transparent", color: "#e74c3c", fontWeight: 600, cursor: "pointer" }}
              >
                Reject
              </button>
            </div>
          )}

          {!isPending && !isTerminal && (
            <div style={{ fontSize: 12, opacity: 0.6 }}>
              Executing — waiting for a workflow_id to attach live progress...
            </div>
          )}
        </div>
      )}
    </div>
  );
}
