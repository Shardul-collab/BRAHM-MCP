import { useState } from "react";
import { HealthPanel } from "./components/HealthPanel";
import { PlanReview } from "./components/PlanReview";
import { RunProgress } from "./components/RunProgress";

const ACTIVE_RUN_KEY = "brahm_active_run_id";

export default function App() {
  // Persisted to localStorage since React state is wiped on refresh --
  // the backend's plan/run state survives fine (it's server-side memory,
  // separate concern from the already-documented run_store/plan_store
  // reset-on-coordinator-restart limitation), but the browser had no way
  // to remember which run_id to reconnect to after a reload.
  const [activeRunId, setActiveRunId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(ACTIVE_RUN_KEY);
    } catch {
      return null;
    }
  });

  function handleRunStarted(runId: string) {
    setActiveRunId(runId);
    try {
      localStorage.setItem(ACTIVE_RUN_KEY, runId);
    } catch {
      // localStorage unavailable (private browsing etc.) - not fatal,
      // just means refresh-persistence silently won't work this session
    }
  }

  return (
    <div
      style={{
        maxWidth: 900,
        margin: "0 auto",
        padding: "24px 16px",
        display: "flex",
        flexDirection: "column",
        gap: 20,
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <header>
        <h1 style={{ fontSize: 20, margin: 0 }}>BRAHM</h1>
        <p style={{ fontSize: 13, opacity: 0.6, margin: "4px 0 0" }}>
          Research operating system — live run dashboard
        </p>
      </header>

      <HealthPanel />

      <div style={{ display: "grid", gridTemplateColumns: activeRunId ? "1fr 1fr" : "1fr", gap: 20 }}>
        <PlanReview onRunStarted={handleRunStarted} />
        {activeRunId && <RunProgress runId={activeRunId} />}
      </div>
    </div>
  );
}
