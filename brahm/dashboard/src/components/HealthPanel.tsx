import { useEffect, useState } from "react";
import { getHealth } from "../api/client";
import type { HealthResponse } from "../api/types";

const POLL_INTERVAL_MS = 5000;

// NOTE: /v2/health mixes two different check types under one uniform-looking response:
//   - http_agents (SHANI, GANESH): real HTTP pings to their ports.
//   - local (SQLite, VIDUR, Vishwakarma, Chitragupta): VIDUR/Vishwakarma/Chitragupta here
//     are checked via local Python import success, NOT real HTTP pings to their actual
//     ports (:8002, :8004, :8003). A dead service could still report "ok" here.
// Per explicit instruction, this panel displays all agents uniformly with no visual
// distinction between the two check types - caveat is documented here only.

interface AgentRow {
  key: string;
  label: string;
  status: string;
  detail?: string;
}

function normalizeAgents(health: HealthResponse): AgentRow[] {
  const rows: AgentRow[] = [];

  for (const a of health.http_agents) {
    rows.push({ key: `http-${a.agent}`, label: a.agent, status: a.status, detail: a.hint || undefined });
  }

  for (const c of health.local) {
    const label = c.agent ?? c.component ?? "unknown";
    rows.push({ key: `local-${label}`, label, status: c.status, detail: c.path });
  }

  return rows;
}

function StatusDot({ status }: { status: string }) {
  const ok = status === "online" || status === "ok" || status === "healthy";
  return (
    <span
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
        marginRight: 8,
        backgroundColor: ok ? "#2ecc71" : "#e74c3c",
        flexShrink: 0,
      }}
    />
  );
}

export function HealthPanel() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const data = await getHealth();
        if (!cancelled) {
          setHealth(data);
          setError(null);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to fetch health");
          setLoading(false);
        }
      }
    }

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (loading) {
    return <div style={{ padding: 12, fontSize: 14, opacity: 0.7 }}>Checking agent health...</div>;
  }

  if (error) {
    return (
      <div style={{ padding: 12, fontSize: 14, color: "#e74c3c" }}>
        Coordinator unreachable: {error}
      </div>
    );
  }

  if (!health) return null;

  const rows = normalizeAgents(health);

  return (
    <div style={{ padding: 12, border: "1px solid #333", borderRadius: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <strong style={{ fontSize: 14 }}>Agent Health</strong>
        <span style={{ fontSize: 12, opacity: 0.7 }}>{health.overall}</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {rows.map((row) => (
          <div key={row.key} style={{ display: "flex", alignItems: "center", fontSize: 13 }}>
            <StatusDot status={row.status} />
            <span style={{ minWidth: 110 }}>{row.label}</span>
            <span style={{ opacity: 0.6, fontSize: 12 }}>{row.status}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
