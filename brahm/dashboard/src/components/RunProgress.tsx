import { useMemo } from "react";
import { useRunSocket } from "../hooks/useRunSocket";
import { TERMINAL_PHASES } from "../api/types";
import type { GaneshSection, Stage } from "../api/types";

interface RunProgressProps {
  runId: string;
}

const PHASE_LABELS: Record<string, string> = {
  shani_running: "SHANI: literature pipeline running",
  shani_done: "SHANI: complete",
  ganesh_writing: "GANESH: drafting sections",
  ganesh_reviewing: "GANESH: reviewing sections",
  complete: "Complete",
  failed: "Failed",
};

function phaseLabel(phase: string): string {
  return PHASE_LABELS[phase] ?? phase; // unknown phases just show the raw string, not "undefined"
}

function StageRow({ stage }: { stage: Stage }) {
  const color =
    stage.status === "completed" ? "#2ecc71" :
    stage.status === "running" ? "#f1c40f" :
    stage.status === "failed" ? "#e74c3c" :
    "#666";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, padding: "3px 0" }}>
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, flexShrink: 0 }} />
      <span style={{ minWidth: 70, fontFamily: "monospace" }}>{stage.stage_name}</span>
      <span style={{ opacity: 0.7 }}>{stage.status}</span>
      {stage.latest_attempt?.error_message && (
        <span style={{ color: "#e74c3c", fontSize: 12 }}>— {stage.latest_attempt.error_message}</span>
      )}
    </div>
  );
}

function SectionRow({ section }: { section: GaneshSection }) {
  const color =
    section.status === "approved" ? "#2ecc71" :
    section.status === "failed" ? "#e74c3c" :
    "#666";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, padding: "3px 0" }}>
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, flexShrink: 0 }} />
      <span style={{ minWidth: 140 }}>{section.section_name}</span>
      <span style={{ opacity: 0.7 }}>{section.status}</span>
      {section.quality_score != null && (
        <span style={{ opacity: 0.5, fontSize: 12 }}>score {section.quality_score.toFixed(2)}</span>
      )}
    </div>
  );
}

export function RunProgress({ runId }: RunProgressProps) {
  const { status, connected, wsError } = useRunSocket(runId);

  // Known API bug: /v2/status and the WS payload have returned duplicate section
  // entries for the same section id (differing only in draft content/draft_version).
  // Dedupe by id, keeping the last occurrence, for display purposes only - not fixed
  // upstream, this is a frontend workaround.
  const dedupedSections = useMemo(() => {
    const sections = status?.ganesh?.document?.sections;
    if (!sections) return [];
    const byId = new Map<number, GaneshSection>();
    for (const s of sections) byId.set(s.id, s);
    return Array.from(byId.values()).sort((a, b) => a.exec_order - b.exec_order);
  }, [status]);

  if (!status) {
    return (
      <div style={{ padding: 12, fontSize: 13, opacity: 0.7 }}>
        {connected ? "Waiting for first update..." : "Connecting..."}
        {wsError && <div style={{ color: "#e74c3c", marginTop: 6 }}>{wsError}</div>}
      </div>
    );
  }

  const isTerminal = TERMINAL_PHASES.has(status.phase);
  const stages = status.shani?.workflow ? status.shani.stages : [];

  return (
    <div style={{ padding: 12, border: "1px solid #333", borderRadius: 8, display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <strong style={{ fontSize: 14 }}>{phaseLabel(status.phase)}</strong>
        <span style={{ fontSize: 11, opacity: 0.5 }}>
          {connected ? "live" : isTerminal ? "finished" : "disconnected"}
        </span>
      </div>

      {status.error && (
        <div style={{ color: "#e74c3c", fontSize: 13, padding: 8, background: "rgba(231,76,60,0.1)", borderRadius: 4 }}>
          {status.error}
        </div>
      )}

      {stages.length > 0 && (
        <div>
          <div style={{ fontSize: 12, opacity: 0.6, marginBottom: 4 }}>SHANI stages</div>
          {stages.map((s) => <StageRow key={s.id} stage={s} />)}
        </div>
      )}

      {dedupedSections.length > 0 && (
        <div>
          <div style={{ fontSize: 12, opacity: 0.6, marginBottom: 4 }}>GANESH sections</div>
          {dedupedSections.map((s) => <SectionRow key={s.id} section={s} />)}
        </div>
      )}
    </div>
  );
}
