import { useState } from "react";
import { createRun } from "../api/client";
import type { RunRequest } from "../api/types";

interface RunFormProps {
  onRunStarted: (runId: string) => void;
}

const DEFAULT_MAX_PAPERS = 8; // per session guidance: scope test runs small on this hardware
                               // (4-core/16GB, RTX 2050 4GB, Ollama S5/S5_5 inference is slow -
                               // full-corpus runs of 195+ papers can take many hours)

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

export function RunForm({ onRunStarted }: RunFormProps) {
  const [material, setMaterial] = useState("");
  const [focus, setFocus] = useState("");
  const [structure, setStructure] = useState("");
  const [method, setMethod] = useState("");
  const [properties, setProperties] = useState("");
  const [characterization, setCharacterization] = useState("");
  const [maxPapers, setMaxPapers] = useState<number>(DEFAULT_MAX_PAPERS);
  const [documentType, setDocumentType] = useState("literature_review");
  const [autoWrite, setAutoWrite] = useState(true);
  const [useLocal, setUseLocal] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = material.trim().length > 0 && focus.trim().length > 0 && !submitting;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;

    setSubmitting(true);
    setError(null);

    const req: RunRequest = {
      name: `${material.trim()} - ${focus.trim()} - ${new Date().toISOString()}`,
      material: material.trim(),
      focus: focus.trim(),
      max_papers: maxPapers,
      document_type: documentType,
      auto_write: autoWrite,
      use_local: useLocal,
    };
    if (structure.trim()) req.structure = structure.trim();
    if (method.trim()) req.method = method.trim();
    if (properties.trim()) req.properties = properties.trim();
    if (characterization.trim()) req.characterization = characterization.trim();

    try {
      const result = await createRun(req);
      if (!result.ok) {
        setError("Run creation returned ok=false");
        setSubmitting(false);
        return;
      }
      onRunStarted(result.run_id);
      // Note: intentionally not resetting the form fields after a successful submit,
      // so the researcher can tweak one field and re-run without retyping everything.
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start run");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 12, padding: 12, border: "1px solid #333", borderRadius: 8 }}>
      <strong style={{ fontSize: 14 }}>New Research Run</strong>

      <div>
        <label style={labelStyle}>Material *</label>
        <input style={{ ...inputStyle, width: "100%" }} value={material} onChange={(e) => setMaterial(e.target.value)} placeholder="e.g. MoS2" required />
      </div>

      <div>
        <label style={labelStyle}>Focus *</label>
        <input style={{ ...inputStyle, width: "100%" }} value={focus} onChange={(e) => setFocus(e.target.value)} placeholder="e.g. synthesis" required />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div>
          <label style={labelStyle}>Structure</label>
          <input style={{ ...inputStyle, width: "100%" }} value={structure} onChange={(e) => setStructure(e.target.value)} placeholder="optional" />
        </div>
        <div>
          <label style={labelStyle}>Method</label>
          <input style={{ ...inputStyle, width: "100%" }} value={method} onChange={(e) => setMethod(e.target.value)} placeholder="optional" />
        </div>
        <div>
          <label style={labelStyle}>Properties</label>
          <input style={{ ...inputStyle, width: "100%" }} value={properties} onChange={(e) => setProperties(e.target.value)} placeholder="optional" />
        </div>
        <div>
          <label style={labelStyle}>Characterization</label>
          <input style={{ ...inputStyle, width: "100%" }} value={characterization} onChange={(e) => setCharacterization(e.target.value)} placeholder="optional" />
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, alignItems: "end" }}>
        <div>
          <label style={labelStyle}>Max Papers</label>
          <input
            type="number"
            min={1}
            style={{ ...inputStyle, width: "100%" }}
            value={maxPapers}
            onChange={(e) => setMaxPapers(Math.max(1, Number(e.target.value) || 1))}
          />
        </div>
        <div>
          <label style={labelStyle}>Document Type</label>
          <select style={{ ...inputStyle, width: "100%" }} value={documentType} onChange={(e) => setDocumentType(e.target.value)}>
            <option value="literature_review">literature_review</option>
            <option value="research_report">research_report</option>
            <option value="technical_summary">technical_summary</option>
            <option value="manuscript_draft">manuscript_draft</option>
            <option value="dft_report">dft_report</option>
          </select>
        </div>
      </div>

      <div style={{ display: "flex", gap: 20 }}>
        <label style={{ fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={autoWrite} onChange={(e) => setAutoWrite(e.target.checked)} />
          Auto-write (chain into GANESH on SHANI completion)
        </label>
        <label style={{ fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={useLocal} onChange={(e) => setUseLocal(e.target.checked)} />
          Use local (Ollama)
        </label>
      </div>

      {error && <div style={{ color: "#e74c3c", fontSize: 13 }}>{error}</div>}

      <button
        type="submit"
        disabled={!canSubmit}
        style={{
          padding: "8px 16px",
          fontSize: 13,
          borderRadius: 4,
          border: "none",
          background: canSubmit ? "#2ecc71" : "#555",
          color: "#111",
          cursor: canSubmit ? "pointer" : "not-allowed",
          fontWeight: 600,
        }}
      >
        {submitting ? "Starting..." : "Start Run"}
      </button>
    </form>
  );
}
