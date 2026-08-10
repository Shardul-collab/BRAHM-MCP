import { useEffect, useRef, useState } from "react";
import type { RunStatus } from "../api/types";

interface UseRunSocketResult {
  status: RunStatus | null;
  connected: boolean;
  wsError: string | null;
}

/**
 * Connects to /v2/ws/{run_id}, which pushes the same shape as GET /v2/status/{run_id}
 * (confirmed against coordinator/app.py — payload = {run_id, workflow_id, document_id,
 * phase, error, shani, ganesh}), sent only when something changed.
 *
 * Falls back to nothing automatic on disconnect — caller decides whether to poll
 * getStatus() as backup. Kept simple since this is the WS's first real test.
 */
export function useRunSocket(runId: string | null): UseRunSocketResult {
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [connected, setConnected] = useState(false);
  const [wsError, setWsError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!runId) {
      setStatus(null);
      setConnected(false);
      setWsError(null);
      return;
    }

    // Vite proxy forwards ws:// too (see vite.config.ts `ws: true`), so a relative
    // path works from the browser exactly like the HTTP endpoints do.
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/v2/ws/${runId}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setWsError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.ok === false) {
          // run_id not found case from the handler
          setWsError(data.error ?? "Unknown run_id");
          return;
        }
        setStatus(data as RunStatus);
      } catch (err) {
        setWsError("Failed to parse WS message");
      }
    };

    ws.onerror = () => {
      setWsError("WebSocket error");
    };

    ws.onclose = () => {
      setConnected(false);
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [runId]);

  return { status, connected, wsError };
}
