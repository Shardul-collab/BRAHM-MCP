#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# BRAHM System Startup Script — one-command startup for all 6 services
# Usage: bash start_brahm.sh
# ═══════════════════════════════════════════════════════════════════

set -u

BRAHM_ROOT="/mnt/d/brahm"
BRAHM_VENV="$BRAHM_ROOT/.venv/bin/python"
LOG_DIR="$BRAHM_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════"
echo "  BRAHM System Startup — $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════"

echo ""
echo "[0/6] Clearing stale processes on ports 8000/8001/8003/8004/8010/5173 ..."
for port in 8000 8001 8003 8004 8010 5173; do
    PID=$(ss -ltnp 2>/dev/null | grep ":$port " | grep -oP '(?<=pid=)\d+' | head -1)
    if [[ -n "${PID:-}" ]]; then
        echo "      port $port held by PID $PID — killing"
        kill -9 "$PID" 2>/dev/null
    fi
done
sleep 1

wait_for_http() {
    local url="$1"
    local name="$2"
    local max_wait="${3:-30}"
    local waited=0
    while ! curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -qE "^(200|307|404)$"; do
        sleep 5
        waited=$((waited + 5))
        echo -n "."
        if [[ $waited -ge $max_wait ]]; then
            echo ""
            echo "      ✗ $name did not respond within ${max_wait}s — check $LOG_DIR/${name,,}.log"
            return 1
        fi
    done
    echo ""
    echo "      ✓ $name ready (${waited}s)"
    return 0
}

echo ""
echo "[1/6] Starting SHANI on :8000 ..."
cd "$BRAHM_ROOT/agents/shani"
export UNPAYWALL_EMAIL="shardul.khanduri.msc23@nsut.ac.in"
nohup "$BRAHM_ROOT/agents/shani/venv/bin/python" -m uvicorn api:app \
    --host 0.0.0.0 --port 8000 > "$LOG_DIR/shani.log" 2>&1 &
echo "      PID=$! — log: $LOG_DIR/shani.log"
echo "      (SHANI is known to take up to ~5 min on this hardware — SentenceTransformer/HF Hub checks at startup)"
wait_for_http "http://localhost:8000/docs" "SHANI" 300

echo ""
echo "[2/6] Starting Chitragupta on :8003 ..."
cd "$BRAHM_ROOT/agents/chitragupta"
nohup "$BRAHM_ROOT/agents/chitragupta/.venv/bin/python" -m uvicorn api_server:app \
    --host 0.0.0.0 --port 8003 > "$LOG_DIR/chitragupta.log" 2>&1 &
echo "      PID=$! — log: $LOG_DIR/chitragupta.log"
wait_for_http "http://localhost:8003/docs" "Chitragupta" 30

echo ""
echo "[3/6] Starting GANESH on :8001 ..."
cd "$BRAHM_ROOT/agents/ganesh"
nohup "$BRAHM_ROOT/agents/ganesh/.venv/bin/python" -m uvicorn ganesh_api:app \
    --host 0.0.0.0 --port 8001 > "$LOG_DIR/ganesh.log" 2>&1 &
echo "      PID=$! — log: $LOG_DIR/ganesh.log"
wait_for_http "http://localhost:8001/docs" "GANESH" 30

echo ""
echo "[4/6] Starting Vishwakarma on :8004 ..."
export QE_BIN_DIR=/mnt/d/miniforge3/bin
cd "$BRAHM_ROOT/agents/vishwakarma"
nohup "$BRAHM_ROOT/agents/vishwakarma/.venv/bin/python" -m uvicorn vishwakarma_api:app \
    --host 0.0.0.0 --port 8004 > "$LOG_DIR/vishwakarma.log" 2>&1 &
echo "      PID=$! — log: $LOG_DIR/vishwakarma.log"
wait_for_http "http://localhost:8004/docs" "Vishwakarma" 30

echo ""
echo "[5/6] Starting Coordinator on :8010 ..."
cd "$BRAHM_ROOT"
nohup "$BRAHM_VENV" -m uvicorn brahm.coordinator.app:app \
    --host 0.0.0.0 --port 8010 > "$LOG_DIR/coordinator.log" 2>&1 &
echo "      PID=$! — log: $LOG_DIR/coordinator.log"
wait_for_http "http://localhost:8010/v2/health" "Coordinator" 20

echo ""
echo "[6/6] Starting Dashboard dev server on :5173 ..."
cd "$BRAHM_ROOT/brahm/dashboard"
nohup npm run dev > "$LOG_DIR/dashboard.log" 2>&1 &
echo "      PID=$! — log: $LOG_DIR/dashboard.log"
wait_for_http "http://localhost:5173" "Dashboard" 20

echo ""
echo "════════════════════════════════════════════════════"
echo "  Consolidated health check:"
echo "════════════════════════════════════════════════════"
curl -s http://localhost:8010/v2/health | python3 -m json.tool 2>/dev/null || echo "  ✗ Coordinator health endpoint not responding"

echo ""
echo "════════════════════════════════════════════════════"
echo "  Dashboard:    http://localhost:5173"
echo "  Coordinator:  http://localhost:8010/v2/health"
echo "  Logs:         $LOG_DIR/"
echo "  Stop all:     bash stop_brahm.sh"
echo "════════════════════════════════════════════════════"
