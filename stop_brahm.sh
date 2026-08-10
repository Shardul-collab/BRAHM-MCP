#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# BRAHM System Shutdown Script — stops all 6 services cleanly
# Usage: bash stop_brahm.sh
# ═══════════════════════════════════════════════════════════════════

echo "Stopping BRAHM services..."

for port in 8000 8001 8003 8004 8010 5173; do
    PID=$(ss -ltnp 2>/dev/null | grep ":$port " | grep -oP '(?<=pid=)\d+' | head -1)
    if [[ -n "$PID" ]]; then
        kill "$PID" 2>/dev/null
        sleep 0.5
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null
        fi
        echo "  ✓ port $port (PID $PID) stopped"
    else
        echo "    port $port was not in use"
    fi
done

echo "Done. All BRAHM services stopped."
