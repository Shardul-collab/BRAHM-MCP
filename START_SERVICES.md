# BRAHM Service Startup Commands

Open **three separate WSL terminals** (one service each). Each command runs in the **foreground** and blocks that terminal — that is expected.

Start in order: **SHANI → Chitragupta → GANESH**. In a fourth terminal, run the health checks after each service is up.

---

## 0. Optional — stop existing instances first

```bash
pkill -f "uvicorn api:app" 2>/dev/null
pkill -f "uvicorn api_server:app" 2>/dev/null
pkill -f "uvicorn ganesh_api:app" 2>/dev/null
sleep 2
```

---

## Terminal 1 — SHANI (port 8000)

**Recommended** (no `activate` — most reliable on `/mnt/d`):

```bash
cd /mnt/d/brahm/agents/shani
/mnt/d/brahm/agents/shani/venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

**Alternative** (with venv activated):

```bash
cd /mnt/d/brahm/agents/shani
source venv/bin/activate
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Wait until you see `Uvicorn running on http://0.0.0.0:8000`, then:

```bash
curl -s http://localhost:8000/health && echo " — SHANI OK"
```

---

## Terminal 2 — Chitragupta (port 8003)

**Recommended:**

```bash
cd /mnt/d/brahm/agents/chitragupta
/mnt/d/brahm/agents/chitragupta/.venv/bin/python -m uvicorn api_server:app --host 0.0.0.0 --port 8003
```

**Alternative:**

```bash
cd /mnt/d/brahm/agents/chitragupta
source .venv/bin/activate
python -m uvicorn api_server:app --host 0.0.0.0 --port 8003
```

```bash
curl -s http://localhost:8003/health && echo " — CHIT OK"
```

---

## Terminal 3 — GANESH (port 8001)

Loads API keys from Chitragupta `.env` on startup. Ensure `GROQ_API_KEY` is set there.

**Recommended:**

```bash
cd /mnt/d/brahm/agents/ganesh
/mnt/d/brahm/agents/ganesh/.venv/bin/python -m uvicorn ganesh_api:app --host 0.0.0.0 --port 8001
```

**Alternative:**

```bash
cd /mnt/d/brahm/agents/ganesh
source .venv/bin/activate
python -m uvicorn ganesh_api:app --host 0.0.0.0 --port 8001
```

```bash
curl -s http://localhost:8001/health && echo " — GANESH OK"
```

---

## All health checks (fourth terminal)

```bash
curl -s http://localhost:8000/health && echo " — SHANI OK" || echo " — SHANI DOWN"
curl -s http://localhost:8003/health && echo " — CHIT OK"   || echo " — CHIT DOWN"
curl -s http://localhost:8001/health && echo " — GANESH OK" || echo " — GANESH DOWN"
```

---

## Common mistakes

| Problem | Fix |
|--------|-----|
| `uvicorn: command not found` after `source activate` | Use `python -m uvicorn` or the **Recommended** full-path command |
| Wrong app module | Must `cd` into the agent directory first (`api:app` needs `agents/shani/api.py`) |
| GANESH typo `source .venv/bin/activate .` | No trailing `.` — use `source .venv/bin/activate` then newline, or skip activate |
| Port already in use | Run the `pkill` block in section 0 |
| Chitragupta on wrong port | Use `--port 8003` (default in its `.env` may be 8000) |

---

## MCP (Claude Desktop)

Not started here. Point Claude at:

`/mnt/d/brahm/.venv/bin/python /mnt/d/brahm/mcp_server.py`
