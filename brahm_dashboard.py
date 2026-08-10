"""
brahm_dashboard.py
==================
Single-file BRAHM control centre.

  • Launches all services as WSL subprocesses
  • Serves a local web dashboard on http://localhost:5000
  • Health-polls every 5 s; BRAHM MCP starts only after SHANI is up

Run from Windows:
    wsl -e /mnt/d/brahm/.venv/bin/python /mnt/d/brahm/brahm_dashboard.py

Or double-click start_brahm.bat
"""

import asyncio
import subprocess
import sys
import time
import httpx
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ─── Service Definitions ──────────────────────────────────────────────────────

SERVICES = {
    "ollama": {
        "label":    "Ollama",
        "subtitle": "mistral:7b · S5 LLM",
        "port":     11434,
        "health":   "http://localhost:11434/api/tags",
        "cmd":      None,  # system service — just check, don't launch
        "color":    "#f97316",
    },
    "shani": {
        "label":    "SHANI",
        "subtitle": "Literature pipeline · :8000",
        "port":     8000,
        "health":   "http://localhost:8000/docs",
        "cmd": [
            "wsl", "-e", "bash", "-c",
            "cd /mnt/d/brahm/agents/shani && "
            "/mnt/d/brahm/agents/shani/venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000"
        ],
        "color": "#6366f1",
    },
    "chitragupta": {
        "label":    "Chitragupta",
        "subtitle": "Notion · knowledge · :8003",
        "port":     8003,
        "health":   "http://localhost:8003/docs",
        "cmd": [
            "wsl", "-e", "bash", "-c",
            "cd /mnt/d/brahm/agents/chitragupta && "
            "/mnt/d/brahm/agents/chitragupta/.venv/bin/python -m uvicorn api_server:app --host 0.0.0.0 --port 8003"
        ],
        "color": "#10b981",
    },
    "vidur": {
        "label":    "VIDUR",
        "subtitle": "Instrument classifier · :8002",
        "port":     8002,
        "health":   "http://localhost:8002/health",
        "cmd": [
            "wsl", "-e", "bash", "-c",
            "cd /mnt/d/brahm/agents/vidur && "
            "/mnt/d/brahm/agents/vidur/.venv/bin/python3 -m uvicorn vidur_api:app --host 0.0.0.0 --port 8002"
        ],
        "color": "#a855f7",
    },
    "vishwakarma": {
        "label":    "Vishwakarma",
        "subtitle": "DFT engine · :8004",
        "port":     8004,
        "health":   "http://localhost:8004/health",
        "cmd": (
            "cd /mnt/d/brahm/agents/vishwakarma && "
            "/mnt/d/brahm/agents/vishwakarma/.venv/bin/python -m uvicorn vishwakarma_api:app --host 0.0.0.0 --port 8004"
        ),
        "state":   "starting",
    },
    "ganesh": {
        "label":    "GANESH",
        "subtitle": "Document synthesis · :8001 · PENDING",
        "port":     8001,
        "health":   "http://localhost:8001/docs",
        "cmd": [
            "wsl", "-e", "bash", "-c",
            "cd /mnt/d/brahm/agents/ganesh && "
            "GROQ_API_KEY=$(grep GROQ_API_KEY /mnt/d/brahm/agents/chitragupta/.env | cut -d= -f2) "
            "echo GANESH not deployed"
        ],
        "color": "#ec4899",
    },
    "mcp": {
        "label":    "BRAHM MCP",
        "subtitle": "52 tools · stdio",
        "port":     None,  # stdio — no port to poll
        "health":   None,
        "cmd": [
            "wsl", "-e", "bash", "-c",
            "cd /mnt/d/brahm && "
            "/mnt/d/brahm/.venv/bin/python mcp_server.py"
        ],
        "color": "#facc15",
    },
}

# ─── State ────────────────────────────────────────────────────────────────────

_status: dict[str, dict] = {
    key: {
        "state":   "starting",   # starting | up | down | no_port
        "checked": None,
        "uptime":  None,
        "started_at": None,
    }
    for key in SERVICES
}
_procs: dict[str, subprocess.Popen] = {}
_start_time = datetime.now()

# ─── Launch ───────────────────────────────────────────────────────────────────

def _launch_service(key: str) -> None:
    svc = SERVICES[key]
    cmd = svc.get("cmd")
    if cmd is None:
        return  # system service (ollama)
    if key in _procs and _procs[key].poll() is None:
        return  # already running
    print(f"[BRAHM] Launching {svc['label']}...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    _procs[key] = proc
    _status[key]["started_at"] = datetime.now().isoformat()


async def _check_health(key: str) -> str:
    svc = SERVICES[key]
    url = svc.get("health")
    if url is None:
        return "no_port"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(url)
            return "up" if r.status_code < 400 else "down"
    except Exception:
        return "down"


async def _health_loop() -> None:
    """Background task: poll all services every 5 s, launch MCP once SHANI is up."""
    mcp_launched = False
    while True:
        for key in SERVICES:
            state = await _check_health(key)
            _status[key]["state"]   = state
            _status[key]["checked"] = datetime.now().isoformat()

        # Launch MCP only after SHANI is confirmed up
        if not mcp_launched and _status["shani"]["state"] == "up":
            _launch_service("mcp")
            mcp_launched = True
            print("[BRAHM] SHANI up — MCP server launched.")

        await asyncio.sleep(5)


def _launch_all_except_mcp() -> None:
    for key in ("shani", "chitragupta", "ganesh", "vidur", "vishwakarma"):
        _launch_service(key)

# ─── FastAPI ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _launch_all_except_mcp()
    asyncio.create_task(_health_loop())
    yield


app = FastAPI(title="BRAHM Dashboard", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/api/health")
async def api_health():
    return JSONResponse({
        key: {**_status[key], "label": SERVICES[key]["label"]}
        for key in SERVICES
    })


@app.get("/api/restart/{key}")
async def api_restart(key: str):
    if key not in SERVICES:
        return JSONResponse({"error": "unknown service"}, status_code=404)
    proc = _procs.get(key)
    if proc and proc.poll() is None:
        proc.terminate()
        await asyncio.sleep(1)
    _launch_service(key)
    return JSONResponse({"restarted": key})


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    uptime = str(datetime.now() - _start_time).split(".")[0]
    return HTMLResponse(HTML_TEMPLATE.replace("{{UPTIME}}", uptime))

# ─── Dashboard HTML ───────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BRAHM Control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #080c10;
    --bg2:      #0d1117;
    --bg3:      #161b22;
    --border:   #21282f;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --up:       #22c55e;
    --down:     #ef4444;
    --starting: #f59e0b;
    --no-port:  #6366f1;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Space Mono', monospace;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* scanline overlay */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.07) 2px,
      rgba(0,0,0,0.07) 4px
    );
    pointer-events: none;
    z-index: 100;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 24px 40px;
    border-bottom: 1px solid var(--border);
    position: relative;
  }

  .logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 22px;
    letter-spacing: 0.12em;
    color: #fff;
  }
  .logo span {
    color: #6366f1;
  }

  .header-meta {
    font-size: 11px;
    color: var(--muted);
    text-align: right;
    line-height: 1.8;
  }

  /* pulse dot */
  .pulse {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--up);
    margin-right: 6px;
    animation: pulse 2s infinite;
    vertical-align: middle;
  }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }
    50%      { box-shadow: 0 0 0 6px rgba(34,197,94,0); }
  }

  /* ── Main grid ── */
  main {
    padding: 40px;
    max-width: 1200px;
    margin: 0 auto;
  }

  .section-label {
    font-family: 'Syne', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
  }

  /* ── Service cards ── */
  .services {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 48px;
  }

  .card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.15s;
    cursor: default;
  }
  .card:hover {
    transform: translateY(-2px);
  }

  /* colored left bar */
  .card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--accent);
    border-radius: 8px 0 0 8px;
  }

  .card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 12px;
  }

  .card-name {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 15px;
    color: #fff;
  }

  .card-subtitle {
    font-size: 10px;
    color: var(--muted);
    margin-top: 3px;
    line-height: 1.5;
  }

  .status-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 4px;
  }
  .status-dot.up       { background: var(--up);       box-shadow: 0 0 8px var(--up); }
  .status-dot.down     { background: var(--down);     box-shadow: 0 0 8px var(--down); }
  .status-dot.starting { background: var(--starting); box-shadow: 0 0 8px var(--starting); animation: blink 1s infinite; }
  .status-dot.no_port  { background: var(--no-port);  box-shadow: 0 0 8px var(--no-port); }

  @keyframes blink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
  }

  .status-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 8px;
  }
  .status-label.up       { color: var(--up); }
  .status-label.down     { color: var(--down); }
  .status-label.starting { color: var(--starting); }
  .status-label.no_port  { color: var(--no-port); }

  .restart-btn {
    margin-top: 14px;
    width: 100%;
    padding: 7px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    cursor: pointer;
    letter-spacing: 0.08em;
    transition: all 0.15s;
  }
  .restart-btn:hover {
    border-color: var(--accent);
    color: #fff;
    background: rgba(255,255,255,0.04);
  }

  /* ── Stats row ── */
  .stats {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 48px;
  }

  .stat {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
  }
  .stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: #fff;
    line-height: 1;
    margin-bottom: 4px;
  }
  .stat-label {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  /* ── Log / activity feed ── */
  .log-box {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 48px;
    max-height: 220px;
    overflow-y: auto;
  }
  .log-box::-webkit-scrollbar { width: 4px; }
  .log-box::-webkit-scrollbar-track { background: transparent; }
  .log-box::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .log-line {
    font-size: 11px;
    line-height: 2;
    color: var(--muted);
    border-bottom: 1px solid rgba(255,255,255,0.03);
    display: flex;
    gap: 16px;
  }
  .log-line .ts   { color: #334155; flex-shrink: 0; }
  .log-line .msg  { color: var(--text); }
  .log-line .svc  { color: var(--muted); flex-shrink: 0; width: 100px; }
  .log-line.up    .msg { color: var(--up); }
  .log-line.down  .msg { color: var(--down); }

  /* ── Pipeline reference ── */
  .pipeline {
    display: flex;
    gap: 0;
    overflow-x: auto;
    padding-bottom: 8px;
    margin-bottom: 48px;
  }
  .pipeline::-webkit-scrollbar { height: 3px; }
  .pipeline::-webkit-scrollbar-thumb { background: var(--border); }

  .stage {
    flex-shrink: 0;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-right: none;
    padding: 14px 18px;
    min-width: 110px;
    position: relative;
  }
  .stage:last-child { border-right: 1px solid var(--border); border-radius: 0 8px 8px 0; }
  .stage:first-child { border-radius: 8px 0 0 8px; }

  .stage-id {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 13px;
    color: #6366f1;
    margin-bottom: 4px;
  }
  .stage-name {
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 0.05em;
    line-height: 1.4;
  }

  /* arrow connector */
  .stage::after {
    content: '›';
    position: absolute;
    right: -11px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--border);
    font-size: 18px;
    z-index: 1;
  }
  .stage:last-child::after { display: none; }

  /* ── Footer ── */
  footer {
    padding: 20px 40px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 10px;
    color: var(--muted);
  }

  /* ── Refresh indicator ── */
  .refresh-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--border);
    z-index: 50;
  }
  .refresh-bar-fill {
    height: 100%;
    background: #6366f1;
    width: 100%;
    animation: countdown 5s linear infinite;
    transform-origin: left;
  }
  @keyframes countdown {
    from { transform: scaleX(1); }
    to   { transform: scaleX(0); }
  }
</style>
</head>
<body>

<header>
  <div>
    <div class="logo">BRAHM <span>//</span> CONTROL</div>
    <div style="font-size:10px;color:var(--muted);margin-top:4px;letter-spacing:0.05em;">
      Bi-directional Research & Analysis Hub for Multimodal Science
    </div>
  </div>
  <div class="header-meta">
    <span class="pulse"></span>DASHBOARD LIVE<br>
    uptime: {{UPTIME}}<br>
    <span id="last-refresh">—</span>
  </div>
</header>

<main>

  <!-- Services -->
  <div class="section-label">System Services</div>
  <div class="services" id="services-grid">
    <!-- injected by JS -->
  </div>

  <!-- Stats -->
  <div class="section-label">At a Glance</div>
  <div class="stats">
    <div class="stat">
      <div class="stat-val" id="stat-up">—</div>
      <div class="stat-label">Services Up</div>
    </div>
    <div class="stat">
      <div class="stat-val" id="stat-total">6</div>
      <div class="stat-label">Total Services</div>
    </div>
    <div class="stat">
      <div class="stat-val">52</div>
      <div class="stat-label">MCP Tools</div>
    </div>
    <div class="stat">
      <div class="stat-val">5</div>
      <div class="stat-label">Agents</div>
    </div>
  </div>

  <!-- Activity log -->
  <div class="section-label">Activity</div>
  <div class="log-box" id="log-box">
    <div class="log-line">
      <span class="ts">—</span>
      <span class="svc">BRAHM</span>
      <span class="msg">Dashboard initialising...</span>
    </div>
  </div>

  <!-- Pipeline reference -->
  <div class="section-label">SHANI Pipeline</div>
  <div class="pipeline">
    <div class="stage"><div class="stage-id">S1</div><div class="stage-name">generate queries</div></div>
    <div class="stage"><div class="stage-id">S2</div><div class="stage-name">search papers</div></div>
    <div class="stage"><div class="stage-id">S2.75</div><div class="stage-name">lightweight extract</div></div>
    <div class="stage"><div class="stage-id">S2.5</div><div class="stage-name">resolve PDF</div></div>
    <div class="stage"><div class="stage-id">S3</div><div class="stage-name">download papers</div></div>
    <div class="stage"><div class="stage-id">S4</div><div class="stage-name">extract content</div></div>
    <div class="stage"><div class="stage-id">S5</div><div class="stage-name">extract knowledge</div></div>
    <div class="stage"><div class="stage-id">G1–G5</div><div class="stage-name">GANESH writing</div></div>
  </div>

</main>

<footer>
  <span>BRAHM MCP v2.0 · May 2026</span>
  <span>ZnSe / ZnSeO Research Platform</span>
  <span>localhost:5000</span>
</footer>

<div class="refresh-bar"><div class="refresh-bar-fill"></div></div>

<script>
const SERVICE_META = {
  ollama:      { color: '#f97316' },
  shani:       { color: '#6366f1' },
  chitragupta: { color: '#10b981' },
  ganesh:      { color: '#ec4899' },
  vidur:       { color: '#a855f7' },
  mcp:         { color: '#facc15' },
};

const STATE_LABELS = {
  up:       'ONLINE',
  down:     'OFFLINE',
  starting: 'STARTING',
  no_port:  'STDIO',
};

let prevStates = {};
const logLines = [];

function ts() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false });
}

function addLog(svc, msg, cls = '') {
  logLines.push({ ts: ts(), svc, msg, cls });
  if (logLines.length > 60) logLines.shift();
  renderLog();
}

function renderLog() {
  const box = document.getElementById('log-box');
  box.innerHTML = [...logLines].reverse().map(l =>
    `<div class="log-line ${l.cls}">
       <span class="ts">${l.ts}</span>
       <span class="svc">${l.svc}</span>
       <span class="msg">${l.msg}</span>
     </div>`
  ).join('');
}

function renderServices(data) {
  const grid = document.getElementById('services-grid');
  let upCount = 0;

  grid.innerHTML = Object.entries(data).map(([key, svc]) => {
    const meta  = SERVICE_META[key] || { color: '#888' };
    const state = svc.state;
    const label = STATE_LABELS[state] || state.toUpperCase();
    if (state === 'up' || state === 'no_port') upCount++;

    // detect state changes for log
    if (prevStates[key] && prevStates[key] !== state) {
      const cls = state === 'up' ? 'up' : state === 'down' ? 'down' : '';
      addLog(svc.label, `${prevStates[key].toUpperCase()} → ${label}`, cls);
    }
    prevStates[key] = state;

    return `
      <div class="card" style="--accent:${meta.color}">
        <div class="card-header">
          <div>
            <div class="card-name">${svc.label}</div>
            <div class="card-subtitle">${getSvcSubtitle(key)}</div>
          </div>
          <div class="status-dot ${state}"></div>
        </div>
        <div class="status-label ${state}">${label}</div>
        ${state !== 'no_port' ? `
        <button class="restart-btn" onclick="restart('${key}')" style="--accent:${meta.color}">
          ↺ restart
        </button>` : ''}
      </div>`;
  }).join('');

  document.getElementById('stat-up').textContent = upCount;
}

function getSvcSubtitle(key) {
  const map = {
    ollama:      'mistral:7b · S5 LLM',
    shani:       'Literature pipeline · :8000',
    chitragupta: 'Notion · knowledge · :8003',
    ganesh:      'Document synthesis · :8001 · PENDING',
    vidur:       'Instrument classifier · :8002',
    mcp:         '52 tools · stdio',
  };
  return map[key] || '';
}

async function restart(key) {
  addLog(key, 'Restart requested...');
  await fetch(`/api/restart/${key}`);
  addLog(key, 'Restarting...', 'starting');
}

async function poll() {
  try {
    const r    = await fetch('/api/health');
    const data = await r.json();
    renderServices(data);
    document.getElementById('last-refresh').textContent =
      'refreshed ' + ts();
  } catch(e) {
    addLog('BRAHM', 'Health poll failed', 'down');
  }
}

// initial log entry
addLog('BRAHM', 'Dashboard started', 'up');

// first poll immediately, then every 5 s
poll();
setInterval(poll, 5000);
</script>
</body>
</html>
"""

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[BRAHM] Starting dashboard on http://localhost:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="warning")
