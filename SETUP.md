# BRAHM — Setup

BRAHM was built and is actively run on WSL2 (Ubuntu) on Windows 11. These
steps assume that environment; adjust paths for plain Linux/macOS if you
try it there (untested).

**Status note:** this is an active personal research project, not a
packaged product. Setup is manual and involves five separate agent
services plus an external DFT engine. Expect friction.

---

## 1. Clone

```bash
git clone https://github.com/Shardul-collab/BRAHM-prototype-1.git
cd BRAHM-prototype-1
```

By default the codebase assumes it lives at `/mnt/d/brahm`. If you clone
somewhere else, set the `BRAHM_ROOT` environment variable to your actual
path (see `brahm/shared/constants.py`) — every other path in the project
is derived from it.

```bash
export BRAHM_ROOT=/path/to/your/clone
```

---

## 2. Per-agent virtual environments

Each agent has its own venv and `requirements.txt` — there is currently
no single install script that sets all of these up for you.

```bash
# SHANI
cd agents/shani
python -m venv venv
venv/bin/pip install -r requirements.txt
cd ../..

# Chitragupta
cd agents/chitragupta
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ../..

# GANESH
cd agents/ganesh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ../..

# VIDUR
cd agents/vidur
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ../..

# Vishwakarma
cd agents/vishwakarma
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ../..

# Root-level MCP server (brahm/, mcp_server.py)
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

---

## 3. Environment variables

Copy the env template and fill in real keys:

```bash
cp agents/chitragupta/.env.example agents/chitragupta/.env
```

See the comments in that file for what each key does and which ones are
actually required for the core pipeline vs. optional legacy features.
At minimum you need `GROQ_API_KEY` for GANESH and SHANI's S5 extraction
stage to run.

---

## 4. Quantum ESPRESSO (required for Vishwakarma / DFT)

Vishwakarma shells out to real Quantum ESPRESSO binaries — QE itself is
not bundled in this repo.

1. Install QE 7.5 via conda-forge (recommended over your distro's
   package manager — Ubuntu's apt-packaged QE 6.7 is known broken on
   WSL2 due to a `__snprintf_chk` buffer overflow):
   ```bash
   conda create -n qe -c conda-forge qe=7.5
   ```
2. Point Vishwakarma at the binaries via `QE_BIN_DIR` (defaults to
   `/mnt/d/miniforge3/bin` — override this if your conda env lives
   elsewhere):
   ```bash
   export QE_BIN_DIR=/path/to/your/conda/envs/qe/bin
   ```
3. Pseudopotentials: this repo ships a working set at
   `agents/vishwakarma/pseudo/` (filenames follow pslibrary's naming
   convention, PAW/PBE). If you need others, `QE_PSEUDO` controls the
   lookup directory.
4. MPI note: if `mpirun` fails with a PRRTE "not enough slots" error
   under WSL2 even though your machine has enough cores, this is a
   known hwloc slot-detection issue — Vishwakarma's `runner.py` already
   works around it with `--map-by :OVERSUBSCRIBE`. If you still hit
   issues, check `mpirun --version` for PRRTE compatibility.

---

## 5. Optional: Ollama (local model for SHANI S5.5)

SHANI's finding-reconstruction stage (S5_5) currently runs against a
local Ollama model rather than Groq (a known docstring/code mismatch —
see limitations). If you want that stage to run:

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.1:8b-instruct-q3_k_m
```

Note: this specific quantization is confirmed to over-hedge on
synthesis tasks — findings quality from this stage should be treated
as provisional, not published as-is.

---

## 6. Start the services

Open one terminal per service. Start in this order:

```bash
# Terminal 1 — SHANI (:8000)
cd agents/shani && venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2 — Chitragupta (:8003)
cd agents/chitragupta && .venv/bin/python -m uvicorn api_server:app --host 0.0.0.0 --port 8003

# Terminal 3 — GANESH (:8001)
cd agents/ganesh && .venv/bin/python -m uvicorn ganesh_api:app --host 0.0.0.0 --port 8001
```

VIDUR and Vishwakarma are invoked as local Python imports through the
MCP tool layer (`brahm/agents/vidur.py`, `brahm/agents/vishwakarma.py`)
rather than run as standalone HTTP services — no separate terminal
needed for them.

Verify each service:
```bash
curl -s http://localhost:8000/health && echo " — SHANI OK"
curl -s http://localhost:8003/health && echo " — CHIT OK"
curl -s http://localhost:8001/health && echo " — GANESH OK"
```

---

## 7. Optional: coordinator + dashboard

A FastAPI coordinator (conversational plan → approve → execute layer)
and a React/TypeScript dashboard sit on top of the core agents. Not
required to use SHANI/GANESH/Vishwakarma directly via MCP, but useful
for a browser-based control panel.

```bash
# Coordinator (:8010) — needs NVIDIA_API_KEY set. Must be run from the
# repo root exactly like this (its own sys.path setup depends on it):
cd {BRAHM_ROOT}
python -m uvicorn brahm.coordinator.app:app --host 0.0.0.0 --port 8010 --reload

# Dashboard (:5173)
cd brahm/dashboard
npm install
npm run dev
```

**Caveat:** `brahm/coordinator/app.py` currently hardcodes `/mnt/d/brahm`
directly in its own `sys.path` setup and `load_dotenv()` call — it does
NOT yet respect the `BRAHM_ROOT` override from step 1. If your clone
lives anywhere else, the coordinator specifically will need that file
edited by hand until this is migrated (core SHANI/GANESH/Vishwakarma
tools via `constants.py` are already fixed; the coordinator layer isn't).

---

## 8. Connect via MCP (Claude Desktop)

Point Claude Desktop's MCP config at:

```
{BRAHM_ROOT}/.venv/bin/python {BRAHM_ROOT}/mcp_server.py
```

This exposes all ~60 agent tools (SHANI, Chitragupta, GANESH, VIDUR,
Vishwakarma) directly to Claude.

---

## Known limitations (see also: project README)

- `brahm/coordinator/app.py` hardcodes `/mnt/d/brahm` in its `sys.path`
  setup and `load_dotenv()` call, independent of the `BRAHM_ROOT` env
  var the rest of the project now respects (see step 7).
- No automated tests yet — a fresh install is not currently verified
  end-to-end by CI.
- SHANI S5_5 uses a local Ollama model despite its own docstring
  claiming Groq — output quality from that stage is unverified.
- JSON parsing in S5/S5_5 has a known greedy-regex bug that can
  silently drop extraction results for certain content patterns.
- Hardware assumptions (GPU, MPI core count, WSL2-specific paths) were
  tuned against one development machine; your mileage on other setups
  is unverified.
