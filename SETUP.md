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
git clone https://github.com/Shardul-collab/BRAHM-MCP.git
cd BRAHM-MCP
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

**Disk/GPU footprint warning:** Chitragupta's and SHANI's
`requirements.txt` pull a full CUDA + OCR + voice-transcription stack
unconditionally (`torch`, `torchvision`, a dozen `nvidia-*` CUDA wheels,
`paddleocr`/`paddlex`, `openai-whisper`, `spacy` with a model wheel).
This is tens of GB, not a lightweight API dependency set, and it's not
behind an optional extra — even just importing Chitragupta's FastAPI app
pulls in `voice/whisper_handler.py` at module load time via its `voice`
router. Confirmed to exhaust disk on a constrained/sandboxed environment
(GPU-only wheels with no CPU-only alternative reachable if your network
doesn't allow `download.pytorch.org`). On a normal desktop/workstation
with tens of GB free this isn't an issue; if you're on a constrained VM
or container, budget for it or expect to trim these two requirements
files down to what you actually need.

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

# Vishwakarma: NO separate venv needed. agents/vishwakarma/requirements.txt
# does not exist, and agents/vishwakarma/vishwakarma_api.py (which a
# per-agent venv would imply you're installing for) is dead code -- it
# imports Notion/config modules that don't exist anywhere in this repo.
# The real, working Vishwakarma entry point is brahm/agents/vishwakarma.py,
# used through the MCP tool layer (step 6/8 below), which only needs the
# root-level venv set up in the next block. Don't create agents/vishwakarma/.venv.

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

At minimum you need `GROQ_API_KEY` for GANESH and SHANI's S5 extraction
stage to run. A few things worth knowing before you fill it in (full
detail is in the file's own comments):

- **`GROQ_MODEL`**: the current default (`llama-3.3-70b-versatile`) was
  deprecated by Groq in June 2026 — confirm it still resolves before
  relying on it; migration targets are `openai/gpt-oss-120b` or
  `qwen/qwen3.6-27b`.
- **Required vs. optional is split explicitly** in the file: `GROQ_API_KEY`,
  `GROQ_MODEL`, `GANESH_LLM`, `NVIDIA_API_KEY` (coordinator only),
  `MP_API_KEY`, `NLP_MODEL`, `LOG_LEVEL`, and `API_KEY` (Chitragupta's own
  optional auth toggle) are the ones that matter for the core pipeline.
  `NOTION_TOKEN`/`NOTION_PAGE_ID`/`NOTION_VERSION`/`WHISPER_MODEL`/
  `SCHEDULE_TIME` are only for Chitragupta's legacy Notion-journal/voice
  features — leave them blank unless you specifically want those.

See `agents/chitragupta/SETUP.md` for more on the legacy-feature split.

---

## 4. Quantum ESPRESSO (required for Vishwakarma / DFT)

**This is the one step in this entire doc that pip cannot help with.**
Vishwakarma shells out to real Quantum ESPRESSO binaries, and QE is not
a Python package — it's a separate Fortran/C scientific computing suite
with no PyPI wheel. You need `conda`/`mamba` (or a from-source build)
specifically; there is no `pip install` path around this step.

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
3. Pseudopotentials: despite earlier notes claiming this repo ships a
   working set, `agents/vishwakarma/pseudo/` does **not** exist in the
   repo (confirmed absent) -- only `pseudo_manager.py`, the code that
   looks them up, is present. You'll need to source your own `.UPF`
   pseudopotential files (e.g. from pslibrary or SSSP) and point
   `QE_PSEUDO` at that directory.
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
