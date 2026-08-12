# BRAHM — Setup

BRAHM was built and is actively run on WSL2 (Ubuntu) on Windows 11. These
steps assume that environment; adjust paths for plain Linux/macOS if you
try it there (untested).

**Status note:** this is an active personal research project, not a
packaged product. Setup is manual and involves five separate agent
services plus an external DFT engine. Expect friction.

---

## Docker (recommended)

As of v1.2.0, `docker compose` replaces the five-manual-venv process
below for SHANI, Chitragupta, GANESH, and Vishwakarma/QE. The manual
path further down is still maintained as a fallback (e.g. this project's
own WSL2 dev machine, or GPU passthrough scenarios where you want more
control than Docker gives you).

**What Docker does and doesn't cover:**
- SHANI, Chitragupta, GANESH run as real networked services with
  `/health` endpoints, same as manual setup.
- Vishwakarma/QE runs in its own container, but **not** as an HTTP
  service — `mcp_server.py` is a stdio MCP server (Claude Desktop spawns
  it as a subprocess over stdin/stdout), so there's no port to `curl`.
  The container stays alive and you reach it via
  `docker exec -i brahm-vishwakarma python /app/mcp_server.py` (this is
  also how Claude Desktop's own MCP config should point at it — see step
  8 below, adjusted for `docker exec`).
- VIDUR has no dedicated port either way (manual or Docker) — it's a
  local import through the MCP layer.
- The coordinator/dashboard layer (step 7) is **not** containerized in
  this pass — it still hardcodes `/mnt/d/brahm` and isn't part of the
  Docker "done" criteria. Run it manually if you need it.

**Steps:**

1. Clone and env files:
   ```bash
   git clone https://github.com/Shardul-collab/BRAHM-MCP.git
   cd BRAHM-MCP
   cp agents/chitragupta/.env.example agents/chitragupta/.env   # fill in GROQ_API_KEY at minimum
   cp .env.example .env                                          # compose-level vars
   ```
   **Also set `NOTION_TOKEN`** in `agents/chitragupta/.env`, even if you
   don't use any Notion features. Confirmed 2026-08-12 via a real
   crash-loop: `config/settings.py` hard-raises at import time if it's
   unset, and that import chain is pulled in unconditionally by
   `api/app.py` — so Chitragupta's container won't start without it
   despite Notion/voice being framed elsewhere as optional legacy
   features. A real token (https://www.notion.so/my-integrations) or any
   non-empty placeholder string both work to get past the check; a
   placeholder just means anything that actually calls the Notion API
   will fail later. This is a real gap in the source, not Docker-specific
   — not fixed here, left as a known issue (see `.env.example`'s comment
   on it for the full explanation).
2. `agents/vishwakarma/pseudo/` is present and populated in this repo
   (confirmed 2026-08-12 — a prior note claiming it was absent was
   stale). `QE_PSEUDO_HOST_DIR` in `.env` already defaults to
   `./agents/vishwakarma/pseudo`, so this step is usually a no-op; only
   override it if you keep your `.UPF` files somewhere else.
3. Bring everything up:
   ```bash
   docker compose --profile full up --build
   ```
   Or, if you only want DFT automation and don't want to pay the
   SHANI/Chitragupta CUDA+OCR image weight:
   ```bash
   docker compose --profile vishwakarma up --build
   ```
4. Verify:
   ```bash
   curl -s http://localhost:8000/health   # SHANI
   curl -s http://localhost:8003/health   # Chitragupta
   curl -s http://localhost:8001/health   # GANESH
   docker exec brahm-vishwakarma /opt/conda/envs/qe/bin/pw.x --version < /dev/null   # Vishwakarma/QE
   ```
   The last one prints a version banner + correct MPI/OpenMP core count
   then exits nonzero — confirmed 2026-08-12, that's expected: `pw.x` has
   no real CLI flag parser, any argument is ignored and it just waits on
   stdin for a namelist. The printed banner is the pass signal, not the
   exit code. For a real end-to-end proof (actual SCF calculation, not
   just the binary launching), use step 7's smoke test script instead.
5. Image sizes and CUDA/torch weight: SHANI's and Chitragupta's images
   stay multi-GB — their `requirements.txt` files pull full CUDA wheels
   and were kept as-is (confirmed decision, not an oversight) rather than
   swapped to CPU-only torch, in case GPU passthrough is added later.
   Chitragupta's whisper/torch *import* cost specifically is avoided via
   `ENABLE_VOICE_FEATURES=false` (default) — the wheel is still in the
   image, but `api/app.py` no longer imports it at startup unless you
   opt in.
6. To connect Claude Desktop to the Dockerized Vishwakarma/VIDUR MCP
   layer instead of a local venv, see step 8 further down for the exact
   `claude_desktop_config.json` entry (`docker exec -i brahm-vishwakarma
   python /app/mcp_server.py`) — it only works while the container is
   running.
7. Real DFT smoke test (not run automatically — launches an actual short
   QE calculation, so it's a manual step):
   ```bash
   ./docker/smoke_test_vishwakarma.sh
   ```
   Needs `Si.pbe-n-kjpaw_psl.1.0.0.UPF` in your `QE_PSEUDO_HOST_DIR`; the
   script tells you clearly if it's missing rather than failing opaquely.

---

## Manual setup (fallback / non-Docker)

### 1. Clone

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

### 2. Per-agent virtual environments

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

### 3. Environment variables

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
  `NOTION_PAGE_ID`/`NOTION_VERSION`/`WHISPER_MODEL`/`SCHEDULE_TIME` are
  only for Chitragupta's legacy Notion-journal/voice features — leave
  them blank unless you specifically want those.
- **`NOTION_TOKEN` is the one exception to that split — confirmed
  2026-08-12 to be a hard requirement, not optional**, despite the file's
  own framing suggesting otherwise. `config/settings.py` raises at import
  time if it's unset, and that import gets pulled in unconditionally by
  `api/app.py`, so Chitragupta's API process won't start at all without
  it — regardless of whether you ever touch a Notion endpoint. Set a
  real token or any non-empty placeholder to get past the check.

See `agents/chitragupta/SETUP.md` for more on the legacy-feature split.

---

### 4. Quantum ESPRESSO (required for Vishwakarma / DFT)

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
3. Pseudopotentials: `agents/vishwakarma/pseudo/` is present and
   populated (confirmed 2026-08-12 via `ls` — a prior note in this file
   claiming it was absent was stale/wrong). Point `QE_PSEUDO` at it:
   ```bash
   export QE_PSEUDO=/path/to/your/clone/agents/vishwakarma/pseudo
   ```
   If you're on a fresh clone and this directory is empty for you, you'll
   need to source your own `.UPF` files (pslibrary or SSSP) instead.
4. MPI note: if `mpirun` fails with a PRRTE "not enough slots" error
   under WSL2 even though your machine has enough cores, this is a
   known hwloc slot-detection issue — Vishwakarma's `runner.py` already
   works around it with `--map-by :OVERSUBSCRIBE`. If you still hit
   issues, check `mpirun --version` for PRRTE compatibility.

---

### 5. Optional: Ollama (local model for SHANI S5.5)

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

### 6. Start the services

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

### 7. Optional: coordinator + dashboard

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

### 8. Connect via MCP (Claude Desktop)

Claude Desktop's config file lives at:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

Open it (create it if it doesn't exist) and add a `brahm` entry under
`mcpServers`. Full working example, assuming the Docker path above:

```json
{
  "mcpServers": {
    "brahm": {
      "command": "docker",
      "args": ["exec", "-i", "brahm-vishwakarma", "python", "/app/mcp_server.py"]
    }
  }
}
```

If there are already other entries under `mcpServers`, add `"brahm": {...}`
as a sibling key, not a replacement for the whole file.

This only works while the container is actually running
(`docker compose ps` shows `brahm-vishwakarma` as `Up`) — `docker exec`
attaches to a live container, it doesn't start one on its own.

For the manual (non-Docker) setup instead, point at the local venv:

```json
{
  "mcpServers": {
    "brahm": {
      "command": "{BRAHM_ROOT}/.venv/bin/python",
      "args": ["{BRAHM_ROOT}/mcp_server.py"]
    }
  }
}
```

Replace `{BRAHM_ROOT}` with your actual absolute clone path in both the
`command` and `args` values — Claude Desktop doesn't expand environment
variables or `~` in this file.

After saving, fully quit and reopen Claude Desktop (a reload isn't
enough — it only reads this file on startup). This exposes all ~60
agent tools (SHANI, Chitragupta, GANESH, VIDUR, Vishwakarma) directly to
Claude.

**Note on secrets:** this config file only tells Claude Desktop how to
*launch* the MCP server — it's not where API keys go. Those live in
`agents/chitragupta/.env` (see step 3 above / the Docker section's step 1),
read by the services themselves, not by Claude Desktop.

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
