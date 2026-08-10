# Chitragupta — Setup

Chitragupta is BRAHM's knowledge custodian: a FastAPI service (port 8003)
that gives the other agents a central data API for projects, papers, DFT
results, and generated documents. It also retains an earlier Notion/voice
"personal journal tracker" feature set (still live in the API, described
below as legacy) from before it was repurposed into BRAHM.

For the full BRAHM install (all five agents), see the root
[`SETUP.md`](../../SETUP.md) — this file covers Chitragupta on its own.

---

## 1. Virtual environment

```bash
cd agents/chitragupta
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Environment variables

```bash
cp .env.example .env
```

See the comments in `.env.example` for what each key does. At minimum,
`GROQ_API_KEY` is needed (GANESH reads its Groq key from this file, not
its own directory — see `brahm/shared/constants.py`'s `ENV_FILE`).

`API_KEY` is optional — leave blank for local dev. If set, it enables
`X-API-Key` header auth on the FastAPI server.

## 3. Run the server

```bash
.venv/bin/python -m uvicorn api_server:app --host 0.0.0.0 --port 8003
```

Verify:
```bash
curl -s http://localhost:8003/health && echo " — CHIT OK"
```

Note the port: `api_server.py`'s own default is 8000, but BRAHM runs
SHANI on 8000 — Chitragupta must be started with `--port 8003` explicitly,
as above.

---

## What Chitragupta actually does in the BRAHM pipeline

- Central `brahm.db` access: projects, papers, DFT results
  (`chitragupta_save_dft_result` etc.), generated documents
- Cross-workflow paper dedup, called by SHANI's S3 download stage
- Analysis endpoints (`analysis/pattern_analyzer.py`,
  `research_analyzer.py`) for trend/pattern queries over stored research
  data

## Legacy features (still live in the API, not part of the core pipeline)

Chitragupta began as a standalone voice-first personal habit/journal
tracker — log a day's stats by voice, store it in Notion, get trend
analysis back. That functionality's API routes
(`api/routers/databases.py`, `entries.py`, `voice.py`) are still wired
into the live server and still work, but SHANI and GANESH never call
them. You only need this if you specifically want the Notion-journal /
voice-logging endpoints:

- **Notion**: set `NOTION_TOKEN`, `NOTION_PAGE_ID`, `NOTION_VERSION` in
  `.env`. Get a token at https://www.notion.so/my-integrations, then
  open your target Notion page → `···` → **Add connections** → select
  your integration.
- **Voice input**: uses Whisper, which needs FFmpeg installed
  (`sudo apt install ffmpeg` on Ubuntu/WSL2). Set `WHISPER_MODEL` in
  `.env` (default `base`).

If you don't need these, leave the relevant `.env` fields blank — the
core BRAHM pipeline does not depend on them.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` on startup | Re-run `.venv/bin/pip install -r requirements.txt` |
| GANESH can't find `GROQ_API_KEY` | Check `agents/chitragupta/.env`, not `agents/ganesh/.env` — GANESH reads Chitragupta's file |
| Port conflict with SHANI | Chitragupta must run on `--port 8003`, not the module default of 8000 |
| `ffmpeg not found` (voice endpoints only) | Install FFmpeg and ensure it's on `PATH` |
| Notion endpoints failing | Confirm the integration has been added to the target page (`···` → Add connections), not just that the token is set |
