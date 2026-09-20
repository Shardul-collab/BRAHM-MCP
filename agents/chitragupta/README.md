# Chitragupta

> Voice-driven structured logging into Notion — speak to create databases, log entries, and analyze patterns.

---

## What problem it solves

Notion is powerful but slow to fill manually. Chitragupta lets you talk to it: describe a database schema in plain English and it's created. Say your log entry aloud and it's structured, validated, and saved — no form, no keyboard, no browser tab.

It runs entirely on your machine. Voice transcription (Whisper) and intent parsing (DistilBERT) are both local. Notion is the only external service.

---

## Key features

| Feature | Details |
|---|---|
| **Voice input** | Local Whisper transcription — no cloud API needed |
| **Schema inference** | Describe a database in plain English; fields are inferred and confirmed interactively |
| **Notion integration** | Creates databases and pages via the Notion API; supports all major property types |
| **Intent parsing** | Hybrid heuristic + DistilBERT QA pipeline extracts structured field values from free-form speech |
| **Schema drift detection** | Warns before logging if your local schema diverges from the live Notion database |
| **Write journal** | Pending writes are journalled locally so no entry is silently lost on a network failure |
| **Database relations** | Create bidirectional relation fields between any two local databases |
| **Pattern analysis** | Fetch database records and generate usage/trend reports |
| **Scheduler** | Schedule recurring log-entry prompts (foreground, background, or run-once) |
| **REST API** | Full FastAPI layer with versioned routes, API key auth, Swagger UI at `/docs` |

---

## Architecture

```
chitragupta/
│
├── main.py              ← Voice-first CLI entry point
├── api_server.py        ← FastAPI / uvicorn entry point
│
├── core/                ← Conversation engine, dialogue FSM, session memory,
│                           JSON builder, field validator
│
├── nlp/                 ← Intent parser (heuristic + DistilBERT),
│                           schema inferencer (field type inference from text)
│
├── voice/               ← Whisper model singleton, audio recording,
│                           hallucination filter, beep tones
│
├── notion/              ← Notion API client, schema manager (local JSON),
│                           relation manager, write journal
│
├── api/                 ← FastAPI app factory, routers (databases, entries,
│   └── routers/            voice, analysis), request models, API key auth
│
├── analysis/            ← Pattern analyzer (fetches + summarises Notion data)
├── scheduler/           ← Recurring task runner (foreground / background)
├── config/              ← Central settings — all env vars loaded here
└── data/                ← Local schema files + session/write-journal state
```

Two independent entry points share the same underlying modules — no logic is duplicated between the CLI and the API.

---

## Requirements

- Python 3.10+
- A Notion account with an [internal integration](https://www.notion.so/my-integrations) and a target page shared with it
- A working microphone (CLI mode only)
- ~2 GB disk space for model weights (Whisper `base` + DistilBERT)

> **CPU-only machines**: replace the `torch` line in `requirements.txt` with the CPU wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before installing.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/your-username/chitragupta.git
cd chitragupta
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
NOTION_TOKEN=secret_...          # Your Notion integration token
NOTION_PAGE_ID=                  # ID of the parent page (optional — can type at runtime)

# API server only
API_KEY=your-secret-key          # Leave empty to disable auth in dev

# Optional overrides
WHISPER_MODEL=base               # tiny / base / small / medium / large
SCHEDULE_TIME=09:00
LOG_LEVEL=INFO
PORT=8000
```

### 3a. Run the voice CLI

```bash
python main.py
```

You'll see a main menu. Say a number or keyword:

```
1 · Create database    2 · Log entry    3 · Analyze data
4 · Link databases     5 · Start scheduler    6 · Exit
```

### 3b. Run the API server

```bash
python api_server.py
# or
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI → `http://localhost:8000/docs`

---

## API overview

All routes are under `/v1`. Every request (except `/health`) requires the `X-API-Key` header.

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — no auth required |
| `GET/POST` | `/v1/databases` | List or create local schemas |
| `GET/POST` | `/v1/entries/{name}` | Read or log entries for a database |
| `POST` | `/v1/voice/log-entry` | Transcribe audio + save entry (returns 202) |
| `GET` | `/v1/analysis/{name}` | Fetch pattern report for a database |
| `POST` | `/v1/relations` | Create a relation field between two databases |
| `GET` | `/v1/session/{name}/last-values` | Last confirmed field values for a database |
| `GET` | `/v1/session/{name}/skip-counts` | Per-field skip counters for a database |
| `DELETE` | `/v1/session/{name}/skip/{field}` | Reset skip counter for a specific field |
| `DELETE` | `/v1/session/{name}` | Clear all session data for a database |

Full request/response schemas are in the Swagger UI at `/docs`.

---

## Folder structure

```
chitragupta/
├── analysis/           Pattern analyzer
├── api/
│   └── routers/        databases · entries · voice · analysis
├── config/             settings.py — single source of truth for all env vars
├── core/               conversation_engine · dialogue_fsm · json_builder
│                       session_memory · validator · confirmation
├── data/
│   └── schemas/        Per-database JSON schema files (auto-created)
├── logs/               chitragupta.log (auto-created)
├── nlp/                intent_parser · schema_inferencer
├── notion/             notion_client · schema_manager · relation_manager
│                       write_journal
├── scheduler/          reminder (foreground / background / run-once)
├── voice/              whisper_handler
├── main.py             CLI entry point
├── api_server.py       API entry point
├── requirements.txt
└── .env.example
```

---

## Notes on first run

- **Model download**: Whisper and DistilBERT weights are downloaded on first use (~1–2 GB total). Subsequent runs load from cache.
- **Token validation**: `config/settings.py` checks that `NOTION_TOKEN` starts with `secret_` or `ntn_` and raises a clear error at startup if it is missing or malformed.
- **Write journal**: if a Notion API call fails mid-session, the entry is preserved in `data/write_journal.json` and you are warned on the next run.

---

## Future scope

- [ ] `.env.example` committed to repo
- [ ] `pytest`-based test suite (replacing ad-hoc `system_test_*.py` scripts)
- [ ] Web UI for non-CLI users
- [ ] Multi-language voice support (Whisper supports 99 languages)
- [ ] Export to formats other than Notion (CSV, SQLite)

---

## License

<!-- Add your license here -->
