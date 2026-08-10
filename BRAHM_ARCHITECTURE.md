# BRAHM Architecture Reference
**Bi-directional Research & Analysis Hub for Multimodal Science**
Version: 1.0 | Date: May 2026 | Status: Canonical Reference

---

## 1. What is BRAHM?

BRAHM is a budget Jarvis for material science research. It is an MCP server that exposes tools to any LLM that connects to it (e.g. Claude). The connected LLM reads the research context and makes suggestions — but **never acts without user approval**.

Intelligence lives in the connected LLM. Primitives live in BRAHM's agents. The user always makes the final decision.

---

## 2. The Five Agents

| Agent | Role | Interface | Port |
|---|---|---|---|
| SHANI | Literature pipeline (S1→S5) | HTTP API | 8000 |
| GANESH | Scientific writing (G1→G5) | HTTP API | 8001 |
| VIDUR | Instrument file classifier | Local import | — |
| Vishwakarma | Quantum ESPRESSO DFT engine | Local subprocess | — |
| Chitragupta | Central data custodian + knowledge hub | HTTP API | 8003 |

---

## 3. Core Design Principle

> **SHANI owns published knowledge. Chitragupta owns our knowledge.**

- **SHANI DB** (`research_workflow.db`) — knowledge derived from published literature. What others have reported. SHANI owns this DB exclusively.
- **Chitragupta DB** (`brahm_knowledge.db`) — knowledge from our own work. What WE have found. Calculations, documents, classifications.

The BRAHM LLM reads **both** when making suggestions to the user.

---

## 4. Data Flow

```
User gives direction
        ↓
Connected LLM (e.g. Claude) calls BRAHM MCP tools
        ↓
BRAHM reads SHANI knowledge DB + Chitragupta DB
        ↓
LLM suggests what to do (calculations, writing, classification)
        ↓
User approves
        ↓
BRAHM triggers agent
        ↓
Chitragupta stores results
```

### Full Pipeline Flow

```
SHANI S1→S5
  [research_workflow.db: papers, knowledge extracted from literature]
        ↓
Chitragupta /context/load
  [filters, ranks, balances knowledge for GANESH]
        ↓
GANESH G1→G5
  [G1: load context into memory]
  [G2: plan document outline]
  [G3: write sections — draft → critique → revise loop via Groq]
  [G4: cross-section coherence check]
  [G5: assemble final document]
        ↓
Chitragupta stores output
  [brahm_knowledge.db: ganesh_documents, ganesh_sections]
```

```
User provides instrument file
        ↓
VIDUR classify → detect → parse
        ↓
Chitragupta stores result
  [brahm_knowledge.db: vidur_classifications]
```

```
BRAHM LLM reads SHANI knowledge DB
        ↓
Suggests DFT calculations based on literature gaps
        ↓
User approves
        ↓
Vishwakarma runs QE calculation
        ↓
Chitragupta stores result
  [brahm_knowledge.db: vishwakarma_calculations]
```

---

## 5. Chitragupta — The Custodian

Chitragupta is the central hub. It:
- Queries SHANI's SQLite directly to serve context to GANESH
- Stores results from GANESH, VIDUR, and Vishwakarma
- Tracks all agent activity in `brahm_activity`
- Logs errors and events per agent

### What Chitragupta Does NOT Do
- Does not write to SHANI's DB
- Does not read from Notion (Notion is export-only, separate concern)
- Does not make decisions — it stores and serves data

### Chitragupta API Endpoints (to build)

| Endpoint | Purpose |
|---|---|
| `GET /context/workflows` | List SHANI workflows available and ready |
| `POST /context/load` | Return curated, filtered context package for GANESH |
| `GET /context/knowledge_summary` | Lightweight summary for GANESH document planning |

### Filtering Layers

- **SHANI filters** — papers must have status `knowledge_ready` or `completed`
- **Chitragupta filters** — relevance score, dedup across workflows, category balance
- **GANESH filters** — selects what to cite per section during writing

---

## 6. brahm_knowledge.db Schema

**Location:** `/mnt/d/brahm/agents/chitragupta/database/brahm_knowledge.db`

### Table: `ganesh_documents`
| Field | Type | Notes |
|---|---|---|
| id | INTEGER PRIMARY KEY | |
| workflow_ids | TEXT | JSON list of SHANI workflow IDs used |
| document_type | TEXT | literature_review, dft_report, etc. |
| status | TEXT | planning, drafting, complete |
| final_document | TEXT | Full document text |
| created_at | DATETIME | |

### Table: `ganesh_sections`
| Field | Type | Notes |
|---|---|---|
| id | INTEGER PRIMARY KEY | |
| document_id | INTEGER | FK → ganesh_documents.id |
| section_name | TEXT | introduction, methodology, etc. |
| draft_text | TEXT | Section content |
| created_at | DATETIME | |

### Table: `vidur_classifications`
| Field | Type | Notes |
|---|---|---|
| id | INTEGER PRIMARY KEY | |
| file_path | TEXT | Absolute path to instrument file |
| technique | TEXT | XRD, UV-Vis, SEM_EDX, Raman |
| confidence | REAL | 0.0 → 1.0 |
| signals | TEXT | JSON list of detection signals |
| parsed_data | TEXT | JSON — axis + intensity arrays |
| created_at | DATETIME | |

### Table: `vishwakarma_calculations`
| Field | Type | Notes |
|---|---|---|
| id | INTEGER PRIMARY KEY | |
| calculation_type | TEXT | scf, relax, bands, dos, phonon, neb, hp |
| material_name | TEXT | e.g. ZnSe, ZnSeO |
| output_file_path | TEXT | Absolute path to QE output file |
| scf_iterations | INTEGER | Number of SCF cycles |
| converged | BOOLEAN | True / False |
| created_at | DATETIME | |

### Table: `brahm_activity`
| Field | Type | Notes |
|---|---|---|
| id | INTEGER PRIMARY KEY | |
| agent | TEXT | ganesh, vidur, vishwakarma, chitragupta |
| action | TEXT | What was triggered |
| triggered_by | TEXT | user or brahm_llm |
| status | TEXT | success, failed |
| timestamp | DATETIME | |

---

## 7. Logs

**Location:** `/mnt/d/brahm/agents/chitragupta/logs/`

| File | Contents |
|---|---|
| `ganesh.log` | GANESH errors, failed sections, LLM failures |
| `vidur.log` | Classification errors, unknown/uncertain results |
| `vishwakarma.log` | QE errors, convergence failures, missing binaries |
| `chitragupta.log` | Context load requests, API errors, DB write failures |

Errors go to logs. Results go to `brahm_knowledge.db`. These never mix.

---

## 8. File Layout

```
/mnt/d/brahm/
├── mcp_server.py                  ← BRAHM MCP entry point
├── .venv/                         ← BRAHM venv (mcp, httpx, python-dotenv)
│
├── agents/
│   ├── shani/
│   │   ├── database/
│   │   │   └── research_workflow.db   ← SHANI owns this. No other agent writes here.
│   │   ├── venv/
│   │   └── api.py                     ← FastAPI port 8000
│   │
│   ├── chitragupta/
│   │   ├── database/
│   │   │   └── brahm_knowledge.db     ← Chitragupta owns this.
│   │   ├── logs/
│   │   │   ├── ganesh.log
│   │   │   ├── vidur.log
│   │   │   ├── vishwakarma.log
│   │   │   └── chitragupta.log
│   │   ├── .env                       ← NOTION_API_KEY, NOTION_PAGE_ID
│   │   ├── .venv/
│   │   └── api_server.py              ← FastAPI port 8003
│   │
│   ├── ganesh/
│   │   ├── .venv/
│   │   └── ganesh_api.py              ← FastAPI port 8001
│   │
│   ├── vidur/
│   │   ├── parsers/
│   │   ├── extractor.py
│   │   ├── auto_detector.py
│   │   └── router.py
│   │
│   └── vishwakarma/
│       ├── vishwakarma/               ← Python package
│       ├── pseudo/                    ← UPF pseudopotential files
│       └── jobs/                      ← Calculation working directories
```

---

## 9. Agent Responsibilities Summary

| Agent | Reads From | Writes To | Never Touches |
|---|---|---|---|
| SHANI | Web APIs, PDFs | research_workflow.db | brahm_knowledge.db |
| GANESH | Chitragupta /context/load | Nothing (Chitragupta pulls from it) | research_workflow.db |
| VIDUR | Local instrument files | Nothing (Chitragupta pulls from it) | Both DBs |
| Vishwakarma | QE binaries + pseudopotentials | Job output files | Both DBs |
| Chitragupta | research_workflow.db + all agent outputs | brahm_knowledge.db + logs | research_workflow.db (read only) |

---

## 10. LLM Stack

BRAHM is the MCP server — it does not contain an LLM. LLMs connect TO BRAHM and call its tools.

| Use | Model | Provider |
|---|---|---|
| Connects to BRAHM MCP (orchestration) | Claude or any MCP-compatible LLM | External |
| S5 knowledge extraction | mistral:7b-instruct | Ollama local |
| GANESH writing (G1–G5) | llama-3.3-70b-versatile | Groq API |
| Supporting tasks | qwen2.5:7b | Ollama local |

---

## 11. Hardware Constraints

- WSL2, Ubuntu, Windows 11
- 16GB RAM
- RTX 2050 4GB GPU
- No OpenAI. Free Claude plan only.

---

## 12. What to Build Next (Priority Order)

1. Create `brahm_knowledge.db` with the 5 tables above
2. Create `logs/` folder with 4 log files
3. Add `/context/load` endpoint to Chitragupta
4. Rewrite GANESH G1 to call `/context/load` instead of querying SQLite directly
5. Implement GANESH G2–G3 (document planning + section writing via Groq)
6. Implement Chitragupta result storage (pull from GANESH, VIDUR, Vishwakarma after each run)
7. Implement `brahm_activity` logging on every agent trigger

---

*This document is the canonical architecture reference for BRAHM development.*
*All implementation decisions should be consistent with the flows defined here.*
*Update this document whenever a major architectural decision changes.*
