# BRAHM
### Bi-directional Research & Analysis Hub for Multimodal Science

A personal project — a self-hosted, multi-agent AI platform for automating materials science research workflows. Built to reduce the time between a scientific idea and a validated result.

---

## What It Does

Modern research is slowed by information overload, fragmented knowledge, repetitive analysis, and documentation overhead. BRAHM automates the mechanical parts of the research process while keeping the researcher in control of every scientific decision.

The pipeline moves from literature to knowledge to insight to communication through a unified, evidence-driven workflow.

---

## Architecture

BRAHM is built as an ecosystem of specialised agents, each responsible for one layer of the research process.

```
Literature Discovery → Knowledge Extraction → Gap Analysis → Simulation → Documentation
      SHANI                  SHANI              SHANI       Vishwakarma      GANESH
                                          Chitragupta
```

### Agents

| Agent | Role |
|---|---|
| **SHANI** | Literature pipeline — discovery, download, content extraction, knowledge extraction |
| **Chitragupta** | Knowledge custodian — context management, research memory, database access |
| **GANESH** | Document synthesis — literature reviews, research reports, manuscript drafts |
| **VIDUR** | Characterisation analysis — XRD, Raman, UV-Vis, SEM/EDS interpretation |
| **Vishwakarma** | Computational engine — structure generation, DFT via Quantum ESPRESSO |

---

## Pipeline Stages

### SHANI — Literature Pipeline

| Stage | Description |
|---|---|
| S1 | Workflow initialisation and query generation |
| S2 | Paper discovery via Semantic Scholar and arXiv |
| S3 | PDF download and resolution |
| S4 | Content extraction with section-aware parsing |
| S4.5 | PaperContent normalisation — canonical section names, noise removal |
| S5 | LLM-driven knowledge extraction into structured records |
| S5.5 | Finding reconstruction — connecting extracted signals into grounded claims |

### GANESH — Document Pipeline

| Stage | Description |
|---|---|
| G1 | Context loading from Chitragupta |
| G2 | Document planning |
| G3 | Section graph construction |
| G4 | Section-by-section generation |
| G5 | Document integration and export |

---

## Tech Stack

- **Runtime:** Python, FastAPI, SQLite, FAISS
- **LLMs:** Groq, Gemini, Cerebras (cloud) + Ollama (local, for sensitive data)
- **Computation:** Quantum ESPRESSO 7.5 for DFT
- **Embeddings:** `all-MiniLM-L6-v2` for vector search
- **Infrastructure:** WSL2 on Windows 11, self-hosted

---

## Design Principles

**Human judgment remains central.** BRAHM assists research. Researchers direct research. Scientific decisions always belong to the researcher.

**Scientific integrity above speed.** Evidence is more important than confidence. Uncertainty is surfaced rather than hidden.

**Privacy by design.** Instrument data and DFT structures never leave the local machine. Only published paper text reaches cloud APIs.

**LLM-agnostic.** Each agent uses the best available model for its task. The orchestration layer is model-independent.

---

## Status

Active development. Core pipeline (S1→S5) is operational. Document generation (GANESH G1→G5) is functional. S5.5 finding reconstruction and full grounded document generation are in progress.

---

## Project Structure

```
brahm/
├── agents/
│   ├── shani/          # Literature pipeline
│   ├── chitragupta/    # Knowledge custodian
│   ├── ganesh/         # Document generation
│   ├── vidur/          # Characterisation analysis
│   └── vishwakarma/    # DFT computation
├── brahm/              # Shared registry and utilities
├── brahm_dashboard.py  # Service health and control UI
└── mcp_server.py       # MCP entry point
```

---

*Built by Shardul Khanduri — MSc Physics, materials science and AI systems.*
