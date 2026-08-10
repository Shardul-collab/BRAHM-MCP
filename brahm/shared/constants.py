"""
brahm/shared/constants.py
==========================
Single source of truth for all BRAHM configuration.
"""

from pathlib import Path
import os

BRAHM_ROOT       = Path(os.environ.get("BRAHM_ROOT", "/mnt/d/brahm"))

SHANI_ROOT       = BRAHM_ROOT / "agents/shani"
CHITRAGUPTA_ROOT = BRAHM_ROOT / "agents/chitragupta"
VIDUR_ROOT       = BRAHM_ROOT / "agents/vidur"
VISHWAKARMA_ROOT = BRAHM_ROOT / "agents/vishwakarma"
GANESH_ROOT      = BRAHM_ROOT / "agents/ganesh"

SHANI_VENV_PY  = str(SHANI_ROOT       / "venv/bin/python")
CHIT_VENV_PY   = str(CHITRAGUPTA_ROOT / ".venv/bin/python")
GANESH_VENV_PY = str(GANESH_ROOT      / ".venv/bin/python")

DB_PATH        = str(SHANI_ROOT / "database/research_workflow.db")
AUDIT_LOG_PATH = str(SHANI_ROOT / "database/mcp_corrections.jsonl")
QUEUE_PATH     = str(SHANI_ROOT / "workflow_queue.json")

QE_WORKDIR = os.environ.get("QE_WORKDIR", "/tmp/brahm_qe_jobs")
QE_BIN_DIR = os.environ.get("QE_BIN_DIR", "/mnt/d/miniforge3/bin")
QE_PSEUDO  = os.environ.get("QE_PSEUDO", str(VISHWAKARMA_ROOT / "pseudo"))

ENV_FILE = str(CHITRAGUPTA_ROOT / ".env")

SHANI_BASE  = "http://localhost:8000"
GANESH_BASE = "http://localhost:8001"
CHITRAGUPTA_BASE = "http://localhost:8003"

RATE_LIMIT_SEC = 0.38

SHANI_STAGE_SEQUENCE = (
    "S1", "S2", "S2_75", "S2_5",
    "S3", "S4", "S5", "S5_5",
)
SHANI_VALID_STAGES = frozenset(SHANI_STAGE_SEQUENCE)

GANESH_STAGE_SEQUENCE = ("G1", "G2", "G3", "G4", "G5")
GANESH_VALID_STAGES   = frozenset(GANESH_STAGE_SEQUENCE)

PAPER_WRITABLE_FIELDS = frozenset({
    "title", "doi", "abstract", "pdf_url", "pdf_status",
    "status", "created_at", "failed_candidates", "last_error",
})
PAPER_IMMUTABLE_FIELDS = frozenset({
    "id", "workflow_id", "source", "raw_text", "file_path",
})
CONFIG_WRITABLE_FIELDS = frozenset({
    "material", "focus", "structure", "method",
    "properties", "characterization", "domain",
})

AGENTS = {
    "SHANI": {
        "description": "Literature acquisition pipeline (S1-S5_5)",
        "api_base":    SHANI_BASE,
        "tool_group":  "shani",
        "type":        "http_api",
    },
    "Chitragupta": {
        "description": "Knowledge management + Notion export",
        "api_base":    None,
        "tool_group":  "chitragupta",
        "type":        "local_import",
    },
    "VIDUR": {
        "description": "Characterization instrument file classifier",
        "api_base":    None,
        "tool_group":  "vidur",
        "type":        "local_import",
    },
    "Vishwakarma": {
        "description": "Quantum ESPRESSO DFT calculation agent",
        "api_base":    None,
        "tool_group":  "vishwakarma",
        "type":        "local_import",
    },
    "CHITRAGUPTA_DB": {
        "description": "Central data API — projects, papers, results, documents",
        "api_base":    CHITRAGUPTA_BASE,
        "tool_group":  "chitragupta",
        "type":        "http_api",
    },
    "GANESH": {
        "description": "Scientific writing + synthesis agent (G1-G5)",
        "api_base":    GANESH_BASE,
        "tool_group":  "ganesh",
        "type":        "http_api",
    },
}
