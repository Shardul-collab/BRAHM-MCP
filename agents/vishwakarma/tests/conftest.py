"""
Shared fixtures for the Vishwakarma test suite.

sys.path handling mirrors what vishwakarma_api.py and mcp_server.py do at
runtime: the `vishwakarma` package sits one level up from this directory and
is imported by bare name, not as part of an installed distribution.
"""

import sys
from pathlib import Path

import pytest

_AGENT_ROOT = Path(__file__).resolve().parents[1]      # agents/vishwakarma
_REPO_ROOT  = Path(__file__).resolve().parents[3]      # repo root, for `brahm`
for _p in (_AGENT_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def real_pw_output() -> str:
    """
    Genuine pw.x SCF output captured from job f67965d4 (si_outdir_link_test_scf,
    a completed silicon SCF). Not hand-written — if a parser change breaks this
    test, the parser stopped reading output QE actually produced.
    """
    return (FIXTURES / "si_scf_real.out").read_text(errors="replace")


@pytest.fixture(scope="session")
def real_dos_output() -> str:
    """Genuine dos.x output captured from job 92a9a899 (si_outdir_link_test_dos)."""
    return (FIXTURES / "si_dos_real.out").read_text(errors="replace")
