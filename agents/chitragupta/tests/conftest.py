"""
Shared fixtures for the Chitragupta test suite.

Every test runs against a throwaway brahm.db in tmp_path. brahm_db.schema
resolves BRAHM_DB_PATH at import time, so the fixture repoints the module
attribute and the repositories pick it up through get_connection().
"""

import sys
from pathlib import Path

import pytest

_AGENT_ROOT = Path(__file__).resolve().parents[1]     # agents/chitragupta
_REPO_ROOT = Path(__file__).resolve().parents[3]      # repo root
for _p in (_AGENT_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture()
def scratch_db(tmp_path, monkeypatch):
    """A fresh, schema-initialised brahm.db isolated to one test."""
    import brahm_db.schema as schema

    db = tmp_path / "brahm.db"
    monkeypatch.setattr(schema, "BRAHM_DB_PATH", db)
    schema.init_db()
    return db
