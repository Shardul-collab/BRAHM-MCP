"""A save that never happened must say so.

Both VIDUR stores held 0 rows on 2026-09-19 and nothing in the tool result said
why: `_chit_save_instrument` swallowed every failure and returned None, and the
auto-save is gated on a `project_id` that no caller passes -- the same dead path
as Vishwakarma's `_chit_save_dft` (2026-09-09). `saved_result_id: None` could not
be told apart from "not attempted".

Needs the `mcp` package (the registry decorator), which lives in brahm/.venv, not
in the shani venv the rest of these tests run under -- so it skips there.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
vidur_mod = pytest.importorskip(
    "brahm.agents.vidur",
    reason="needs the mcp package; run with brahm/.venv/bin/python",
)


def _classify(**kw):
    return asyncio.run(vidur_mod.vidur_classify(kw))


def test_missing_project_id_is_reported_not_silent(tmp_path):
    f = tmp_path / "scan.xy"
    f.write_text("# 2Theta Intensity\n" + "\n".join(
        f"{10 + 0.5 * i} {100 + i}" for i in range(60)))
    r = _classify(file_path=str(f))
    assert r["status"] == "success"
    assert r["technique"] == "XRD"
    assert r["persisted"] is False
    assert "project_id" in r["persist_error"]


def test_result_always_carries_a_persisted_flag(tmp_path):
    f = tmp_path / "scan.xy"
    f.write_text("# 2Theta Intensity\n" + "\n".join(
        f"{10 + 0.5 * i} {100 + i}" for i in range(60)))
    r = _classify(file_path=str(f))
    assert "persisted" in r and "persist_error" in r


def test_missing_file_is_an_error(tmp_path):
    r = _classify(file_path=str(tmp_path / "nope.xy"))
    assert r["status"] == "error"


def test_classify_does_not_dump_the_whole_scan(tmp_path):
    """Measured through the live connector 2026-09-19: one 1,800-point .xrdml
    came back as 52.9 KB of JSON. VIDUR's caller is a model, so the default
    response has to be a summary."""
    import json
    f = tmp_path / "scan.xy"
    f.write_text("# 2Theta Intensity\n" + "\n".join(
        f"{10 + 0.01 * i} {100 + i}" for i in range(3000)))
    r = _classify(file_path=str(f))
    assert len(json.dumps(r)) < 4000
    assert r["summary"]["points"] == 3000
    assert r["parsed_data"].get("axis") is None
    full = _classify(file_path=str(f), include_data=True)
    assert len(full["parsed_data"]["axis"]) == 3000


def test_health_checks_every_registered_parser(tmp_path):
    """The health list was hardcoded and omitted `sourcemeter` on the day it
    was added, so it reported everything healthy without checking it."""
    import asyncio
    from router import _get_parser_map
    h = asyncio.run(vidur_mod.vidur_health({}))
    for mod in _get_parser_map().values():
        assert f"parser:{mod.__name__.split('.')[-1]}" in h["modules"]
