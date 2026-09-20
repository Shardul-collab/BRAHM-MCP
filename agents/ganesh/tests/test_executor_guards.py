"""Grounded mode must never fall back to the uncited path or approve an empty section (2026-09-11 pm)."""
import os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from section_executor import SectionExecutor  # noqa: E402


class Node:
    section_id, section_name, brief = 1, "Characterization", {"section_name": "Characterization"}


def _executor(packets):
    ex = SectionExecutor(repo=None, document_id=1, context_bundle={"evidence_packets": packets},
                         llm_client=lambda *a, **k: "")
    ex._update_section_status = lambda *a, **k: None
    ex._save_draft = lambda *a, **k: pytest.fail("an empty/unpacketed section must not be saved")
    return ex


def test_missing_packet_raises_instead_of_ungrounded_fallback(monkeypatch):
    monkeypatch.setenv("GANESH_WRITING_MODE", "L4")
    with pytest.raises(RuntimeError, match="no evidence packet"):
        _executor({}).run(Node())


def test_empty_packet_is_not_approved(monkeypatch):
    monkeypatch.setenv("GANESH_WRITING_MODE", "L4")
    with pytest.raises(RuntimeError, match="produced no text"):
        _executor({"Characterization": []}).run(Node())


def test_mode_map_overrides_the_default_mode(monkeypatch):
    # Shardul's blind read: L5 for Synthesis Methods, L3 elsewhere (2026-09-12)
    monkeypatch.setenv("GANESH_WRITING_MODE", "L3")
    monkeypatch.setenv("GANESH_WRITING_MODE_MAP", '{"Characterization": "L5"}')
    seen = []
    ex = _executor({"Characterization": [1]})
    ex._run_grounded = lambda section, mode, packet: seen.append(mode)
    ex.run(Node())
    monkeypatch.setenv("GANESH_WRITING_MODE_MAP", "{}")
    ex2 = _executor({"Characterization": [1]})
    ex2._run_grounded = lambda section, mode, packet: seen.append(mode)
    ex2.run(Node())
    assert seen == ["L5", "L3"]
