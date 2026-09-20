"""The registry is the only place a tool's declared schema is enforced.

mcp_server.py dispatches straight through (`registry.dispatch(name, arguments)`),
so before 2026-09-20 a missing required argument reached the handler and came
back as an unhandled KeyError. Measured across the 64 registered tools, seven
did this: analysis_technique_frequency, analysis_trend_report,
analysis_find_gaps, analysis_parameter_distribution,
research_find_papers_by_topic, shani_get_papers, shani_get_paper_content.

It matters because the caller is a model. A traceback tells it nothing; the name
of the argument it forgot tells it everything.

Run: brahm/.venv/bin/python -m pytest tests/ -q     (needs the mcp package)
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BRAHM_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT))

from brahm.brahm_registry import ToolRegistry  # noqa: E402


def _registry():
    reg = ToolRegistry()

    async def handler(args):
        return {"status": "success", "got": args["needed"]}

    reg.register(
        name="demo_tool", group="test", description="demo",
        input_schema={
            "type": "object",
            "properties": {"needed": {"type": "string",
                                      "description": "the thing it cannot work without"},
                           "optional": {"type": "string"}},
            "required": ["needed"],
        },
        handler=handler,
    )
    return reg


def test_a_missing_required_argument_is_an_error_not_an_exception():
    res = asyncio.run(_registry().dispatch("demo_tool", {}))
    assert res["status"] == "error"
    assert "needed" in res["error"]


def test_the_error_names_the_argument_and_quotes_its_description():
    res = asyncio.run(_registry().dispatch("demo_tool", {"optional": "x"}))
    assert "demo_tool" in res["error"] and "needed" in res["error"]
    assert "cannot work without" in res["detail"]


def test_an_empty_string_counts_as_missing():
    assert asyncio.run(_registry().dispatch("demo_tool", {"needed": ""}))["status"] == "error"


def test_a_valid_call_still_reaches_the_handler():
    res = asyncio.run(_registry().dispatch("demo_tool", {"needed": "value"}))
    assert res == {"status": "success", "got": "value"}


def test_an_unknown_tool_is_an_error():
    assert asyncio.run(_registry().dispatch("nope", {}))["status"] == "error"


def test_no_registered_tool_raises_on_an_empty_argument_dict():
    """The regression guard for all 64: dispatching with {} must never raise,
    whatever the tool does with the arguments it then has."""
    for p in (ROOT/"agents/shani", ROOT/"agents/chitragupta/analysis", ROOT/"agents/vidur",
              ROOT/"agents/vishwakarma", ROOT/"agents/ganesh", ROOT/"agents/chitragupta"):
        sys.path.insert(0, str(p))
    import brahm.agents.analysis, brahm.agents.research  # noqa: F401
    from brahm.brahm_registry import registry

    checked = 0
    for name in ("analysis_technique_frequency", "analysis_trend_report",
                 "analysis_find_gaps", "analysis_parameter_distribution",
                 "research_find_papers_by_topic"):
        if name not in registry._tools:
            continue
        res = asyncio.run(registry.dispatch(name, {}))   # must not raise
        assert res["status"] == "error", name
        checked += 1
    assert checked, "none of the named tools were registered"


def test_every_required_argument_documents_itself():
    """The schema is the documentation: VIDUR's caller, and every other tool's,
    is a model. On 2026-09-20 65 of 83 required arguments across 44 tools had no
    description, which the new dispatch validator surfaced as errors reading
    "category: no description"."""
    for p in (ROOT/"agents/shani", ROOT/"agents/chitragupta/analysis", ROOT/"agents/vidur",
              ROOT/"agents/vishwakarma", ROOT/"agents/ganesh", ROOT/"agents/chitragupta"):
        sys.path.insert(0, str(p))
    import brahm.agents.analysis, brahm.agents.chitragupta, brahm.agents.db_tools  # noqa
    import brahm.agents.ganesh, brahm.agents.meta, brahm.agents.research  # noqa
    import brahm.agents.shani, brahm.agents.vidur, brahm.agents.vishwakarma  # noqa
    from brahm.brahm_registry import registry

    undocumented = []
    for name, tool in registry._tools.items():
        schema = tool.inputSchema or {}
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if not (props.get(key) or {}).get("description"):
                undocumented.append(f"{name}.{key}")
    assert not undocumented, f"required arguments with no description: {undocumented}"


def test_a_destructive_tool_says_so_in_the_argument_that_arms_it():
    """shani_clear_database(confirm=True) and db_bulk_fix are irreversible. The
    argument that arms them has to say that where a caller will read it."""
    from brahm.brahm_registry import registry
    for tool_name, arg in (("shani_clear_database", "confirm"),
                           ("db_bulk_fix", "match_pattern")):
        if tool_name not in registry._tools:
            continue
        props = (registry._tools[tool_name].inputSchema or {}).get("properties") or {}
        desc = (props.get(arg) or {}).get("description", "")
        assert "DESTRUCTIVE" in desc.upper() or "no undo" in desc.lower(), f"{tool_name}.{arg}"
