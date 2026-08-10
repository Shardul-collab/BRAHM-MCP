"""
brahm/coordinator/planning.py
===============================
Layer 2 — conversational plan generation.

Flow: prompt -> state-check (real registry dispatches) -> NIM LLM call
-> validated plan object -> PlanStore (pending_approval).

Execution of an APPROVED plan reuses registry.dispatch exactly like
orchestration.py does for /v2/run — no new dispatch logic, only new
sequencing. No SHANI or GANESH orchestrator file is read from or
modified.

SAFETY: ALLOWED_ACTIONS is an explicit whitelist. A plan step whose
tool_name is not in this set is rejected before it is ever shown to the
researcher, let alone executed. This blocks destructive/direct-mutation
tools (shani_clear_database, db_bulk_fix, db_update_*, shani_reset_workflow)
from ever being LLM-selectable, even behind human approval.

PARAM SCHEMA FIX (this revision): the first live test showed the model
guessing plausible-but-unverified param names (e.g. "material"/"process"
for analysis_find_gaps) because earlier system prompts only listed tool
NAMES, not their actual inputSchema. This revision pulls each allowed
tool's real inputSchema straight from the registry and injects it into
the system prompt, then does a required-field presence check on every
proposed step before it's ever stored. This is NOT full JSON Schema
validation (no type/enum checking) — just a "did the model forget a
required key" guard. Upgrade to jsonschema.validate() later if stronger
guarantees are needed.

KNOWN GAP (flagged, not solved here): /v2/run's RunRequest has no
project_id field, so no workflow created through this coordinator is
currently linked to a Chitragupta project. State-check below therefore
falls back to raw shani_get_all_status + ganesh_list_documents,
keyword-matched against the prompt client-side, rather than querying
Chitragupta for prior work on a material. Revisit once /v2/run threads
project_id through.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import httpx

from brahm.brahm_registry import registry

log = logging.getLogger("coordinator.planning")

NIM_URL   = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL = "openai/gpt-oss-120b"

# --- Safety whitelist - the ONLY tools an LLM-generated plan may reference ---
# Deliberately excludes: shani_clear_database, db_bulk_fix, db_update_paper,
# db_update_workflow_config, shani_reset_workflow, and anything else that
# mutates/destroys existing data rather than creating new research output.
ALLOWED_ACTIONS = {
    "shani_create_workflow",
    "shani_run_workflow",
    "shani_run_knowledge_extraction",
    "analysis_find_gaps",
    "analysis_trend_report",
    "analysis_parameter_distribution",
    "ganesh_write_review",
    "ganesh_synthesize",
    "vishwakarma_run_scf",
    "vishwakarma_run_relax",
    "vishwakarma_run_bands",
    "vishwakarma_run_dos",
    "vidur_classify",
}


def _get_allowed_tool_schemas() -> dict:
    """Pulls real inputSchema for every whitelisted tool straight from
    the registry — this is the ground truth the LLM should be shown,
    not just bare tool names."""
    schemas = {}
    for tool in registry.all_tools():
        if tool.name in ALLOWED_ACTIONS:
            schemas[tool.name] = {
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            }
    missing = ALLOWED_ACTIONS - schemas.keys()
    if missing:
        log.warning("ALLOWED_ACTIONS references tools not found in registry: %s", missing)
    return schemas


async def _check_current_state(prompt: str) -> dict:
    """Cheap, real-tool state check using only tools confirmed to exist
    in the registry (shani_get_all_status, ganesh_list_documents)."""
    workflows = await registry.dispatch("shani_get_all_status", {})
    documents = await registry.dispatch("ganesh_list_documents", {"status_filter": "all"})

    keywords = [w.lower() for w in prompt.split() if len(w) > 2]

    def _matches(text: str) -> bool:
        text = (text or "").lower()
        return any(kw in text for kw in keywords)

    matching_workflows = [
        w for w in workflows.get("workflows", [])
        if _matches(w.get("name", ""))
    ]
    matching_documents = [
        d for d in documents.get("documents", [])
        if _matches(d.get("title", ""))
    ]

    return {
        "matching_workflows": matching_workflows,
        "matching_documents": matching_documents,
    }


def _build_system_prompt(state: dict, schemas: dict) -> str:
    return (
        "You are BRAHM's research planning assistant. Given a researcher's "
        "natural-language intent, propose a plan using ONLY these tools. "
        "Each tool's real parameter schema is given below — use the exact "
        "property names from inputSchema.properties, and include every "
        "field listed in inputSchema.required:\n\n"
        f"{json.dumps(schemas, indent=2)}\n\n"
        "Current system state (already-existing work - avoid redundant steps):\n"
        f"{json.dumps(state, indent=2)}\n\n"
        "CHAINING STEPS TOGETHER: a step that creates something (e.g. "
        "shani_create_workflow) will only be known to have succeeded, and its "
        "real ID assigned, once it actually runs. So a LATER step that needs "
        "that ID (e.g. shani_run_workflow's workflow_id, or "
        "shani_run_knowledge_extraction's workflow_ids) must NEVER contain a "
        "guessed or placeholder number or string. Instead, use the exact "
        'placeholder "$workflow_id" (a string starting with "$", matching the '
        "output key name of the earlier step) wherever that value is needed — "
        "including inside a list, e.g. \"workflow_ids\": [\"$workflow_id\"]. "
        "Do NOT guess a number, do NOT write descriptive placeholder text like "
        '"<the id from step 1>", and do NOT invent a different key name — use '
        'exactly "$workflow_id" (or "$document_id" if a step\'s result includes '
        "that key instead). Example of a correctly chained two-step plan:\n"
        "{\n"
        '  "steps": [\n'
        "    {\n"
        '      "order": 1,\n'
        '      "tool_name": "shani_create_workflow",\n'
        '      "reason": "Create a new workflow for this material/focus.",\n'
        '      "params": {"name": "...", "material": "...", "focus": "..."},\n'
        '      "skip": false\n'
        "    },\n"
        "    {\n"
        '      "order": 2,\n'
        '      "tool_name": "shani_run_workflow",\n'
        '      "reason": "Run the workflow just created above.",\n'
        '      "params": {"workflow_id": "$workflow_id", "stop_after_stage": "S5"},\n'
        '      "skip": false\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Respond with ONLY a JSON object of this exact shape:\n"
        "{\n"
        '  "steps": [\n'
        "    {\n"
        '      "order": 1,\n'
        '      "tool_name": "<one of the allowed tools above>",\n'
        '      "reason": "<why this step, referencing current state if relevant>",\n'
        '      "params": {"<exact property names from that tool\'s inputSchema>": "..."},\n'
        '      "skip": false\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "If existing state already satisfies part of the request, include that "
        'step with "skip": true and explain why in "reason". Do not invent tool '
        "names or parameter names outside what is shown above."
    )


def _validate_step_params(step: dict, schemas: dict) -> None:
    """Required-field presence check only — not full JSON Schema
    validation. Raises ValueError on a missing required key so the
    caller can surface a clear 502 rather than storing a plan whose
    params will fail at dispatch time."""
    tool_name = step.get("tool_name")
    schema = schemas.get(tool_name, {}).get("inputSchema", {})
    required = schema.get("required", [])
    params = step.get("params", {})

    missing = [r for r in required if r not in params]
    if missing:
        raise ValueError(
            f"Step for '{tool_name}' is missing required params {missing} "
            f"(schema requires: {required}, got: {list(params.keys())})"
        )


async def generate_plan(prompt: str, plan_store) -> dict:
    state = await _check_current_state(prompt)
    schemas = _get_allowed_tool_schemas()

    api_key = os.environ["NVIDIA_API_KEY"]
    payload = {
        "model": NIM_MODEL,
        "messages": [
            {"role": "system", "content": _build_system_prompt(state, schemas)},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 3000,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=170.0) as client:
        # Raised 60 -> 110 -> 170 over 2026-07-23/24. Root cause confirmed via
        # direct minimal repro (bare "Say OK", max_tokens=20): NIM itself took
        # 57.9s even for a trivial call, independent of our prompt size or the
        # chaining-instructions addition. This is NIM-side latency, not a bug
        # in this code. Accepting slow responses for now per explicit decision
        # rather than adding retry logic. Must stay BELOW the Vite proxy's
        # timeout/proxyTimeout in vite.config.ts, which needs to move to 180s+
        # alongside this change or the proxy will cut off first.
        resp = await client.post(
            NIM_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    if content is None:
        raise ValueError("NIM returned null content - likely truncated by max_tokens, raise it")

    parsed = json.loads(content)  # response_format=json_object guarantees valid JSON here
    steps = parsed.get("steps", [])

    # --- Validate every step: whitelist membership, THEN required params ---
    for step in steps:
        if step.get("tool_name") not in ALLOWED_ACTIONS:
            raise ValueError(f"Model proposed disallowed tool: {step.get('tool_name')}")
        if not step.get("skip"):
            _validate_step_params(step, schemas)

    plan_id = plan_store.create(prompt=prompt, state_check=state, steps=steps)
    return plan_store.get(plan_id)


async def execute_plan(plan_id: str, plan_store, run_store) -> None:
    """Executes only non-skip steps, in order. Threads simple outputs
    (workflow_id, document_id) forward via a "$key" placeholder convention
    in step params. Also mirrors workflow_id/document_id into run_store
    (record-only -- no background watcher is spawned, since the plan's own
    steps are the orchestration; spawning run_shani_then_ganesh here would
    race an explicit ganesh_write_review/synthesize step later in the same
    plan). This lets RunProgress.tsx / the existing /v2/ws/{run_id} feed
    work unmodified against plan-originated runs.

    KNOWN LIMITATION: document_type is hardcoded to "literature_review" in
    the run_store.create() call below, since a Plan has no document_type
    concept today (only RunRequest does). Cosmetically wrong if a plan's
    ganesh_write_review step uses a different document_type -- functionally
    inert today since neither RunProgress.tsx nor the WS handler read
    document_type off the run record, only workflow_id/document_id/phase."""
    plan = plan_store.get(plan_id)
    if plan is None:
        return

    plan_store.set_status(plan_id, "executing")
    context: dict = {}
    run_id: Optional[str] = None

    SHANI_TOOLS = {"shani_create_workflow", "shani_run_workflow", "shani_run_knowledge_extraction"}
    GANESH_TOOLS = {"ganesh_write_review", "ganesh_synthesize"}

    for step in sorted(plan["steps"], key=lambda s: s["order"]):
        if step.get("skip"):
            continue

        def _substitute(value):
            """Recursively resolves "$key" placeholder strings against context,
            including inside lists (e.g. workflow_ids: ["$workflow_id"]) and dicts.
            Originally only checked top-level string params -- a list containing
            a "$workflow_id" string passed through unresolved, since
            isinstance(v, str) is False for the list itself. Fixed here rather
            than only in the frontend, since no amount of step-editor UI can work
            around a substitution bug that only fires on exact top-level string
            match."""
            if isinstance(value, str) and value.startswith("$"):
                return context.get(value[1:])
            if isinstance(value, list):
                return [_substitute(v) for v in value]
            if isinstance(value, dict):
                return {k: _substitute(v) for k, v in value.items()}
            return value

        params = {k: _substitute(v) for k, v in step.get("params", {}).items()}

        result = await registry.dispatch(step["tool_name"], params)
        if result.get("status") == "error":
            plan_store.set_error(plan_id, f"{step['tool_name']} failed: {result.get('error')}")
            if run_id is not None:
                run_store.set_error(run_id, f"{step['tool_name']} failed: {result.get('error')}")
            return

        if "workflow_id" in result:
            context["workflow_id"] = result["workflow_id"]
            if run_id is None:
                run_id = run_store.create(
                    workflow_id=result["workflow_id"],
                    document_type="literature_review",
                    auto_write=False,
                )
                plan_store.set_run_id(plan_id, run_id)

        if "document_id" in result:
            context["document_id"] = result["document_id"]
            if run_id is not None:
                run_store.set_document_id(run_id, result["document_id"])

        if run_id is not None:
            if step["tool_name"] in SHANI_TOOLS:
                run_store.set_phase(run_id, "shani_running")
            elif step["tool_name"] in GANESH_TOOLS:
                run_store.set_phase(run_id, "ganesh_writing")

    plan_store.set_status(plan_id, "complete")
    if run_id is not None:
        run_store.set_phase(run_id, "complete")
