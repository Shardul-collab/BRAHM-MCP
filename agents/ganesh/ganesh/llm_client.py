"""
ganesh/llm_client.py
=====================
LLM client for GANESH — supports Groq (cloud) and Ollama (local).

Default is LOCAL (Ollama) - confidentiality requirement, 2026-09-11.
Groq only when GANESH_LLM=groq is set explicitly; no silent cloud fallback.

Config via environment variables:
  GANESH_LLM            - 'ollama' (default) | 'groq'
  OLLAMA_BASE_URL       - default: http://localhost:11434
  OLLAMA_MODEL          - default model for any role without its own setting
  GANESH_WRITER_MODEL / GANESH_POLISH_MODEL / GANESH_CRITIC_MODEL - per role
  GROQ_API_KEY, GROQ_MODEL - only read when GANESH_LLM=groq
"""

from __future__ import annotations

import json
import logging
import os
import time

log = logging.getLogger("ganesh.llm")

GROQ_MODEL   = os.environ.get("GROQ_MODEL",   "llama-3.1-70b-versatile")
OLLAMA_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q3_K_M")
# 2026-09-11: LOCAL BY DEFAULT. GANESH handles unpublished research context
# and Shardul requires that nothing leave the machine. "auto" used to mean
# "Groq first, Ollama on failure"; it now means Ollama. Groq is used only when
# GANESH_LLM=groq is set explicitly. (Measured that day: the Groq key GANESH
# loads was invalid and its model no longer existed on Groq, so every call
# slept through five 15 s "rate-limit" retries before falling back.)
GANESH_LLM   = os.environ.get("GANESH_LLM", "ollama")

# Per-role local models (writer / polish / critic). Unset roles use OLLAMA_MODEL.
ROLE_MODELS = {
    "writer": os.environ.get("GANESH_WRITER_MODEL"),
    "polish": os.environ.get("GANESH_POLISH_MODEL"),
    "critic": os.environ.get("GANESH_CRITIC_MODEL"),
    "repair": os.environ.get("GANESH_REPAIR_MODEL"),
}
GROQ_PREFIX = "groq:"
# free-tier output-tokens-per-minute ceiling; a single call may not ask for more
GROQ_MAX_OUTPUT = int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "1000"))


_OTPM = int(os.environ.get("GROQ_OUTPUT_TOKENS_PER_MINUTE", "1000"))
_OTPM_WINDOW: list = []      # (timestamp, requested max_tokens) over the last 60 s


def _throttle_output(requested: int) -> None:
    """Keep the requested output tokens of the last 60 s under the free-tier OTPM budget."""
    while True:
        now = time.time()
        _OTPM_WINDOW[:] = [(t, n) for t, n in _OTPM_WINDOW if now - t < 60]
        if sum(n for _, n in _OTPM_WINDOW) + requested <= _OTPM or not _OTPM_WINDOW:
            _OTPM_WINDOW.append((now, requested))
            return
        wait = 60 - (now - _OTPM_WINDOW[0][0]) + 0.5
        log.info("Groq OTPM pacing: waiting %.1fs", wait)
        time.sleep(max(0.1, wait))


def max_output_for(model: str) -> int:
    """Largest max_tokens one call may request for this model."""
    return GROQ_MAX_OUTPUT if (model or "").startswith(GROQ_PREFIX) else 100_000

# Provenance: the backend/model that served the most recent call.
LAST_CALL: dict = {}


class LLMError(Exception):
    pass


class LLMRateLimited(LLMError):
    pass


def model_for(role: str | None = None, model: str | None = None) -> str:
    return model or (ROLE_MODELS.get(role) if role else None) or OLLAMA_MODEL


def ensure_local_models(models) -> None:
    """Fail once, loudly, if a configured model is not installed."""
    import requests
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
    except Exception as e:
        raise LLMError(f"Cannot reach Ollama at {OLLAMA_URL}: {e}")
    have = {t.get("name") for t in tags} | {t.get("model") for t in tags}
    if any(m and m.startswith(GROQ_PREFIX) for m in models) and not _groq_key():
        raise LLMError("a groq: model is configured but no GROQ_API_KEY was found")
    missing = [m for m in models if m and not m.startswith(GROQ_PREFIX) and m not in have and f"{m}:latest" not in have]
    if missing:
        raise LLMError(f"Ollama models not installed: {missing}. Fix with: ollama pull <model>")


def _groq_key() -> str:
    """GROQ_API_KEY from the environment, else from agents/shani/.env (the repo's working key,
    checked 2026-09-11; chitragupta/.env's is empty). Never logged."""
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    import re
    from pathlib import Path
    env = Path(__file__).resolve().parents[2] / "shani" / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^GROQ_API_KEY=(.*)$", line.strip())
            if m:
                return m.group(1).strip().strip('"').strip("'")
    return ""


def _call_groq(prompt: str, max_tokens: int = 4096, model: str | None = None,
               temperature: float = 0.3) -> str:
    import requests
    api_key = _groq_key()
    if not api_key:
        raise LLMError("GROQ_API_KEY not set")
    model = model or GROQ_MODEL
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(max_tokens, GROQ_MAX_OUTPUT), "temperature": temperature}
    if "gpt-oss" in model:
        # reasoning model: keep the hidden reasoning short so it does not eat the token budget
        body.update(reasoning_effort="low", include_reasoning=False)
    elif "qwen3" in model:
        # qwen3.6 otherwise writes a visible <think> block, and hiding the reasoning still spends the
        # token budget on it (measured 2026-09-12: 902 chars of reasoning, empty content)
        body.update(reasoning_effort="none")
    _throttle_output(body["max_tokens"])
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                      json=body, timeout=120)
    if r.status_code == 429 and "too large" in r.text:
        # a limit on the request itself, not a transient rate limit: retrying changes nothing
        raise LLMError(f"Groq rejected the request size ({model}): {r.text[:200]}")
    if r.status_code == 429 or r.status_code >= 500:
        wait = r.headers.get("retry-after")
        raise LLMRateLimited(f"Groq {r.status_code} retry-after={wait}: {r.text[:300]}")
    if r.status_code != 200:
        raise LLMError(f"Groq API error {r.status_code} ({model}): {r.text[:300]}")
    d = r.json()
    ch = d["choices"][0]
    usage = d.get("usage") or {}
    LAST_CALL.update(backend="groq", model=model, done_reason=ch.get("finish_reason"),
                     eval_count=usage.get("completion_tokens"), prompt_eval_count=usage.get("prompt_tokens"),
                     seconds=round(float(usage.get("total_time") or 0), 1))
    return ch["message"].get("content") or ""


def _groq_with_retry(prompt, max_tokens, model, temperature, tries: int = 14):
    import re as _re
    for attempt in range(tries):
        try:
            return _call_groq(prompt, max_tokens, model, temperature)
        except LLMRateLimited as e:
            m = _re.search(r"retry-after=([\d.]+)", str(e)) or _re.search(r"try again in ([\d.]+)s", str(e))
            # the limiter is per-minute, so a suggested wait of several minutes is pointless to
            # honour in full: the budget refills within 60 s (measured 2026-09-12). Retry sooner,
            # more often, and let _throttle_output do the real pacing.
            wait = min(65.0, float(m.group(1)) + 1) if m else 20.0
            if attempt == tries - 1:
                raise
            log.warning("Groq %s - waiting %.1fs (retry %d)", str(e)[:40], wait, attempt + 1)
            time.sleep(wait)


def _call_ollama(prompt: str, max_tokens: int = 4096, model: str | None = None,
                 temperature: float = 0.3, num_ctx: int = 8192) -> str:
    import requests
    model = model or OLLAMA_MODEL
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model":  model,
            "prompt": prompt,
            "stream": False,
            # think:false - hybrid-reasoning models (qwen3.x) would otherwise
            # spend the token budget on hidden reasoning; ignored by others.
            "think":  False,
            "options": {"num_predict": max_tokens, "temperature": temperature, "num_ctx": num_ctx},
        },
        timeout=1800,
    )
    if r.status_code != 200:
        raise LLMError(f"Ollama error {r.status_code} ({model}): {r.text[:300]}")
    d = r.json()
    LAST_CALL.update(backend="ollama", model=model, eval_count=d.get("eval_count"),
                     prompt_eval_count=d.get("prompt_eval_count"),
                     seconds=round((d.get("total_duration") or 0) / 1e9, 1),
                     done_reason=d.get("done_reason"))
    return d.get("response", "")


def call_llm(prompt: str, max_tokens: int = 4096, prefer: str = "auto", _retry: int = 0,
             model: str | None = None, role: str | None = None, temperature: float = 0.3) -> str:
    """
    prefer: 'groq' | 'ollama' | 'auto' ('auto' = GANESH_LLM, which defaults to ollama).
    Groq is retried only on HTTP 429, and never falls back silently.
    """
    resolved = model_for(role, model)
    if resolved.startswith(GROQ_PREFIX):
        return _groq_with_retry(prompt, max_tokens, resolved[len(GROQ_PREFIX):], temperature)
    backend = prefer if prefer != "auto" else GANESH_LLM
    if backend == "groq":
        import re as _re
        try:
            return _call_groq(prompt, max_tokens)
        except LLMRateLimited as e:
            m = _re.search(r"try again in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 1 if m else 15
            if _retry < 5:
                log.warning("Groq 429 - waiting %.1fs (retry %d)", wait, _retry + 1)
                time.sleep(wait)
                return call_llm(prompt, max_tokens, prefer, _retry + 1, model, role, temperature)
            raise
    return _call_ollama(prompt, max_tokens, model_for(role, model), temperature)


def call_llm_json(prompt: str, max_tokens: int = 2048, prefer: str = "auto") -> dict:
    """
    Call LLM expecting a JSON response. Strips markdown fences.
    Retries once with a correction prompt if JSON parse fails.
    """
    raw = call_llm(prompt, max_tokens, prefer)

    def _parse(text: str) -> dict | None:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    result = _parse(raw)
    if result is not None:
        return result

    # Correction attempt
    correction = (
        "Your previous response was not valid JSON.\n"
        "Return ONLY a valid JSON object — no preamble, no markdown fences.\n\n"
        f"Original response:\n{raw}"
    )
    raw2 = call_llm(correction, max_tokens, prefer)
    result = _parse(raw2)
    if result is None:
        raise LLMError(f"LLM returned invalid JSON after correction.\nRaw: {raw[:500]}")
    return result
