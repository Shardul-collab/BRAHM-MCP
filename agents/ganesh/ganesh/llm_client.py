"""
ganesh/llm_client.py
=====================
LLM client for GANESH — supports Groq (cloud) and Ollama (local).

Priority:
  1. Groq (fast, good quality) — used for G2, G4, G5 and section writing
  2. Ollama (local fallback) — used if Groq fails or key not set

Config via environment variables:
  GROQ_API_KEY      — Groq API key (required for Groq)
  GROQ_MODEL        — default: llama-3.1-70b-versatile
  OLLAMA_BASE_URL   — default: http://localhost:11434
  OLLAMA_MODEL      — default: qwen2.5:7b-instruct-q3_K_M
  GANESH_LLM        — 'groq' | 'ollama' | 'auto' (default: auto)
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
GANESH_LLM   = os.environ.get("GANESH_LLM", "auto")


class LLMError(Exception):
    pass


def _call_groq(prompt: str, max_tokens: int = 4096) -> str:
    import requests
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise LLMError("GROQ_API_KEY not set")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model":       GROQ_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  max_tokens,
            "temperature": 0.3,
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise LLMError(f"Groq API error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _call_ollama(prompt: str, max_tokens: int = 4096) -> str:
    import requests
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3, "num_ctx": 8192},
        },
        timeout=600,
    )
    if r.status_code != 200:
        raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
    return r.json().get("response", "")


def call_llm(prompt: str, max_tokens: int = 4096, prefer: str = "auto", _retry: int = 0) -> str:
    """
    Call LLM with automatic fallback.
    prefer: 'groq' | 'ollama' | 'auto'
    """
    backend = prefer if prefer != "auto" else GANESH_LLM

    import time, re
    if backend == "groq":
        try:
            return _call_groq(prompt, max_tokens)
        except LLMError as e:
            m = re.search(r"try again in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 1 if m else 15
            if _retry < 5:
                log.warning("Groq 429 — waiting %.1fs (retry %d)", wait, _retry+1)
                time.sleep(wait)
                return call_llm(prompt, max_tokens, prefer, _retry+1)
            raise
    if backend == "ollama":
        return _call_ollama(prompt, max_tokens)

    # auto: try groq first with retry, fall back to ollama
    try:
        return _call_groq(prompt, max_tokens)
    except LLMError as e:
        m = re.search(r"try again in ([\d.]+)s", str(e))
        wait = float(m.group(1)) + 1 if m else 15
        if _retry < 5:
            log.warning("Groq 429 — waiting %.1fs (retry %d)", wait, _retry+1)
            time.sleep(wait)
            return call_llm(prompt, max_tokens, prefer, _retry+1)
        log.warning("Groq failed (%s), falling back to Ollama", e)
        return _call_ollama(prompt, max_tokens)


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
