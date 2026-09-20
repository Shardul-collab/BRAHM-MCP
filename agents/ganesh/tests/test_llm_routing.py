"""Hybrid routing (2026-09-11 pm): 'groq:<id>' goes to Groq, everything else stays local."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ganesh.llm_client as L  # noqa: E402


def test_groq_prefix_routes_to_groq_and_plain_names_stay_local(monkeypatch):
    seen = []
    monkeypatch.setattr(L, "_call_groq", lambda p, mt, model, temp: seen.append(("groq", model)) or "g")
    monkeypatch.setattr(L, "_call_ollama", lambda p, mt, model, temp: seen.append(("ollama", model)) or "o")
    monkeypatch.setattr(L, "GANESH_LLM", "ollama")
    assert L.call_llm("x", model="groq:qwen/qwen3.8-27b") == "g"
    assert L.call_llm("x", model="qwen2.5:7b") == "o"
    assert seen == [("groq", "qwen/qwen3.8-27b"), ("ollama", "qwen2.5:7b")]


def test_rate_limit_is_retried(monkeypatch):
    calls = []
    def flaky(p, mt, model, temp):
        calls.append(1)
        if len(calls) < 3:
            raise L.LLMRateLimited("Groq 429 retry-after=0: slow down")
        return "ok"
    monkeypatch.setattr(L, "_call_groq", flaky)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    assert L.call_llm("x", model="groq:m") == "ok" and len(calls) == 3


def test_output_throttle_waits_when_the_minute_budget_is_spent(monkeypatch):
    slept = []
    monkeypatch.setattr(L.time, "sleep", lambda s: slept.append(s))
    clock = [1000.0]
    monkeypatch.setattr(L.time, "time", lambda: clock[0])
    L._OTPM_WINDOW.clear()
    monkeypatch.setattr(L, "_OTPM", 1000)
    L._throttle_output(400); L._throttle_output(400)
    assert slept == []                       # 800 of 1000 in this minute: no wait
    def advance(s):
        slept.append(s); clock[0] += s
    monkeypatch.setattr(L.time, "sleep", advance)
    L._throttle_output(400)                  # would be 1200: must wait out the window
    assert slept and abs(sum(slept) - 60.5) < 1
