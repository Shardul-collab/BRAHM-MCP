import re
import json
import os
import time as _time
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"),
    override=False,
)


class LLMResponseError(Exception):
    pass


# ============================================================
# DEBUG LOGGING UTILITIES
# ============================================================

def _ensure_dirs():
    os.makedirs("logs/llm_payloads", exist_ok=True)
    os.makedirs("logs/llm_payloads_readable", exist_ok=True)


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _log_payload(prompt, stage=None, extra=None):
    _ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stage = stage or "unknown"
    payload = {"stage": stage, "prompt": prompt, "extra": extra or {}}
    filepath = f"logs/llm_payloads/{stage}_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _log_readable(prompt, stage=None):
    _ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stage = stage or "unknown"
    filepath = f"logs/llm_payloads_readable/{stage}_{timestamp}.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(prompt)


# ============================================================
# OLLAMA CLIENT — REST API
# ============================================================

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")


class OllamaClient:

    def __init__(self, model="mistral:7b-instruct", timeout=None):
        self.model = model
        self.timeout = timeout or 240

    def generate(self, prompt, max_tokens=800, temperature=0.7):
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_API_URL}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama request timed out after {self.timeout} seconds.")

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama API returned status {response.status_code}: {response.text[:300]}"
            )
        try:
            data = response.json()
        except Exception:
            raise RuntimeError(f"Ollama returned non-JSON response: {response.text[:300]}")

        output = data.get("response", "").strip()
        if not output:
            raise RuntimeError("Empty response from Ollama API.")
        return output


# ============================================================
# SHARED JSON-ARRAY PARSER
#
# Used by LLMService._parse_json (S5 + S5_eq) and
# tools/reconstruct_findings.py (S5_5) -- single implementation,
# no duplicated parsing logic across files.
#
# FIX: the previous fallback used a greedy regex r"\[.*\]" that
# matched from the FIRST '[' to the LAST ']' in the raw response.
# Because the S5 schema mandates bracket-qualifier notation inside
# every numeric value (e.g. "550 C [CVD substrate temperature]"),
# any response with extra brackets outside the array (markdown
# fences, trailing notes) caused the regex to span past the
# array's real end and produce an invalid JSON fragment -- silently
# dropping the entire chunk.
#
# This version tracks bracket depth and JSON-string-literal state
# so brackets inside quoted values don't affect boundary detection.
# Returns the first complete top-level JSON array, ignoring any
# commentary the LLM appended before or after it.
# ============================================================

def parse_json_array(raw: str):
    """Parse the first complete top-level JSON array out of `raw`."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = raw[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


# ============================================================
# LLM SERVICE
# ============================================================

class LLMService:

    ALLOWED_CATEGORIES = {
        "material",
        "synthesis_method",
        "characterization",
        "application",
        "computational_method",
        "software",
        "exchange_correlation",
        "calculated_property",
        "defect_type",
        "doping_parameter",
        "annealing_condition",
        "optical_property",
        "electrical_property",
        # synthesis condition sub-categories
        "growth_temperature",
        "chamber_pressure",
        "gas_flow",
        "growth_duration",
        # device parameter sub-categories
        "field_effect_mobility",
        "on_off_ratio",
        "threshold_voltage",
        "subthreshold_swing",
        "contact_resistance",
        "photoresponsivity",
    }

    def __init__(self, llm_client):
        self.llm = llm_client

    def extract(self, prompt, stage="S5"):
        token_estimate = _estimate_tokens(prompt)
        print(f"[LLM DEBUG] Stage={stage} | Tokens≈{token_estimate}")
        if token_estimate > 8000:
            print(f"[WARNING] Large payload detected in {stage}: {token_estimate}")
        _log_payload(prompt, stage=stage, extra={"type": "extract"})
        _log_readable(prompt, stage=stage)
        try:
            raw = self.llm.generate(prompt, temperature=0.0)
        except Exception as e:
            raise LLMResponseError(f"LLM extraction failed: {e}")
        data = self._parse_json(raw)
        if data is None:
            raise LLMResponseError(f"Invalid JSON returned by LLM\nRaw output:\n{raw[:500]}")
        return self._validate(data)

    def generate_text(self, prompt, max_tokens=800, temperature=0.5, stage="S6"):
        token_estimate = _estimate_tokens(prompt)
        print(f"[LLM DEBUG] Stage={stage} | Tokens≈{token_estimate}")
        if token_estimate > 8000:
            print(f"[WARNING] Large payload detected in {stage}: {token_estimate}")
        _log_payload(prompt, stage=stage, extra={"type": "generation", "max_tokens": max_tokens, "temperature": temperature})
        _log_readable(prompt, stage=stage)
        try:
            output = self.llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            return output.strip()
        except Exception as e:
            print(f"[LLM ERROR FULL] {repr(e)}")
            return None

    def _parse_json(self, raw):
        return parse_json_array(raw)

    def _validate(self, data):
        if not isinstance(data, list):
            raise LLMResponseError("LLM output must be a list")
        validated = []
        for item in data:
            if "category" not in item or "value" not in item:
                raise LLMResponseError("Invalid knowledge format")
            category = item["category"]
            value    = item["value"]
            if category not in self.ALLOWED_CATEGORIES:
                continue
            validated.append({
                "category":       category,
                "value":          str(value),
                "section_source": "llm",
            })
        return validated


# ============================================================
# GEMINI CLIENT — Google AI Studio REST API
# Key: GOOGLE_API_KEY in agents/shani/.env
# ============================================================

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


class GeminiClient:

    def __init__(self, model=GEMINI_MODEL, timeout=60):
        self.model   = model
        self.timeout = timeout
        if not GEMINI_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set. Add it to agents/shani/.env")

    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        try:
            response = requests.post(
                GEMINI_API_URL,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Cannot connect to Gemini API. Check network.")
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Gemini request timed out after {self.timeout}s.")

        if response.status_code != 200:
            raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:300]}")

        try:
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, Exception) as exc:
            raise RuntimeError(f"Unexpected Gemini response structure: {exc}\nRaw: {response.text[:300]}")

        if not text:
            raise RuntimeError("Empty response from Gemini API.")
        return text


# ============================================================
# CEREBRAS CLIENT — OpenAI-compatible REST API
# Key: CEREBRAS_API_KEY in agents/shani/.env
# ============================================================

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL   = "gpt-oss-120b"


class CerebrasClient:

    def __init__(self, model=CEREBRAS_MODEL, timeout=60):
        self.model   = model
        self.timeout = timeout
        if not CEREBRAS_API_KEY:
            raise RuntimeError("CEREBRAS_API_KEY is not set. Add it to agents/shani/.env")

    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        try:
            response = requests.post(
                CEREBRAS_API_URL,
                headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Cannot connect to Cerebras API. Check network.")
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Cerebras request timed out after {self.timeout}s.")

        if response.status_code != 200:
            raise RuntimeError(f"Cerebras API error {response.status_code}: {response.text[:300]}")

        try:
            msg  = response.json()["choices"][0]["message"]
            text = (msg.get("content") or msg.get("reasoning") or "").strip()
        except (KeyError, IndexError, Exception) as exc:
            raise RuntimeError(f"Unexpected Cerebras response: {exc}\nRaw: {response.text[:300]}")

        if not text:
            raise RuntimeError("Empty response from Cerebras API.")
        return text


# ============================================================
# GROQ CLIENT — OpenAI-compatible REST API
#
# SHANI S5 extraction only.
# Uses GROQ_API_KEY_2 (dedicated S5 account — separate from
# GANESH which uses GROQ_API_KEY).
#
# On 429: reads Retry-After header (or defaults to 15s),
# sleeps, and retries up to MAX_RETRIES times.
# No key rotation — single dedicated key per agent.
# ============================================================

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_S5 = "llama-3.3-70b-versatile"

_GROQ_KEY_S5 = os.environ.get("GROQ_API_KEY_2", "")


class GroqClient:

    MAX_RETRIES    = 4
    DEFAULT_WAIT_S = 15

    def __init__(self, model=GROQ_MODEL_S5, timeout=60):
        self.model   = model
        self.timeout = timeout
        if not _GROQ_KEY_S5:
            raise RuntimeError(
                "GROQ_API_KEY_2 is not set. "
                "Add it to agents/shani/.env (dedicated S5 extraction key)."
            )

    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {_GROQ_KEY_S5}",
                        "Content-Type":  "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.exceptions.ConnectionError:
                raise RuntimeError("Cannot connect to Groq API. Check network.")
            except requests.exceptions.Timeout:
                raise RuntimeError(f"Groq request timed out after {self.timeout}s.")

            if response.status_code == 200:
                try:
                    text = response.json()["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError, Exception) as exc:
                    raise RuntimeError(f"Unexpected Groq response: {exc}\nRaw: {response.text[:300]}")
                if not text:
                    raise RuntimeError("Empty response from Groq API.")
                return text

            if response.status_code == 429:
                wait = self.DEFAULT_WAIT_S
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(int(retry_after), 1)
                    except ValueError:
                        pass
                print(f"[GROQ] 429 rate limit (attempt {attempt + 1}/{self.MAX_RETRIES}). Sleeping {wait}s.")
                _time.sleep(wait)
                last_error = f"429 rate limit after {self.MAX_RETRIES} attempts"
                continue

            raise RuntimeError(f"Groq API error {response.status_code}: {response.text[:300]}")

        raise RuntimeError(f"Groq API: {last_error}")
