"""
brahm/shared/http.py
=====================
HTTP client helpers, one set per agent that exposes an HTTP API.
"""

from __future__ import annotations
import logging
from brahm.shared.constants import (SHANI_BASE, GANESH_BASE, BRAHM_ROOT,
                                    SHANI_ROOT, CHITRAGUPTA_ROOT, GANESH_ROOT)

log = logging.getLogger("mcp.brahm.http")


async def _shani_get(path: str) -> dict:
    import httpx
    from brahm.shared.helpers import _err
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{SHANI_BASE}{path}")
            if r.status_code >= 400:
                return _err(f"SHANI API error {r.status_code}", r.text[:500])
            return r.json()
    except Exception as exc:
        return _err("SHANI API unreachable", str(exc))


async def _shani_post(path: str, body: dict) -> dict:
    import httpx
    from brahm.shared.helpers import _err
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{SHANI_BASE}{path}", json=body)
            if r.status_code >= 400:
                return _err(f"SHANI API error {r.status_code}", r.text[:500])
            return r.json()
    except Exception as exc:
        return _err("SHANI API unreachable", str(exc))


async def _check_shani() -> bool:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{SHANI_BASE}/docs")
            return r.status_code == 200
    except Exception:
        return False


SHANI_START_HINT = (
    "Start SHANI with: "
    f"cd {SHANI_ROOT} && "
    "source venv/bin/activate && "
    "python -m uvicorn api:app --host 0.0.0.0 --port 8000"
)


async def _ganesh_get(path: str) -> dict:
    import httpx
    from brahm.shared.helpers import _err
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(f"{GANESH_BASE}{path}")
            if r.status_code >= 400:
                return _err(f"GANESH API error {r.status_code}", r.text[:500])
            return r.json()
    except Exception as exc:
        return _err("GANESH API unreachable", str(exc))


async def _ganesh_post(path: str, body: dict) -> dict:
    import httpx
    from brahm.shared.helpers import _err
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{GANESH_BASE}{path}", json=body)
            if r.status_code >= 400:
                return _err(f"GANESH API error {r.status_code}", r.text[:500])
            return r.json()
    except Exception as exc:
        return _err("GANESH API unreachable", str(exc))


async def _check_ganesh() -> bool:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{GANESH_BASE}/docs")
            return r.status_code == 200
    except Exception:
        return False


GANESH_START_HINT = (
    "Start GANESH with: "
    f"cd {GANESH_ROOT} && "
    "source .venv/bin/activate && "
    "python -m uvicorn ganesh_api:app --host 0.0.0.0 --port 8001"
)


def _chit_api_key() -> str:
    """Chitragupta's API_KEY, for the X-API-Key header.

    2026-09-20: `api/app.py` refuses to start without an API_KEY and every
    protected router depends on `api_key_auth`, but these three helpers sent no
    headers at all -- so the moment the key existed, every chitragupta_* MCP
    tool would have started returning 401. This is step one of the two-step fix
    recorded on 2026-09-09: teaching the callers to send the key is safe on its
    own, because unauthenticated routes ignore it.

    Env var wins; otherwise read Chitragupta's own .env, the same fallback
    pattern GANESH's _groq_key() uses.
    """
    import os
    key = os.environ.get("CHITRAGUPTA_API_KEY", "").strip()
    if key:
        return key
    from brahm.shared.constants import ENV_FILE
    try:
        for line in open(ENV_FILE, errors="ignore"):
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def _chit_headers() -> dict:
    key = _chit_api_key()
    return {"X-API-Key": key} if key else {}


async def _chitragupta_get(path: str) -> dict:
    import httpx
    from brahm.shared.helpers import _err
    from brahm.shared.constants import CHITRAGUPTA_BASE
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{CHITRAGUPTA_BASE}{path}", headers=_chit_headers())
            if r.status_code >= 400:
                return _err(f"CHITRAGUPTA API error {r.status_code}", r.text[:500])
            return r.json()
    except Exception as exc:
        return _err("CHITRAGUPTA API unreachable", str(exc))


async def _chitragupta_post(path: str, body: dict) -> dict:
    import httpx
    from brahm.shared.helpers import _err
    from brahm.shared.constants import CHITRAGUPTA_BASE
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{CHITRAGUPTA_BASE}{path}", json=body, headers=_chit_headers())
            if r.status_code >= 400:
                return _err(f"CHITRAGUPTA API error {r.status_code}", r.text[:500])
            return r.json()
    except Exception as exc:
        return _err("CHITRAGUPTA API unreachable", str(exc))


async def _chitragupta_patch(path: str, body: dict) -> dict:
    import httpx
    from brahm.shared.helpers import _err
    from brahm.shared.constants import CHITRAGUPTA_BASE
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.patch(f"{CHITRAGUPTA_BASE}{path}", json=body, headers=_chit_headers())
            if r.status_code >= 400:
                return _err(f"CHITRAGUPTA API error {r.status_code}", r.text[:500])
            return r.json()
    except Exception as exc:
        return _err("CHITRAGUPTA API unreachable", str(exc))


async def _check_chitragupta() -> bool:
    import httpx
    from brahm.shared.constants import CHITRAGUPTA_BASE
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{CHITRAGUPTA_BASE}/health", headers=_chit_headers())
            return r.status_code == 200
    except Exception:
        return False


CHITRAGUPTA_START_HINT = (
    "Start CHITRAGUPTA with: "
    f"cd {CHITRAGUPTA_ROOT} && "
    f"{BRAHM_ROOT}/.venv/bin/python api_server.py"
)


async def _chit_store_async(endpoint: str, payload: dict) -> None:
    """Fire-and-forget POST to a Chitragupta /v1/store/* endpoint. Non-fatal."""
    import httpx
    import os
    import logging as _log
    _logger = _log.getLogger("mcp.brahm.http")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"http://localhost:8003{endpoint}",
                json=payload,
                # 2026-09-20: was os.environ["API_KEY"], which is empty in the
                # MCP server process -- it sent a blank key and would have 401'd
                # the moment Chitragupta required one.
                headers=_chit_headers(),
            )
            if r.status_code >= 400:
                _logger.warning("Chitragupta store %s returned %d: %s", endpoint, r.status_code, r.text[:200])
            else:
                _logger.info("Chitragupta store %s OK", endpoint)
    except Exception as exc:
        _logger.warning("Chitragupta store %s failed (non-fatal): %s", endpoint, exc)
