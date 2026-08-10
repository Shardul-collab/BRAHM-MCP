"""
brahm/shared/http.py
=====================
HTTP client helpers, one set per agent that exposes an HTTP API.
"""

from __future__ import annotations
import logging
from brahm.shared.constants import SHANI_BASE, GANESH_BASE

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
    "cd /mnt/d/brahm/agents/shani && "
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
    "cd /mnt/d/brahm/agents/ganesh && "
    "source .venv/bin/activate && "
    "python -m uvicorn ganesh_api:app --host 0.0.0.0 --port 8001"
)


async def _chitragupta_get(path: str) -> dict:
    import httpx
    from brahm.shared.helpers import _err
    from brahm.shared.constants import CHITRAGUPTA_BASE
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{CHITRAGUPTA_BASE}{path}")
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
            r = await client.post(f"{CHITRAGUPTA_BASE}{path}", json=body)
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
            r = await client.patch(f"{CHITRAGUPTA_BASE}{path}", json=body)
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
            r = await client.get(f"{CHITRAGUPTA_BASE}/health")
            return r.status_code == 200
    except Exception:
        return False


CHITRAGUPTA_START_HINT = (
    "Start CHITRAGUPTA with: "
    "cd /mnt/d/brahm/agents/chitragupta && "
    "/mnt/d/brahm/.venv/bin/python -m uvicorn api_server:app --host 0.0.0.0 --port 8003"
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
                headers={"X-API-Key": os.environ.get("API_KEY", "")},
            )
            if r.status_code >= 400:
                _logger.warning("Chitragupta store %s returned %d: %s", endpoint, r.status_code, r.text[:200])
            else:
                _logger.info("Chitragupta store %s OK", endpoint)
    except Exception as exc:
        _logger.warning("Chitragupta store %s failed (non-fatal): %s", endpoint, exc)


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
                headers={"X-API-Key": os.environ.get("API_KEY", "")},
            )
            if r.status_code >= 400:
                _logger.warning("Chitragupta store %s returned %d: %s", endpoint, r.status_code, r.text[:200])
            else:
                _logger.info("Chitragupta store %s OK", endpoint)
    except Exception as exc:
        _logger.warning("Chitragupta store %s failed (non-fatal): %s", endpoint, exc)
