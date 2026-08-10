# api_server.py

"""
Chitragupta API Server — uvicorn entry point.

Run with:
    python api_server.py

Or directly via uvicorn:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Environment variables (in .env):
    API_KEY      — set to enable X-API-Key authentication (leave empty for dev)
    CORS_ORIGINS — comma-separated allowed origins, or * for all (default)
    LOG_LEVEL    — DEBUG / INFO / WARNING (default: INFO)
    PORT         — server port (default: 8000)

The CLI (main.py) and the API server are independent entry points.
Both use the same underlying modules — no code is duplicated.
"""

import os
import logging

from dotenv import load_dotenv
load_dotenv()

from api.app import create_app

logger = logging.getLogger("chitragupta.api_server")

app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"

    logger.info("Starting Chitragupta API on %s:%d (reload=%s)", host, port, reload)

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
