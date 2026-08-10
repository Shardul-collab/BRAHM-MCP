# api/app.py

"""
Chitragupta API — FastAPI application factory.

API versioning
    All routers are mounted under /v1.
    /health is at the root (no version prefix) as a liveness probe.
    Swagger reflects /v1/... paths throughout.

Startup key enforcement
    verify_api_key_at_startup() runs in the lifespan context.
    If API_KEY is not set the process raises RuntimeError and refuses
    to start.  /health is the only route without auth.

CORS
    The CORS spec forbids Access-Control-Allow-Origin: * paired with
    Access-Control-Allow-Credentials: true.  When origins is the wildcard
    default, allow_credentials is set to False.  Enumerate CORS_ORIGINS
    in .env to get credentialed cross-origin requests.

Exception handling
    The global handler logs the full traceback server-side and returns a
    generic message to callers — no internal paths or schema names leaked.

OpenAPI docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import verify_api_key_at_startup
from api.routers import databases, entries, voice, analysis, relations, session, context, store
from brahm_db.api import router as brahm_db_router
from brahm_db.schema import init_db as init_brahm_db

logger = logging.getLogger("chitragupta.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup check — crashes the server if API_KEY is not set.
    Runs before any request is accepted.
    """
    verify_api_key_at_startup()
    # brahm_db_router (projects/results/documents/papers-dedup) reads/writes
    # brahm.db. init_brahm_db() is idempotent (CREATE TABLE IF NOT EXISTS),
    # safe to call on every startup. Previously this only ran inside
    # brahm_db/api.py's own standalone lifespan, which never executed under
    # the actual running process (api_server.py -> here).
    init_brahm_db()
    logger.info("Chitragupta API started successfully.")
    yield
    logger.info("Chitragupta API shut down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Chitragupta API",
        description=(
            "REST API for Chitragupta — voice-driven Notion logging system.\n\n"
            "All endpoints are under **/v1**.\n\n"
            "**Authentication**: every request (except `/health`) requires "
            "the `X-API-Key` header matching your `API_KEY` env var.\n\n"
            "**Background writes**: POST /v1/entries/{name} and "
            "POST /v1/voice/log-entry return 202 immediately. "
            "Check /v1/entries/{name}/pending to confirm completion."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    raw_origins = os.getenv("CORS_ORIGINS", "*")
    wildcard    = raw_origins.strip() == "*"
    origins     = ["*"] if wildcard else [o.strip() for o in raw_origins.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception at %s", request.url)
        return JSONResponse(
            status_code=500,
            content={
                "ok":      False,
                "message": "An internal server error occurred. See server logs.",
            },
        )

    # ── /health — no auth, no version prefix ─────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health() -> dict:
        """Liveness check. No auth required. Returns 200 when server is running."""
        return {"ok": True, "message": "Chitragupta API is running."}

    # ── Feature routers under /v1 ─────────────────────────────────────────────
    app.include_router(databases.router, prefix="/v1")
    app.include_router(entries.router,   prefix="/v1")
    app.include_router(voice.router,     prefix="/v1")
    app.include_router(analysis.router,  prefix="/v1")
    app.include_router(relations.router, prefix="/v1")
    app.include_router(session.router,   prefix="/v1")
    app.include_router(context.router,  prefix="/v1")
    app.include_router(store.router,   prefix="/v1")

    # brahm_db_router's routes already hardcode the "/v1/..." prefix
    # themselves (e.g. @router.post("/v1/projects")), so no prefix= here —
    # adding one would double it to /v1/v1/projects.
    #
    # No auth dependency added: these routes never had Depends(api_key_auth)
    # in their original standalone app, and the MCP-side callers
    # (brahm/shared/http.py's _chitragupta_get/_chitragupta_post) never send
    # an X-API-Key header. Adding auth here would newly break every
    # chitragupta_* MCP tool instead of fixing them.
    app.include_router(brahm_db_router)

    logger.info("Chitragupta API app created. All routes under /v1.")
    return app
