# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# NOTE (confirmed decision 2026-08-11): CUDA/torch wheels kept as-is (no
# CPU-only swap). Whisper/torch import cost is instead avoided via
# ENABLE_VOICE_FEATURES (default false, see api/app.py) -- the lazy-import
# fix applied this session -- so a container that never enables voice
# features never pays the import cost even though the wheel is present.
ENV PYTHONUNBUFFERED=1 \
    BRAHM_ROOT=/app \
    ENABLE_VOICE_FEATURES=false

WORKDIR /app

COPY agents/chitragupta/requirements.txt agents/chitragupta/requirements.txt
# --mount=type=cache persists pip's download cache across builds (not
# baked into the final image layer), so a retry after a network drop --
# confirmed 2026-08-12: torch's 526MB wheel timed out mid-download on a
# flaky connection -- resumes from cache instead of redownloading
# everything. --timeout 120 (default 15s) gives slow/unstable connections
# more room per read before pip gives up.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 -r agents/chitragupta/requirements.txt

COPY agents/chitragupta agents/chitragupta

RUN mkdir -p agents/chitragupta/database agents/chitragupta/data

WORKDIR /app/agents/chitragupta
EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8003/health',timeout=3).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8003"]
