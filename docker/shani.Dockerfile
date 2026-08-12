# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# NOTE (confirmed decision 2026-08-11): SHANI keeps its full CUDA/OCR/torch
# requirements.txt as-is (no CPU-only swap) even though no GPU is actually
# used at runtime -- image will be large. See SETUP.md Docker section.
ENV PYTHONUNBUFFERED=1 \
    BRAHM_ROOT=/app

WORKDIR /app

COPY agents/shani/requirements.txt agents/shani/requirements.txt
# --mount=type=cache persists pip's download cache across builds, so a
# retry after a network drop resumes from cache instead of redownloading
# everything (confirmed 2026-08-12: chitragupta's identically-sized torch
# pull timed out mid-download on a flaky connection -- same risk applies
# here). --timeout 120 gives slow/unstable connections more room per read.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 -r agents/shani/requirements.txt

COPY agents/shani agents/shani

# database/ dir needs to exist as a mount point even before the volume
# populates it (init_db.py creates the .db file on first run if absent).
RUN mkdir -p agents/shani/database

WORKDIR /app/agents/shani
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
