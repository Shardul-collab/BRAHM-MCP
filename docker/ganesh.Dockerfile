FROM python:3.11-slim

# GANESH's own requirements.txt is light (fastapi/uvicorn/pydantic/
# python-dotenv/requests) -- confirmed via agents/ganesh/requirements.txt.
# But ganesh_api.py's own sys.path setup (see its "Path setup" comment
# block) pulls repositories.repository from agents/shani and shared code
# from brahm/ at import time -- both must be present in this image even
# though we deliberately do NOT install SHANI's heavy requirements.txt
# here. If GANESH fails to start with an ImportError inside
# repositories/repository.py, that means it depends on something from
# SHANI's heavy stack after all -- flag that as a follow-up, don't just
# add torch to this image to make it go away.
ENV PYTHONUNBUFFERED=1 \
    BRAHM_ROOT=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY agents/ganesh/requirements.txt agents/ganesh/requirements.txt
RUN pip install -r agents/ganesh/requirements.txt

COPY brahm brahm
COPY agents/ganesh agents/ganesh

# Only the pieces ganesh_api.py's sys.path insertion actually reaches into --
# not all of agents/shani, to keep this image from silently growing SHANI's
# full tree (and its heavy deps) into a "light" service.
COPY agents/shani/repositories agents/shani/repositories

RUN mkdir -p agents/shani/database

WORKDIR /app/agents/ganesh
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health',timeout=3).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "ganesh_api:app", "--host", "0.0.0.0", "--port", "8001"]
