FROM condaforge/miniforge3:latest

# IMPORTANT — READ BEFORE CHANGING THIS FILE:
# This container does NOT run mcp_server.py as a foreground service.
# mcp_server.py is an stdio MCP server (Claude Desktop spawns it as a
# subprocess and talks over stdin/stdout) -- it has no HTTP port, so there
# is nothing to EXPOSE and nothing for a curl-based /health check to hit.
# The container's job is to stay alive with QE + the MCP layer installed
# so Claude Desktop (or you, for a smoke test) can run:
#   docker exec -i brahm-vishwakarma python /app/mcp_server.py
# See SETUP.md's "Docker" section for the full explanation and the
# --profile vishwakarma usage.

ENV PYTHONUNBUFFERED=1 \
    BRAHM_ROOT=/app \
    QE_BIN_DIR=/opt/conda/envs/qe/bin \
    QE_WORKDIR=/data/qe_jobs \
    VISHWAKARMA_WORKDIR=/data/qe_jobs \
    QE_PSEUDO=/data/pseudo \
    PATH="/opt/conda/envs/qe/bin:${PATH}"

# QE 7.5 via conda-forge -- matches the tested working dev-machine setup
# (SETUP.md step 4). This pulls in openmpi/PRRTE as a transitive dep,
# which is what makes the --map-by :OVERSUBSCRIBE fix in runner.py
# relevant inside the container too.
RUN mamba create -y -n qe -c conda-forge qe=7.5 && mamba clean -afy

WORKDIR /app

# Root-level requirements.txt is the light MCP-layer one (no CUDA/torch --
# confirmed via requirements.txt audit), safe to install with base conda's pip.
COPY requirements.txt requirements.txt
RUN /opt/conda/bin/pip install --no-cache-dir -r requirements.txt

COPY brahm brahm
COPY mcp_server.py mcp_server.py

# No pseudopotentials baked into the image (confirmed decision 2026-08-11:
# bind-mount instead, via QE_PSEUDO_HOST_DIR in .env -- see docker-compose.yml).
# Note: agents/vishwakarma/pseudo/ on the host IS populated (confirmed via
# `ls` 2026-08-12, includes Si.pbe-n-kjpaw_psl.1.0.0.UPF) -- an earlier note
# here claiming it was absent was stale. QE_PSEUDO_HOST_DIR already defaults
# to that directory, so bind-mounting it is usually automatic, not manual.
RUN mkdir -p /data/qe_jobs /data/pseudo

# Confirmed 2026-08-12 (real build/run test): the container runs as root by
# default, and Open MPI correctly REFUSES to mpirun as root without
# --allow-run-as-root -- this isn't just a smoke-test annoyance, it would
# block every real DFT job runner.py launches through this container.
# Fixing it here (non-root user) rather than adding --allow-run-as-root to
# runner.py's mpirun call -- that's app code, out of scope for a packaging
# pass, and baking in a root-MPI override permanently is a real security
# smell we don't need since a non-root user solves it cleanly.
RUN useradd --create-home --shell /bin/bash brahm \
    && chown -R brahm:brahm /data/qe_jobs
USER brahm
ENV HOME=/home/brahm

# Confirmed 2026-08-12: pw.x has no real CLI flag parser -- any argument
# (including --version) is ignored and it exits nonzero waiting on stdin,
# so invoking it here would always report unhealthy even on a working
# container. Check binary presence/executability instead.
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD test -x /opt/conda/envs/qe/bin/pw.x || exit 1

CMD ["tail", "-f", "/dev/null"]
