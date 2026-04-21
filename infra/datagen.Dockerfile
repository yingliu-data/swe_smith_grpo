# Pod A — CPU datagen. Runs Nebius-driven mutation + three-way pytest validation.
# Pinned to python:3.12-slim because the training image uses a CUDA base; keeping
# datagen separate avoids a 10 GB image on a CPU-only pod.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: git (repo cloning), build-essential (for some wheels),
# docker-cli (datagen calls `docker save` to pre-bake per-repo images).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
        curl \
        docker.io \
    && rm -rf /var/lib/apt/lists/*

# uv — single static binary, cached across layers.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# Copy path-dep packages first so they're cached if datagen/ changes alone.
COPY agent/ /app/agent/
COPY common/ /app/common/
COPY datagen/ /app/datagen/

WORKDIR /app/datagen
RUN uv sync --frozen

# Default entrypoint: the CLI. Override at docker-compose level.
ENTRYPOINT ["uv", "run", "datagen"]
CMD ["--repo", "fastapi/fastapi"]
