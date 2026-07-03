# Pod A — CPU datagen. Runs Nebius-driven mutation + three-way pytest validation.
# Pinned to python:3.12-slim because the training image uses a CUDA base; keeping
# datagen separate avoids a 10 GB image on a CPU-only pod.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: git (repo cloning), build-essential (for some wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# uv — single static binary, cached across layers. Must be new enough to read
# the committed uv.lock schema (revision 3 needs >= 0.8).
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app

# Copy path-dep packages first so they're cached if datagen/ changes alone.
COPY agent/ /app/agent/
COPY common/ /app/common/
COPY datagen/ /app/datagen/

WORKDIR /app/datagen
RUN uv sync --frozen

# Default entrypoint: the CLI (--frozen: never re-resolve at container start).
# Default mode is SWE-Smith mutation (requires NEBIUS_API_KEY); pass --base
# for plain diff extraction, which needs no LLM key.
ENTRYPOINT ["uv", "run", "--frozen", "datagen"]
CMD ["--repo", "fastapi/fastapi"]
