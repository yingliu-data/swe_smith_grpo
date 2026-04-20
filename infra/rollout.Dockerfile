# Per-task rollout image — this is what DockerEnvironment spawns per rollout.
# It must be small, self-contained, and network-isolated at runtime.
# All Python deps for the *task* (fastapi, anyio, pytest) must be baked in;
# we cannot pip-install inside the rollout because the container runs with
# --network none.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pre-install the task-specific test deps. In a production build, datagen would
# parametrise this per-repo; here we pre-bake the fastapi pilot.
RUN pip install --no-cache-dir \
        "fastapi==0.110.0" \
        "starlette==0.36.3" \
        "pydantic==2.6.4" \
        "anyio==3.5.0" \
        "httpx==0.27.0" \
        "pytest==8.1.1" \
        "pytest-asyncio==0.23.6" \
        "trio==0.24.0"

WORKDIR /workspace/repo
# Repo contents are bind-mounted at runtime (:ro). A writable tmpfs is mounted
# at /workspace/repo by DockerEnvironment so the agent can edit source files.
