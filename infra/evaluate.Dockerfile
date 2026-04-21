# Pod B eval image — same pod as train, but without prime-rl/flash-attn pins.
# Rollouts run in-process via agent.AsyncLocalEnvironment (no docker-in-docker);
# the fastapi repo is pre-mirrored into /workspace/src/fastapi__fastapi at pod
# bootstrap so each rollout can shutil.copytree from there.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

COPY agent/ /app/agent/
COPY common/ /app/common/
COPY evaluation/ /app/evaluation/

WORKDIR /app/evaluation
RUN uv sync --extra gpu

# Bake the fastapi template + its pytest deps into the eval venv so the
# rollout's `python -m pytest` finds them when invoked via cwd=<scratch-dir>.
RUN git clone https://github.com/fastapi/fastapi.git /workspace/src/fastapi__fastapi
RUN uv pip install \
        -e /workspace/src/fastapi__fastapi \
        pytest pytest-asyncio anyio httpx dirty-equals

ENTRYPOINT ["uv", "run", "evaluate"]
