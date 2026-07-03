# Pod B eval image — same pod as train, but without prime-rl/flash-attn pins.
# Rollouts run in-process via agent.AsyncLocalEnvironment (no docker-in-docker).
# The fastapi repo is baked into /opt/repo-cache/fastapi__fastapi (outside the
# /workspace bind mount so the clone survives runtime) and each rollout
# shutil.copytree's from there into /tmp/eval-rollouts/<instance_id>.

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
        python3.12-dev \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app

COPY agent/ /app/agent/
COPY common/ /app/common/
COPY evaluation/ /app/evaluation/

WORKDIR /app/evaluation
RUN uv sync --extra gpu

# Rollout target repo + its own test venv — deliberately ISOLATED from the
# eval venv. The eval venv carries vLLM, whose API server runs on fastapi;
# installing the target's main HEAD -e into it replaced vLLM's fastapi pin
# and 500'd every route (prometheus-fastapi-instrumentator's `route.path` vs
# fastapi-main's `_IncludedRouter`) — same bug as infra/train.Dockerfile.
# Rollout pytest runs via $SWE_TARGET_PYTHON (see evaluation/sample.py).
# Cloned under /opt (not /workspace) so the compose bind mount doesn't shadow
# the clone at runtime; mirror name matches {git_mirror_root}/{repo_slug}.
# Full clone: rollouts `git checkout -f <base_commit>` historical commits.
RUN git clone https://github.com/fastapi/fastapi.git /opt/repo-cache/fastapi__fastapi
RUN uv venv /opt/target-venv --python /usr/bin/python3.12 \
    && uv pip install --python /opt/target-venv/bin/python \
        -e /opt/repo-cache/fastapi__fastapi \
        pytest pytest-asyncio anyio httpx dirty-equals
ENV SWE_TARGET_PYTHON=/opt/target-venv/bin/python
# Regression guards: target venv runs pytest standalone; the eval venv's
# fastapi (vLLM's dependency) must NOT resolve from /opt/repo-cache.
RUN /opt/target-venv/bin/python -m pytest --version \
    && /app/evaluation/.venv/bin/python -c \
       "import fastapi; assert '/opt/repo-cache' not in fastapi.__file__, fastapi.__file__; print('serving fastapi:', fastapi.__version__)"

ENTRYPOINT ["uv", "run", "evaluate"]

CMD ["--swebench-n", "20", \
     "--checkpoint", "/workspace/checkpoints/final", \
     "--heldout", "/workspace/datasets/pilot/heldout.jsonl", \
     "--sessions-root", "/workspace/sessions", \
     "--heldout-n", "10"]