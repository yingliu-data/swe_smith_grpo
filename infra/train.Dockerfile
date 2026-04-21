# Pod B — H100 trainer. Runs prime-rl + vLLM (co-located).
# CUDA 12.8 base matches prime-rl v0.5.0 pins (torch 2.9 cu128 / vLLM 0.14).

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

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
        docker.io \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

# ---------------------------------------------------------------------------
# prime-rl lives in its own project so uv honors its tool.uv.sources block
# (verifiers/torchtitan/transformers git pins, pytorch-cu128 index, etc.).
# Pinned to v0.5.0 tag (63331ad, 2026-03-30).
# ---------------------------------------------------------------------------
ARG PRIME_RL_REV=63331ad8b17048b6d5f2051b2bf159e1392924b7
RUN git clone https://github.com/PrimeIntellect-ai/prime-rl.git /opt/prime-rl \
    && cd /opt/prime-rl && git checkout "${PRIME_RL_REV}"
WORKDIR /opt/prime-rl
RUN uv sync

# ---------------------------------------------------------------------------
# Our training/agent/common packages are installed editable into prime-rl's
# venv, so `uv run rl --env training.swe_env:SWEAgentEnv` can import them.
# ---------------------------------------------------------------------------
WORKDIR /app
COPY agent/    /app/agent/
COPY common/   /app/common/
COPY training/ /app/training/
RUN UV_PROJECT=/opt/prime-rl uv pip install \
        --python /opt/prime-rl/.venv/bin/python \
        -e /app/agent -e /app/common -e /app/training \
        aiodocker>=0.24 aiofiles>=24.1

# train.py shells out to `$PRIME_RL_CMD ...`; point it at prime-rl's venv.
ENV PRIME_RL_CMD="uv --project /opt/prime-rl run rl"

WORKDIR /app/training
# Minimal sync for train.py itself (no gpu extras needed here — prime-rl's
# venv carries torch/vllm; train.py only needs the wrapper's own deps).
RUN uv sync

ENTRYPOINT ["uv", "run", "train"]
CMD ["--dataset", "/workspace/datasets/pilot/pilot.jsonl", \
     "--profile", "smoke", \
     "--output-dir", "/workspace/checkpoints", \
     "--sessions-root", "/workspace/sessions"]
