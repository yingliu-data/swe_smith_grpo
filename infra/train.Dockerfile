# Pod B — H100 trainer. Runs prime-rl + vLLM (co-located).
# CUDA 12.4 base matches flash-attn-3 / vLLM 0.6 / torch 2.5 wheels.

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
        docker.io \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# Path-dep packages first (ordering matters for layer cache).
COPY agent/ /app/agent/
COPY common/ /app/common/
COPY training/ /app/training/

WORKDIR /app/training
# GPU extras pull in torch/vllm/peft; flash-attn-3 is compiled at first import.
# No --frozen: CUDA/vllm/flash-attn wheels are Linux-only, so resolution happens
# at build time (macOS cannot produce a lockfile that covers these wheels).
RUN uv sync --extra gpu

# Trainer CLI. Wrapped by infra/run_train.sh supervisor loop in production.
ENTRYPOINT ["uv", "run", "train"]
