# Pod B eval image — same pod as train, but without prime-rl/flash-attn pins.
# A smaller image means a faster cold-start on the vLLM re-launch after train.

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

COPY agent/ /app/agent/
COPY common/ /app/common/
COPY evaluation/ /app/evaluation/

WORKDIR /app/evaluation
RUN uv sync --extra gpu

ENTRYPOINT ["uv", "run", "evaluate"]
