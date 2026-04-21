# Pod B — H100 trainer. Runs prime-rl + vLLM (co-located).
# CUDA 12.8 base matches prime-rl v0.5.0 pins (torch 2.10 cu128 / vLLM 0.14).

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
        python3.12-dev \
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
# `flash-attn` is an optional extra (prebuilt wheel for cu128/py3.12); prime-rl's
# trainer imports `ring_flash_attn` → `flash_attn` unconditionally via
# prime_rl.utils.cp, so the extra is effectively required for training.
RUN uv sync --extra flash-attn

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

# Fail the build loud if flash-attn didn't land. prime-rl's trainer imports
# `flash_attn` unconditionally via `ring_flash_attn` in prime_rl.utils.cp, so
# a missing wheel otherwise surfaces as a mid-rollout ModuleNotFoundError on
# Pod B. We've been burned by this: (1) `uv sync` alone drops the extra,
# (2) running `uv sync --extra flash-attn` against a populated venv wipes the
# editable overlay packages, so fixes in-pod are order-sensitive. Running the
# check *after* the overlay install catches regressions from either step.
RUN /opt/prime-rl/.venv/bin/python -c "import flash_attn, ring_flash_attn; print(f'flash_attn {flash_attn.__version__}')"

# train.py shells out to `$PRIME_RL_CMD ...`; invoke the baked console script
# directly to skip `uv run`'s implicit sync/resolve (which needs network).
ENV PRIME_RL_CMD="/opt/prime-rl/.venv/bin/rl"
# prime-rl's `rl` launcher internally spawns `torchrun` by bare name, so that
# venv's bin must be on PATH. `uv run train` only activates /app/training's
# venv, which doesn't carry torch/torchrun.
ENV PATH="/opt/prime-rl/.venv/bin:${PATH}"

# 1-GPU co-location (Pod B): prime-rl's deployment check requires
# num_train_gpus + num_infer_gpus = 2 physical devices. Listing device 0 twice
# maps both logical slots to the same GPU; vLLM is capped at
# gpu_memory_utilization=0.45 (see configs/infer.toml) leaving the rest for
# the LoRA trainer. docker-compose.train.yml sets the same value for local
# dev; baking it here covers direct `runpodctl pod create` launches.
ENV CUDA_VISIBLE_DEVICES="0,0"

WORKDIR /app/training
# Minimal sync for train.py itself (no gpu extras needed here — prime-rl's
# venv carries torch/vllm; train.py only needs the wrapper's own deps).
RUN uv sync

ENTRYPOINT ["uv", "run", "train"]
CMD ["--dataset", "/workspace/datasets/pilot/pilot.jsonl", \
     "--profile", "smoke", \
     "--output-dir", "/workspace/checkpoints", \
     "--sessions-root", "/workspace/sessions"]
