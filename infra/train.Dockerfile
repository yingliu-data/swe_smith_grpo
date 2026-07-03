# Pod B — 1-GPU co-located trainer. prime-rl + vLLM share a single
# RTX PRO 6000 Blackwell (96 GB) via the CUDA_VISIBLE_DEVICES="0,0" alias:
# prime-rl's launcher asserts num_train_gpus + num_infer_gpus ≤
# len(physical_gpu_ids), and duplicating device 0 in the list makes that
# check pass on a 1-GPU pod. Capacity stopgap; the 2-GPU layout is the
# intended production topology. Trainer and vLLM are separate processes;
# vLLM's gpu_memory_utilization (see infer.toml) is set with colocation
# in mind, since vLLM's KV-cache profiler subtracts memory already held
# by the trainer process on the same physical device.
# CUDA 12.8 base matches prime-rl v0.5.0 pins (torch 2.10 cu128 / vLLM 0.14).

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1

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

# ---------------------------------------------------------------------------
# prime-rl lives in its own project so uv honors its tool.uv.sources block
# (verifiers/torchtitan/transformers git pins, pytorch-cu128 index, etc.).
# Pinned to v0.5.0 tag (63331ad, 2026-03-30).
# ---------------------------------------------------------------------------
ARG PRIME_RL_REV=63331ad8b17048b6d5f2051b2bf159e1392924b7
# Blobless clone: keeps the full ref/tag graph (so prime-rl's version
# detection and an arbitrary-SHA checkout still work) but defers file blobs,
# fetching only those for the checked-out rev — far smaller than a full clone.
RUN git clone --filter=blob:none https://github.com/PrimeIntellect-ai/prime-rl.git /opt/prime-rl \
    && cd /opt/prime-rl && git checkout "${PRIME_RL_REV}"
WORKDIR /opt/prime-rl
# `flash-attn` is an optional extra (prebuilt wheel for cu128/py3.12); prime-rl's
# trainer imports `ring_flash_attn` → `flash_attn` unconditionally via
# prime_rl.utils.cp, so the extra is effectively required for training.
RUN uv sync --extra flash-attn

# ---------------------------------------------------------------------------
# Install our `training` package editable into prime-rl's venv so verifiers'
# importlib-based discovery finds `swe_agent_env` (shipped in training/src)
# when orch.toml names `id = "swe-agent-env"`. Its `agent`/`common` path deps
# come in via training's [tool.uv.sources] as non-editable installs — do NOT
# add a second `-e /app/agent -e /app/common` here: uv 0.11 rejects the same
# path referenced both editable (the `-e`) and non-editable (common's own
# source) as conflicting URLs, which breaks the build.
# ---------------------------------------------------------------------------
WORKDIR /app
COPY agent/    /app/agent/
COPY common/   /app/common/
COPY training/ /app/training/
RUN UV_PROJECT=/opt/prime-rl uv pip install \
        --python /opt/prime-rl/.venv/bin/python \
        -e /app/training \
        aiodocker>=0.24 aiofiles>=24.1

# Fail the build loud if flash-attn didn't land. prime-rl's trainer imports
# `flash_attn` unconditionally via `ring_flash_attn` in prime_rl.utils.cp, so
# a missing wheel otherwise surfaces as a mid-rollout ModuleNotFoundError on
# Pod B. We've been burned by this: (1) `uv sync` alone drops the extra,
# (2) running `uv sync --extra flash-attn` against a populated venv wipes the
# editable overlay packages, so fixes in-pod are order-sensitive. Running the
# check *after* the overlay install catches regressions from either step.
RUN /opt/prime-rl/.venv/bin/python -c "import flash_attn, ring_flash_attn; print(f'flash_attn {flash_attn.__version__}')"

# ---------------------------------------------------------------------------
# Rollout target repo + its own test venv — deliberately ISOLATED from
# prime-rl's venv. The target is training *data*, not infrastructure:
# installing it -e into the serving venv clobbered vLLM's own fastapi pin
# with the target's main HEAD, and every API route 500'd
# (prometheus-fastapi-instrumentator's `route.path` vs fastapi-main's
# `_IncludedRouter`). Rollout `evaluate` runs pytest via $SWE_TARGET_PYTHON
# (see swe_agent_env._task_from_row), so target deps never touch prime-rl's
# resolver. Each rollout copies /opt/repo-cache/target into
# /tmp/rollout-workspace/current (agent/src/agent/async_local_env.py::prepare).
# ---------------------------------------------------------------------------
ARG TARGET_REPO_URL=https://github.com/fastapi/fastapi.git
# Full clone (NOT shallow/blobless): each rollout runs `git checkout -f
# <task.base_commit>` (agent/async_local_env.py:53), so every task's
# historical base commit must be present offline. A --depth 1 clone has only
# HEAD; a blobless clone would need network mid-rollout.
RUN git clone "${TARGET_REPO_URL}" /opt/repo-cache/target
# The trailing dep list is the target's test-runner deps (fastapi's pytest
# stack for the default target); swap alongside TARGET_REPO_URL.
RUN uv venv /opt/target-venv --python /usr/bin/python3.12 \
    && uv pip install --python /opt/target-venv/bin/python \
        -e /opt/repo-cache/target \
        pytest pytest-asyncio anyio httpx dirty-equals
ENV SWE_TARGET_PYTHON=/opt/target-venv/bin/python \
    SWE_TARGET_TEMPLATE=/opt/repo-cache/target
# Regression guards: the target venv must run pytest standalone, and
# prime-rl's venv must NOT have picked up the target repo — vLLM's own
# fastapi pin has to survive (see isolation note above).
RUN /opt/target-venv/bin/python -m pytest --version \
    && /opt/prime-rl/.venv/bin/python -c \
       "import fastapi; assert '/opt/repo-cache' not in fastapi.__file__, fastapi.__file__; print('serving fastapi:', fastapi.__version__)"

# Import-check: verifiers resolves `swe-agent-env` via importlib.import_module,
# so the module must be at the top level of prime-rl's venv. Fail the build if
# hatch didn't ship src/swe_agent_env (pyproject.toml [tool.hatch.build...]).
RUN /opt/prime-rl/.venv/bin/python -c \
    "import swe_agent_env; assert callable(swe_agent_env.load_environment)"

# train.py shells out to `$PRIME_RL_CMD ...`; invoke the baked console script
# directly to skip `uv run`'s implicit sync/resolve (which needs network).
ENV PRIME_RL_CMD="/opt/prime-rl/.venv/bin/rl"
# prime-rl's `rl` launcher internally spawns `torchrun` by bare name, so that
# venv's bin must be on PATH. `uv run train` only activates /app/training's
# venv, which doesn't carry torch/torchrun.
ENV PATH="/opt/prime-rl/.venv/bin:${PATH}"
# Co-location alias: one physical GPU masqueraded as two entries so
# prime-rl's deployment check passes. RunPod often resets this at pod
# start based on the GPU-count assigned, so train.py re-pins it in the
# subprocess env; this ENV is the baseline for anyone running the image
# outside the training entrypoint (e.g. an exec shell).
ENV CUDA_VISIBLE_DEVICES="0,0"
# Reduce allocator fragmentation for the colocated trainer+vLLM squeeze.
# docker-compose.train.yml sets this too; baked here so plain `docker run`
# gets the same allocator behaviour.
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

WORKDIR /app/training
# Minimal sync for train.py itself (no gpu extras needed here — prime-rl's
# venv carries torch/vllm; train.py only needs the wrapper's own deps).
RUN uv sync

ENTRYPOINT ["uv", "run", "train"]
CMD ["--dataset", "/workspace/datasets/pilot/pilot.jsonl", \
     "--profile", "smoke", \
     "--output-dir", "/workspace/checkpoints", \
     "--sessions-root", "/workspace/sessions"]
