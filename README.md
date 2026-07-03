# ML Systems Take-Home — Two-Phase RL Training

Three-pipeline system that (a) attempts synthetic SWE-Smith-style training
data generation from target-repo PRs, (b) trains `SWE-bench/SWE-agent-LM-7B`
with GRPO+ via prime-rl on that data, and (c) evaluates the resulting
policy on SWE-bench Verified + a heldout split. Each pipeline is its own
Docker image on its own pod, sized to the workload. Three `uv` projects
share two path-dep libraries (`agent/`, `common/`); $13 smoke budget.

> **Current status — partial.**
> - **Datagen**: all four variance methods (`lm_modify`, `lm_rewrite`,
>   `procedural`, `pr_mirror`) returned **0 valid candidates** on the
>   fastapi target — F2P/P2P validation gated them all out. Fell back to
>   pulling bug-fix PRs directly from `fastapi/fastapi` and using them as
>   the SWE-bench-format dataset.
> - **Training**: GRPO smoke with 16 samples produced **0 reward across
>   all rollouts**. Prime-rl `MultiTurnEnv` integration is the leading
>   suspect; `common/reward.py` is unit-tested and green.
> - **Eval**: wired end-to-end, clean once the fastapi template-path bug
>   in `evaluate.Dockerfile` is rebuilt.

## What runs where

Three pods, one per pipeline. Each mounts the same `/workspace` network
volume for dataset → checkpoint → session hand-off across stages.

| # | Pipeline   | Pod hardware                     | Image                       | Project       | Wall-clock |
| - | ---------- | -------------------------------- | --------------------------- | ------------- | ---------- |
| 1 | Datagen    | CPU, 8 vCPU, 32 GB               | `infra/datagen.Dockerfile`  | `datagen/`    | ~2 hr      |
| 2 | Training   | 1× RTX PRO 6000 Blackwell, 96 GB | `infra/train.Dockerfile`    | `training/`   | ~3.5 hr    |
| 3 | Evaluation | 1× RTX 5090, 24 GB               | `infra/evaluate.Dockerfile` | `evaluation/` | ~25 min    |

Training needs the 96 GB Blackwell to fit 7B bf16 + LoRA optimiser +
activations (seq=16384) + vLLM KV cache + vLLM weight copy — the 5090
can't (commit `28a124e`). Training is 1-GPU colocated: prime-rl trainer
and vLLM share the card via the `CUDA_VISIBLE_DEVICES="0,0"` alias
prime-rl's launcher accepts. The dedicated trainer/inference split over
≥2 GPUs is the intended production topology. Eval runs only vLLM
inference so the 5090 is sufficient and ~$2/hr cheaper.

## Project structure

```
ml_systems/
├── agent/                              # path-dep: async Environment ABC + impls
│   ├── pyproject.toml
│   └── src/agent/
│       ├── environment.py              # Abstract 7-method async Environment contract
│       ├── local_env.py                # Sync subprocess impl (datagen PR validation)
│       ├── async_local_env.py          # asyncio impl (train + eval rollouts, in-process)
│       ├── docker_env.py               # Containerised impl (reward-defense #1/#2 land here)
│       └── models.py                   # TaskSpec / ToolCall / ToolResult / EvaluationResult
│
├── common/                             # path-dep: reward + session + tool dispatch
│   ├── pyproject.toml
│   └── src/common/
│       ├── reward.py                   # compute_reward + DefenseEvent (per-eval gates #4–6)
│       ├── tool_surface.py             # Tool schema + dispatch(ToolCall, Environment)
│       ├── config.py                   # Cross-project knob types
│       ├── ids.py, ipc.py, logging.py
│       └── session/                    # Shared session-dir layout
│           ├── session_dir.py          # trace.jsonl + tickets/ + manifests/
│           ├── manifest.py             # Hash-verified output manifest
│           ├── ticket.py               # Audit tickets (agent, eval, train)
│           ├── memory.py
│           └── state.py                # FSM transitions for session lifecycle
│
├── datagen/                            # uv project #1 — synthetic datagen, lockfile #1
│   ├── pyproject.toml
│   └── src/datagen/
│       ├── pipeline.py                 # Top-level asyncio driver
│       ├── pilot_gen.py                # PR → 4-method × T=15 candidate fan-out
│       ├── methods/                    # lm_modify, lm_rewrite, procedural, pr_mirror
│       │                               # (all 4 yielded 0 valid candidates — see §Current status)
│       ├── nebius_client.py            # Qwen3-30B-A3B-Instruct-2507 API wrapper
│       ├── validator.py                # F2P/P2P gate via LocalWorkspaceEnvironment
│       ├── repo_manager.py             # Clone + pinned checkout of target repos
│       ├── writers/swebench_jsonl.py   # Emits pilot.jsonl + heldout.jsonl
│       ├── yield_logger.py             # yield.csv audit trail
│       └── config.py
│
├── training/                           # uv project #2 — trainer, lockfile #2, prime-rl v0.5.0
│   ├── pyproject.toml
│   └── src/
│       ├── training/
│       │   ├── train.py                # CLI wrapper around prime-rl's `rl` launcher
│       │   ├── checkpoint.py           # save / list / prune + hash-manifest integrity
│       │   ├── watchdog.py             # Stall detector (no step progress)
│       │   ├── session_logger.py       # trace.jsonl writer
│       │   ├── config.py
│       │   └── configs/
│       │       ├── train.toml          # Trainer (GRPO+, LoRA, seq=16384)
│       │       ├── orch.toml           # Prime-rl orchestrator (G=4 rollouts)
│       │       └── infer.toml          # vLLM colocation config
│       └── swe_agent_env/
│           └── __init__.py             # verifiers `MultiTurnEnv` wrapping AsyncLocalEnvironment
│                                       # (top-level package: verifiers resolves by import name)
│
├── evaluation/                         # uv project #3 — eval, lockfile #3, no prime-rl
│   ├── pyproject.toml
│   └── src/evaluation/
│       ├── eval_cli.py                 # `uv run evaluate` entry
│       ├── runner.py                   # SWE-bench-Verified (n=20) + heldout (n=10) driver
│       ├── rollout.py                  # AsyncLocalEnvironment rollout → common.reward.compute_reward
│       ├── sample.py                   # Deterministic random.Random(42) sampler
│       ├── vllm_server.py              # Re-launches vLLM (prefix-caching on, T=0)
│       └── config.py
│
└── infra/
    ├── datagen.Dockerfile              # CPU pod image
    ├── train.Dockerfile                # Training pod (cu128, prime-rl, flash-attn, fastapi template)
    ├── evaluate.Dockerfile             # Eval pod (cu124, vLLM, fastapi template at /opt/repo-cache)
    ├── docker-compose.datagen.yml
    ├── docker-compose.train.yml
    └── docker-compose.evaluate.yml
```

## Running

### Local pre-pod smoke

```bash
cd agent       && uv sync && uv run pytest tests -q   # 2 passed, 1 skipped (docker-gated)
cd ../common   && uv sync && uv run pytest tests -q   # 18 passed
cd ../datagen  && uv sync && uv run pytest tests -q   # 4 passed
cd ../training && uv sync && uv run pytest tests -q   # 22 passed
cd ../evaluation && uv sync && uv run pytest tests -q # 12 passed
```

GPU deps sit behind `[gpu]` extras so tests run on macOS/CPU. Docker-path
tests require `DOCKER_TESTS=1` and a live daemon.

### Running the datagen image

The datagen container needs **two env keys** passed in with `-e` at run
time (neither is baked into the image):

| Env var          | Used for                                                | Required?                                                        |
| ---------------- | ------------------------------------------------------- | ---------------------------------------------------------------- |
| `NEBIUS_API_KEY` | Qwen3 calls for the `lm_modify`/`lm_rewrite` methods    | Yes in mutation mode (the default). Not needed with `--base` or `--offline`. |
| `GITHUB_TOKEN`   | GitHub API for PR enumeration + patch downloads         | Strongly recommended — anonymous search is capped at 10 req/min and rate-limits mid-run. |

Pull the CI-built image and run it; `--repo` is required, and outputs land
under the `/workspace` mount (`pilot.jsonl`, `heldout.jsonl`, `yield.csv`,
plus `sessions/<id>/logs/trace.jsonl` for debugging):

```bash
docker run --rm \
  -v /workspace:/workspace \
  -e NEBIUS_API_KEY="TOKEN" \
  -e GITHUB_TOKEN="TOKEN" \
  ghcr.io/yingliu-data/ml-systems-datagen:latest \
  --repo fastapi/fastapi \
  --t 5 \
  --max-prs 15 \
  --output-root /workspace/datasets/pilot
```

The default mode runs the four SWE-Smith mutation methods with F2P
validation; on first contact with a base commit it builds a cached test
venv under `/workspace/repos/_envs/` (minutes, once per commit). Add
`--base` for plain bug-fix-PR diff extraction — no `NEBIUS_API_KEY`
needed. `docker compose -f infra/docker-compose.datagen.yml up` does the
same and reads both keys from the host environment.

### Running the train image

The train image runs the GRPO loop (prime-rl trainer + colocated vLLM) over a
datagen-produced dataset. It needs a **GPU**, the host `/workspace` mount for
the dataset and checkpoints, and `--ipc=host` plus a large `/dev/shm` for
prime-rl's shared-memory IPC:

| Env var         | Used for                                          | Required?                                                                       |
| --------------- | ------------------------------------------------- | ------------------------------------------------------------------------------- |
| `HF_TOKEN`      | pulling the base policy model + tokenizer from HF | Recommended — avoids HF rate limits and unlocks gated repos.                     |
| `WANDB_API_KEY` | streaming metrics to Weights & Biases             | Optional — the image defaults to `WANDB_MODE=offline`; set `-e WANDB_MODE=online` to use it. |

The single-GPU colocation alias (`CUDA_VISIBLE_DEVICES=0,0`) is baked into the
image, so one physical card is enough. Point `--dataset` at the `pilot.jsonl`
produced by the datagen image; checkpoints land under the `/workspace` mount:

```bash
docker run --rm \
  --gpus all \
  --ipc=host --shm-size=16g \
  -v /workspace:/workspace \
  -e HF_TOKEN="TOKEN" \
  ghcr.io/yingliu-data/ml-systems-train:latest \
  --dataset /workspace/datasets/pilot/pilot.jsonl \
  --profile smoke \
  --output-dir /workspace/checkpoints \
  --sessions-root /workspace/sessions
```

`--profile smoke` is a fast wiring check; `--profile full` runs the real
schedule. Resume an interrupted run with `--resume latest`.
`docker compose -f infra/docker-compose.train.yml up` does the same and adds
the GPU reservation, `HF_HOME`, and offline W&B defaults.

Loop geometry can be retuned per run without rebuilding the image — append
any of `--seq-len`, `--batch-size`, `--rollouts-per-example`, `--max-steps`,
`--max-async-level`, `--max-off-policy-steps` to the `docker run` above
(e.g. `--seq-len 4096 --batch-size 8 --max-steps 3`). They're applied on top
of the baked prime-rl tomls at start-up; cross-file invariants are enforced
automatically (`--seq-len` and `--max-steps` fan out to both train.toml and
orch.toml, `--seq-len` is capped at infer max_model_len, and `--batch-size`
must divide by `--rollouts-per-example`). The patched tomls land in the
session dir for provenance.

### Pod provisioning (RunPod)

`runpodctl`'s quickstart CLI doesn't expose `--privileged`, volume-mount,
or env-var flags, so each pod is built as a template in the web UI. Shared
across all three: 250 GB network volume at `/workspace`; env vars
`NEBIUS_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`.

| Pod        | Hardware                         | Privileged                        | Image                       |
| ---------- | -------------------------------- | --------------------------------- | --------------------------- |
| Datagen    | CPU, 8 vCPU, 32 GB               | **on** (Docker for PR validation) | `infra/datagen.Dockerfile`  |
| Training   | 1× RTX PRO 6000 Blackwell, 96 GB | off (rollouts in-process)         | `infra/train.Dockerfile`    |
| Evaluation | 1× RTX 5090, 24 GB               | off (rollouts in-process)         | `infra/evaluate.Dockerfile` |

**Sync.** Mutagen two-way per pod (sessions: `<PROJECT>-{datagen,train,eval}`)
plus a one-way remote→local session for `/workspace/sessions/` outputs.

**Iteration.** Edit on laptop → Mutagen pushes to active pod → `ssh` in
and run `docker compose -f infra/docker-compose.<stage>.yml up`. Session
dirs under `/workspace/sessions/` stream back for inspection of
`trace.jsonl`, tickets, rollouts.

## Design decisions

**Three-pod hardware split.** See §What runs where. Net saves ~$3.50 on
the eval window vs keeping the Blackwell idle, at the cost of a second
vLLM cold-start and a volume reattach.

**Three separate lockfiles.** `torch≥2.5 + flash-attn-3 + prime-rl`
conflicts with CPU datagen; eval doesn't need prime-rl/flash-attn. Path
deps (`agent/`, `common/`) shared via `[tool.uv.sources]` without
duplication.

**Rollouts run in-process on GPU pods.** Train and eval both use
`agent.AsyncLocalEnvironment` with `shutil.copytree` from a baked-in
template repo (`/opt/repo-cache/<slug>/`) into per-rollout scratch dirs
(`/tmp/<train|eval>-rollouts/<id>/`). No docker-in-docker, no privileged
mode on the GPU pods. Only datagen runs Docker — for the containerised
PR-replay path used by `validator.py`.

**G=4, not G=8** (deliberate departure from `manual_prompts.md`).
7B bf16 (14 GB) + LoRA optim (~0.3) + activations (~10, seq=16384 w/
grad-ckpt) + vLLM KV (~18 at G=4) + vLLM weights (~14) ≈ 56 GB. G=8
requires a second card for prime-rl's trainer/inference split.

**SWE-bench Verified (full, n=20)** rather than Verified Lite (second
deliberate departure from `manual_prompts.md`): the 500-instance set gives
tighter confidence intervals on the 20-instance sample.

**Group-failure semantics.** Prime-rl / verifiers drives G parallel
rollouts and discards whole groups when any rollout is missing — missing
members poison group-relative advantage. `SWEAgentEnv`
(`training/src/swe_agent_env/__init__.py`) propagates rollout exceptions
up to the verifiers orchestrator unchanged.

**Heldout split from datagen.** 10 instances reserved *before* training
via `random.Random(42).sample()` and never written to `pilot.jsonl`.
Split discipline survives the raw-PR fallback; corpus is just smaller.

**Mock-free tests.** `common.reward` is pure-Python so tested directly;
Docker paths are exercised via `DOCKER_TESTS=1` opt-in against a live
daemon. No `MagicMock` — keeps defense-test coverage anchored to real
container behaviour.

## Cost (smoke)

| Item                                 | Cost    |
| ------------------------------------ | ------- |
| Datagen pod (CPU) × 2 hr             | $0.20   |
| Training pod (RTX PRO 6000) × 3.5 hr | $9.45   |
| Eval pod (RTX 5090) × 0.5 hr         | $0.25   |
| Nebius API (~1K calls)               | $1.00   |
| 250 GB volume × 1 mo (prorated)      | $1.40   |
| **Total**                            | **~$12.30** |

## Known issues

- **Training GRPO smoke returns 0 reward across 16 rollouts.** Open;
  likely suspects in priority order:
  1. `SWEAgentEnv` tool-call parsing diverging from what prime-rl's
     `MultiTurnEnv` emits — the agent never successfully calls `evaluate`,
     so `final_head` stays on `base_commit` and defense #4 zeros reward.
  2. `AsyncLocalEnvironment`'s `shutil.copytree` vs prime-rl's rollout
     worker concurrency — races on the same scratch dir clobber each
     other.
  3. Reward arithmetic itself — ruled out: `common/tests/test_reward.py`
     is green on the same inputs.
- **Datagen 0-yield.** All four variance methods gated out by F2P/P2P
  validation on fastapi. Raw-PR fallback works, but the training corpus
  has no synthetic variance — every instance is a real merged PR, so
  memorisation risk is higher than intended.
- **prime-rl `MultiTurnEnv` contract** moves between releases;
  `swe_agent_env/__init__.py` is pinned via `PRIME_RL_REV` in
  `infra/train.Dockerfile`. Highest-risk integration — see training
  failure above.
- **OOM contingency.** If training OOMs from fragmentation: drop
  G=4 → G=2 or seq=16384 → 8192.

## Things to improve

1. **Task cleaning and filtering.** The raw-PR fallback corpus is noisy
   — docs-only PRs, style-only diffs, and PRs whose tests don't exercise
   the fix all survive into `pilot.jsonl`. A filter pass (LLM-judge or
   heuristic on diff shape + test coverage) before training is probably
   the cheapest way to move the 0-reward smoke off zero.
2. **Data generation spawns in a dedicated test-environment container to
   filter.** `datagen/src/datagen/validator.py` runs F2P/P2P via
   `LocalWorkspaceEnvironment` (host subprocess + pytest), so per-candidate
   Python and system deps inherit from the datagen image. A per-candidate
   container pinned to the target repo's own environment (fastapi's CI
   matrix, etc.) would isolate env-mismatch failures from real candidate
   failures — plausibly a large contributor to the 0-yield across all
   four variance methods.
3. **Allow different coding environments.** The whole pipeline is wired
   to fastapi: `/opt/repo-cache/fastapi__fastapi`, fastapi's pytest deps
   baked into the eval venv, `repo_slug` hardcoded. Config-driven
   `test_command`, per-repo template paths, and per-repo pytest plugin
   sets would turn this from a one-repo demo into a SWE-bench-scale
   harness.
4. **Distribute G across multiple GPUs.** Current 1-GPU colocation
   (`CUDA_VISIBLE_DEVICES="0,0"`, G=4) is a stopgap. Prime-rl's intended
   topology is a dedicated trainer/inference split over ≥2 GPUs, which
   also lifts the G=8 memory ceiling and cuts the trainer↔vLLM weight-copy
   stall every step.
5. **Local prod-like environment for pre-build testing.** Every
   Dockerfile or compose change needs a full `docker build` (minutes on
   cu128/flash-attn) before the failure surfaces. A devcontainer or a
   compose overlay with editable path-dep mounts would catch bugs like
   the `/workspace/src/fastapi__fastapi` clone-path issue in
   `evaluate.Dockerfile` in seconds instead of minutes.
