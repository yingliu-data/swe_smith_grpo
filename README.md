# ML Systems Take-Home — Two-Phase RL Training

Build a system that generates synthetic SWE-Smith-style training data, trains
`SWE-bench/SWE-agent-LM-7B` with GRPO+ on that data, and evaluates the
resulting policy. Two pods, three uv projects, shared reward/session
abstractions, $13 smoke budget.

## What runs where

| Pod | Hardware | Phase | Project | Wall-clock |
|---|---|---|---|---|
| A | CPU, 8 vCPU, 32 GB | Data gen | `datagen/` | ~2 hr |
| B | 1× H100 SXM (privileged) | Training + Eval | `training/`, `evaluation/` | ~3.5 hr train + ~25 min eval |

Pod C (L40S-eval) was considered and rejected: same-pod eval saves ~$0.65 and
avoids a second volume re-attach + vLLM cold-start. See §Trade-offs.

## Layout

```
ml_systems/
├── agent/        # path-dep: Environment ABC + Local/Docker implementations
├── common/       # path-dep: reward (6 defenses), tool_surface, session, config
├── datagen/      # uv project #1 — lockfile #1 (CPU)
├── training/     # uv project #2 — lockfile #2 (H100, prime-rl)
├── evaluation/   # uv project #3 — lockfile #3 (H100, no prime-rl)
└── infra/        # Dockerfiles, compose files, runpod-notes, supervisor
```

Three lockfiles because `torch>=2.5 + flash-attn-3 + prime-rl` conflicts with
CPU datagen, and eval doesn't need prime-rl/flash-attn. Path deps share
`agent/` and `common/` without duplication.

## Reward defenses (6-layer)

Split into **structural** (impossible to breach without breaking infra) and
**per-eval gates** (checked at rollout finalise). The structural layer makes
the eval gates' failure modes infrastructurally unreachable, which is how we
reduced v3's 9-defense proposal to 6.

| # | Type | Defense | Enforced in |
|---|---|---|---|
| 1 | Structural | Read-only test mounts + edit-file test-glob allowlist | `agent/docker_env.py` |
| 2 | Structural | Rollout container `--network none` | `agent/docker_env.py` |
| 3 | Structural | Step (20) / token (16K) / wall (120s) budgets | Container + caller |
| 4 | Per-eval | Base-commit `git rev-parse` before ≡ after → else reward=0 | `common/reward.py` |
| 5 | Per-eval | `git apply --check` on final diff → else reward=0 | `common/reward.py` |
| 6 | Per-eval | F2P ∧ ¬P2P-regression → reward=1 | `common/reward.py` |

Each per-eval defense has a failing-fixture unit test in
`training/tests/test_reward_defenses.py`.

## GRPO correctness — group failure

Prime-rl / verifiers drives G parallel rollouts and handles group-failure
semantics internally; missing rollouts poison group-relative advantage, so
the whole group gets discarded rather than fire-and-forget. Our
`SWEAgentEnv` (in `training/src/swe_agent_env/__init__.py`) propagates
rollout exceptions up to the verifiers orchestrator unchanged.

## Run

### Local (pre-RunPod smoke)

```bash
cd agent       && uv sync && uv run pytest tests -q   # 2 passed, 1 skipped (docker-gated)
cd ../common   && uv sync && uv run pytest tests -q   # 18 passed
cd ../datagen  && uv sync && uv run pytest tests -q   # 4 passed
cd ../training && uv sync && uv run pytest tests -q   # 22 passed
cd ../evaluation && uv sync && uv run pytest tests -q # 12 passed
```

(GPU deps are behind `[gpu]` extras so tests run on macOS/CPU. Docker-path
tests require `DOCKER_TESTS=1` and a live daemon.)

> `_old/` contains the previous iteration scratch; it's gitignored and can be
> removed with `rm -rf _old/` once you've confirmed the new tree works.

### Pod provisioning & sync (RunPod)

**First-time setup — web UI.** `runpodctl`'s quickstart CLI doesn't expose
`--privileged`, volume-mount, docker-socket bind-mount, or env-var flags, so
the first pod has to be built as a template in the web UI. Required settings:

- Privileged mode **on** (required for docker-in-docker during rollouts)
- Bind-mount host `/var/run/docker.sock` (avoids nested DinD — simpler + faster)
- 500 GB network volume mounted at `/workspace`
- Env vars: `NEBIUS_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`
- Image: Pod A uses `infra/datagen.Dockerfile`; Pod B uses `infra/train.Dockerfile`

**Subsequent pods — CLI.** Once the template exists, iterate with
`runpodctl pod create --template-id <id>` → `runpodctl ssh info $POD_ID` for
the SSH string. Full flags in `infra/runpod-notes.md`.

**Sync (code):** Mutagen two-way (session: `<PROJECT_NAME>`)
**Sync (outputs):** Mutagen one-way remote→local (session: `<PROJECT_NAME>-output`)
Changes made locally or remotely are automatically synced.

**Test loop.** Edit on laptop → Mutagen pushes to pod → `ssh` in and run
`docker compose -f infra/docker-compose.<datagen|train|evaluate>.yml up` →
session dirs under `/workspace/sessions/` stream back to the laptop for local
inspection of `trace.jsonl`, tickets, and rollouts.

### Pod A — datagen

```bash
docker compose -f infra/docker-compose.datagen.yml up datagen
# → /workspace/datasets/pilot/{pilot.jsonl,heldout.jsonl,harbor/}
# → /workspace/docker-cache/*.tar.gz
```

### Pod B — train + eval

```bash
docker load < /workspace/docker-cache/*.tar.gz
docker compose -f infra/docker-compose.train.yml up train
# ...kill-test at step 12 per infra/runpod-notes.md...
docker compose -f infra/docker-compose.train.yml down
docker compose -f infra/docker-compose.evaluate.yml up evaluate
# → /workspace/sessions/eval-*/logs/{summary.json,results.jsonl}
```

## Cost (smoke)

| Item | Cost |
|---|---|
| Pod A (CPU) × 2 hr | $0.20 |
| Pod B (H100) × 3.9 hr | $10.49 |
| Nebius API (~1K calls) | $1.00 |
| 500 GB volume × 1 mo | $1.40 |
| **Total** | **~$13.10** (buffer ~$6.90 in $20 envelope) |

Full profile (max_steps=150, G=8, 2× H100) documented in plan, not built:
~$70.

## Discrepancies with `manual_prompts.md`

Two deliberate departures, with reasoning:

1. **G=4 (not G=8)**: GPU-memory fit on 1× H100. 7B bf16 (14 GB) + LoRA
   optimiser (~0.3 GB) + activations (~10 GB, seq=16384 w/ grad-ckpt) + vLLM KV
   cache (~18 GB at G=4) + weights in vLLM (~14 GB) ≈ 56 GB. G=8 would need
   2× H100 for the prime-rl trainer/inference split, which the full profile
   specifies.
2. **SWE-bench Verified (full, sample 20)** rather than Verified Lite: the full
   500-instance set gives tighter confidence intervals on the 20-instance
   sample.

## Trade-offs

- **One pod for train+eval** saves ~$0.65 and a second vLLM cold-start, at the
  cost of coupling train to eval config drift. Mitigated by re-launching vLLM
  with distinct configs (prefix caching + greedy for eval).
- **Mock-free tests**: `common.reward` is pure-Python so tested directly; the
  Docker paths are exercised via `DOCKER_TESTS=1` opt-in because they need a
  live daemon. No `MagicMock` in the test suite — that was what let us keep
  defense-test coverage high without drifting from real container behaviour.
- **Heldout split from datagen**: 10 of the 40–80 generated instances are
  reserved *before* training via `random.Random(42).sample()` and never appear
  in `pilot.jsonl`. README below has "no leakage" assertion.

## Cut-list if strictly time-boxed to 5 hr

| Drop | Saves | Loses |
|---|---|---|
| Three lockfiles → one workspace | 30 min | CPU/GPU dep isolation |
| `common/session/*` → `shared.py` | 1 hr | Ticket audit, manifest verify |
| Watchdog + supervisor | 30 min | Stall recovery |
| Heldout eval half, SWE-bench only | 30 min | In-dist sanity signal |

## Known issues

- **prime-rl MultiTurnEnv contract** moves between releases; our
  `swe_env.py` is pinned to a specific prime-rl commit in
  `training/pyproject.toml`. This is the highest-risk integration.
- **Nebius seed** is best-effort (provider load balancing); `yield.csv` will
  vary ±1–2 candidates per method per run.
- **OOM contingency**: if the first run OOMs from fragmentation, drop G=4→G=2
  or seq=16384→8192. Plan documents this.

## What I would do with more time

- 2× H100 full profile with proper trainer/inference split (prime-rl
  orchestrator over two nodes).
- Batched SWE-Smith corpus hydration from the 50K HF dataset; current plan
  skips it because the ~295 GB of per-repo images doesn't fit the $20
  envelope.
- Hydra-style config composition for `train.toml` / `orch.toml` / `infer.toml`
  so sweep experiments don't require three separate file edits.
- Reward function v2: LLM-judge of "did the agent do the minimal thing"
  (shorter diffs → higher reward) to combat reward hacking via complexity
  bloat. Requires its own defense suite.
