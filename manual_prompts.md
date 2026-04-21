# 1st prompt, plan + auto-edit
I need to build a infrastructure based on the @DESCRIPTION.md. There are two phases. One phase is to build a SWE-smith synthetic data generation pipeline. The second phas is to build a Agentic Training pipeline.  Read it and construct a plan. Here is my design:
┌──────────────────────────────┼──────────────────────────────┐
│ RunPod                                                     │
│                                                             │
│  ═══════════════════════════════════════════════════════   │
│  PHASE 1: Data Generation  (~2 hrs, sequential)              │
│  ═══════════════════════════════════════════════════════   │
│  ┌──────────────────────────────────────────────────┐      │
│  │ Pod A: CPU-only  ($0.10/hr × 2hr = $0.20)         │      │
│  │                                                    │      │
│  │  data-gen container:                              │      │
│  │   • SWE-smith CLI                                 │      │
│  │   • OpenRouter client (calls Qwen3-30B)          │      │
│  │   • Docker via /var/run/docker.sock              │      │
│  │   • pilot_gen.py (logs yield.csv)                │      │
│  └─────────────────────┬────────────────────────────┘      │
│                        │ writes                              │
│                        ▼                                     │
│                ┌───────────────────────┐                     │
│                │ Network Volume        │                     │
│                │ swe-workspace (500 GB)│                     │
│                │ /workspace/           │                     │
│                │  ├─ datasets/         │                     │
│                │  │   ├─ pilot/        │  ← 40–80 instances  │
│                │  │   └─ yield.csv     │                     │
│                │  ├─ hf-cache/         │                     │
│                │  ├─ docker-cache/     │  ← saved .tar.gz    │
│                │  ├─ models/           │                     │
│                │  └─ logs/             │                     │
│                └───────────────────────┘                     │
│                        ▲                                     │
│                        │                                     │
│                 [stop Pod A]                                 │
│                 [Volume detaches,                            │
│                  keeps all data]                             │
│                        │                                     │
│                 [deploy Pod B]                               │
│                 [attach same Volume]                         │
│                        │                                     │
│                        ▼                                     │
│  ═══════════════════════════════════════════════════════   │
│  PHASE 2: Training Smoke-Test  (0.5 hr = $1.35)              │
│  PHASE 3: Training Full        (3 hr  = $8.07)               │
│  PHASE 4: Eval                 (1 hr  = $2.69)               │
│  ═══════════════════════════════════════════════════════   │
│  ┌──────────────────────────────────────────────────┐      │
│  │ Pod B: 1× H100 SXM                               │      │
│  │                                                    │      │
│  │  rl-train container:                              │      │
│  │   • prime-rl (trainer + orchestrator + inference) │      │
│  │   • vLLM serving SWE-bench/SWE-agent-LM-7B        │      │
│  │   • Reads /workspace/datasets/pilot/              │      │
│  │   • docker load < /workspace/docker-cache/*.tar   │      │
│  │   • GRPO+ amalgam: 30 steps × 16 prompts × G=4    │      │
│  │   • Checkpoints → /workspace/checkpoints/         │      │
│  │   • W&B offline → /workspace/logs/                │      │
│  └──────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
1. use the existing 50K SWE-smith dataset on HF and generate small amount of data from scratch (tiny pilot generation with 10 instances from customised repo which is an argument). synthetic data generation from Mutation-based fully-synthetic method SWE-Smith, with LM Modify, LM Rewrite, Procedural Modification and PR Mirror (Invert PRs) from the same PR. yield T (argements) samples from each method, log yield rate. 120-second timeout per candidate.
    1.1 Use Qwen/Qwen3-30B-A3B-Instruct-2507 model API call as the LLM in the data generation.
    ```import os
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY")
        )

        response = client.chat.completions.create(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            messages=[
                {
                    "role": "system",
                    "content": """SYSTEM_PROMPT"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """USER_MESSAGE"""
                        }
                    ]
                }
            ]
        )

        print(response.to_json())
    ```
2. save the SWE-Smith output in SWE-bench-style JSONL dataset. Then write a harbor schema copy for the training purpose
3. Build Agentic Training pipeline loop to run the training process using  RL algorithm. Learn from the @.claude/skills/second-brain/PROJECT_STRUCTURE.md about what the pipeline architecture is like. NOTE: agent/ and session/ folder strucuture within the second-brain project. Do not inheritate the whole structure from second-brain, ONLY take small parts like orchestration and multi-agent. DO NOT allow flexibal tool-calling, use exisiting tools from @agent/environment.py. DO NOT inheritate the full memeory system, ONLY keep the Procedural Memory.  Use @agent/ abstract classes as function template.  There are something to pay attention to: detailed log for session.
4. for training, use model `SWE-bench/SWE-agent-LM-7B`, use H100 SXM gpu in runpod,  spin-off docker containers. GRPO+ RL run (1 epoch, G=8, ~1K prompts). Add a buffer : the ability to kill a broken run at hour 1 of training and restart. convert SWE-bench-style JSONL dataset to Harbor style described in @DESCRIPTION.md before feed into the training loop
5. Eval on SWE-bench Verified Lite (50 instances), log Pass@1 number, F2P, P2P and other values you see fit.
6. Add procedure to prevent reward hacking
7.  Use uv system, control system with packages-lock for  SWE-Smith data generation pipeline, agent-loop pipeline and trining pipeline.
8. Fix random seed in all initialisation system. add to data config.
9. Use Async framework for all pipelines.

# Plans

> **Revision history**
> - v1 → v2: fixed 10 real issues (engineering-vs-compute-time conflation, prime-rl API misuse, single shared lock, DinD unaddressed, brittle reward defenses, infeasible 50K-HF training corpus).
> - v2 → v3: three-pod layout (CPU datagen / H100 training / L40S eval), moved `reward.py` + `tool_surface.py` into `common/` to avoid duplication, dropped GGUF conversion step, all nine reward defenses retained, full `common/` session infrastructure retained, three separate uv lockfiles retained.
> - v3 → v4: **(1)** dropped Pod C — eval now runs on Pod B at end of training, against a mixed 30-instance sample (20 SWE-bench Verified for cross-repo signal + 10 held-out from datagen for in-distribution sanity). User originally proposed 5 fastapi-only instances; pushed back because N=5 is statistically meaningless and same-distribution-as-training measures memorization rather than capability — the user's quoted prompt was about Phase 1's ≥5 *generated tasks*, which is already satisfied by datagen's 40–80 output. **(2)** Reward defenses reduced from 9 to 6: kept structural (#1–3) and per-eval gates (#4–6); dropped audit defenses (#7 SHA-256 snapshot, #8 patch entropy, #9 suspicious-API scan) as belt-and-suspenders given that #1–3 make their failure modes structurally impossible. **(3)** Kill-at-stall test moved to step 12 (between checkpoints at step 10) with explicit assertion that post-resume rollout hashes for steps 11/12 equal pre-kill hashes — proves seed determinism, not just resume mechanics.

---

## Context

Build two pipelines for a coding-model RL training system, meeting DESCRIPTION.md's minimums and `manual_prompts.md`:

- **Phase 1 — SWE-Smith synthetic data generation** (Pod A, CPU-only). Fresh pilot of ~40–80 instances from a configurable target repo (default `fastapi/fastapi`, satisfying DESCRIPTION.md's FastAPI bug-fix requirement and the ≥5-tasks bar with significant headroom). The 50K HF SWE-smith dataset is **not** used as a training corpus (295 GB of Docker images needed; impossible at this budget); it's referenced only in README §Scaling.
- **Phase 2 — Agentic RL training + evaluation** (Pod B, 1× H100 SXM, privileged). GRPO+ on `SWE-bench/SWE-agent-LM-7B` driven by **prime-rl**, vLLM for inference, rollouts executed inside per-task Docker containers. At end of training, an evaluation container runs on the same pod against the trained checkpoint. Eval shares `DockerEnvironment`, `tool_surface`, and `reward` with training via `common/`.

### Why two pods instead of three (v3 → v4)

Originally Pod C (L40S 48 GB) ran eval separately. Cost analysis killed it:

- **Eval is short.** 30 instances × ~80 s wall-time ≈ 40 min of useful work.
- **Setup overhead is fixed.** Docker image hydration (~3 min) + vLLM warmup (~2 min) is the same on either pod. On Pod C this is amortized across only ~40 min of work; on Pod B it's amortized across hours of training plus eval.
- **Cost.** Pod C runs ~$1.19, but Pod B's incremental ~25 min for eval costs ~$0.55. Saves ~$0.65 *and* eliminates a pod-management step (volume detach/reattach, image re-hydration, separate compose file).
- **Eval reuses training-time vLLM.** Same checkpoint, same model, same KV cache configuration — only `temperature` and prefix-caching settings differ. Re-launching vLLM for eval on Pod B with eval-specific sampling params takes ~2 min vs ~7 min cold-start on Pod C.

### Honest scope and budget

DESCRIPTION.md's **"≤5 hours"** is **engineering time**, not compute wall-clock. This plan targets a smoke-test-scale training run that demonstrates the full architecture (rollout loop, reward shaping, GRPO update, checkpoint, resume, eval):

| Profile | Steps | Batch | G | Total rollouts | H100-hrs | Cost |
|---|---|---|---|---|---|---|
| **smoke** (default) | 30 | 16 | 4 | 1,920 | ~3 (train) + ~0.4 (eval) | ~$10.50 |
| full (documented, not built) | 150 | 32 | 8 | 38,400 | ~25 + ~0.4 | ~$70 |

A CLI flag `--profile {smoke,full}` flips between the two; README documents that `full` is a configuration switch, not a reimplementation.

**Total smoke-profile cost** (incl. API + storage): **~$13.10** — fits the `manual_prompts.md` $20 envelope with ~$7 of buffer.

### Stubs present

At `/Users/sophia/Library/Mobile Documents/com~apple~CloudDocs/Work/application/2026/Matt Johnson/ml_systems/`:

- `agent/environment.py` — abstract `Environment` ABC (step, read_file, edit_file, delete_file, evaluate; 120 s timeout default)
- `agent/models.py` — `TaskSpec`, `ToolCall`, `ToolResult`, `EvaluationResult`, `StepResult`
- `agent/__init__.py` — **has dangling `LocalWorkspaceEnvironment` import** (must implement)
- `DESCRIPTION.md`, `manual_prompts.md`

The directory also contains prior-iteration directories (`data_pipeline/`, `training/`, `session/`, `reward_hacking/`, `util/`, `eval/`, `infra/`, `configs/`, `knowledge_base/`, existing `pyproject.toml`, `uv.lock`, `README.md`, `.venv/`). **Step 0 of execution**: `mv` these to `_old/`, rebuild from scratch, delete `_old/` at the end. The `agent/` ABC stubs are the only required-preserved artifacts.

### Reuse from `/Users/sophia/Local Projects/SecondBrain/agent-api/app/`

- `asyncio.gather(..., return_exceptions=True)` + `Semaphore` — the *shape* of the wave-based pattern, adapted (GRPO requires all-succeed-or-discard-group semantics, not fire-and-forget).
- `Ticket.start`/`finish` append-only audit — `session/ticket.py:36-84`
- `SessionDir` layout — `session/session_dir.py:59-180`
- Forward-only state machine — `session/state.py:11-30`
- `atomic_write_json`, `read_json_once` — `util/ipc.py`
- `MemoryStore` procedural-facts pattern — `user/memory.py`

**Do NOT inherit**: full skills registry, device-tool routing, LLM-in-the-loop planning, SQLite sessions, multi-session concurrent queue.

---

## Target directory layout

**Three separate uv projects** (req #7 taken literally; also technically required — their dep trees conflict):

```
ml_systems/                                # repo root (NOT a uv workspace)
├── DESCRIPTION.md                         # provided
├── manual_prompts.md                      # provided
├── README.md                              # design, run instructions, trade-offs, DinD note, scaling
├── .gitignore
│
├── agent/                                 # shared abstracts + env impls (path dep)
│   ├── pyproject.toml                     # [project] name = "agent"
│   ├── src/agent/
│   │   ├── __init__.py                    # fix dangling LocalWorkspaceEnvironment import
│   │   ├── environment.py                 # provided ABC, UNCHANGED
│   │   ├── models.py                      # provided dataclasses, UNCHANGED
│   │   ├── local_env.py                   # LocalWorkspaceEnvironment (datagen validation; sync subprocess OK — no event loop)
│   │   └── docker_env.py                  # DockerEnvironment (aiodocker streaming exec; used by training + eval)
│   └── tests/
│
├── common/                                # shared infra + algorithm code (path dep)
│   ├── pyproject.toml                     # [project] name = "common"
│   ├── src/common/
│   │   ├── __init__.py
│   │   ├── config.py                      # SEED=42, paths, apply_seed()
│   │   ├── ids.py                         # make_session_id, make_ticket_id, safe_key
│   │   ├── ipc.py                         # atomic_write_json, read_json_once, await_ipc_file
│   │   ├── logging.py                     # structured trace.jsonl writer
│   │   ├── reward.py                      # 6-layer reward defenses (shared by training + eval)
│   │   ├── tool_surface.py                # fixed 4 tools (read_file, edit_file, delete_file, run_tests)
│   │   ├── session/
│   │   │   ├── session_dir.py             # workspace, tickets, logs/trace.jsonl, ipc, checkpoints/
│   │   │   ├── ticket.py                  # Ticket.start/finish append-only
│   │   │   ├── manifest.py                # compute/verify file manifest (sha256)
│   │   │   └── state.py                   # forward-only state: active → complete | failed | escalated
│   │   └── memory/
│   │       └── procedural.py              # MemoryStore (run-scoped facts, YAML frontmatter + markdown body)
│   └── tests/
│
├── datagen/                               # Phase 1 — standalone uv project
│   ├── pyproject.toml                     # CPU-only deps + path sources for agent, common
│   ├── uv.lock                            # ← lockfile #1
│   ├── .python-version                    # 3.12
│   ├── src/datagen/
│   │   ├── __init__.py
│   │   ├── config.py                      # T=15, TIMEOUT=120, NEBIUS_MODEL
│   │   ├── nebius_client.py               # async Qwen3-30B-A3B-Instruct-2507 wrapper
│   │   ├── repo_manager.py                # git clone, checkout commit, PR enumeration via GitHub API
│   │   ├── methods/
│   │   │   ├── __init__.py                # register_method decorator (narrow surface, no plugins)
│   │   │   ├── base.py                    # BaseMutationMethod ABC
│   │   │   ├── lm_modify.py               # LM subtly breaks a fix
│   │   │   ├── lm_rewrite.py              # LM rewrites source file with injected bug
│   │   │   ├── procedural.py              # AST-level deterministic mutations
│   │   │   └── pr_mirror.py               # invert merged PR diff → synthetic bug
│   │   ├── validator.py                   # LocalWorkspaceEnvironment wrapper, 120 s timeout
│   │   ├── yield_logger.py                # yield.csv writer
│   │   ├── writers/
│   │   │   ├── swebench_jsonl.py          # primary output
│   │   │   └── harbor_dir.py              # derived from JSONL
│   │   └── pipeline.py                    # async orchestrator (PR × method × T → validate → write)
│   ├── pilot_gen.py                       # CLI entry: uv run datagen
│   └── tests/
│       ├── test_methods.py
│       ├── test_validator.py
│       └── test_writers.py
│
├── training/                              # Phase 2 — standalone uv project
│   ├── pyproject.toml                     # CUDA 12 + torch + vllm + prime-rl + verifiers + aiodocker
│   ├── uv.lock                            # ← lockfile #2
│   ├── .python-version                    # 3.12
│   ├── src/training/
│   │   ├── __init__.py
│   │   ├── config.py                      # smoke/full profiles, LLM_SEM=8, DOCKER_SEM=4
│   │   ├── schema/harbor.py               # JSONL → Harbor converter for prompts
│   │   ├── swe_env.py                     # verifiers.MultiTurnEnv wrapping DockerEnvironment
│   │   ├── session_logger.py              # per-rollout / per-update / per-ckpt Tickets
│   │   ├── checkpoint.py                  # save/load + Manifest verify
│   │   ├── watchdog.py                    # heartbeat stall detector (asyncio task)
│   │   └── configs/
│   │       ├── train.toml                 # prime-rl trainer config
│   │       ├── orch.toml                  # prime-rl orchestrator config
│   │       └── infer.toml                 # prime-rl inference (vLLM) config
│   ├── train.py                           # CLI entry: thin wrapper around uv run rl + SessionDir
│   └── tests/
│       ├── test_rollout.py                # fixed tool surface enforced
│       ├── test_reward_defenses.py        # one fixture per defense (imports common.reward)
│       ├── test_checkpoint.py             # save→corrupt→verify detects; resume round-trip; rollout-hash equality
│       └── test_orchestrator.py           # G parallel, semaphore cap, group-failure contract
│
├── evaluation/                            # Phase 3 — standalone uv project (deployed on Pod B)
│   ├── pyproject.toml                     # torch + vllm + aiodocker + datasets; NO prime-rl, NO flash-attn
│   ├── uv.lock                            # ← lockfile #3
│   ├── .python-version                    # 3.12
│   ├── src/evaluation/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── vllm_server.py                 # launch local vLLM server with trained checkpoint
│   │   ├── sample.py                      # mixed sample: 20 SWE-bench Verified + 10 held-out datagen
│   │   ├── rollout.py                     # uses common.tool_surface + agent.docker_env
│   │   ├── metrics.py                     # Pass@1, F2P, P2P, wall-time, tokens, tool-calls; split by source
│   │   └── runner.py                      # async eval loop, Semaphore(4) on Docker
│   ├── eval_cli.py                        # CLI entry: uv run evaluate
│   └── tests/
│
└── infra/
    ├── datagen.Dockerfile                 # CPU, git, python 3.12, uv
    ├── rollout.Dockerfile                 # per-task image base (pytest + pip deps pre-baked; network-none at run)
    ├── train.Dockerfile                   # CUDA 12 + uv + prime-rl + vllm; privileged + docker.sock mount
    ├── evaluate.Dockerfile                # CUDA 12 + uv + vllm + aiodocker; privileged + docker.sock mount
    ├── docker-compose.datagen.yml
    ├── docker-compose.train.yml           # vllm + train services on Pod B
    ├── docker-compose.evaluate.yml        # vllm + evaluate services on Pod B (run after train completes)
    └── runpod-notes.md                    # DinD requirement, privileged flag, volume mounts, H100 sizing
```

---

## Component details

### Phase 1 — Pod A: datagen on CPU

**Flow** (async, `Semaphore(8)` cap on Nebius, `Semaphore(4)` on local pytest):

1. Clone `fastapi/fastapi` to `/workspace/repos/fastapi__fastapi/`; checkout pinned base commit.
2. Enumerate merged PRs labeled `bug`/`fix` via GitHub API → ground-truth `(repo, base, patch, test_patch, F2P, P2P)`.
3. For each PR × 4 methods × `T=15` candidates:
   - **LM methods** (`lm_modify`, `lm_rewrite`): call Qwen3-30B-A3B-Instruct-2507 via Nebius
   - **Procedural**: AST-level deterministic mutations (invert condition, swap args, negate return)
   - **PR mirror**: invert the merged diff
4. Validate each candidate via `LocalWorkspaceEnvironment.evaluate()` (120 s timeout):
   - Apply candidate patch
   - Run `FAIL_TO_PASS` tests → expected FAIL
   - Run `PASS_TO_PASS` tests → expected PASS
   - Reverse candidate, apply reference patch → expected PASS on F2P
5. Emit passing candidates as SWE-bench JSONL: `datagen/output/pilot.jsonl`.
6. For each passing instance, also write Harbor directory: `datagen/output/harbor/<instance_id>/`:
   - `instruction.md` ← `problem_statement`
   - `test.sh` ← shell script running F2P + P2P
   - `reference.diff` ← `patch`
   - `task.json` ← `{repository, base_commit, test_command, metadata}`
7. **Held-out split for evaluation**: `random.Random(42).sample(passing_instances, 10)` is reserved as eval-only and NOT written to `pilot.jsonl`. Stored separately in `datagen/output/heldout.jsonl`. README documents this split explicitly so reviewers can verify no leakage.
8. Append row to `yield.csv` per method: `{method, repo, attempted, passed, rate, avg_seconds}`.
9. Populate `common.memory.procedural.MemoryStore` with per-instance env-setup hints (e.g. `pip install anyio==3.5`) discovered during validation.
10. `docker save` each repo image to `/workspace/docker-cache/*.tar.gz` (~3 GB compressed each) for Pod B.

**Nebius call** (exact shape per user spec):

```python
client = openai.AsyncOpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
)
resp = await client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": USER_MESSAGE}]},
    ],
    seed=SEED,
)
```

**Caveat**: Nebius `seed=` provides best-effort determinism but does NOT guarantee bit-exact reproducibility across calls — provider load-balancing across replicas can produce different completions even with seed set. README documents this; `yield.csv` is expected to vary by ±1–2 candidates per method per run.

**Output layout on Network Volume**:

```
/workspace/datasets/pilot/
├── pilot.jsonl                     # SWE-bench format, training-eligible instances
├── heldout.jsonl                   # 10 instances reserved for eval (in-distribution)
├── yield.csv                       # method, repo, attempted, passed, rate, avg_seconds
├── setup_facts/                    # MemoryStore output (one .md file per instance)
└── harbor/<instance_id>/
    ├── instruction.md
    ├── test.sh
    ├── reference.diff
    └── task.json
/workspace/docker-cache/
├── fastapi-fastapi_abc123.tar.gz
└── ...
```

---

### Phase 2 — Pod B: agentic training + evaluation on H100 (privileged)

**Runtime**: `docker run --privileged -v /var/run/docker.sock:/var/run/docker.sock -v /workspace:/workspace ...` — Pod B's container has Docker access to spawn per-task rollout containers.

**Pod B execution sequence**:

1. `docker compose -f infra/docker-compose.train.yml up vllm train` → smoke profile runs 30 GRPO steps (~3 hr).
2. On training completion, `docker compose down` for the train stack.
3. `docker compose -f infra/docker-compose.evaluate.yml up vllm evaluate` → re-launches vLLM with eval sampling params (temperature=0, prefix-caching enabled), runs 30-instance eval (~25 min).
4. Stop Pod B; volume persists with checkpoints + eval results.

**Integration with prime-rl via verifiers.MultiTurnEnv**:

```python
# training/src/training/swe_env.py
from verifiers import MultiTurnEnv
from agent.docker_env import DockerEnvironment
from common.tool_surface import FIXED_TOOL_DEFS, dispatch, parse_tool_call
from common.reward import compute_reward

class SWEAgentEnv(MultiTurnEnv):
    """prime-rl calls reset(task) to load a prompt, then step(model_output) after each LLM turn.
    On terminal, evaluate() → reward via common.reward.compute_reward."""

    def __init__(self, docker_sem, llm_sem):
        self.docker_sem = docker_sem
        self.llm_sem = llm_sem
        self.dockenv: DockerEnvironment | None = None

    async def reset(self, task):
        self.dockenv = DockerEnvironment(workspace_root=..., task=task)
        await self.dockenv.prepare()
        return self._render_prompt(task)

    async def step(self, model_output):
        tool_call = parse_tool_call(model_output)  # fixed surface enforced here
        async with self.docker_sem:
            obs = await dispatch(tool_call, self.dockenv)
        done = self._should_terminate(tool_call, obs)
        info = {}
        if done:
            result = await self.dockenv.evaluate()
            reward_result = compute_reward(
                task=self.task,
                trajectory=self._trace,
                container=self.dockenv.container,
                rollout_trace=self._trace,
            )
            info.update(reward=reward_result.reward, defense_log=reward_result.defense_log)
        return obs, 0.0, done, info

    async def close(self):
        await self.dockenv.teardown()
```

**Risk note**: prime-rl's `MultiTurnEnv` contract has been moving across releases. `training/pyproject.toml` pins a specific prime-rl commit; the `swe_env.py` interface is verified against that exact version before declaring this component done. This is the highest-risk integration in the project.

**Group-level rollout failure handling** (GRPO-correct, not fire-and-forget):

```python
rollouts = await asyncio.gather(
    *[run_one_rollout(prompt, i, llm_sem, docker_sem) for i in range(G)],
    return_exceptions=True,
)
good = [r for r in rollouts if not isinstance(r, Exception)]
if len(good) < G:
    # Missing rollouts poison group-relative advantage; DISCARD group.
    trace.log("rollout.group_discarded", n_failed=G - len(good))
    raise GroupFailure(prompt_id, dropped=G - len(good))
return good
```

**Invocation** (what `train.py` runs):

```bash
uv run rl \
  --trainer @training/src/training/configs/train.toml \
  --orchestrator @training/src/training/configs/orch.toml \
  --inference @training/src/training/configs/infer.toml \
  --env training.swe_env:SWEAgentEnv \
  --dataset /workspace/datasets/pilot/pilot.jsonl \
  --output-dir /workspace/checkpoints/run_${run_id}
```

`train.py` is a thin CLI wrapper: applies seed, creates `SessionDir`, writes a `train.run` Ticket with config hashes, starts prime-rl as subprocess, streams its stdout into `trace.jsonl`, runs the watchdog.

**Two decoupled semaphores**:

- `LLM_SEM = asyncio.Semaphore(8)` — 8 concurrent `chat/completions` into vLLM. vLLM's continuous batching handles the GPU side.
- `DOCKER_SEM = asyncio.Semaphore(4)` — 4 concurrent `docker exec pytest`. Protects host CPU/RAM (single H100 pod has ~16 vCPU).

**Blocking-exec prevention**: `DockerEnvironment` MUST use `aiodocker.Container.exec(..., stream=True)` iterating via `async for chunk`; wall-clock via `asyncio.wait_for(..., timeout=120)`. Host subprocesses (`git apply`, diff checks) use `asyncio.create_subprocess_exec`. A unit test launches a 5 s sleep inside a container and asserts a concurrent `asyncio.sleep(0.1)` tick counter advances.

**GPU memory fit on 1× H100 (80 GB)** — LoRA required:

- `SWE-agent-LM-7B` bf16 weights ≈ 14 GB
- LoRA adapter (r=16, α=32, all q/k/v/o + gate/up/down projections) → ~40M trainable params → optimizer state ~0.3 GB
- Activations with `gradient_checkpointing=true`, `max_seq_len=16384`, batch=16 (microbatch=1 + grad accum) ≈ 8–12 GB
- vLLM inference co-located: `gpu_memory_utilization=0.45`, `max_model_len=16384`, KV cache ~18 GB across G=4 rollouts
- Total: 14 + 0.3 + 10 + 18 + 20 (headroom) ≈ **62 GB → fits with 18 GB margin**

**Co-location risk**: training allocator and vLLM allocator can fragment GPU memory in practice. Contingency: drop G=4 → G=2 *or* `max_seq_len=16384` → `8192` if first run OOMs. Documented in README §Known-issues.

**Full fine-tune is forbidden on 1× H100 without FSDP**. `full` profile in README recommends 2× H100 with trainer-GPU / inference-GPU split. Not built in this deliverable.

**`train.toml` (prime-rl trainer)**:

```toml
[model]
name = "SWE-bench/SWE-agent-LM-7B"
dtype = "bfloat16"

[model.peft]
enabled = true
type = "lora"
r = 16
alpha = 32
target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

[train]
profile = "smoke"
gradient_checkpointing = true
learning_rate = 1e-6
lr_scheduler = "constant"
grad_clip = 0.05
optimizer = "adamw"
micro_batch_size = 1
batch_size = 16
max_steps = 30
epochs = 1

[loss]
loss_type = "grpo"
clip_low = 0.20                       # DAPO Clip-Higher
clip_high = 0.28
kl_coef = 0.0                         # DAPO / DeepSWE
entropy_coef = 0.0
advantage = "leave_one_out"           # RLOO
length_normalize = "constant"         # Dr.GRPO
max_length_constant = 16384
importance_sampling_level = "token"
std_normalize = false                 # Dr.GRPO
compact_filter = true                 # mask overlong/timeout trajectories

[ckpt]
interval = 10
keep_last = 3

[monitor.wandb]
project = "ml-systems-takehome"
mode = "offline"
```

**`infer.toml` (vLLM inference for training)**:

```toml
[model]
name = "SWE-bench/SWE-agent-LM-7B"
dtype = "bfloat16"
max_model_len = 16384
enforce_eager = false

[server]
port = 8000
host = "0.0.0.0"

[engine]
gpu_memory_utilization = 0.45
tensor_parallel_size = 1
enable_prefix_caching = false         # required for seed determinism during training
seed = 42
```

For eval, vLLM is re-launched with `enable_prefix_caching = true` and `temperature = 0.0`; determinism via greedy decoding rather than seed-controlled sampling.

**Session logging per rollout**:

```
sessions/train-20260420T143022Z/
├── logs/
│   └── trace.jsonl                   # every event
├── tickets/
│   ├── tk_001_train_run.json
│   ├── tk_002_grpo_step_0.json
│   ├── tk_003_rollout_p0_r0.json
│   ├── ...
│   └── tk_NNN_train_checkpoint_step_30.json
├── workspace/
│   └── rollouts/<prompt_id>/<rollout_id>/
│       ├── trajectory.jsonl          # every tool_call + result
│       ├── final_patch.diff
│       ├── eval_output.txt
│       └── reward.json               # full defense_log from common.reward
└── ipc/
    └── heartbeat.json                # watchdog input
```

---

### Phase 3 — Evaluation (Pod B, after training)

**Sample composition** (30 instances total):

- **20 instances**: seeded sample from `princeton-nlp/SWE-bench_Verified` for cross-repo generalization signal.
  ```python
  ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
  all_ids = sorted(ds["instance_id"])
  swebench_ids = random.Random(42).sample(all_ids, 20)
  ```
- **10 instances**: `heldout.jsonl` from datagen (in-distribution sanity check). These are passing candidates from the datagen pipeline that were *withheld* from `pilot.jsonl` — the model has not seen them during training.

Metrics are reported separately per source so a high in-distribution score with low cross-repo score (memorization without generalization) is visible rather than averaged away.

**Flow**:

1. `apply_seed(42)`; create `SessionDir.create(kind="eval")`.
2. Load both samples; merge into `eval_set: list[TaskSpec]` with `source` field annotated.
3. Re-launch vLLM on the trained checkpoint with eval sampling params:
   ```bash
   uv run python -m vllm.entrypoints.openai.api_server \
     --model /workspace/checkpoints/run_${run_id}/final \
     --dtype bfloat16 \
     --max-model-len 16384 \
     --gpu-memory-utilization 0.85 \
     --enable-prefix-caching \
     --seed 42 \
     --port 8000 &
   ```
4. For each instance (with `Semaphore(4)` on Docker, `Semaphore(8)` on vLLM):
   - Spawn per-task container via `agent.docker_env.DockerEnvironment` (same class as training)
   - Run rollout against local vLLM server (reuses `common.tool_surface`)
   - Evaluate via `common.reward.compute_reward` (same 6 defenses as training)
5. Aggregate metrics → `/workspace/logs/eval/summary.json`:
   ```json
   {
     "checkpoint": "run_20260420T143022Z",
     "total": {
       "n_instances": 30,
       "pass_at_1": 0.30,
       "f2p_rate": 0.33,
       "p2p_rate": 0.96,
       "mean_wall_seconds": 82.4,
       "mean_tokens": 3214,
       "mean_tool_calls": 11.2
     },
     "by_source": {
       "swebench_verified": {
         "n": 20, "pass_at_1": 0.25, "f2p_rate": 0.30, "p2p_rate": 0.95
       },
       "datagen_heldout": {
         "n": 10, "pass_at_1": 0.40, "f2p_rate": 0.40, "p2p_rate": 0.98
       }
     },
     "failures_by_category": {
       "timeout": 5, "bad_patch": 12, "syntax": 2, "test_regression": 1
     }
   }
   ```
6. Per-instance detail in `/workspace/logs/eval/results.jsonl`.

**Cross-pipeline code reuse**: evaluation imports `agent.docker_env`, `common.tool_surface`, `common.reward`, `common.session`, `common.config`. **Zero duplication** of rollout or reward logic between training and eval. The only training-only code is `training/swe_env.py` (prime-rl integration) and `training/configs/*.toml` (GRPO hyperparameters).

---

## Cross-cutting concerns

### uv: three separate projects, three separate locks (req #7)

| Project | Deps summary | Hardware target |
|---|---|---|
| `datagen` | `openai`, `aiodocker`, `gitpython`, `unidiff` | CPU (Pod A) |
| `training` | `torch>=2.5`, `vllm`, `prime-rl`, `flash-attn-3`, `aiodocker`, `verifiers` | H100 Hopper (Pod B) |
| `evaluation` | `torch>=2.5`, `vllm`, `aiodocker`, `datasets` (no prime-rl, no flash-attn) | H100 Hopper (Pod B, post-training) |

**Justification** (beyond literal requirement):
- Training vs datagen have **conflicting deps**: `torch>=2.5`, `flash-attn-3` (wheels only for CUDA 12 + SM90), `prime-rl` (pins an unreleased `transformers` commit). datagen runs on CPU; these are useless and wheel-unavailable on macOS.
- A single workspace lock would force one conflict-resolved tree that's either fat-for-CPU or broken-for-GPU.
- `agent/` + `common/` shared via `[tool.uv.sources]` path entries — single source of truth, three locks.
- Evaluation kept as third disjoint set even though it deploys on Pod B: it doesn't need `prime-rl` or `flash-attn`, and isolating it means the eval CLI can be re-run later against any checkpoint without dragging in the training stack.

### Fixed random seed (req #8)

`common/config.py:apply_seed(seed=42)` called at every CLI entry and rollout spawn:

- `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- `torch.use_deterministic_algorithms(True)` (warn if CUDA op non-deterministic)
- vLLM `SamplingParams(seed=SEED, temperature=1.0)` for training; `temperature=0.0` (greedy) for eval
- vLLM `enable_prefix_caching=false` for training (determinism); `=true` for eval (greedy decoding makes prefix caching deterministic)
- DataLoader `generator=torch.Generator().manual_seed(SEED)`
- Rollout container name suffix `{seed}_{rollout_idx}` for reproducible naming
- Seed value written into every Ticket's `inputs_hash` preimage

### Async framework (req #9)

- `asyncio` only. No threads except unavoidable library-internal ones.
- HTTP: `httpx.AsyncClient` with `Limits(max_connections=16)`, `tenacity` retry with exponential backoff on 429/5xx.
- Docker: **`aiodocker` MANDATORY**. No `docker-py` sync `exec_run`, no `subprocess.run` inside `DockerEnvironment`. Streaming exec via `async for`, wall-clock via `asyncio.wait_for(timeout=120)`.
- Host-side commands: `asyncio.create_subprocess_exec`.
- File I/O: sync for atomic JSON (sub-ms); `aiofiles` only for high-throughput trajectory appends.
- CI test `training/tests/test_docker_env.py::test_exec_does_not_block_loop` asserts concurrent `asyncio.sleep(0.1)` tick counter advances while a 5 s container exec runs.

**Concurrency caps** (documented per pipeline):

| Pipeline | LLM | Docker | Notes |
|---|---|---|---|
| datagen | Nebius Semaphore(8) | Pytest Semaphore(4) | CPU-bound validators |
| training | vLLM Semaphore(8) | Docker Semaphore(4) | H100 batches; host has ~16 vCPU |
| evaluation | vLLM Semaphore(8) | Docker Semaphore(4) | Same Pod B host |

### Reward-hacking prevention (req #6) — 6 layered defenses in `common/reward.py`

Reduced from 9 to 6 in v4. The dropped audit defenses (#7 SHA-256 snapshot, #8 patch entropy, #9 suspicious-API scan) are belt-and-suspenders given that the structural defenses (#1–3) make their failure modes infrastructurally impossible — if defense #1 (read-only mounts) is breached, that's a hard infra bug requiring the run to be aborted, not a reward signal to be denied. README explicitly notes this trade-off and lists what was cut.

**Primary: structural (infrastructure-level)**

1. **Read-only test mounts**: test directories mounted `:ro` in the container. `edit_file` path-allowlist in `DockerEnvironment` rejects any path matching the test glob → `ToolResult(ok=False, error="read-only")`.
2. **Network isolation**: rollout container `--network none`. All deps pre-baked into `rollout.Dockerfile`.
3. **Step / token / wall-clock budgets**: 20 tool calls / 16K context / 120 s per rollout.

**Per-eval gates**

4. **Base-commit verification**: `git rev-parse HEAD` before + after; drift ⇒ reward=0.
5. **Diff-applies-cleanly**: `git apply --check` on final diff; failure ⇒ reward=0.
6. **F2P ∧ ¬P2P-regression**: reward=1 only if every F2P passes AND every P2P still passes. **This is the reward function itself.**

**Each defense has a unit test with a failing fixture**:

```python
# training/tests/test_reward_defenses.py (imports common.reward)

def test_defense_1_test_file_edit_rejected():
    """edit_file targeting tests/ returns ok=False without modifying disk."""
    ...

def test_defense_4_base_commit_drift():
    """Agent resets repo to a different commit → reward=0."""
    ...

def test_defense_6_p2p_regression_zeros_reward():
    """All F2P pass but a P2P now fails → reward=0."""
    ...
```

### Full `common/` session infrastructure (req-adjacent, preserved)

Complete infrastructure for per-operation auditing:

- `SessionDir.create(kind)` → directory with `workspace/`, `tickets/`, `logs/trace.jsonl`, `ipc/`, `checkpoints/`
- `Ticket.start(op, inputs)` / `Ticket.finish(outputs, state)` — append-only, `Manifest` binds ticket to produced files via sha256
- Forward-only state machine: `active → complete | failed | escalated`
- `MemoryStore` (run-scoped facts only, YAML frontmatter + markdown body) for per-instance env-setup hints

**Why preserved in this pilot**: per-rollout debugging. Every `reward=0` outcome is traceable from the per-rollout Ticket → to the trajectory → to the tool calls → to the LLM output → to the prompt → to the group → to the gradient step. This is real debugging value for RL training, not ceremony.

### Kill-at-stall + restart (req #4)

- **Checkpoint cadence**: every 10 GRPO updates → `/workspace/checkpoints/run_<id>/step_<N>/{adapter.safetensors (LoRA), optimizer.pt, rng.pt, meta.json}`. `Manifest.compute()` stored in `train.checkpoint` Ticket.
- **Resume**: `train.py --resume latest` scans tickets reverse-chronologically for `train.checkpoint && state=complete`, runs `Manifest.verify()`, reloads via prime-rl's `--resume-from`. RNG state restored → deterministic continuation.
- **Watchdog**: asyncio task inside `train.py` polls `sessions/<run_id>/ipc/heartbeat.json` written every 30 s by prime-rl's trainer. Stale > 10 min ⇒ watchdog sends `SIGTERM` to prime-rl subprocess; `train.py` exits with code 42 ("stall-detected").
- **Supervisor**: `infra/train.Dockerfile`'s `CMD` runs `infra/run_train.sh`:
  ```bash
  #!/usr/bin/env bash
  set -u
  RETRIES=${RETRIES:-5}
  COOLDOWN=${COOLDOWN:-30}
  for i in $(seq 1 $RETRIES); do
      uv run train --resume latest --profile ${PROFILE:-smoke}
      code=$?
      [ $code -eq 0 ] && exit 0
      echo "[supervisor] train exited $code (attempt $i/$RETRIES); sleeping $COOLDOWN s" >&2
      sleep $COOLDOWN
  done
  echo "[supervisor] exhausted $RETRIES retries" >&2
  exit 1
  ```
  No systemd, no manual operator. Pod B's container restart-policy is `on-failure` as a second safety net for OOM kills.
- **Retention**: keep N-10 checkpoint as safety net; prune older than 2 back.

**Kill test scenario** (v3 → v4 update):
1. Run smoke training to step 12 (one checkpoint already written at step 10).
2. `docker kill ml-train` mid-rollout in step 12.
3. Capture pre-kill rollout hashes for steps 11 and 12 from `trace.jsonl` events `rollout.complete` (hash = sha256 of trajectory.jsonl).
4. Supervisor restarts; `train.py --resume latest` loads step 10 checkpoint.
5. Re-run reaches step 12; assert post-resume rollout hashes for steps 11/12 **byte-equal** the pre-kill hashes.

This is the test that catches RNG-state bugs (e.g., forgetting to restore `numpy.random` state, or vLLM's sampling state drifting). Pure resume-from-disk-without-hash-equality is necessary but insufficient — it only proves the file loaded, not that determinism survived the round-trip.

### Pod B Docker-in-Docker requirement

H100 pods on RunPod are themselves containers; they have no outer Docker daemon by default. Rollouts need to spawn per-task containers.

**Deployment requirement**: deploy with either
- (a) `--privileged` + Docker installed inside (DinD), or
- (b) host `/var/run/docker.sock` bind-mounted into the pod (requires `--privileged`).

`infra/train.Dockerfile` and `infra/evaluate.Dockerfile` are authored assuming the host-docker.sock pattern (simpler, no nested daemon); fall back to DinD if the mount isn't present. `infra/runpod-notes.md` documents the exact RunPod pod-create flags. README's Run Instructions section calls this out explicitly.

---

## Deliverables (per DESCRIPTION.md)

1. **`README.md`** — design decisions, trade-offs ($13 smoke vs $70 full run), DinD requirements on Pod B, scaling-to-50K discussion (image-management needed), v3 → v4 changes, "what I'd do with more time" list.
2. **Task dataset** — `datagen/output/pilot.jsonl` + `datagen/output/heldout.jsonl` + `datagen/output/harbor/` (≥5 Harbor directories; all training-eligible passing instances promoted; README highlights 5 exemplars per DESCRIPTION.md minimum).
3. **Runnable code** via three CLI entry points, one per pipeline:
   - `cd datagen && uv run datagen --repo fastapi/fastapi --t 15`
   - `cd training && uv run train --dataset ../datagen/output/pilot.jsonl --profile smoke`
   - `cd evaluation && uv run evaluate --checkpoint /workspace/checkpoints/run_<id>/final --swebench-n 20 --heldout ../datagen/output/heldout.jsonl`
4. **Zipped repo or private GH link** at submission.

---

## Verification

**Local laptop (before RunPod)** — each pipeline resolves independently:

```bash
cd datagen    && uv sync && uv run pytest && uv run datagen --dry-run --offline  # 12 candidates, stubs for Nebius
cd ../training && uv sync && uv run pytest                                       # GPU deps no-op on macOS; tests mock aiodocker
cd ../evaluation && uv sync && uv run pytest && uv run evaluate --dry-run        # 3 instances stubbed
```

**Pod A (CPU, datagen)**:
1. Attach 500 GB volume at `/workspace/`.
2. `docker compose -f infra/docker-compose.datagen.yml up` → 40–80 instances into `/workspace/datasets/pilot/` (10 reserved as heldout); `docker save` → `/workspace/docker-cache/*.tar.gz`.
3. Stop Pod A; volume persists.

**Pod B (H100 SXM, privileged, training + evaluation)**:
1. Attach volume; verify Docker works (`docker ps` from pod shell).
2. `docker load < /workspace/docker-cache/*.tar.gz` (~3 min).
3. `docker compose -f infra/docker-compose.train.yml up vllm train` → smoke profile runs 30 GRPO steps; checkpoints land in `/workspace/checkpoints/run_<id>/`.
4. **Simulated kill at step 12** (~1.2 hr in): `docker kill ml-train` → supervisor restarts `train.py --resume latest` → resumes from step 10 checkpoint → reaches step 12. Assert rollout-hash equality for steps 11 and 12 (proves seed determinism, not just resume mechanics).
5. On training completion: `docker compose -f infra/docker-compose.train.yml down`.
6. `docker compose -f infra/docker-compose.evaluate.yml up vllm evaluate` → 30-instance eval (~25 min). Results in `/workspace/logs/eval/summary.json` with per-source breakdown.
7. Stop Pod B; volume persists.

**Engineering budget** (per DESCRIPTION.md ≤ 5 hr — honest estimate with full scope):

| Task | Time |
|---|---|
| `agent/` path-dep package + `local_env` + `docker_env` | 45 min |
| `common/` full infra (session, ticket, manifest, state, ids, ipc, logging, memory, reward, tool_surface) | 1 h 15 min |
| `datagen/` project + 4 methods + validator + writers + pipeline + heldout split + CLI + tests | 2 h |
| `training/` project + swe_env + configs + watchdog + checkpoint + CLI + tests (incl. hash-equality resume test) | 2 h |
| `evaluation/` project + vllm_server + mixed-sample loader + runner + per-source metrics + CLI + tests | 1 h |
| `infra/` Dockerfiles + compose files + runpod-notes | 30 min |
| README + local smoke test | 1 h |
| **Total** | **~8.5 hours** |

This is **~3.5 hours over the 5-hour target** (down from ~4 in v3). The extra is genuine production-grade structure (ticket-level audit, three-lock isolation, layered reward defenses, DinD documentation). README §Trade-offs acknowledges:

> *"Scope exceeds the 5-hour target by ~3.5 hours; the additions are structural engineering (ticket-level audit, per-lockfile isolation, layered reward defenses) judged valuable for demonstrating production readiness. I can trim if strictly time-boxed — the likely cuts would be: collapse three lockfiles to one (-30 min), reduce `common/session/` to a 30-line `shared.py` (-1 hr), drop watchdog + supervisor (-30 min), drop the in-distribution heldout half of eval and run only SWE-bench Verified (-30 min). All four cuts would bring this to ~5 hours and sacrifice audit depth / iso / debuggability / generalization signal."*

---

## Cost summary

| Item | Cost |
|---|---|
| Pod A (CPU) × 2 hr | $0.20 |
| Pod B (H100) × 3.9 hr (3.5 train + 0.4 eval) | $10.49 |
| Nebius API (Qwen3-30B, ~1K calls) | $1.00 |
| Network Volume (500 GB × 1 month) | $1.40 |
| **Subtotal** | **$13.09** |
| Buffer for re-runs / debugging | ~$7 of $20 envelope |

Under the `manual_prompts.md` $20 envelope with ~$7 unused for buffer.

Full-profile run (documented, not built): ~$70 (Pod A 2 hr + Pod B ~25 hr training + 0.4 hr eval + API + storage).


# 2nd 
Check @'/Users/sophia/Library/Mobile Documents/com~apple~CloudDocs/Work/application/2026/Matt Johnson/ml_systems project'. I need to work on the training part which uses GRPO + AMALGAM to train vLLM serving SWE-bench/SWE-agent-LM-7B. when G = 4, i should have 4 roll out docker container, each container has the correct functioning environment (for example, fastapi python environement). It also should have a sequential agentic environement such as 
```
import json, asyncio, time, logging
from datetime import datetime
from app.config import MAX_TOOLS, SYSTEM_PROMPT, TOOL_TIMEOUT
from app.agent.sanitize import sanitize

logger = logging.getLogger("agent")

# Shared state for tool result handoff (iPhone → agent loop)
tool_result_events: dict[str, asyncio.Event] = {}
tool_results: dict[str, str] = {}

SSE_HEADERS = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-store, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

# Avatar tools that emit a separate SSE event for the frontend renderer.
_AVATAR_TOOLS = {"set_pose", "move_joints", "animate_sequence", "plan_movement"}


async def run_agent_loop(message: str, history: list, registry, llm):
    """Generator that yields SSE events.
    - registry: routes tool calls to the right skill
    - llm: LLMProvider instance (local, cloud, or fallback chain)"""

    system = SYSTEM_PROMPT.format(current_time=datetime.now().isoformat())
    history.append({"role": "user", "content": message})
    messages = [{"role": "system", "content": system}] + history[-20:]

    # Get tool definitions filtered by query relevance (not all tools)
    tool_defs = registry.get_tools_for_query(message)
    server_tools = registry.get_server_tool_names()
    device_tools = registry.get_device_tool_names()

    for loop_idx in range(MAX_TOOLS):
        # Call LLM via provider abstraction (local → cloud fallback)
        try:
            resp = await llm.chat_completion(messages, tools=tool_defs or None)
        except Exception as e:
            logger.error(f"LLM unavailable after retries: {e}")
            error_msg = "I'm temporarily unable to respond — the server may be restarting. Please try again in a moment."
            history.append({"role": "assistant", "content": error_msg})
            yield f"event: token\ndata: {json.dumps({'text': error_msg})}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        choice = resp["choices"][0]
        assistant_msg = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        # ── Tool calls ──────────────────────────────────
        if finish_reason == "tool_calls" and assistant_msg.get("tool_calls"):
            messages.append(assistant_msg)

            for tc in assistant_msg["tool_calls"]:
                tool_name = tc["function"]["name"]
                try:
                    arguments = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}
                tc_id = tc.get("id", f"tc_{int(time.time()*1000)}")

                logger.info(f"Tool call: {tool_name}({arguments}) [loop {loop_idx+1}]")

                if tool_name in server_tools:
                    # Server skill — execute via registry
                    result = await registry.execute_server_tool(tool_name, arguments)

                    # Emit avatar commands as a separate SSE event so the
                    # frontend can act on them before the LLM text response.
                    if tool_name in _AVATAR_TOOLS:
                        try:
                            parsed = json.loads(result)
                            yield f"event: avatar_command\ndata: {json.dumps({'name': tool_name, 'result': parsed})}\n\n"
                            # For plan_movement, feed compact summary to LLM
                            # instead of full frame data to save context tokens.
                            if tool_name == "plan_movement":
                                n_frames = len(parsed.get("frames", []))
                                result = json.dumps({
                                    "status": "ok",
                                    "frames_generated": n_frames,
                                    "loop": parsed.get("loop", False),
                                })
                        except (json.JSONDecodeError, TypeError):
                            pass
                elif tool_name in device_tools:
                    # Device skill — delegate to iPhone via SSE
                    yield f"event: tool_call\ndata: {json.dumps({'id': tc_id, 'name': tool_name, 'arguments': arguments})}\n\n"

                    evt = asyncio.Event()
                    tool_result_events[tc_id] = evt
                    try:
                        await asyncio.wait_for(evt.wait(), timeout=TOOL_TIMEOUT)
                        result = tool_results.pop(tc_id, "No result")
                    except asyncio.TimeoutError:
                        result = f"Error: {tool_name} timed out after {TOOL_TIMEOUT}s. The iPhone may be unreachable."
                    finally:
                        tool_result_events.pop(tc_id, None)
                else:
                    all_tools = server_tools | device_tools
                    result = f"Error: Unknown tool '{tool_name}'. Available: {', '.join(all_tools)}"

                result = sanitize(result)
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
            continue

        # ── Final text response ─────────────────────────
        text = assistant_msg.get("content", "")
        history.append({"role": "assistant", "content": text})
        for word in text.split(" "):
            yield f"event: token\ndata: {json.dumps({'text': word + ' '})}\n\n"
        yield "event: done\ndata: {}\n\n"
        return
    else:
        # Tool loop exhausted without a final text response
        logger.warning(f"Agent loop exhausted {MAX_TOOLS} tool iterations without final response")
        fallback = "I'm sorry, I wasn't able to complete that request. Please try again."
        history.append({"role": "assistant", "content": fallback})
        yield f"event: token\ndata: {json.dumps({'text': fallback})}\n\n"
        yield "event: done\ndata: {}\n\n"
```
Use only tools defined here @'/Users/sophia/Library/Mobile Documents/com~apple~CloudDocs/Work/application/2026/Matt Johnson/ml_systems/agent/src/agent/environment.py'. It also has maximum iterations defined in configuration. the evaluation should happend in the same roll-out environment. 
Check if this project did that correctly.


# Things could be improved:
1. task cleaning and filtering. 
2. data generation spawn in different test environment container to filter.
3. allow different coding environment
4. the G=n in GRPO can be distributed in multiple GPUs.
5. Local prod environment to test before building the docker image
