# Architecture

Three-pipeline RL training system: **(1) datagen** synthesizes SWE-bench-style training
tasks from a target repo's bug-fix PRs, **(2) training** runs GRPO on
`SWE-bench/SWE-agent-LM-7B` over those tasks via prime-rl, and **(3) evaluation** scores
the resulting policy on SWE-bench Verified plus a held-out split of the synthetic data.
Each pipeline is its own `uv` project and Docker image; all three share two path-dep
libraries (`agent/`, `common/`) and hand artifacts to each other through a `/workspace`
volume.

*Reflects the `staging` branch as of 2026-07-03.*

```
            ┌─────────────────────────────────────────────────────────────────┐
            │                        /workspace volume                        │
            │  datasets/pilot/{pilot,heldout}.jsonl   checkpoints/   sessions/│
            └─────▲──────────────────────▲─────────────────▲─────────────────┘
                  │ writes               │ reads pilot     │ reads heldout + ckpt
   ┌──────────────┴───┐      ┌───────────┴──────────┐      ┌┴──────────────────┐
   │ 1. datagen (CPU) │      │ 2. training (1 GPU)  │      │ 3. evaluation(GPU)│
   │ PR mining +      │      │ prime-rl trainer +   │      │ vLLM (greedy) +   │
   │ 4 mutation       │      │ vLLM colocated,      │      │ 30-instance mixed │
   │ methods + F2P    │      │ GRPO G=4, LoRA r16   │      │ eval, per-source  │
   │ validation       │      │ rollouts in-process  │      │ pass rates        │
   └──────────────────┘      └──────────────────────┘      └───────────────────┘
        shared libs:  agent/ (Environment impls)   common/ (reward, tools, session)
```

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Shared library: `agent/`](#shared-library-agent)
3. [Shared library: `common/`](#shared-library-common)
4. [Pipeline 1: `datagen/`](#pipeline-1-datagen)
5. [Pipeline 2: `training/`](#pipeline-2-training)
6. [Pipeline 3: `evaluation/`](#pipeline-3-evaluation)
7. [Workflow traces, function by function](#workflow-traces)
8. [Infrastructure: images, compose, CI](#infrastructure)
9. [Cross-cutting invariants](#cross-cutting-invariants)
10. [Known quirks and dead code](#known-quirks)

---

## Repository layout

```
swe_smith_grpo/
├── agent/                    # path-dep lib: Environment ABC + 3 implementations
│   └── src/agent/            #   models, environment, local_env, async_local_env, docker_env
├── common/                   # path-dep lib: reward math, tool surface, session infra
│   └── src/common/           #   reward, tool_surface, config, ids, ipc, logging, session/
├── datagen/                  # uv project 1 — synthetic task generation (CPU pod)
│   └── src/datagen/          #   pilot_gen (CLI), pipeline, validator, methods/, writers/
├── training/                 # uv project 2 — GRPO wrapper around prime-rl (GPU pod)
│   └── src/
│       ├── training/         #   train (CLI), config, checkpoint, watchdog, configs/*.toml
│       └── swe_agent_env/    #   verifiers MultiTurnEnv plugin (top-level by design)
├── evaluation/               # uv project 3 — post-training eval (GPU pod)
│   └── src/evaluation/       #   eval_cli, runner, rollout, sample, vllm_server, config
├── infra/                    # 3 Dockerfiles + 3 compose files
└── .github/workflows/        # build.yml — image build+push on main/staging
```

Dependency direction: `datagen`, `training`, `evaluation` each depend on `agent` and
`common` as **non-editable path sources** (`[tool.uv.sources]`). `common` depends on
`agent`. Every project resolving the shared libs must agree on editability — uv 0.11
rejects mixed editable/non-editable references to the same path (see
[invariants](#cross-cutting-invariants)).

---

## Shared library: `agent/`

The environment abstraction every pipeline drives rollouts and validation through.

### `agent/src/agent/models.py`
Value types (`@dataclass(slots=True)`) shared across the system:

| Type | Fields | Role |
|---|---|---|
| `TaskSpec` | `repository, base_commit, instruction, test_command, reference_patch, issue_url, fix_commit, metadata` | The task definition every Environment consumes. Built in `datagen/validator.py`, `swe_agent_env._task_from_row`, `evaluation/rollout.py`. |
| `ToolCall` | `name, arguments` | A parsed model tool invocation (from `common.tool_surface.parse_tool_call`). |
| `ToolResult` | `name, ok, output, error, exit_code, path` | Result of one tool step. |
| `EvaluationResult` | `reward, passed, output, exit_code` | Returned by every `evaluate()`. |

### `agent/src/agent/environment.py`
- **`Environment(ABC)`** — `__init__(workspace_root, command_timeout_seconds=120)`;
  abstract `step`, `read_file`, `edit_file`, `delete_file`, `evaluate`. Base of all
  three implementations below.

### `agent/src/agent/async_local_env.py` — training + eval rollouts
- **`AsyncLocalEnvironment(Environment)`** — async, subprocess-backed, no Docker.
  Not concurrency-safe by itself; callers serialize (training) or give each instance
  its own dir (eval).
  - `prepare(*, test_patch="")` — rmtree + copytree from `template_path` (ignoring
    `.venv`/`__pycache__`), `git checkout -f base_commit`, `git clean -fdx`, apply
    `test_patch`. **Requires the template to be a full git clone** (historical base
    commits must be checkoutable offline).
  - `step(tool_call)` — name-dispatch to the four tools; unknown tool → failed result.
  - `read_file(path)` — path-containment check (`_resolve_inside`) then read.
  - `edit_file(path, patch)` / `delete_file(path)` — **reject test-glob paths**
    (reward defense #1: agent cannot edit or delete tests), else `git apply` / unlink.
  - `evaluate()` — runs `task.test_command`; reward 1.0/0.0 by exit code.
  - `current_head()` / `current_diff()` / `git_apply_check(diff)` — git plumbing used
    by the reward path (drift detection, diff extraction, applicability check).
  - `_run` / `_run_stdin` — subprocess exec with `asyncio.wait_for` wall-clock timeout
    (reward defense #3); `[TIMEOUT]` sentinel on expiry.

### `agent/src/agent/local_env.py` — datagen validation
- **`LocalWorkspaceEnvironment(Environment)`** — synchronous sibling used by datagen's
  `Validator` (runs inside a thread executor, no event loop).
  - `reset(task)` — checkout base commit + `git clean -fdx`.
  - `apply_patch_text(patch)` / `reverse_patch_text(patch)` — forward/reverse
    `git apply`; the A/B/C validation stages are built on these.
  - `evaluate()` — runs `task.test_command` with subprocess timeout.
  - Note: no test-glob edit block here (not needed — datagen constructs patches itself).

### `agent/src/agent/docker_env.py` — containerized variant (tests only)
- **`DockerEnvironment(Environment)`** — aiodocker-backed per-task container with
  `NetworkMode:none` (defense #2), memory/pids limits, tmpfs repo. Same 7-method async
  surface as `AsyncLocalEnvironment`. Currently exercised only by tests
  (`test_docker_env.py`, `test_reward_defenses.py`); production rollouts run in-process.

---

## Shared library: `common/`

### `common/src/common/config.py`
- `SEED = 42` — the project-wide seed.
- `workspace_root()` / `sessions_root()` — `$ML_SYSTEMS_WORKSPACE` or `/workspace`.
- `apply_seed(seed=SEED)` — seeds PYTHONHASHSEED, `random`, numpy, torch (+CUDA,
  deterministic algorithms), tolerating missing imports. Called at the top of all
  three CLIs.

### `common/src/common/reward.py` — the reward function
- **`DefenseEvent(defense, passed, detail)`** — one audit entry.
- **`RewardResult(reward, passed, defense_log)`**.
- **`compute_reward(*, initial_head, final_head, apply_check_ok, f2p_results,
  p2p_results, structural_log=None)`** — layered gates, short-circuiting to 0:
  - **#4 drift**: `final_head != initial_head` (agent moved the base commit) → 0.
  - **#5 applicability**: final diff doesn't `git apply --check` → 0.
  - **#6 tests**: all F2P pass AND no P2P regression → 1.0, else 0. Empty
    `f2p_results` fails safe (0).
  - Defenses #1–#3 (test-file immutability, network isolation, wall-clock timeout)
    are structural — enforced by the Environments, logged via `structural_log`.
  - Binary reward by design: GRPO group-relative advantage supplies the gradient
    signal; there is no partial credit.

### `common/src/common/tool_surface.py` — the fixed 4-tool surface
- **`FIXED_TOOL_DEFS`** — exactly four OpenAI-style schemas: `read_file`, `edit_file`,
  `delete_file`, `evaluate`.
- **`parse_tool_call(model_output)`** — accepts OpenAI JSON (`{"name", "arguments"}`,
  including `function.name` shapes with stringified args) or the SWE-agent pipe form
  `TOOL|<name>|<json>`; validates against `VALID_TOOL_NAMES`; raises
  `ToolSurfaceError` otherwise.
- **`dispatch(tool_call, env)`** — final gate on tool name, then `await env.step(...)`.
  Used by both training (`swe_agent_env`) and eval (`rollout.py`).

### `common/src/common/session/` — session-dir infrastructure
- **`session_dir.SessionDir`** — canonical on-disk layout: `workspace/ tickets/ logs/
  ipc/ checkpoints/ memory/` under `sessions/<kind>-<utc-timestamp>/`.
  `create(kind=...)` mints it; `open(root)` reopens for resume. `trace_path` =
  `logs/trace.jsonl`; `heartbeat_path` = `ipc/heartbeat.json` (watchdog).
- **`ticket.Ticket`** — durable per-operation JSON record; `start(...)` /
  `finish(outputs, state, manifest, error)` with atomic writes.
- **`state.transition(current, target)`** — ticket FSM: `active → complete | failed |
  escalated`; raises `InvalidTransition` otherwise.
- **`manifest.Manifest`** — sha256 file manifests; `compute(root, files)` /
  `verify(root)` — checkpoint integrity checking.
- **`memory.MemoryStore` / `MemoryRecord`** — markdown+frontmatter setup-fact store
  (datagen writes one record per passing instance).

### Small modules
- `ids.py` — `make_session_id(kind)`, `make_ticket_id(seq, op)`, `safe_key`.
- `ipc.py` — `atomic_write_json` (temp file + fsync + `os.replace`).
- `logging.py` — **`TraceLogger`**: lock-guarded JSONL appender
  (`{"ts", "event", **fields}`), the structured log used by all pipelines.

---

## Pipeline 1: `datagen/`

Console entry: `datagen = "datagen.pilot_gen:main"`.

### `config.py`
- `NEBIUS_BASE_URL`, `NEBIUS_MODEL` (`Qwen/Qwen3-30B-A3B-Instruct-2507`).
- **`DatagenConfig`** — all knobs: `repo` (required), `t_per_method=5`, `base=False`,
  `max_prs=15`, `heldout_count=5`, `llm_concurrency=8`, `docker_concurrency=4`,
  `validation_timeout_seconds=120`, `seed=42`, path roots
  (`output_root=/workspace/datasets/pilot`, `repos_root=/workspace/repos`,
  `sessions_root=/workspace/sessions`), `methods=("lm_modify","lm_rewrite",
  "procedural","pr_mirror")`, `offline`, `dry_run`. `from_env()` variant reads
  `DATAGEN_*` env vars (not used by the CLI path).

### `pilot_gen.py` — CLI entry
- `_say(msg)` — stderr breadcrumb (`[datagen] ...`).
- `_parse_args()` — flags mirror `DatagenConfig` (defaults come from a template
  instance so CLI and config can't drift).
- `_run(cfg)` — seed → `SessionDir.create(kind="datagen")` → `TraceLogger` →
  dry-run short-circuit → strip `lm_*` methods if `--offline` → `Pipeline.run()`.
- `main()` — argparse → `asyncio.run(_run)` → `SystemExit(code)`; uncaught exceptions
  printed with traceback.

### `pipeline.py` — orchestrator
- **`Pipeline.__init__(cfg, *, trace)`** — builds `_llm_sem` (8), `_docker_sem` (4),
  `RepoManager`, `Validator`.
- **`Pipeline.run()`** — the whole flow (see [trace](#trace-1--datagen-run)).
- **`_fetch_patches(prs)`** — concurrent patch downloads under a local `Semaphore(4)`;
  a failed fetch skips only that PR.
- **`_generate_validate_write(...)`** — one (method, trial) task: generate a candidate
  (LLM methods under `_llm_sem`) → validate under `_docker_sem` → on pass, build an
  `InstanceRecord` + `MemoryRecord`, append to the shared `passing` list.
- `_build_problem_statement(pr, cand)` — PR title/body + regression instruction.

### `validator.py` — the quality gate
- **`extract_f2p_nodeids(test_patch)`** — heuristic extraction of pytest node-ids for
  newly-added `test_*` functions (tracks enclosing classes via hunk section headers
  and indentation; drops methods whose class is unresolvable).
- **`split_reference_patch(patch)`** — splits a SWE-bench diff into
  `(src_patch, test_patch)` by `_is_test_path`; returns `("","")` for unparseable or
  binary-bearing patches (traps all parse exceptions — a bad PR is a skip, not a crash).
- **`Validator.validate(...)`** — async facade; offloads to a thread executor.
- **`Validator._validate_sync(...)`** — three-stage check in a tempdir copy of the
  clone, running pytest with the per-commit test venv's python
  (`[python_bin, -B, -m, pytest, -x, --tb=short, -p, no:cacheprovider, *f2p]`):
  - **Stage A**: apply `test_patch` (the F2P tests must exist).
  - **Stage B**: apply the candidate `buggy_patch` → F2P must **FAIL** (bug bites).
  - **Stage C**: reverse buggy, apply reference `src_patch` → F2P must **PASS**
    (reference fix repairs it). `passed = B ∧ C`.

### `repo_manager.py` — GitHub + git
- **`PullRequestInfo`** — number, base/merge commit, title, body, labels, patch_url.
- **`RepoManager`** — `ensure_clone` (blobless `--filter=blob:none` into
  `repos_root/<owner>__<name>`), `checkout`, `list_bug_prs` (GitHub search for merged
  PRs labeled `bug`/`fix`, hydrated per-PR), `fetch_patch` (raw `.patch`).
  Auth from `GITHUB_TOKEN` (unauthenticated search rate-limits mid-run).

### `test_env.py` — per-commit test venvs
- **`ensure_test_env(repo_dir, envs_root, key)`** — builds/caches a venv per
  (repo, base-commit) under `/workspace/repos/_envs/<key>/`: `pip install .`,
  pytest, best-effort test extras and `requirements-test*.txt`; `.ready` marker;
  returns the venv's python (used by the Validator, never the datagen venv itself).

### `methods/` — the four mutation methods
All implement **`BaseMutationMethod.generate(ctx: Context) -> Candidate | None`**
(`base.py` defines `Context` — repo/PR/patch/seed/trial — and `Candidate` —
`buggy_patch` + rationale).

| Method | File | Strategy |
|---|---|---|
| `lm_modify` | `lm.py` | LLM makes a subtle 1–3 line bug in a file the reference patch touched; `<edit><old/><new/></edit>` output parsed, unique-replacement applied, `ast.parse`-checked, emitted as a unified diff. |
| `lm_rewrite` | `lm.py` | Same machinery (`_LMBase`), larger 5–15 line rewrite prompt. |
| `procedural` | `procedural.py` | Seeded AST walk collects comparison/binop sites, flips one operator textually (`==`→`!=`, `+`→`-`, …), re-parses, emits diff. No LLM. |
| `pr_mirror` | `pr_mirror.py` | Takes the reference src diff and drops one random hunk — the "almost-fixed" bug. No LLM. |

Registry in `methods/__init__.py`: `get_method(name)` / `iter_methods()`.

### `nebius_client.py`
- **`NebiusClient.complete(*, system, user, seed, max_tokens=2048, temperature=0.7)`**
  — one chat completion against Nebius's OpenAI-compatible API with tenacity retry
  (5 attempts, exponential backoff) on rate-limit/timeout errors. Key from
  `NEBIUS_API_KEY`.

### `writers/swebench_jsonl.py` + `yield_logger.py`
- **`InstanceRecord`** — the SWE-bench-format row: `instance_id, repo, base_commit,
  problem_statement, patch, test_patch, FAIL_TO_PASS, PASS_TO_PASS, created_at,
  version, metadata`.
- **`SWEBenchJSONLWriter.write(rec)`** — lock-guarded JSONL append.
- **`YieldLogger.append(MethodYield)`** — per-method `attempted/passed/rate/
  avg_seconds` rows in `yield.csv`.

---

## Pipeline 2: `training/`

Console entry: `train = "training.train:main"`. The heavy lifting is prime-rl
(pinned v0.5.0, its own venv at `/opt/prime-rl` in the image); this project is the
wrapper plus the verifiers environment plugin.

### `training/train.py` — CLI wrapper
- `_parse_args()` — `--dataset` (required), `--profile {smoke,full}`, `--resume`,
  `--output-dir`, `--sessions-root`, `--prime-rl` (default `$PRIME_RL_CMD`), plus six
  **run-time config overrides**: `--seq-len, --batch-size, --rollouts-per-example,
  --max-steps, --max-async-level, --max-off-policy-steps` (None = keep baked value).
- **`_materialize_configs(cfg_root, out_root, overrides)`** — the invariant enforcer.
  No overrides → baked configs returned untouched. Otherwise: patched copies of
  train/orch/infer.toml written to the session dir. `--seq-len` fans out to *both*
  `train.model.seq_len` and `orch.seq_len` (and is rejected if > infer
  `max_model_len`); `--max-steps` fans out to both files; `batch_size %
  rollouts_per_example == 0` asserted. The CLI cannot desynchronize the TOMLs.
- **`_run(args)`** — seed → profile → session (create or `--resume` reopen) →
  `RunLogger` ticket → prime-rl resolvability probe (`_prime_rl_missing`, exit 127)
  → materialize configs → spawn `rl --trainer @… --orchestrator @… --inference @…
  --output-dir …` with `ML_SYSTEMS_SESSION` set and `CUDA_VISIBLE_DEVICES` pinned →
  race `proc.wait()` against the watchdog. Exit 42 = heartbeat stall.
- `_pin_coloc_cuda_visible_devices(env)` — forces `"0,0"` so prime-rl's launcher sees
  two GPU entries and colocates trainer + vLLM on one physical card (the 1-GPU
  stopgap; 2 GPUs is the intended topology).
- `_stream_to_trace(proc, log)` — mirrors subprocess stdout into `trace.jsonl` as
  `prime_rl.stdout` events.
- `_open_or_create_session` / `_pick_latest` / `_resume_path` — resume plumbing.
- `_sigterm(proc)` — watchdog stall callback.

### `training/src/swe_agent_env/__init__.py` — the verifiers plugin
Discovered by module name: orch.toml's `[[env]] id="swe-agent-env"` →
`importlib.import_module("swe_agent_env")` → `load_environment(**args)`. That's why
the package sits at the top of `training/src/` and is hatch-shipped alongside
`training`.

Environment wiring (baked by `infra/train.Dockerfile`):
- `SWE_TARGET_TEMPLATE` (default `/opt/repo-cache/target`) — full git clone of the
  target repo, copied per rollout.
- `SWE_TARGET_PYTHON` (default `python`) — the **target venv's** interpreter used for
  the rollout's pytest; never bare `python`, which PATH-resolves into prime-rl's
  serving venv (see [invariants](#cross-cutting-invariants)).

- **`SWEAgentEnv(vf.MultiTurnEnv)`** — one shared workspace
  (`/tmp/rollout-workspace/current`) serialized by an `asyncio.Lock`:
  - `setup_state` — acquire lock → `_task_from_row(info)` → `AsyncLocalEnvironment`
    → `prepare(test_patch)` → record `initial_head`, zero `tool_calls`, seed the
    `step_budget` DefenseEvent.
  - `env_response` — parse the model turn (`parse_tool_call`); tool-surface errors
    become `[tool-surface-error]` user turns; `evaluate` triggers `_finalise`;
    everything else goes through `dispatch(call, env)`.
  - `budget_exhausted` (`@vf.stop`) — trips at `max_turns`; finalizes if unscored so
    budget exhaustion still yields a reward signal.
  - `cleanup_rollout` (`@vf.cleanup`) — teardown + release lock.
- **`_finalise(state)`** — `current_head`/`current_diff`/`git_apply_check` →
  `env.evaluate()` (F2P pytest under `SWE_TARGET_PYTHON`) → `compute_reward(...)` →
  stores `reward_value` for the rubric.
- **`_SWERubric` / `_reward_from_state`** — the verifiers rubric just reads back the
  reward computed once at finalize.
- **`_task_from_row(row)`** — dataset row → `TaskSpec`;
  `test_command=[SWE_TARGET_PYTHON, -m, pytest, -x, --tb=short, *FAIL_TO_PASS]`.
- **`_load_jsonl_dataset(path)`** — rows become `{"prompt": problem_statement,
  "info": row}`; the split matters because prime-rl's buffer dedups on hashable
  `task`/`prompt` keys.
- **`load_environment(*, dataset, max_tool_calls=20, template_path=…,
  workspace_path=…, **_)`** — the factory; swallows legacy kwargs.

### `training/src/training/configs/*.toml` — prime-rl configs
- **`train.toml`** (TrainerConfig): `seq_len=4096` (80 GB budget; 8192 fits the
  96 GB card the comments describe), LoRA r16/α32 on all attention+MLP projections,
  AdamW `lr=1e-6`, grad-clip 0.05, full activation checkpointing, `[ckpt]
  interval=10 keep_last=3`, `max_steps=10`.
- **`orch.toml`** (OrchestratorConfig): `batch_size=16`, `rollouts_per_example=4`
  (GRPO group G — 4 unique prompts × 4 rollouts per step), `seq_len=4096`,
  `max_async_level=1`, `max_off_policy_steps=16`, sampling `temp=1.0
  max_tokens=2048`, and the `[[env]]` block naming `swe-agent-env` + dataset path.
- **`infer.toml`** (InferenceConfig): vLLM on `0.0.0.0:8000`, bf16,
  `max_model_len=16384` (generation bound only), `gpu_memory_utilization=0.75`
  (tuned for colocation — vLLM's KV profiler subtracts the trainer's resident
  memory). LoRA serving flags are auto-derived from train.toml by prime-rl.

**Invariants:** same model name in all three; `train.model.seq_len == orch.seq_len`;
training seq_len ≤ infer `max_model_len`; `batch_size % rollouts_per_example == 0`.
All are enforced by `_materialize_configs` when overridden via CLI.

### Support modules
- **`config.py`** — `TrainingConfig` + `SMOKE`/`FULL` profiles: loop geometry
  (`max_steps, batch_size, rollouts_per_example, seq_len`) plus watchdog policy.
  The geometry is pushed into the TOMLs at start-up with precedence **CLI flag >
  profile > baked toml** (`_profile_geometry` + `_materialize_configs`), so
  `--profile full` genuinely runs the full schedule (150 steps, batch 32, G=8).
  SMOKE matches the baked TOMLs exactly (asserted by a test), so smoke behavior
  is identical with or without the profile plumbing. Everything else (lr, LoRA,
  ckpt cadence) lives only in the TOMLs.
- **`checkpoint.py`** — `Checkpoint.from_dir` (manifest-hashed), `list_checkpoints`,
  `latest_valid` (newest with clean manifest), `prune_old(keep_last)`.
- **`watchdog.py`** — `watchdog_loop(heartbeat_path, stale_after_seconds, on_stall)`:
  polls `ipc/heartbeat.json` every 30 s; stale/missing → `on_stall()` (SIGTERM) +
  `StaleHeartbeatError`.
- **`session_logger.py`** — `RunLogger`: binds SessionDir + TraceLogger + monotonic
  ticket sequence (`next_ticket`).

---

## Pipeline 3: `evaluation/`

Console entry: `evaluate = "evaluation.eval_cli:main"`. No prime-rl — this project
launches vLLM itself and hand-rolls the rollout loop.

### `eval_cli.py`
- `_parse_args()` — `--checkpoint` (required), `--swebench-n` (20), `--heldout`,
  `--heldout-n` (10), `--sessions-root`, `--seed`, `--dry-run`, `--offline`.
- `_run(args)` — seed → `EvalConfig` → `mixed_sample(...)` → (dry-run prints and
  exits) → `SessionDir.create(kind="eval")` → `vllm_server.launch` + `wait_ready` →
  `run_eval(...)` → print `{total, per_source, session}` → `finally`: terminate vLLM.

### `sample.py` — instance set construction
- **`EvalInstance`** — instance_id, source (`swebench_verified` | `heldout`), repo,
  base_commit, instruction, test_command, patches, F2P/P2P.
- **`sample_swebench_verified(n, seed)`** — deterministic sample from HF
  `princeton-nlp/SWE-bench_Verified` (sorted ids + `Random(seed)`).
- **`load_heldout_jsonl(path, n)`** — reads datagen's held-out split.
- **`_resolve_test_command(row)`** — explicit `test_command` if present, else
  `[SWE_TARGET_PYTHON, -m, pytest, -x, *FAIL_TO_PASS]`.
- **`_pin_interpreter(cmd)`** — rewrites bare `pytest`/`python`/absolute foreign
  interpreter heads onto `SWE_TARGET_PYTHON`; passes `tox`/`make` through.
- **`mixed_sample(...)`** — merges both sources (default 20 + 10).

### `runner.py` — eval orchestration
- **`run_eval(*, instances, cfg, session, checkpoint, vllm_base_url)`** — ticket +
  trace, two semaphores (`llm_concurrency=8`, `rollout_concurrency=4`), one shared
  `VllmClient`, `asyncio.gather` over instances. Per-instance exceptions become
  `passed=False, reward=0.0` results (errors count as failures — no survivorship
  bias). Results stream to `logs/results.jsonl`; `compute_metrics` buckets per
  source; `EvalSummary` → `logs/summary.json`.
- **`compute_metrics(results)`** — per-source `SourceMetrics(n, n_passed, pass_rate)`
  — heldout vs Verified kept separate so in-distribution memorization stays visible.

### `rollout.py` — one eval rollout
- **`VllmClient.complete(prompt, *, history)`** — `/v1/chat/completions` under the
  LLM semaphore with tenacity retry (3 attempts).
- **`run_single_rollout(*, instance, cfg, llm, rollout_sem)`** — `TaskSpec` →
  `AsyncLocalEnvironment` with a **per-instance** workspace
  (`/tmp/eval-rollouts/<instance_id>`) copied from `/opt/repo-cache/<repo_slug>` →
  `prepare(test_patch)` → manual loop (≤ `max_tool_calls=20`): complete → parse →
  `dispatch` → history; break on `evaluate`, else `step_budget_exhausted` → score via
  `current_head`/`current_diff`/`git_apply_check`/`evaluate` →
  `compute_reward(...)` → `RolloutResult`; `teardown` in `finally`.

### `vllm_server.py`
- **`launch(checkpoint, cfg)`** — spawns `sys.executable -m
  vllm.entrypoints.openai.api_server` (never PATH-resolved `python`) with the eval
  flags: greedy (`temperature=0` client-side), `--enable-prefix-caching`,
  `gpu_memory_utilization=0.85` (no trainer contention).
- **`wait_ready(cfg, timeout=600)`** — polls `/v1/models` until 200.

### `config.py`
- **`EvalConfig`** (frozen) — all eval knobs; notable: `max_wall_seconds=120` per
  command, `git_mirror_root=/opt/repo-cache` (template mirrors baked by the image,
  outside `/workspace` so the bind mount can't shadow them).

---

## Workflow traces

### Trace 1 — datagen run

`datagen --repo fastapi/fastapi --t 5 --max-prs 15`

1. `pilot_gen.main()` → `_parse_args()` → `DatagenConfig` → `asyncio.run(_run(cfg))`.
2. `_run`: `apply_seed(42)` → `SessionDir.create("datagen")` → `TraceLogger` →
   (`--offline` strips `lm_*`) → `Pipeline(cfg).run()`.
3. `Pipeline.run()`:
   1. `RepoManager.ensure_clone` → blobless clone at `/workspace/repos/<slug>`.
   2. `RepoManager.list_bug_prs` → GitHub search (merged, `bug`/`fix`), 0 PRs = hard
      error.
   3. Instantiate methods (LLM ones get a `NebiusClient`); delete stale
      `pilot.jsonl`/`heldout.jsonl`; open writers, `YieldLogger`, `MemoryStore`.
   4. `_fetch_patches` — all PR patches, ≤4 concurrent.
   5. **Per-PR loop (sequential** — the shared clone sits at one base commit):
      `split_reference_patch` → `extract_f2p_nodeids` → skip if unusable →
      `checkout(base_commit)` →
      - `--base` mode: emit the real PR directly as an instance (no LLM, no
        validation);
      - mutation mode: `ensure_test_env` (cached per-commit venv, thread executor)
        → **fan out `methods × t` concurrent tasks** (default 4×5=20) of
        `_generate_validate_write`.
   6. Each task: `method.generate(ctx)` (LLM under `_llm_sem` ≤8) →
      `Validator.validate` under `_docker_sem` ≤4 (thread executor: tempdir copy,
      Stage A/B/C) → on pass: `InstanceRecord` + memory record → `passing`.
   7. Yield aggregation → `yield.csv`; seeded shuffle → heldout split
      (`heldout_count`, capped at half if scarce) → `pilot.jsonl` + `heldout.jsonl`.

Outputs: `/workspace/datasets/pilot/{pilot.jsonl, heldout.jsonl, yield.csv,
setup_facts/}`, session trace, cached clones + `_envs` venvs.

### Trace 2 — training run

`uv run train --dataset /workspace/datasets/pilot/pilot.jsonl --profile smoke`

1. `train.main()` → `_run`: seed → profile → session → ticket → prime-rl probe →
   `_materialize_configs` (overrides → patched TOMLs in the session dir) → spawn
   `rl --trainer @train.toml --orchestrator @orch.toml --inference @infer.toml`
   with `CUDA_VISIBLE_DEVICES=0,0`.
2. prime-rl's launcher spawns three processes on the shared GPU: **trainer** (LoRA
   GRPO per train.toml), **vLLM** (infer.toml, port 8000), **orchestrator**
   (orch.toml).
3. Orchestrator reads `[[env]] id="swe-agent-env"` → imports `swe_agent_env` →
   `load_environment(dataset=…)` → `SWEAgentEnv` over the JSONL dataset.
4. Per rollout (async with the trainer, serialized among themselves by the lock):
   `setup_state` (copytree template → checkout base → apply test_patch) → loop of
   vLLM generation ⇄ `env_response` (`parse_tool_call` → `dispatch`; test files
   read-only; path escapes rejected) → `evaluate` call or budget exhaustion →
   `_finalise`: apply-check + F2P pytest under `SWE_TARGET_PYTHON` →
   `compute_reward` (drift/apply/tests gates) → rubric reads `reward_value`.
5. GRPO: 4 rollouts per prompt form a group; group-relative advantage; LoRA update.
   Per `[ckpt] interval=10`, the run dir gets two distinct trees:
   `weights/step_<N>/` — a **full HF-format model export** (merged LoRA,
   config.json + tokenizer + sharded safetensors, `STABLE` marker; this is what
   eval's `--checkpoint` takes) — and `checkpoints/step_<N>/trainer/` — torch
   distributed **resume state** (`.distcp`, not vLLM-loadable). The trainer
   heartbeats `ipc/heartbeat.json`; `watchdog_loop` SIGTERMs on >600 s staleness
   (exit 42). All prime-rl stdout mirrored into the session trace.

### Trace 3 — evaluation run

`uv run evaluate --checkpoint …/step_150 --heldout …/heldout.jsonl`

1. `eval_cli.main()` → `_run`: seed → `mixed_sample` (20 Verified + 10 heldout,
   deterministic; test commands pinned onto `SWE_TARGET_PYTHON`).
2. `SessionDir.create("eval")` → `vllm_server.launch(checkpoint)` (greedy, prefix
   caching, util 0.85) → `wait_ready`.
3. `run_eval`: gather over instances (≤4 concurrent rollouts, ≤8 concurrent LLM
   calls, one shared `VllmClient`).
4. `run_single_rollout` per instance: per-instance workspace copied from
   `/opt/repo-cache/<repo_slug>` → manual tool loop → score with the *same*
   `compute_reward` as training.
5. Outputs: `logs/results.jsonl` (streamed), `logs/summary.json`
   (per-source pass rates), ticket, stdout JSON summary. vLLM terminated in
   `finally`.

---

## Infrastructure

### Images (`infra/*.Dockerfile`)

| Image | Base | Contents | Notable |
|---|---|---|---|
| `datagen` | python slim | agent+common+datagen via `uv sync --frozen` | CPU-only; needs Docker-capable host for validation envs |
| `train` | `nvidia/cuda:12.8.1-devel` | prime-rl v0.5.0 in `/opt/prime-rl` (own venv, flash-attn extra); `training` installed **editable-only** into that venv (agent/common arrive non-editable via its sources); target repo full-cloned to `/opt/repo-cache/target` with **its own venv** `/opt/target-venv` | `ARG TARGET_REPO_URL` (repo-agnostic); `ENV SWE_TARGET_PYTHON/SWE_TARGET_TEMPLATE`; `CUDA_VISIBLE_DEVICES=0,0`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; build-time guards (flash-attn import, `swe_agent_env` import, target venv pytest, serving fastapi ∉ `/opt/repo-cache`) |
| `evaluate` | `nvidia/cuda:12.4.1-devel` | evaluation venv (`uv sync --extra gpu`, includes vLLM); target repo full-cloned to `/opt/repo-cache/fastapi__fastapi` with its own `/opt/target-venv` | same isolation guards as train |

### Compose files (`infra/docker-compose.*.yml`)
Per-pod runtime wiring: `/workspace` bind mount, env keys (`NEBIUS_API_KEY`,
`GITHUB_TOKEN`, `HF_HOME`, W&B), GPU reservations, `ipc: host` + 16 GB shm for
training, and the default commands (datagen: `--repo fastapi/fastapi --t 15`;
train: smoke profile; evaluate: 20+10 instances).

### CI (`.github/workflows/build.yml`)
Push to `main` **or `staging`** (paths: `infra/*.Dockerfile`, `agent/**`, `common/**`,
`datagen/**`, `training/**`, `evaluation/**`) → 3-job matrix building all images →
GHCR as `ghcr.io/<owner>/ml-systems-{datagen,train,evaluate}` tagged with the commit
SHA, the branch name (`:staging`, `:main`), and `:latest` **only on the default
branch**. GHA layer cache per image (`scope=<name>`).

---

## Cross-cutting invariants

1. **The target repo is data, not infrastructure.** The repo being trained/evaluated
   against lives at `/opt/repo-cache/...` with its **own venv** (`/opt/target-venv`)
   and never enters a venv that serves vLLM or prime-rl. History: installing fastapi
   `main` `-e` into the serving venv replaced vLLM's own fastapi pin and 500'd every
   API route. Corollary: any subprocess interpreter must be explicit —
   `sys.executable` or `$SWE_TARGET_PYTHON`, never PATH-resolved `python`/`pytest`.
   Build-time guards in both GPU images enforce this.

2. **Rollout templates are full git clones.** `AsyncLocalEnvironment.prepare` runs
   `git checkout -f <base_commit>` for arbitrary historical commits, offline.
   Shallow/blobless clones break rollouts.

3. **uv editability consistency.** All `[tool.uv.sources]` path deps are
   non-editable. The train image installs only `-e /app/training` and lets agent/
   common resolve through training's sources; adding `-e /app/agent -e /app/common`
   recreates a uv 0.11 "conflicting URLs" build failure.

4. **prime-rl TOML invariants** (`seq_len` equality, `max_model_len` bound,
   batch divisibility) are enforced in one place — `_materialize_configs` — and the
   CLI override flags are the only sanctioned way to vary them per run.

5. **Reward is layered and fails safe.** Structural defenses live in the
   Environments (#1 test-file immutability, #2 network isolation in the Docker
   variant, #3 wall-clock timeouts); mathematical gates in `compute_reward`
   (#4 drift, #5 applicability, #6 F2P∧P2P). Empty F2P → 0. Eval errors → counted
   failures.

6. **Determinism knobs**: `SEED=42` everywhere (`apply_seed`), seeded sampling in
   datagen split / SWE-bench sampling / mutation methods (per-trial derived seeds),
   greedy eval decoding, sorted-then-shuffled splits.

7. **Sessions are the audit trail.** Every pipeline run creates
   `sessions/<kind>-<ts>/` with `trace.jsonl` (all structured events + mirrored
   subprocess output), `tickets/` (durable op records with FSM states), and
   `ipc/heartbeat.json` (training watchdog).

---

## Known quirks

Resolved quirks (dead types `StepResult`/`short_hash`/`read_json_once` removed;
`dispatch` re-hinted to the `Environment` base; profile geometry now actually
reaches prime-rl) are gone from the code. What remains is intentional:

- **`DockerEnvironment` is kept deliberately** despite production rollouts being
  in-process (`AsyncLocalEnvironment`). It is the network-isolation upgrade path
  (reward defense #2 — a container boundary is the only way to enforce it), is
  test-covered, and `orch.toml`'s ignored legacy `docker_image` arg is its config
  vestige. Resurrect it if reward hacking via network access becomes a concern.
- **`LocalWorkspaceEnvironment` has no test-glob guard by design** (documented in
  its docstring): it is driven only by datagen's Validator, which applies whole
  patches it constructed itself with `path=""` — a path-argument guard could
  never fire. Defense #1 lives in the agent-steerable envs.
- Training serializes rollouts (one shared workspace + lock); eval parallelizes
  (per-instance workspaces). Same environment class, opposite concurrency models.
- Eval bakes only the fastapi mirror; cross-repo SWE-bench instances are
  provisioned on demand by `evaluation/repo_setup.py` (full clone into
  `/workspace/repo-cache`, per-(repo, commit) test venv into
  `/workspace/eval-envs`, test command retargeted onto that venv). First contact
  with a new repo pays a clone+pip cost (can be minutes for heavy repos); caches
  persist on the volume. Venv builds follow datagen's shadowing recipe, so
  src/-layout repos may still fail their test run — a counted reward-0, not a
  crash.
