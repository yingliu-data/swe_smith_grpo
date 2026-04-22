"""Verifiers environment entry point for prime-rl.

Discovery: `verifiers.utils.env_utils.load_environment("swe-agent-env", **args)`
resolves the id to `swe_agent_env` via `importlib.import_module`, then calls
this module's `load_environment(**args)`. That's why this package sits at the
top level of `training/src/` (alongside `training/`) rather than nested — the
verifiers discovery is by importable module name, not entry-point groups.
"""
from __future__ import annotations

import asyncio
from typing import Any

import verifiers as vf
from datasets import Dataset

from agent import AsyncLocalEnvironment, TaskSpec
from common.reward import DefenseEvent, compute_reward
from common.tool_surface import ToolSurfaceError, dispatch, parse_tool_call

__all__ = ["SWEAgentEnv", "load_environment"]

# Baked into the training image at build time (see infra/train.Dockerfile).
_REPO_CACHE_TEMPLATE = "/opt/repo-cache/fastapi"
# Ephemeral rollout scratch dir. Not on network storage — local to the
# training container, recreated per rollout.
_ROLLOUT_WORKSPACE = "/tmp/rollout-workspace/current"


class SWEAgentEnv(vf.MultiTurnEnv):
    """Sequential, container-less SWE-agent rollouts driven by verifiers.

    One shared workspace (`_ROLLOUT_WORKSPACE`) reused across rollouts with
    `shutil.rmtree` + `shutil.copytree` from the baked template between each.
    A rollout-level asyncio.Lock serialises access so prime-rl's parallel
    rollout workers block rather than corrupt the shared checkout.
    """

    def __init__(self, *, template_path: str, workspace_path: str,
                 max_tool_calls: int = 20, **kw: Any):
        super().__init__(max_turns=max_tool_calls, **kw)
        self._template = template_path
        self._workspace = workspace_path
        self._rollout_lock = asyncio.Lock()

    async def setup_state(self, state):
        await self._rollout_lock.acquire()
        row = state.get("info") or {}
        task = _task_from_row(row)
        env = AsyncLocalEnvironment(
            workspace_root=self._workspace,
            template_path=self._template,
            task=task,
        )
        try:
            await env.prepare(test_patch=row.get("test_patch", ""))
        except Exception:
            self._rollout_lock.release()
            raise
        state["env"] = env
        state["task_spec"] = task
        state["test_patch"] = row.get("test_patch", "")
        state["initial_head"] = await env.current_head()
        state["tool_calls"] = 0
        state["defense_log"] = [
            DefenseEvent("step_budget", True, f"<= {self.max_turns} tool calls"),
        ]
        state["reward_value"] = None
        state["reward_breakdown"] = None
        return state

    async def env_response(self, messages, state, **_):
        state["tool_calls"] = state.get("tool_calls", 0) + 1
        last = messages[-1]
        model_text = last.get("content", "") if isinstance(last, dict) else str(last)

        try:
            call = parse_tool_call(model_text)
        except ToolSurfaceError as exc:
            return [{"role": "user", "content": f"[tool-surface-error] {exc}"}]

        if call.name == "evaluate":
            output = await _finalise(state)
            final = [{"role": "user", "content": output}]
            state["final_env_response"] = final
            return final

        result = await dispatch(call, state["env"])
        return [{"role": "user", "content": result.output or result.error or ""}]

    @vf.stop
    async def budget_exhausted(self, state) -> bool:
        if state.get("tool_calls", 0) >= self.max_turns:
          if state.get("reward_value") is None:
              state["final_env_response"] = [                                                                
                  {"role": "user", "content": await _finalise(state)}
              ]                                                                                              
          return True                                   
        return False

    @vf.cleanup
    async def cleanup_rollout(self, state) -> None:
        env = state.get("env")
        try:
            if env is not None:
                await env.teardown()
        finally:
            if self._rollout_lock.locked():
                self._rollout_lock.release()


async def _finalise(state) -> str:
    """Finalise the rollout: compute reward from the diff + F2P pytest result."""
    env: AsyncLocalEnvironment = state["env"]
    task: TaskSpec = state["task_spec"]
    try:
        final_head = await env.current_head()
        final_diff = await env.current_diff()
    except Exception:
        final_head = state["initial_head"]
        final_diff = ""
    apply_ok = await env.git_apply_check(final_diff) if final_diff else True
    eval_result = await env.evaluate()
    reward = compute_reward(
        initial_head=state["initial_head"],
        final_head=final_head,
        apply_check_ok=apply_ok,
        f2p_results={",".join(task.test_command): eval_result.passed},
        p2p_results={},
        structural_log=state["defense_log"],
    )
    state["reward_value"] = reward.reward
    state["reward_breakdown"] = reward
    return eval_result.output


class _SWERubric(vf.Rubric):
    def __init__(self):
        super().__init__()
        self.add_reward_func(_reward_from_state)


async def _reward_from_state(state, **_):
    val = state.get("reward_value")
    return 0.0 if val is None else float(val)


def _task_from_row(row: dict[str, Any]) -> TaskSpec:
    f2p = list(row.get("FAIL_TO_PASS") or [])
    return TaskSpec(
        repository=row.get("repo", ""),
        base_commit=row.get("base_commit", ""),
        instruction=row.get("problem_statement", ""),
        test_command=["python", "-m", "pytest", "-x", "--tb=short", *f2p],
        reference_patch=row.get("patch"),
        metadata={
            **(row.get("metadata") or {}),
            "instance_id": row.get("instance_id"),
        },
    )


def _load_jsonl_dataset(path: str) -> Dataset:
    ds = Dataset.from_json(path)

    def _row_to_example(row):
        # `task` must be a hashable scalar: prime-rl's buffer dedups via
        # hash_keys=['task', 'prompt'] (see orchestrator.stdout) and will fail
        # with `unhashable type: 'dict'` if left to fall back to the row dict.
        return {
            "prompt": row.get("problem_statement", ""),
            "info": row,
        }

    return ds.map(_row_to_example)


def load_environment(
    *,
    dataset: str,
    max_tool_calls: int = 20,
    template_path: str = _REPO_CACHE_TEMPLATE,
    workspace_path: str = _ROLLOUT_WORKSPACE,
    **_: Any,
) -> SWEAgentEnv:
    """Factory called by verifiers.load_environment. Extra kwargs are ignored
    so legacy orch.toml args (docker_image, workspace_root, max_wall_seconds)
    don't break the call."""
    return SWEAgentEnv(
        dataset=_load_jsonl_dataset(dataset),
        rubric=_SWERubric(),
        template_path=template_path,
        workspace_path=workspace_path,
        max_tool_calls=max_tool_calls,
    )
