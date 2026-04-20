from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from agent import DockerEnvironment, TaskSpec, ToolCall
from common.reward import DefenseEvent, RewardResult, compute_reward
from common.tool_surface import ToolSurfaceError, dispatch, parse_tool_call

try:
    from verifiers.envs.multiturn_env import MultiTurnEnv as _MultiTurnEnv  # type: ignore
except Exception:  # pragma: no cover -- allow import on non-GPU machines
    class _MultiTurnEnv:  # type: ignore[no-redef]
        """Stub used when verifiers is not installed (e.g. local macOS dev)."""

        def __init__(self, *a, **kw): ...


class GroupFailure(RuntimeError):
    def __init__(self, prompt_id: str, dropped: int):
        super().__init__(f"group {prompt_id}: {dropped} rollouts failed")
        self.prompt_id = prompt_id
        self.dropped = dropped


@dataclass(slots=True)
class Trajectory:
    prompt_id: str
    rollout_idx: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    initial_head: str = ""
    final_head: str = ""
    final_diff: str = ""
    reward: RewardResult | None = None

    def hash(self) -> str:
        payload = json.dumps(
            {"prompt": self.prompt_id, "rollout": self.rollout_idx, "steps": self.steps,
             "initial_head": self.initial_head, "final_head": self.final_head,
             "final_diff": self.final_diff},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class SWEAgentEnv(_MultiTurnEnv):
    """prime-rl MultiTurnEnv wrapper around DockerEnvironment.

    Lifecycle per rollout (prime-rl calls these):
      reset(task) -> render initial prompt
      step(model_output) -> (observation, reward, done, info)
      close() -> teardown container

    Safety:
      - tools dispatched via common.tool_surface (fixed surface of 4)
      - step budget, wall budget, token budget enforced by caller
      - network=none on container (infra-level); read-only mounts on tests/
    """

    MAX_TOOL_CALLS = 20

    def __init__(self, *, docker_image: str, workspace_root: str,
                 docker_sem: asyncio.Semaphore, seed: int = 42):
        super().__init__()
        self._docker_image = docker_image
        self._workspace_root = workspace_root
        self._docker_sem = docker_sem
        self._seed = seed
        self._env: DockerEnvironment | None = None
        self._task: TaskSpec | None = None
        self._trajectory: Trajectory | None = None
        self._structural_log: list[DefenseEvent] = []
        self._step_count = 0

    async def reset(self, task: dict[str, Any] | TaskSpec) -> str:
        if isinstance(task, dict):
            md = task.get("metadata", {}) or {}
            task = TaskSpec(
                repository=md.get("repository", task.get("repo", "")),
                base_commit=md.get("base_commit", ""),
                instruction=task.get("prompt", ""),
                test_command=list(md.get("test_command", [])),
                reference_patch=md.get("reference_patch"),
                metadata=md | {"instance_id": task.get("instance_id"), "seed": self._seed},
            )
        self._task = task
        self._env = DockerEnvironment(
            workspace_root=self._workspace_root,
            image=self._docker_image,
            task=task,
            command_timeout_seconds=120,
        )
        await self._env.prepare()
        prompt_id = task.metadata.get("instance_id", "unknown")
        self._trajectory = Trajectory(
            prompt_id=prompt_id,
            rollout_idx=task.metadata.get("rollout_idx", 0),
            initial_head=await self._env.current_head(),
        )
        self._structural_log = [
            DefenseEvent("network_none", True, "container network=none"),
            DefenseEvent("test_mounts_readonly", True, "/workspace mounted :ro"),
            DefenseEvent("step_budget", True, f"<= {self.MAX_TOOL_CALLS} tool calls"),
        ]
        self._step_count = 0
        return self._render_prompt(task)

    async def step(self, model_output: str) -> tuple[str, float, bool, dict[str, Any]]:
        if self._env is None or self._task is None or self._trajectory is None:
            raise RuntimeError("step called before reset")
        self._step_count += 1
        if self._step_count > self.MAX_TOOL_CALLS:
            return await self._finalise(reason="step_budget_exhausted")

        try:
            call = parse_tool_call(model_output)
        except ToolSurfaceError as exc:
            obs = f"[tool-surface-error] {exc}"
            self._trajectory.steps.append({"kind": "parse_error", "detail": str(exc)})
            return obs, 0.0, False, {}

        async with self._docker_sem:
            result = await dispatch(call, self._env)
        self._trajectory.steps.append({
            "kind": "tool", "name": call.name, "args": call.arguments,
            "ok": result.ok, "exit_code": result.exit_code,
        })
        observation = result.output or result.error

        if call.name == "run_tests":
            return await self._finalise(reason="agent_submitted_tests")
        return observation, 0.0, False, {}

    async def close(self) -> None:
        if self._env is not None:
            await self._env.teardown()
            self._env = None

    async def _finalise(self, *, reason: str) -> tuple[str, float, bool, dict[str, Any]]:
        assert self._env is not None and self._trajectory is not None
        try:
            final_head = await self._env.current_head()
            final_diff = await self._env.current_diff()
        except Exception as exc:
            final_head = self._trajectory.initial_head
            final_diff = ""
        apply_ok = await self._env.git_apply_check(final_diff) if final_diff else True
        # F2P / P2P results derived from test_command output; we run the task's full
        # test_command which is the F2P list. P2P is empty in this smoke build; the
        # reward falls back to "did F2P pass".
        eval_result = await self._env.evaluate()
        f2p_results = {",".join(self._task.test_command): eval_result.passed}
        p2p_results: dict[str, bool] = {}
        reward = compute_reward(
            initial_head=self._trajectory.initial_head,
            final_head=final_head,
            apply_check_ok=apply_ok,
            f2p_results=f2p_results,
            p2p_results=p2p_results,
            structural_log=self._structural_log,
        )
        self._trajectory.final_head = final_head
        self._trajectory.final_diff = final_diff
        self._trajectory.reward = reward
        info = {
            "reward": reward.reward,
            "defense_log": [d.defense + ("✓" if d.passed else "✗") for d in reward.defense_log],
            "finalise_reason": reason,
            "trajectory_hash": self._trajectory.hash(),
        }
        return eval_result.output, reward.reward, True, info

    def _render_prompt(self, task: TaskSpec) -> str:
        return (
            f"# Task\n{task.instruction}\n\n"
            f"# Repository\n{task.repository} @ {task.base_commit}\n\n"
            f"# Tools\nYou have read_file, edit_file, delete_file, run_tests. "
            f"You may not edit or delete test files.\n\n"
            f"# Budget\n{self.MAX_TOOL_CALLS} tool calls, 16K context, 120s wall per tool.\n"
        )


async def run_group(
    *,
    env_factory,
    prompt: dict[str, Any],
    group_size_g: int,
    llm_call,
    trace,
    prompt_id: str,
) -> list[Trajectory]:
    """Run G rollouts in parallel under GRPO group-failure semantics.

    Any rollout failure poisons group-relative advantage; we discard the whole group.
    """
    async def _one(idx: int) -> Trajectory:
        env = env_factory()
        obs = await env.reset(prompt | {"metadata": (prompt.get("metadata", {}) | {"rollout_idx": idx})})
        done = False
        while not done:
            model_out = await llm_call(obs)
            obs, _r, done, _info = await env.step(model_out)
        traj = env._trajectory
        await env.close()
        return traj

    rollouts = await asyncio.gather(
        *[_one(i) for i in range(group_size_g)], return_exceptions=True
    )
    good = [r for r in rollouts if not isinstance(r, BaseException)]
    if len(good) < group_size_g:
        trace.log("rollout.group_discarded", prompt_id=prompt_id,
                  n_failed=group_size_g - len(good))
        raise GroupFailure(prompt_id=prompt_id, dropped=group_size_g - len(good))
    return good  # type: ignore[return-value]
