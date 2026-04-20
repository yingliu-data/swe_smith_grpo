from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent import DockerEnvironment, TaskSpec
from common.reward import DefenseEvent, RewardResult, compute_reward
from common.tool_surface import ToolSurfaceError, dispatch, parse_tool_call

from .config import EvalConfig
from .sample import EvalInstance


@dataclass(slots=True)
class RolloutResult:
    instance_id: str
    source: str
    reward: float
    passed: bool
    n_tool_calls: int
    finalise_reason: str
    final_diff: str
    trajectory: list[dict[str, Any]]
    defense_log: list[DefenseEvent]


class VllmClient:
    """Minimal OpenAI-compatible wrapper over the vLLM server we launched."""

    def __init__(self, *, base_url: str, model: str, llm_sem: asyncio.Semaphore,
                 temperature: float, top_p: float, max_tokens: int, seed: int):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._sem = llm_sem
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._seed = seed
        self._client = httpx.AsyncClient(timeout=180.0)

    async def close(self):
        await self._client.aclose()

    async def complete(self, prompt: str, *, history: list[dict[str, str]]) -> str:
        async with self._sem:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=20),
                retry=retry_if_exception_type((httpx.HTTPError,)),
                reraise=True,
            ):
                with attempt:
                    resp = await self._client.post(
                        f"{self._base_url}/v1/chat/completions",
                        json={
                            "model": self._model,
                            "messages": [*history, {"role": "user", "content": prompt}],
                            "temperature": self._temperature,
                            "top_p": self._top_p,
                            "max_tokens": self._max_tokens,
                            "seed": self._seed,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
        raise RuntimeError("unreachable")  # pragma: no cover


async def run_single_rollout(
    *,
    instance: EvalInstance,
    cfg: EvalConfig,
    llm: VllmClient,
    docker_sem: asyncio.Semaphore,
) -> RolloutResult:
    task = TaskSpec(
        repository=instance.repo,
        base_commit=instance.base_commit,
        instruction=instance.instruction,
        test_command=instance.test_command,
        reference_patch=instance.reference_patch,
        metadata={"instance_id": instance.instance_id, "source": instance.source,
                  "seed": cfg.seed},
    )
    env = DockerEnvironment(
        workspace_root=cfg.workspace_root,
        image=cfg.rollout_image,
        task=task,
        command_timeout_seconds=cfg.max_wall_seconds,
    )
    await env.prepare()

    history: list[dict[str, str]] = [
        {"role": "system",
         "content": "You are a software-engineering agent. Tools: read_file, edit_file, "
                    "delete_file, run_tests. You may not edit or delete test files. "
                    "Submit your fix by calling run_tests when ready."},
    ]
    observation = (f"Task: {instance.instruction}\nRepo: {instance.repo} @ "
                   f"{instance.base_commit}\nAvailable tools: read_file, edit_file, "
                   f"delete_file, run_tests.")
    trajectory: list[dict[str, Any]] = []
    initial_head = await env.current_head()
    structural = [
        DefenseEvent("network_none", True, "container network=none"),
        DefenseEvent("test_mounts_readonly", True, "/workspace mounted :ro"),
        DefenseEvent("step_budget", True, f"<= {cfg.max_tool_calls} tool calls"),
    ]

    finalise_reason = "agent_submitted_tests"
    calls = 0
    try:
        while calls < cfg.max_tool_calls:
            calls += 1
            model_output = await llm.complete(observation, history=history)
            history.append({"role": "assistant", "content": model_output})

            try:
                call = parse_tool_call(model_output)
            except ToolSurfaceError as exc:
                observation = f"[tool-surface-error] {exc}"
                trajectory.append({"kind": "parse_error", "detail": str(exc)})
                continue

            async with docker_sem:
                result = await dispatch(call, env)
            trajectory.append({
                "kind": "tool", "name": call.name, "args": call.arguments,
                "ok": result.ok, "exit_code": result.exit_code,
            })
            history.append({"role": "tool", "content": result.output or result.error or ""})

            if call.name == "run_tests":
                break
            observation = result.output or result.error or ""
        else:
            finalise_reason = "step_budget_exhausted"
        # Score the final state.
        final_head = await env.current_head()
        final_diff = await env.current_diff()
        apply_ok = await env.git_apply_check(final_diff) if final_diff else True
        eval_result = await env.evaluate()
        f2p_results = {",".join(instance.test_command): eval_result.passed}
        p2p_results: dict[str, bool] = {}
        reward = compute_reward(
            initial_head=initial_head,
            final_head=final_head,
            apply_check_ok=apply_ok,
            f2p_results=f2p_results,
            p2p_results=p2p_results,
            structural_log=structural,
        )
        return RolloutResult(
            instance_id=instance.instance_id,
            source=instance.source,
            reward=reward.reward,
            passed=reward.passed,
            n_tool_calls=calls,
            finalise_reason=finalise_reason,
            final_diff=final_diff,
            trajectory=trajectory,
            defense_log=list(reward.defense_log),
        )
    finally:
        await env.teardown()
