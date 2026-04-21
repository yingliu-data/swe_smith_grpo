"""Group-orchestrator tests.

Exercise three properties of training.swe_env.run_group:
  1. All G rollouts actually run in parallel (no serialisation).
  2. The docker_sem cap is respected: at most N concurrent docker exec.
  3. Group-failure contract: any single rollout raising discards the group
     (GroupFailure), as required by GRPO group-relative-advantage math.

We use a stub env that doesn't touch docker — just exercises the
gather+return_exceptions pipeline in run_group. A full docker round-trip is
covered in tests/test_docker_env.py under DOCKER_TESTS=1.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from training.swe_env import GroupFailure, Trajectory, run_group


class _StubTrace:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def log(self, event: str, **kw):  # pragma: no cover - trivial
        self.events.append((event, kw))


class _StubEnv:
    """Minimal MultiTurnEnv stand-in for orchestrator tests."""

    def __init__(self, *, sem: asyncio.Semaphore, fail: bool = False,
                 work_seconds: float = 0.05, concurrency_tracker: list[int] | None = None):
        self._sem = sem
        self._fail = fail
        self._work = work_seconds
        self._tracker = concurrency_tracker
        self._traj: Trajectory | None = None
        self._closed = False

    # The signature mirrors SWEAgentEnv.reset for run_group's call site.
    async def reset(self, task):
        md = task.get("metadata", {}) if isinstance(task, dict) else {}
        self._traj = Trajectory(prompt_id="p", rollout_idx=md.get("rollout_idx", 0))
        return ""

    async def step(self, model_output):
        async with self._sem:
            if self._tracker is not None:
                self._tracker.append(1)
            try:
                await asyncio.sleep(self._work)
                if self._fail:
                    raise RuntimeError("synthetic rollout failure")
                assert self._traj is not None
                self._traj.steps.append({"kind": "tool", "name": "evaluate", "ok": True, "exit_code": 0})
            finally:
                if self._tracker is not None:
                    self._tracker.append(-1)
        return "", 1.0, True, {}  # done=True

    async def close(self):
        self._closed = True

    @property
    def _trajectory(self) -> Trajectory:  # matches SWEAgentEnv attribute name
        assert self._traj is not None
        return self._traj


async def _llm_noop(_obs):
    return "TOOL|evaluate|"


# ---------------------------------------------------------------------------
# 1. Parallel execution
# ---------------------------------------------------------------------------

async def test_run_group_runs_g_rollouts_in_parallel():
    sem = asyncio.Semaphore(4)
    work = 0.2

    def factory():
        return _StubEnv(sem=sem, work_seconds=work)

    t0 = time.monotonic()
    trajectories = await run_group(
        env_factory=factory,
        prompt={"metadata": {"instance_id": "p"}},
        group_size_g=4,
        llm_call=_llm_noop,
        trace=_StubTrace(),
        prompt_id="p",
    )
    elapsed = time.monotonic() - t0

    assert len(trajectories) == 4
    # 4× serial would be 0.8s; parallel should be ~0.2s. Require < 0.6s as a
    # loose cap that still fails under a regression to serial.
    assert elapsed < 0.6, f"expected parallel execution, got {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# 2. Semaphore respected
# ---------------------------------------------------------------------------

async def test_run_group_respects_docker_semaphore():
    sem = asyncio.Semaphore(2)
    tracker: list[int] = []

    def factory():
        return _StubEnv(sem=sem, work_seconds=0.08, concurrency_tracker=tracker)

    # G=8 but semaphore cap is 2; at no point should active count exceed 2.
    await run_group(
        env_factory=factory,
        prompt={"metadata": {"instance_id": "p"}},
        group_size_g=8,
        llm_call=_llm_noop,
        trace=_StubTrace(),
        prompt_id="p",
    )

    # Replay the tracker deltas; max concurrent should be <= 2.
    active = 0
    peak = 0
    for delta in tracker:
        active += delta
        peak = max(peak, active)
    assert peak <= 2, f"semaphore breached: peak={peak}"


# ---------------------------------------------------------------------------
# 3. Group-failure contract (GRPO-correct)
# ---------------------------------------------------------------------------

async def test_run_group_discards_group_on_any_failure():
    sem = asyncio.Semaphore(4)

    def factory():
        # Decide fail-vs-succeed per-instance via a closure over the next idx.
        counter = factory.__dict__.setdefault("_n", [0])
        idx = counter[0]
        counter[0] += 1
        return _StubEnv(sem=sem, fail=(idx == 1), work_seconds=0.02)

    trace = _StubTrace()
    with pytest.raises(GroupFailure) as exc:
        await run_group(
            env_factory=factory,
            prompt={"metadata": {"instance_id": "prompt_X"}},
            group_size_g=4,
            llm_call=_llm_noop,
            trace=trace,
            prompt_id="prompt_X",
        )
    assert exc.value.prompt_id == "prompt_X"
    assert exc.value.dropped == 1
    # Trace must have logged the group-discard event for downstream audit.
    assert any(ev == "rollout.group_discarded" for ev, _ in trace.events)


async def test_run_group_discards_group_on_multiple_failures():
    sem = asyncio.Semaphore(4)

    def factory():
        counter = factory.__dict__.setdefault("_n", [0])
        idx = counter[0]
        counter[0] += 1
        # fail indices 0 and 2 — 2 of 4 fail
        return _StubEnv(sem=sem, fail=(idx in (0, 2)), work_seconds=0.02)

    with pytest.raises(GroupFailure) as exc:
        await run_group(
            env_factory=factory,
            prompt={"metadata": {"instance_id": "pX"}},
            group_size_g=4,
            llm_call=_llm_noop,
            trace=_StubTrace(),
            prompt_id="pX",
        )
    assert exc.value.dropped == 2
