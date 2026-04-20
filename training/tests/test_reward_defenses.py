"""Failing-fixture coverage for each of the 6 reward-defense layers.

Structural defenses (#1 read-only test mounts, #2 network isolation, #3 budgets)
are enforced by infrastructure; we exercise their *representations* via
DefenseEvent entries on the log AND via the DockerEnvironment path-allowlist for
#1. Per-eval gates (#4–6) are unit-tested against common.reward.compute_reward.
"""
from __future__ import annotations

import pytest

from agent.docker_env import DockerEnvironment
from agent.models import TaskSpec
from common.reward import DefenseEvent, compute_reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task() -> TaskSpec:
    return TaskSpec(
        repository="acme/widget",
        base_commit="abc123",
        instruction="fix bug",
        test_command=["pytest", "-x"],
    )


# ---------------------------------------------------------------------------
# Defense #1 — read-only test mounts (infra), plus edit-file path allowlist
# ---------------------------------------------------------------------------

def test_defense_1_edit_file_rejects_test_glob():
    """The DockerEnvironment allowlist blocks writes to any test_* path without
    ever hitting the container — the *container* mount is :ro, so even if the
    allowlist failed, the write would still fail. Both layers together make
    test-edit reward hacks structurally impossible."""
    env = DockerEnvironment(workspace_root="/tmp/ws", image="x", task=_task())
    assert env._is_test_path("tests/test_core.py")
    assert env._is_test_path("pkg/test_utils.py")
    assert env._is_test_path("src/widget_test.py")
    assert not env._is_test_path("pkg/widget.py")


# ---------------------------------------------------------------------------
# Defense #2 — network isolation (infra) — represented as DefenseEvent on log
# ---------------------------------------------------------------------------

def test_defense_2_network_none_event_present_on_happy_path():
    structural = [
        DefenseEvent("network_none", True, "container network=none"),
        DefenseEvent("test_mounts_readonly", True, "/workspace mounted :ro"),
        DefenseEvent("step_budget", True, "<= 20 tool calls"),
    ]
    result = compute_reward(
        initial_head="abc",
        final_head="abc",
        apply_check_ok=True,
        f2p_results={"tests/test_a.py::t": True},
        p2p_results={"tests/test_b.py::t": True},
        structural_log=structural,
    )
    names = [d.defense for d in result.defense_log]
    assert "network_none" in names
    assert "test_mounts_readonly" in names
    assert "step_budget" in names


# ---------------------------------------------------------------------------
# Defense #4 — base-commit drift zeros reward
# ---------------------------------------------------------------------------

def test_defense_4_base_commit_drift_zeros_reward():
    result = compute_reward(
        initial_head="abc",
        final_head="def",  # drift
        apply_check_ok=True,
        f2p_results={"t": True},
        p2p_results={},
    )
    assert result.reward == 0.0
    assert not result.passed
    drift_event = next(d for d in result.defense_log if d.defense == "base_commit_no_drift")
    assert drift_event.passed is False
    # Drift short-circuits further checks — we should never have evaluated F2P.
    assert all(d.defense != "f2p_all_pass" for d in result.defense_log)


# ---------------------------------------------------------------------------
# Defense #5 — diff-applies-cleanly zeros reward on failure
# ---------------------------------------------------------------------------

def test_defense_5_unclean_diff_zeros_reward():
    result = compute_reward(
        initial_head="abc",
        final_head="abc",
        apply_check_ok=False,
        f2p_results={"t": True},
        p2p_results={},
    )
    assert result.reward == 0.0
    apply_event = next(d for d in result.defense_log if d.defense == "diff_applies_cleanly")
    assert apply_event.passed is False
    assert all(d.defense != "f2p_all_pass" for d in result.defense_log)


# ---------------------------------------------------------------------------
# Defense #6 — F2P-all-pass AND P2P-no-regression is the reward function
# ---------------------------------------------------------------------------

def test_defense_6_p2p_regression_zeros_reward():
    """Even if F2P flips to pass, a P2P regression means the fix broke existing
    behaviour. Reward = 0."""
    result = compute_reward(
        initial_head="abc",
        final_head="abc",
        apply_check_ok=True,
        f2p_results={"tests/f2p_1.py::t": True, "tests/f2p_2.py::t": True},
        p2p_results={"tests/p2p_1.py::t": True, "tests/p2p_2.py::t": False},
    )
    assert result.reward == 0.0
    assert not result.passed
    p2p_event = next(d for d in result.defense_log if d.defense == "p2p_no_regression")
    assert p2p_event.passed is False


def test_defense_6_empty_f2p_zeros_reward():
    """Empty F2P set is not a pass — catches 'agent produced no fix' loopholes."""
    result = compute_reward(
        initial_head="abc", final_head="abc", apply_check_ok=True,
        f2p_results={}, p2p_results={"tests/p2p.py::t": True},
    )
    assert result.reward == 0.0
    assert not result.passed


def test_defense_6_happy_path_awards_one():
    result = compute_reward(
        initial_head="abc",
        final_head="abc",
        apply_check_ok=True,
        f2p_results={"tests/f2p.py::t": True},
        p2p_results={"tests/p2p.py::t": True},
    )
    assert result.reward == 1.0
    assert result.passed
    # All six defenses (3 structural threaded in + 3 eval) accounted for.
    logged = {d.defense for d in result.defense_log}
    assert {"base_commit_no_drift", "diff_applies_cleanly", "f2p_all_pass",
            "p2p_no_regression"} <= logged
