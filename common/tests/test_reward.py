from __future__ import annotations

from common.reward import DefenseEvent, compute_reward


def _ok_inputs():
    return dict(
        initial_head="abc123deadbeef",
        final_head="abc123deadbeef",
        apply_check_ok=True,
        f2p_results={"t::f2p_1": True, "t::f2p_2": True},
        p2p_results={"t::p2p_1": True, "t::p2p_2": True, "t::p2p_3": True},
    )


def test_defense_4_base_commit_drift_zeros_reward():
    inp = _ok_inputs() | {"final_head": "deadbeefabc123"}
    r = compute_reward(**inp)
    assert r.reward == 0.0
    assert r.passed is False
    events = {e.defense: e.passed for e in r.defense_log}
    assert events["base_commit_no_drift"] is False


def test_defense_5_diff_apply_failure_zeros_reward():
    inp = _ok_inputs() | {"apply_check_ok": False}
    r = compute_reward(**inp)
    assert r.reward == 0.0
    events = {e.defense: e.passed for e in r.defense_log}
    assert events["base_commit_no_drift"] is True
    assert events["diff_applies_cleanly"] is False


def test_defense_6_f2p_failure_zeros_reward():
    inp = _ok_inputs()
    inp["f2p_results"] = {"t::f2p_1": True, "t::f2p_2": False}
    r = compute_reward(**inp)
    assert r.reward == 0.0
    events = {e.defense: e.passed for e in r.defense_log}
    assert events["f2p_all_pass"] is False


def test_defense_6_p2p_regression_zeros_reward():
    inp = _ok_inputs()
    inp["p2p_results"] = {"t::p2p_1": True, "t::p2p_2": False, "t::p2p_3": True}
    r = compute_reward(**inp)
    assert r.reward == 0.0
    events = {e.defense: e.passed for e in r.defense_log}
    assert events["f2p_all_pass"] is True
    assert events["p2p_no_regression"] is False


def test_happy_path_reward_one():
    r = compute_reward(**_ok_inputs())
    assert r.reward == 1.0
    assert r.passed is True


def test_structural_log_threaded_through():
    log = [DefenseEvent("network_none", True, "container network=none"),
           DefenseEvent("test_mounts_readonly", True)]
    r = compute_reward(**_ok_inputs(), structural_log=log)
    defenses = [e.defense for e in r.defense_log]
    assert defenses[:2] == ["network_none", "test_mounts_readonly"]


def test_empty_f2p_fails_safely():
    inp = _ok_inputs() | {"f2p_results": {}}
    r = compute_reward(**inp)
    assert r.reward == 0.0
    assert r.passed is False
