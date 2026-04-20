"""Runner metric-aggregation tests.

Per-source metrics are the whole point of the split sample: a single number
(pass_rate=0.5) hides the case where heldout=1.0 but SWE-bench=0.0, which
would indicate pure memorisation. Verify the aggregator keeps these visible.
"""
from __future__ import annotations

from evaluation.rollout import RolloutResult
from evaluation.runner import compute_metrics


def _r(instance_id: str, source: str, passed: bool) -> RolloutResult:
    return RolloutResult(
        instance_id=instance_id, source=source,
        reward=1.0 if passed else 0.0, passed=passed,
        n_tool_calls=3, finalise_reason="agent_submitted_tests",
        final_diff="", trajectory=[], defense_log=[],
    )


def test_compute_metrics_splits_by_source():
    results = [
        _r("a1", "swebench_verified", passed=True),
        _r("a2", "swebench_verified", passed=False),
        _r("a3", "swebench_verified", passed=False),
        _r("b1", "heldout", passed=True),
        _r("b2", "heldout", passed=True),
    ]
    m = compute_metrics(results)
    assert set(m.keys()) == {"swebench_verified", "heldout"}
    assert m["swebench_verified"].n == 3
    assert m["swebench_verified"].n_passed == 1
    assert m["swebench_verified"].pass_rate == round(1 / 3, 4)
    assert m["heldout"].n == 2
    assert m["heldout"].n_passed == 2
    assert m["heldout"].pass_rate == 1.0


def test_compute_metrics_empty_when_no_results():
    assert compute_metrics([]) == {}


def test_compute_metrics_exposes_memorisation_gap():
    """Construct a result set where average hides a huge gap; verify the split
    surfaces it."""
    results = [
        *[_r(f"v{i}", "swebench_verified", passed=False) for i in range(20)],
        *[_r(f"h{i}", "heldout", passed=True) for i in range(10)],
    ]
    m = compute_metrics(results)
    assert m["swebench_verified"].pass_rate == 0.0
    assert m["heldout"].pass_rate == 1.0
    # The overall "average" would be 10/30 ≈ 0.33 — not reported, intentionally.
    assert "overall" not in m
