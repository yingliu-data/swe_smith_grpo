from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DefenseEvent:
    defense: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class RewardResult:
    reward: float
    passed: bool
    defense_log: list[DefenseEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "passed": self.passed,
            "defense_log": [asdict(d) for d in self.defense_log],
        }


def compute_reward(
    *,
    initial_head: str,
    final_head: str,
    apply_check_ok: bool,
    f2p_results: dict[str, bool],
    p2p_results: dict[str, bool],
    structural_log: list[DefenseEvent] | None = None,
) -> RewardResult:
    """6-layer reward computation.

    Structural defenses (#1 read-only test mounts, #2 network isolation, #3 step/token/wall
    budgets) are enforced BY CONSTRUCTION in DockerEnvironment + rollout container config.
    Their outcomes are threaded through `structural_log` for auditing; reward math is driven
    by per-eval gates #4–6:
      #4 base-commit drift zeros reward
      #5 diff-applies-cleanly zeros reward on failure
      #6 F2P-all-pass ∧ P2P-no-regression → reward=1
    """
    log: list[DefenseEvent] = list(structural_log or [])

    no_drift = initial_head == final_head
    log.append(
        DefenseEvent(
            "base_commit_no_drift",
            no_drift,
            detail=f"{initial_head[:12]} -> {final_head[:12]}",
        )
    )
    if not no_drift:
        return RewardResult(reward=0.0, passed=False, defense_log=log)

    log.append(DefenseEvent("diff_applies_cleanly", apply_check_ok))
    if not apply_check_ok:
        return RewardResult(reward=0.0, passed=False, defense_log=log)

    f2p_all = bool(f2p_results) and all(f2p_results.values())
    p2p_ok = all(p2p_results.values())
    log.append(
        DefenseEvent(
            "f2p_all_pass",
            f2p_all,
            detail=f"{sum(f2p_results.values())}/{len(f2p_results)}",
        )
    )
    log.append(
        DefenseEvent(
            "p2p_no_regression",
            p2p_ok,
            detail=f"{sum(p2p_results.values())}/{len(p2p_results)}",
        )
    )

    passed = f2p_all and p2p_ok
    return RewardResult(reward=1.0 if passed else 0.0, passed=passed, defense_log=log)
