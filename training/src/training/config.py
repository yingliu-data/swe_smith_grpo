from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Profile = Literal["smoke", "full"]

_PROFILES: dict[str, "TrainingConfig"] = {}


@dataclass(slots=True)
class TrainingConfig:
    """Wrapper-level profile: loop geometry + watchdog policy.

    The geometry fields are pushed into the prime-rl tomls at start-up via
    train._materialize_configs with precedence CLI flag > profile > baked
    toml, so `--profile full` genuinely changes the schedule. Everything
    else (lr, LoRA, ckpt cadence, sampling, GPU budget) lives in
    configs/*.toml — the single source of truth prime-rl reads.
    """

    profile: Profile
    max_steps: int
    batch_size: int
    rollouts_per_example: int  # GRPO group size G; must divide batch_size
    seq_len: int
    heartbeat_stale_seconds: int


# Matches the baked tomls exactly — smoke is the fast wiring check.
SMOKE = TrainingConfig(
    profile="smoke",
    max_steps=10,
    batch_size=16,
    rollouts_per_example=4,
    seq_len=4096,
    heartbeat_stale_seconds=600,
)

# The real schedule. seq_len stays 4096: sized for the 80 GB colocated pod
# (see train.toml [model] for the memory math).
FULL = TrainingConfig(
    profile="full",
    max_steps=150,
    batch_size=32,
    rollouts_per_example=8,
    seq_len=4096,
    heartbeat_stale_seconds=600,
)

_PROFILES["smoke"] = SMOKE
_PROFILES["full"] = FULL


def load_profile(name: str) -> TrainingConfig:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"unknown profile {name!r}; known={sorted(_PROFILES)}") from exc
