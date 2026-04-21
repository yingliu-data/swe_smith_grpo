from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Profile = Literal["smoke", "full"]

_PROFILES: dict[str, "TrainingConfig"] = {}


@dataclass(slots=True)
class TrainingConfig:
    profile: Profile
    max_steps: int
    batch_size: int
    group_size_g: int
    micro_batch_size: int
    max_seq_len: int
    lr: float
    ckpt_interval: int
    ckpt_keep_last: int
    llm_concurrency: int
    docker_concurrency: int
    heartbeat_stale_seconds: int
    max_tool_calls: int


SMOKE = TrainingConfig(
    profile="smoke",
    max_steps=30,
    batch_size=16,
    group_size_g=4,
    micro_batch_size=1,
    max_seq_len=16384,
    lr=1e-6,
    ckpt_interval=10,
    ckpt_keep_last=3,
    llm_concurrency=8,
    docker_concurrency=4,
    heartbeat_stale_seconds=600,
    max_tool_calls=20,
)

FULL = TrainingConfig(
    profile="full",
    max_steps=150,
    batch_size=32,
    group_size_g=8,
    micro_batch_size=1,
    max_seq_len=16384,
    lr=1e-6,
    ckpt_interval=25,
    ckpt_keep_last=3,
    llm_concurrency=8,
    docker_concurrency=4,
    heartbeat_stale_seconds=600,
    max_tool_calls=20,
)

_PROFILES["smoke"] = SMOKE
_PROFILES["full"] = FULL


def load_profile(name: str) -> TrainingConfig:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"unknown profile {name!r}; known={sorted(_PROFILES)}") from exc
