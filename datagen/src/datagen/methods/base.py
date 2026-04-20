from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Context:
    repo: str
    repo_dir: Path
    pr_number: int
    base_commit: str
    merge_commit: str
    pr_title: str
    pr_body: str
    reference_patch: str
    seed: int
    trial_index: int


@dataclass(slots=True)
class Candidate:
    method: str
    buggy_patch: str
    rationale: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class BaseMutationMethod(ABC):
    name: str = "base"

    @abstractmethod
    async def generate(self, ctx: Context) -> Candidate | None:
        ...
