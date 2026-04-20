from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TaskSpec:
    repository: str
    base_commit: str
    instruction: str
    test_command: list[str]
    reference_patch: str | None = None
    issue_url: str | None = None
    fix_commit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    name: str
    ok: bool
    output: str = ""
    error: str = ""
    exit_code: int | None = None
    path: Path | None = None


@dataclass(slots=True)
class EvaluationResult:
    reward: float
    passed: bool
    output: str
    exit_code: int


@dataclass(slots=True)
class StepResult:
    observation: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
