from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from .environment import Environment
from .models import EvaluationResult, TaskSpec, ToolCall, ToolResult


class LocalWorkspaceEnvironment(Environment):
    """Sync subprocess-backed environment for datagen candidate validation.

    Used when no event loop is active. For training/eval rollouts that need async +
    Docker isolation, use DockerEnvironment.
    """

    def __init__(self, workspace_root, command_timeout_seconds: int = 120):
        super().__init__(workspace_root, command_timeout_seconds)

    def reset(self, task: TaskSpec) -> None:
        self.task = task
        self._git("checkout", "-f", task.base_commit)
        self._git("clean", "-fdx")

    def step(self, tool_call: ToolCall) -> ToolResult:
        name = tool_call.name
        args = tool_call.arguments
        if name == "read_file":
            return self.read_file(args["path"])
        if name == "edit_file":
            return self.edit_file(args["path"], args["patch"])
        if name == "delete_file":
            return self.delete_file(args["path"])
        if name == "evaluate":
            eval_result = self.evaluate()
            return ToolResult(
                name="evaluate",
                ok=eval_result.passed,
                output=eval_result.output,
                exit_code=eval_result.exit_code,
            )
        return ToolResult(name=name, ok=False, error=f"unknown tool: {name}")

    def read_file(self, path: str) -> ToolResult:
        target = self._resolve_inside(path)
        if target is None:
            return ToolResult(name="read_file", ok=False, error="path escapes workspace")
        try:
            return ToolResult(name="read_file", ok=True, output=target.read_text(), path=target)
        except Exception as exc:
            return ToolResult(name="read_file", ok=False, error=str(exc))

    def edit_file(self, path: str, patch: str) -> ToolResult:
        proc = subprocess.run(
            ["git", "-C", str(self.workspace_root), "apply", "--whitespace=nowarn", "-"],
            input=patch,
            capture_output=True,
            text=True,
            timeout=self.command_timeout_seconds,
        )
        return ToolResult(
            name="edit_file",
            ok=proc.returncode == 0,
            output=proc.stdout,
            error=proc.stderr,
            exit_code=proc.returncode,
        )

    def delete_file(self, path: str) -> ToolResult:
        target = self._resolve_inside(path)
        if target is None:
            return ToolResult(name="delete_file", ok=False, error="path escapes workspace")
        try:
            target.unlink(missing_ok=False)
        except Exception as exc:
            return ToolResult(name="delete_file", ok=False, error=str(exc))
        return ToolResult(name="delete_file", ok=True, path=target)

    def evaluate(self) -> EvaluationResult:
        if self.task is None:
            raise RuntimeError("LocalWorkspaceEnvironment.evaluate called before reset")
        return self._eval_cmd(self._test_argv())

    def apply_patch_text(self, patch: str) -> ToolResult:
        return self.edit_file(path="", patch=patch)

    def reverse_patch_text(self, patch: str) -> ToolResult:
        proc = subprocess.run(
            ["git", "-C", str(self.workspace_root), "apply", "--reverse", "--whitespace=nowarn", "-"],
            input=patch,
            capture_output=True,
            text=True,
            timeout=self.command_timeout_seconds,
        )
        return ToolResult(
            name="reverse_patch",
            ok=proc.returncode == 0,
            output=proc.stdout,
            error=proc.stderr,
            exit_code=proc.returncode,
        )

    def _test_argv(self) -> list[str]:
        return list(self.task.test_command)

    def _resolve_inside(self, path: str) -> Path | None:
        target = (self.workspace_root / path).resolve()
        try:
            target.relative_to(self.workspace_root)
            return target
        except ValueError:
            return None

    def _git(self, *argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.workspace_root), *argv],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def _eval_cmd(self, argv: Iterable[str]) -> EvaluationResult:
        try:
            proc = subprocess.run(
                list(argv),
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=self.command_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return EvaluationResult(reward=0.0, passed=False, output=str(exc), exit_code=-1)
        return EvaluationResult(
            reward=1.0 if proc.returncode == 0 else 0.0,
            passed=proc.returncode == 0,
            output=proc.stdout + proc.stderr,
            exit_code=proc.returncode,
        )
