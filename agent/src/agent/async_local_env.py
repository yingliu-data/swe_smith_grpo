from __future__ import annotations

import asyncio
import fnmatch
import shutil
from pathlib import Path
from typing import Iterable

from .environment import Environment
from .models import EvaluationResult, TaskSpec, ToolCall, ToolResult


class AsyncLocalEnvironment(Environment):
    """Async subprocess-backed rollout env — no Docker, no per-rollout container.

    Sibling of DockerEnvironment: same 7-method async surface
    (prepare, step, current_head, current_diff, git_apply_check, evaluate,
    teardown) so common.tool_surface.dispatch can treat them interchangeably.

    Not concurrency-safe on its own: one AsyncLocalEnvironment instance owns
    one workspace dir and must not be driven by two rollouts in parallel.
    The caller (SWEAgentEnv in training/src/swe_agent_env) serialises via an
    asyncio.Lock.
    """

    DEFAULT_TEST_GLOBS: tuple[str, ...] = (
        "tests/*", "tests/**", "test/*", "test/**", "test_*.py", "*_test.py",
    )

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        template_path: str | Path,
        task: TaskSpec,
        command_timeout_seconds: int = 120,
        test_globs: Iterable[str] | None = None,
    ):
        super().__init__(workspace_root, command_timeout_seconds)
        self.task = task
        self._template = Path(template_path).expanduser().resolve()
        self._test_globs = tuple(test_globs) if test_globs else self.DEFAULT_TEST_GLOBS
        self._cwd: Path = self.workspace_root

    async def prepare(self, *, test_patch: str = "") -> None:
        if self._cwd.exists():
            shutil.rmtree(self._cwd)
        self._cwd.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            shutil.copytree, self._template, self._cwd,
            ignore=shutil.ignore_patterns(".venv", "__pycache__"),
        )
        await self._run_checked(["git", "checkout", "-f", self.task.base_commit])
        await self._run_checked(["git", "clean", "-fdx"])
        if test_patch.strip():
            rc, out = await self._run_stdin(
                ["git", "apply", "--whitespace=nowarn", "-"], test_patch,
            )
            if rc != 0:
                raise RuntimeError(f"test_patch failed to apply: {out}")

    async def teardown(self) -> None:
        if self._cwd.exists():
            await asyncio.to_thread(shutil.rmtree, self._cwd, ignore_errors=True)

    async def step(self, tool_call: ToolCall) -> ToolResult:
        name = tool_call.name
        args = tool_call.arguments
        if name == "read_file":
            return await self.read_file(args["path"])
        if name == "edit_file":
            return await self.edit_file(args["path"], args["patch"])
        if name == "delete_file":
            return await self.delete_file(args["path"])
        if name == "evaluate":
            eval_result = await self.evaluate()
            return ToolResult(
                name="evaluate",
                ok=eval_result.passed,
                output=eval_result.output,
                exit_code=eval_result.exit_code,
            )
        return ToolResult(name=name, ok=False, error=f"unknown tool: {name}")

    async def read_file(self, path: str) -> ToolResult:
        safe = self._resolve_inside(path)
        if safe is None:
            return ToolResult(name="read_file", ok=False, error="path escapes workspace")
        try:
            text = await asyncio.to_thread(safe.read_text)
        except Exception as exc:
            return ToolResult(name="read_file", ok=False, error=str(exc))
        return ToolResult(name="read_file", ok=True, output=text, path=safe)

    async def edit_file(self, path: str, patch: str) -> ToolResult:
        safe = self._resolve_inside(path)
        if safe is None:
            return ToolResult(name="edit_file", ok=False, error="path escapes workspace")
        if self._is_test_path(path):
            return ToolResult(name="edit_file", ok=False, error="read-only: test files not editable")
        rc, out = await self._run_stdin(["git", "apply", "--whitespace=nowarn", "-"], patch)
        return ToolResult(name="edit_file", ok=rc == 0, output=out, exit_code=rc)

    async def delete_file(self, path: str) -> ToolResult:
        safe = self._resolve_inside(path)
        if safe is None:
            return ToolResult(name="delete_file", ok=False, error="path escapes workspace")
        if self._is_test_path(path):
            return ToolResult(name="delete_file", ok=False, error="read-only: test files not deletable")
        try:
            await asyncio.to_thread(safe.unlink, missing_ok=False)
        except Exception as exc:
            return ToolResult(name="delete_file", ok=False, error=str(exc))
        return ToolResult(name="delete_file", ok=True, path=safe)

    async def evaluate(self) -> EvaluationResult:
        if self.task is None:
            raise RuntimeError("AsyncLocalEnvironment.evaluate called before prepare")
        rc, out = await self._run(list(self.task.test_command))
        return EvaluationResult(
            reward=1.0 if rc == 0 else 0.0,
            passed=rc == 0,
            output=out,
            exit_code=rc,
        )

    async def current_head(self) -> str:
        rc, out = await self._run(["git", "rev-parse", "HEAD"])
        if rc != 0:
            raise RuntimeError(f"git rev-parse failed: {out}")
        return out.strip()

    async def current_diff(self) -> str:
        rc, out = await self._run(["git", "diff", "HEAD"])
        if rc != 0:
            raise RuntimeError(f"git diff failed: {out}")
        return out

    async def git_apply_check(self, diff: str) -> bool:
        rc, _ = await self._run_stdin(["git", "apply", "--check", "-"], diff)
        return rc == 0

    def _resolve_inside(self, rel_path: str) -> Path | None:
        if rel_path.startswith("/") or ".." in rel_path.split("/"):
            return None
        target = (self._cwd / rel_path.lstrip("./")).resolve()
        try:
            target.relative_to(self._cwd)
        except ValueError:
            return None
        return target

    def _is_test_path(self, rel_path: str) -> bool:
        name = Path(rel_path).name
        return any(
            fnmatch.fnmatch(rel_path, g) or fnmatch.fnmatch(name, g)
            for g in self._test_globs
        )

    async def _run(self, argv: list[str]) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(self._cwd),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self.command_timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, "[TIMEOUT]"
        return int(proc.returncode or 0), stdout.decode(errors="replace")

    async def _run_stdin(self, argv: list[str], stdin_text: str) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(self._cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(input=stdin_text.encode()),
                timeout=self.command_timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, "[TIMEOUT]"
        return int(proc.returncode or 0), stdout.decode(errors="replace")

    async def _run_checked(self, argv: list[str]) -> None:
        rc, out = await self._run(argv)
        if rc != 0:
            raise RuntimeError(f"{' '.join(argv)} failed (rc={rc}): {out}")
