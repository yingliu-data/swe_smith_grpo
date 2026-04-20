from __future__ import annotations

import asyncio
import fnmatch
import shlex
from pathlib import Path
from typing import Iterable

import aiodocker

from .environment import Environment
from .models import EvaluationResult, TaskSpec, ToolCall, ToolResult


class DockerEnvironment(Environment):
    """Async, network-isolated per-task container used for training/eval rollouts.

    Reward defense touchpoints enforced here:
      #1 test-file edits rejected via edit_file path-allowlist (test-glob block)
      #2 network-none at container create
      #3 wall-clock timeout (asyncio.wait_for) + token/step budgets enforced by caller
    """

    DEFAULT_TEST_GLOBS: tuple[str, ...] = (
        "tests/*", "tests/**", "test/*", "test/**", "test_*.py", "*_test.py",
    )

    def __init__(
        self,
        workspace_root: str | Path,
        image: str,
        task: TaskSpec,
        command_timeout_seconds: int = 120,
        test_globs: Iterable[str] | None = None,
        docker_sock: str = "/var/run/docker.sock",
    ):
        super().__init__(workspace_root, command_timeout_seconds)
        self.task = task
        self.image = image
        self._test_globs = tuple(test_globs) if test_globs else self.DEFAULT_TEST_GLOBS
        self._docker_sock = docker_sock
        self._docker: aiodocker.Docker | None = None
        self._container: aiodocker.containers.DockerContainer | None = None

    @property
    def container(self) -> aiodocker.containers.DockerContainer:
        if self._container is None:
            raise RuntimeError("DockerEnvironment.prepare() not called")
        return self._container

    async def prepare(self, *, container_name: str | None = None) -> None:
        self._docker = aiodocker.Docker(url=f"unix://{self._docker_sock}")
        config = {
            "Image": self.image,
            "Cmd": ["sleep", "infinity"],
            "WorkingDir": "/workspace/repo",
            "HostConfig": {
                "NetworkMode": "none",
                "Binds": [f"{self.workspace_root}:/workspace:ro"],
                "Memory": 4 * 1024**3,
                "PidsLimit": 256,
                "AutoRemove": False,
                "Tmpfs": {"/workspace/repo": "rw,size=512m"},
            },
            "Env": [f"SEED={self.task.metadata.get('seed', 42)}"],
            "Labels": {"ml_systems.role": "rollout", "ml_systems.task": self.task.repository},
        }
        name = container_name or f"rollout-{self.task.metadata.get('seed', 42)}-{self.task.metadata.get('rollout_idx', 0)}"
        self._container = await self._docker.containers.create(config=config, name=name)
        await self._container.start()
        await self._exec_checked(["git", "init", "-q"])
        await self._exec_checked(["git", "remote", "add", "origin", f"/workspace/src/{self.task.repository.replace('/', '__')}"])
        await self._exec_checked(["git", "fetch", "--depth", "1", "origin", self.task.base_commit])
        await self._exec_checked(["git", "checkout", "-f", self.task.base_commit])

    async def teardown(self) -> None:
        if self._container is not None:
            try:
                await self._container.kill()
            except aiodocker.exceptions.DockerError:
                pass
            try:
                await self._container.delete(force=True)
            except aiodocker.exceptions.DockerError:
                pass
            self._container = None
        if self._docker is not None:
            await self._docker.close()
            self._docker = None

    async def step(self, tool_call: ToolCall) -> ToolResult:
        name = tool_call.name
        args = tool_call.arguments
        if name == "read_file":
            return await self.read_file(args["path"])
        if name == "edit_file":
            return await self.edit_file(args["path"], args["patch"])
        if name == "delete_file":
            return await self.delete_file(args["path"])
        if name == "run_tests":
            return await self._run_tests()
        return ToolResult(name=name, ok=False, error=f"unknown tool: {name}")

    async def read_file(self, path: str) -> ToolResult:
        safe = self._check_inside(path)
        if safe is None:
            return ToolResult(name="read_file", ok=False, error="path escapes /workspace/repo")
        rc, out = await self._exec(["cat", safe])
        return ToolResult(name="read_file", ok=rc == 0, output=out, exit_code=rc)

    async def edit_file(self, path: str, patch: str) -> ToolResult:
        safe = self._check_inside(path)
        if safe is None:
            return ToolResult(name="edit_file", ok=False, error="path escapes /workspace/repo")
        if self._is_test_path(safe):
            return ToolResult(name="edit_file", ok=False, error="read-only: test files not editable")
        rc, out = await self._exec(
            ["bash", "-lc", f"git apply --whitespace=nowarn <<'PATCH_EOF'\n{patch}\nPATCH_EOF"]
        )
        return ToolResult(name="edit_file", ok=rc == 0, output=out, exit_code=rc)

    async def delete_file(self, path: str) -> ToolResult:
        safe = self._check_inside(path)
        if safe is None:
            return ToolResult(name="delete_file", ok=False, error="path escapes /workspace/repo")
        if self._is_test_path(safe):
            return ToolResult(name="delete_file", ok=False, error="read-only: test files not deletable")
        rc, out = await self._exec(["rm", "-f", safe])
        return ToolResult(name="delete_file", ok=rc == 0, output=out, exit_code=rc)

    async def evaluate(self) -> EvaluationResult:
        if self.task is None:
            raise RuntimeError("DockerEnvironment.evaluate called before prepare")
        rc, out = await self._exec(self.task.test_command)
        return EvaluationResult(
            reward=1.0 if rc == 0 else 0.0,
            passed=rc == 0,
            output=out,
            exit_code=rc,
        )

    async def current_head(self) -> str:
        rc, out = await self._exec(["git", "rev-parse", "HEAD"])
        if rc != 0:
            raise RuntimeError(f"git rev-parse failed: {out}")
        return out.strip()

    async def git_apply_check(self, diff: str) -> bool:
        rc, _ = await self._exec(
            ["bash", "-lc", f"git apply --check <<'DIFF_EOF'\n{diff}\nDIFF_EOF"]
        )
        return rc == 0

    async def current_diff(self) -> str:
        rc, out = await self._exec(["git", "diff", "HEAD"])
        if rc != 0:
            raise RuntimeError(f"git diff failed: {out}")
        return out

    def _check_inside(self, rel_path: str) -> str | None:
        if rel_path.startswith("/") or ".." in rel_path.split("/"):
            return None
        return rel_path.lstrip("./")

    def _is_test_path(self, rel_path: str) -> bool:
        name = Path(rel_path).name
        return any(
            fnmatch.fnmatch(rel_path, g) or fnmatch.fnmatch(name, g) for g in self._test_globs
        )

    async def _exec(self, argv: list[str]) -> tuple[int, str]:
        exec_obj = await self.container.exec(cmd=argv, stdout=True, stderr=True)
        chunks: list[bytes] = []
        try:
            async with exec_obj.start(detach=False) as stream:
                await asyncio.wait_for(
                    self._drain(stream, chunks), timeout=self.command_timeout_seconds
                )
        except asyncio.TimeoutError:
            return -1, b"".join(chunks).decode(errors="replace") + "\n[TIMEOUT]"
        info = await exec_obj.inspect()
        return int(info.get("ExitCode") or 0), b"".join(chunks).decode(errors="replace")

    async def _exec_checked(self, argv: list[str]) -> None:
        rc, out = await self._exec(argv)
        if rc != 0:
            raise RuntimeError(f"{shlex.join(argv)} failed (rc={rc}): {out}")

    @staticmethod
    async def _drain(stream, sink: list[bytes]) -> None:
        async for message in stream:
            if message is None:
                break
            sink.append(message.data if hasattr(message, "data") else bytes(message))

    async def _run_tests(self) -> ToolResult:
        rc, out = await self._exec(list(self.task.test_command))
        return ToolResult(name="run_tests", ok=rc == 0, output=out, exit_code=rc)
