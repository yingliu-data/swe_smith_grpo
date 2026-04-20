from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent import LocalWorkspaceEnvironment, TaskSpec, ToolCall
from unidiff import PatchSet


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    reason: str
    f2p_with_buggy_failed: bool = False
    p2p_with_buggy_passed: bool = False
    f2p_with_reference_passed: bool = False
    wall_seconds: float = 0.0


def extract_f2p_nodeids(test_patch: str) -> list[str]:
    """Extract pytest node-ids for newly added test functions from a test_patch.

    Heuristic: for every test file in the patch, scan added lines for `def test_*(`
    and emit nodeids of form ``<path>::<name>``.
    """
    out: list[str] = []
    ps = PatchSet(test_patch)
    test_def = re.compile(r"^\+\s*(?:async\s+)?def\s+(test_\w+)\s*\(")
    for pf in ps:
        path = pf.path
        if not path.endswith(".py"):
            continue
        for hunk in pf:
            for line in hunk:
                if not line.is_added:
                    continue
                m = test_def.match(f"+{line.value}" if not line.value.startswith("+") else line.value)
                if m:
                    out.append(f"{path}::{m.group(1)}")
    return out


def split_reference_patch(reference_patch: str) -> tuple[str, str]:
    """Split a SWE-bench reference diff into (src_patch, test_patch)."""
    ps = PatchSet(reference_patch)
    src_files, test_files = [], []
    for pf in ps:
        if _is_test_path(pf.path):
            test_files.append(str(pf))
        else:
            src_files.append(str(pf))
    return "".join(src_files), "".join(test_files)


def _is_test_path(p: str) -> bool:
    parts = p.split("/")
    name = parts[-1] if parts else ""
    return (
        any(seg in ("tests", "test") for seg in parts)
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


class Validator:
    """Three-way validation of a candidate mutation.

    Uses an ephemeral copy of the repo (tempdir clone) per candidate so concurrent
    validations don't stomp on each other.
    """

    def __init__(self, *, command_timeout_seconds: int = 120):
        self._timeout = command_timeout_seconds

    async def validate(
        self,
        *,
        repo_dir: Path,
        base_commit: str,
        reference_patch: str,
        buggy_patch: str,
    ) -> ValidationResult:
        src_patch, test_patch = split_reference_patch(reference_patch)
        f2p = extract_f2p_nodeids(test_patch)
        if not f2p:
            return ValidationResult(passed=False, reason="no F2P tests found in reference test_patch")
        return await asyncio.get_running_loop().run_in_executor(
            None,
            self._validate_sync,
            repo_dir,
            base_commit,
            reference_patch,
            test_patch,
            src_patch,
            buggy_patch,
            f2p,
        )

    def _validate_sync(
        self,
        repo_dir: Path,
        base_commit: str,
        reference_patch: str,
        test_patch: str,
        src_patch: str,
        buggy_patch: str,
        f2p: list[str],
    ) -> ValidationResult:
        import time

        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="validator-") as tmp:
            workdir = Path(tmp) / "repo"
            shutil.copytree(repo_dir, workdir, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
            env = LocalWorkspaceEnvironment(workdir, command_timeout_seconds=self._timeout)
            task = TaskSpec(
                repository="validator-local",
                base_commit=base_commit,
                instruction="",
                test_command=["python", "-m", "pytest", "-x", "--tb=short", *f2p],
            )
            try:
                env.reset(task)
            except subprocess.CalledProcessError as exc:
                return ValidationResult(passed=False, reason=f"checkout failed: {exc}")

            # Stage A: apply test_patch so the F2P tests exist at all
            if test_patch.strip():
                r = env.apply_patch_text(test_patch)
                if not r.ok:
                    return ValidationResult(passed=False, reason=f"test_patch failed to apply: {r.error}")

            # Stage B: apply buggy_patch; F2P must FAIL
            r = env.apply_patch_text(buggy_patch)
            if not r.ok:
                return ValidationResult(passed=False, reason=f"buggy_patch failed to apply: {r.error}")
            eval_b = env.evaluate()
            f2p_with_buggy_failed = not eval_b.passed

            # Stage C: reverse buggy_patch then apply reference src_patch; F2P must PASS
            rr = env.reverse_patch_text(buggy_patch)
            if not rr.ok:
                return ValidationResult(
                    passed=False,
                    reason=f"reverse buggy_patch failed: {rr.error}",
                    f2p_with_buggy_failed=f2p_with_buggy_failed,
                )
            if src_patch.strip():
                rp = env.apply_patch_text(src_patch)
                if not rp.ok:
                    return ValidationResult(
                        passed=False,
                        reason=f"reference src_patch failed to apply: {rp.error}",
                        f2p_with_buggy_failed=f2p_with_buggy_failed,
                    )
            eval_c = env.evaluate()
            f2p_with_reference_passed = eval_c.passed

            wall = time.monotonic() - start
            passed = f2p_with_buggy_failed and f2p_with_reference_passed
            return ValidationResult(
                passed=passed,
                reason="ok" if passed else f"buggy_fail={f2p_with_buggy_failed} ref_pass={f2p_with_reference_passed}",
                f2p_with_buggy_failed=f2p_with_buggy_failed,
                f2p_with_reference_passed=f2p_with_reference_passed,
                wall_seconds=wall,
            )
