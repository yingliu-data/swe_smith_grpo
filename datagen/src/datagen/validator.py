from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import sys
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

    Heuristic: for every test file in the patch, scan hunk lines for added
    `def test_*(`, tracking enclosing ``class`` statements by indentation so
    class-based tests get fully qualified node-ids. Indented test defs whose
    enclosing class never appears in the hunks are dropped — an unqualified
    node-id for a method cannot be collected by pytest.
    """
    out: list[str] = []
    test_def = re.compile(r"^(\s*)(?:async\s+)?def\s+(test_\w+)\s*\(")
    class_def = re.compile(r"^(\s*)class\s+(\w+)")
    ps = PatchSet(test_patch)
    for pf in ps:
        path = pf.path
        if not path.endswith(".py"):
            continue
        class_stack: list[tuple[int, str]] = []
        for hunk in pf:
            # git puts the enclosing declaration in the hunk section header
            # (`@@ ... @@ class Foo:`) — often the only place the class of an
            # appended test method is visible.
            sh = class_def.match(hunk.section_header or "")
            if sh:
                indent = len(sh.group(1).expandtabs())
                while class_stack and class_stack[-1][0] >= indent:
                    class_stack.pop()
                class_stack.append((indent, sh.group(2)))
            for line in hunk:
                if line.is_removed:
                    continue
                text = line.value
                cm = class_def.match(text)
                if cm:
                    indent = len(cm.group(1).expandtabs())
                    while class_stack and class_stack[-1][0] >= indent:
                        class_stack.pop()
                    class_stack.append((indent, cm.group(2)))
                    continue
                if not line.is_added:
                    continue
                m = test_def.match(text)
                if not m:
                    continue
                indent = len(m.group(1).expandtabs())
                while class_stack and class_stack[-1][0] >= indent:
                    class_stack.pop()
                if indent and not class_stack:
                    continue
                qualifier = "::".join(name for _, name in class_stack)
                out.append(f"{path}::{qualifier}::{m.group(2)}" if qualifier else f"{path}::{m.group(2)}")
    return out


def split_reference_patch(reference_patch: str) -> tuple[str, str]:
    """Split a SWE-bench reference diff into (src_patch, test_patch).

    Returns ("", "") for patches that cannot be used faithfully: unparseable
    input (unidiff raises more than UnidiffParseError, so trap broadly) or
    patches touching binary files — unidiff drops 'GIT binary patch' payloads,
    so re-rendering those sections would silently materialize empty files.
    """
    try:
        ps = PatchSet(reference_patch)
    except Exception as e:
        print(f"[datagen] split_reference_patch: skipping unparseable patch ({e})", file=sys.stderr, flush=True)
        return "", ""
    if any(pf.is_binary_file for pf in ps):
        print("[datagen] split_reference_patch: skipping patch with binary files", file=sys.stderr, flush=True)
        return "", ""
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
        src_patch: str,
        test_patch: str,
        f2p: list[str],
        buggy_patch: str,
        python_bin: str,
    ) -> ValidationResult:
        if not f2p:
            return ValidationResult(passed=False, reason="no F2P tests found in reference test_patch")
        return await asyncio.get_running_loop().run_in_executor(
            None,
            self._validate_sync,
            repo_dir,
            base_commit,
            test_patch,
            src_patch,
            buggy_patch,
            f2p,
            python_bin,
        )

    def _validate_sync(
        self,
        repo_dir: Path,
        base_commit: str,
        test_patch: str,
        src_patch: str,
        buggy_patch: str,
        f2p: list[str],
        python_bin: str,
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
                # python_bin comes from the per-(repo, commit) test venv built by
                # test_env.ensure_test_env — the datagen venv itself does not
                # have the target repo's dependencies. -B: stages B and C run
                # against the same files rewritten within the same second and
                # with equal size, so cached pyc (mtime+size validated) can go
                # stale between stages and report the wrong verdict.
                test_command=[python_bin, "-B", "-m", "pytest", "-x", "--tb=short",
                              "-p", "no:cacheprovider", *f2p],
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
