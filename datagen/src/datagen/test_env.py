from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# pip resolving + building a real repo (fastapi[all] etc.) takes minutes on a
# cold cache — far beyond the per-test validation timeout.
_BUILD_TIMEOUT_SECONDS = 900


def ensure_test_env(repo_dir: Path, envs_root: Path, key: str) -> Path:
    """Create (or reuse) a venv holding the target repo's runtime + test deps.

    Returns the venv's python executable. Validation runs
    ``<venv python> -m pytest`` with the candidate workdir as cwd, so the
    workdir's (patched) package shadows the copy installed here — the venv
    only needs to supply *dependencies*. This shadowing relies on the repo
    keeping its package at the top level (flat layout, like fastapi); for
    src/-layout repos the installed pristine copy would win and stage B
    would reject every candidate — a safe failure, but zero yield.

    ``repo_dir`` must already be checked out at the commit ``key`` encodes.
    Build failures raise; the caller skips the PR. The venv is cached under
    ``envs_root/key`` (marker file ``.ready``), so re-runs and PRs sharing a
    base commit pay the pip cost once.
    """
    env_dir = envs_root / key
    python = env_dir / "bin" / "python"
    marker = env_dir / ".ready"
    if marker.exists():
        return python
    if env_dir.exists():
        shutil.rmtree(env_dir)  # half-built leftover from an interrupted run
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "venv", str(env_dir)],
        check=True, capture_output=True, timeout=120,
    )

    def pip(*args: str, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", *args],
            cwd=repo_dir, check=check, capture_output=True, text=True,
            timeout=_BUILD_TIMEOUT_SECONDS,
        )

    # Runtime deps (the package itself is shadowed by the workdir at test time).
    pip(".", check=True)
    # The venv needs its own test runner.
    pip("pytest", check=True)
    # Best-effort test extras: pip warns on extras the project doesn't declare
    # and installs the ones it does.
    pip(".[test,tests,testing,dev]")
    # Repo-convention test requirements, when present.
    for req in ("requirements-tests.txt", "requirements-test.txt", "requirements_test.txt"):
        if (repo_dir / req).exists():
            pip("-r", req)
            break
    marker.touch()
    return python
