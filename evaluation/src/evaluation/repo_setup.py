"""On-demand repo mirror + test-venv provisioning for cross-repo eval.

The evaluate image bakes only the in-distribution target (fastapi) at
{git_mirror_root}/{slug} with its deps in the /opt/target-venv interpreter
(SWE_TARGET_PYTHON). SWE-bench Verified instances span many other repos;
without provisioning, those fail at env.prepare (missing template) or at
evaluate (missing deps) and score an automatic 0.

For a non-baked repo this module:
  1. clones a full mirror into {runtime_mirror_root}/{slug} (full clone —
     rollouts `git checkout -f <base_commit>` arbitrary history offline);
  2. builds a per-(repo, base-commit) venv under {runtime_envs_root},
     mirroring datagen/src/datagen/test_env.py's recipe, using the rollout's
     own prepared workspace (already at base_commit) as the pip source;
  3. retargets the instance's pytest command onto that venv's python.

Both steps cache under /workspace and are guarded by per-key asyncio locks
so concurrent rollouts of the same repo/commit provision once. Failures
propagate: the runner counts them as reward-0 results, same as any other
rollout error.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from .config import EvalConfig
from .sample import EvalInstance

_CLONE_TIMEOUT_SECONDS = 600
_BUILD_TIMEOUT_SECONDS = 900

# Per-key critical sections (mirror clones, venv builds). Module-level is
# fine: one event loop per eval process.
_locks: dict[str, asyncio.Lock] = {}


def _lock(key: str) -> asyncio.Lock:
    return _locks.setdefault(key, asyncio.Lock())


def _slug(repo: str) -> str:
    return repo.replace("/", "__")


def _clone_url(repo: str) -> str:
    # Separate function so tests can monkeypatch to a local file:// repo.
    return f"https://github.com/{repo}.git"


async def ensure_template(instance: EvalInstance, cfg: EvalConfig) -> Path:
    """Return a full-clone template dir for the instance's repo.

    Prefers the image-baked mirror; otherwise clones once into
    runtime_mirror_root (persisted on /workspace across runs).
    """
    slug = _slug(instance.repo)
    baked = cfg.git_mirror_root / slug
    if baked.exists():
        return baked
    runtime = cfg.runtime_mirror_root / slug
    async with _lock(f"mirror:{slug}"):
        if not (runtime / ".git").exists():
            if runtime.exists():
                shutil.rmtree(runtime)  # half-cloned leftover
            runtime.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                subprocess.run,
                ["git", "clone", _clone_url(instance.repo), str(runtime)],
                check=True, capture_output=True, timeout=_CLONE_TIMEOUT_SECONDS,
            )
    return runtime


async def ensure_python(
    *, instance: EvalInstance, cfg: EvalConfig, template: Path, repo_dir: Path,
) -> Path | None:
    """Return the python to run this instance's tests with, or None to keep
    the sample-time interpreter (the baked target venv).

    ``repo_dir`` must already be checked out at instance.base_commit (the
    rollout's prepared workspace) — it is the pip-install source.
    """
    if template == cfg.git_mirror_root / _slug(instance.repo):
        return None  # baked target: SWE_TARGET_PYTHON already has the deps
    key = f"{_slug(instance.repo)}__{instance.base_commit[:12]}"
    async with _lock(f"env:{key}"):
        return await asyncio.to_thread(
            _ensure_test_env_sync, repo_dir, cfg.runtime_envs_root, key)


def retarget_test_command(cmd: list[str], python_bin: Path | None) -> list[str]:
    """Point an interpreter-headed test command at ``python_bin``.

    Sample-time pinning already normalized pytest-shaped commands onto an
    interpreter head (see sample._pin_interpreter); here we only swap which
    interpreter. Non-interpreter heads (tox, make) pass through.
    """
    if python_bin is None or not cmd:
        return list(cmd)
    head = Path(cmd[0]).name
    if head in ("python", "python3") or head.startswith("python3."):
        return [str(python_bin), *cmd[1:]]
    return list(cmd)


def _ensure_test_env_sync(repo_dir: Path, envs_root: Path, key: str) -> Path:
    """Create (or reuse) a venv with the repo's runtime + test deps.

    Adapted from datagen/src/datagen/test_env.py::ensure_test_env (kept
    separate — evaluation does not depend on the datagen project). Same
    contract: repo_dir must sit at the target commit; the venv supplies
    *dependencies* while pytest's cwd shadows the installed package copy;
    cached via a ``.ready`` marker so instances sharing a base commit pay
    the pip cost once.
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

    pip(".", check=True)
    pip("pytest", check=True)
    pip(".[test,tests,testing,dev]")  # best-effort extras
    for req in ("requirements-tests.txt", "requirements-test.txt", "requirements_test.txt"):
        if (repo_dir / req).exists():
            pip("-r", req)
            break
    marker.touch()
    return python
