"""On-demand mirror/venv provisioning (evaluation.repo_setup).

Network-free: the clone path is exercised against a local git repo via a
monkeypatched _clone_url; the venv build is monkeypatched to count calls
(the real pip recipe is datagen-proven and needs network).
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from evaluation import repo_setup
from evaluation.config import EvalConfig
from evaluation.sample import EvalInstance


def _instance(repo: str = "acme/widget", commit: str = "deadbeefcafe0000") -> EvalInstance:
    return EvalInstance(
        instance_id=f"{repo.replace('/', '__')}-1", source="swebench_verified",
        repo=repo, base_commit=commit, instruction="", test_command=["python", "-m", "pytest"],
    )


def _cfg(tmp_path: Path) -> EvalConfig:
    return EvalConfig(
        git_mirror_root=tmp_path / "baked",
        runtime_mirror_root=tmp_path / "runtime",
        runtime_envs_root=tmp_path / "envs",
    )


def _make_local_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "README").write_text("x")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"], check=True)
    return path


async def test_ensure_template_prefers_baked(tmp_path):
    cfg = _cfg(tmp_path)
    inst = _instance()
    baked = cfg.git_mirror_root / "acme__widget"
    baked.mkdir(parents=True)
    assert await repo_setup.ensure_template(inst, cfg) == baked


async def test_ensure_template_clones_once_under_concurrency(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    inst = _instance()
    src = _make_local_repo(tmp_path / "src-repo")
    clones: list[str] = []
    real_run = subprocess.run

    def counting_run(argv, **kw):
        if argv[:2] == ["git", "clone"]:
            clones.append(argv[2])
        return real_run(argv, **kw)

    monkeypatch.setattr(repo_setup, "_clone_url", lambda repo: str(src))
    monkeypatch.setattr(repo_setup.subprocess, "run", counting_run)

    results = await asyncio.gather(*(repo_setup.ensure_template(inst, cfg) for _ in range(3)))
    expected = cfg.runtime_mirror_root / "acme__widget"
    assert all(r == expected for r in results)
    assert (expected / ".git").exists()
    assert len(clones) == 1  # per-slug lock: concurrent callers share one clone


async def test_ensure_python_baked_target_keeps_sample_interpreter(tmp_path):
    cfg = _cfg(tmp_path)
    inst = _instance()
    baked = cfg.git_mirror_root / "acme__widget"
    baked.mkdir(parents=True)
    assert await repo_setup.ensure_python(
        instance=inst, cfg=cfg, template=baked, repo_dir=tmp_path / "w") is None


async def test_ensure_python_builds_per_repo_commit_venv(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    inst = _instance()
    built: list[str] = []

    def fake_build(repo_dir, envs_root, key):
        built.append(key)
        return envs_root / key / "bin" / "python"

    monkeypatch.setattr(repo_setup, "_ensure_test_env_sync", fake_build)
    template = cfg.runtime_mirror_root / "acme__widget"  # non-baked path
    results = await asyncio.gather(*(
        repo_setup.ensure_python(
            instance=inst, cfg=cfg, template=template, repo_dir=tmp_path / "w")
        for _ in range(2)))
    assert all(r == cfg.runtime_envs_root / "acme__widget__deadbeefcafe" / "bin" / "python"
               for r in results)
    assert built == ["acme__widget__deadbeefcafe", "acme__widget__deadbeefcafe"]


@pytest.mark.parametrize("cmd,expected_head", [
    (["python", "-m", "pytest", "-x"], "SWAP"),
    (["/opt/target-venv/bin/python", "-m", "pytest"], "SWAP"),
    (["python3.12", "-m", "pytest"], "SWAP"),
    (["tox", "-e", "py312"], "tox"),
])
def test_retarget_test_command(cmd, expected_head):
    new_python = Path("/workspace/eval-envs/x/bin/python")
    out = repo_setup.retarget_test_command(cmd, new_python)
    if expected_head == "SWAP":
        assert out[0] == str(new_python) and out[1:] == cmd[1:]
    else:
        assert out == cmd


def test_retarget_noop_when_python_none():
    cmd = ["python", "-m", "pytest"]
    assert repo_setup.retarget_test_command(cmd, None) == cmd
