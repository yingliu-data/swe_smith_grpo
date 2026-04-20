from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent import LocalWorkspaceEnvironment, TaskSpec, ToolCall


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n")
    (tmp_path / "test_mod.py").write_text("from mod import f\n\ndef test_f():\n    assert f() == 1\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path


def _task(repo: Path) -> TaskSpec:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    return TaskSpec(
        repository="local/test",
        base_commit=head,
        instruction="",
        test_command=["python", "-m", "pytest", "-q", "test_mod.py"],
    )


def test_read_edit_evaluate(git_repo: Path) -> None:
    env = LocalWorkspaceEnvironment(git_repo)
    env.reset(_task(git_repo))

    r = env.step(ToolCall(name="read_file", arguments={"path": "mod.py"}))
    assert r.ok and "def f" in r.output

    patch = (
        "diff --git a/mod.py b/mod.py\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def f():\n"
        "-    return 1\n"
        "+def f():\n"
        "+    return 2\n"
    )
    e = env.step(ToolCall(name="edit_file", arguments={"path": "mod.py", "patch": patch}))
    assert e.ok, e.error

    res = env.evaluate()
    assert res.passed is False
    assert res.exit_code != 0


def test_path_escape_rejected(git_repo: Path) -> None:
    env = LocalWorkspaceEnvironment(git_repo)
    env.reset(_task(git_repo))
    r = env.step(ToolCall(name="read_file", arguments={"path": "../../etc/passwd"}))
    assert r.ok is False
    assert "escapes workspace" in r.error
