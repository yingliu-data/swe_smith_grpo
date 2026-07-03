"""Sample-construction tests.

We do not exercise the HuggingFace code path locally — that requires network +
a large download. Instead we verify:
  * heldout JSONL parsing + the FAIL_TO_PASS / test_command fallback chain
  * mixed_sample orders sources consistently and respects offline mode
  * load_heldout_jsonl respects the `n` cap without re-shuffling
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.sample import (
    EvalInstance,
    _resolve_test_command,
    load_heldout_jsonl,
    mixed_sample,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows))


# _TARGET_PYTHON is "python" outside the eval image (no SWE_TARGET_PYTHON set);
# in-image it's the isolated target venv's interpreter. What matters is that
# every pytest-shaped command is pinned onto it rather than PATH-resolved —
# bare `pytest` from the eval venv once meant target deps had to pollute the
# venv that also serves vLLM.
from evaluation.sample import _TARGET_PYTHON


def test_resolve_test_command_pins_explicit_pytest_list():
    assert _resolve_test_command({"test_command": ["pytest", "-xvs"]}) == \
        [_TARGET_PYTHON, "-m", "pytest", "-xvs"]


def test_resolve_test_command_parses_string_form():
    assert _resolve_test_command({"test_command": "pytest -x tests/"}) == \
        [_TARGET_PYTHON, "-m", "pytest", "-x", "tests/"]


def test_resolve_test_command_rewrites_foreign_python_bin():
    # Absolute interpreter paths from the datagen container don't exist here.
    row = {"test_command": ["/workspace/repos/_envs/x/bin/python", "-m", "pytest", "-x"]}
    assert _resolve_test_command(row) == [_TARGET_PYTHON, "-m", "pytest", "-x"]


def test_resolve_test_command_passes_through_non_interpreter_heads():
    assert _resolve_test_command({"test_command": ["tox", "-e", "py312"]}) == \
        ["tox", "-e", "py312"]


def test_resolve_test_command_falls_back_to_f2p_nodeids():
    row = {"FAIL_TO_PASS": ["tests/test_a.py::t", "tests/test_b.py::t"]}
    assert _resolve_test_command(row) == \
        [_TARGET_PYTHON, "-m", "pytest", "-x", "tests/test_a.py::t", "tests/test_b.py::t"]


def test_resolve_test_command_parses_json_string_f2p():
    row = {"FAIL_TO_PASS": '["tests/test_a.py::t"]'}
    assert _resolve_test_command(row) == [_TARGET_PYTHON, "-m", "pytest", "-x", "tests/test_a.py::t"]


def test_resolve_test_command_default_when_nothing_specified():
    assert _resolve_test_command({}) == [_TARGET_PYTHON, "-m", "pytest", "-x"]


def test_load_heldout_jsonl_roundtrip(tmp_path):
    rows = [
        {
            "instance_id": f"held-{i}",
            "repo": "fastapi/fastapi",
            "base_commit": "abc",
            "problem_statement": f"bug #{i}",
            "test_command": ["pytest", "-x", f"tests/test_{i}.py"],
            "patch": "diff --git a/x b/x\n...",
            "FAIL_TO_PASS": [f"tests/test_{i}.py::t"],
            "PASS_TO_PASS": [],
        }
        for i in range(5)
    ]
    p = tmp_path / "heldout.jsonl"
    _write_jsonl(p, rows)

    instances = load_heldout_jsonl(p)
    assert len(instances) == 5
    assert all(isinstance(i, EvalInstance) for i in instances)
    assert [i.instance_id for i in instances] == [f"held-{i}" for i in range(5)]
    assert all(i.source == "heldout" for i in instances)
    assert instances[0].fail_to_pass == ["tests/test_0.py::t"]


def test_load_heldout_jsonl_respects_n_cap(tmp_path):
    rows = [{"instance_id": f"h-{i}", "repo": "r", "base_commit": "c",
             "problem_statement": "", "test_command": ["pytest"]} for i in range(10)]
    p = tmp_path / "heldout.jsonl"
    _write_jsonl(p, rows)

    assert len(load_heldout_jsonl(p, n=3)) == 3
    # Order preserved — no re-shuffle.
    assert [i.instance_id for i in load_heldout_jsonl(p, n=3)] == ["h-0", "h-1", "h-2"]


def test_mixed_sample_offline_returns_heldout_only(tmp_path):
    rows = [{"instance_id": "h-0", "repo": "r", "base_commit": "c",
             "problem_statement": "", "test_command": ["pytest"]}]
    p = tmp_path / "heldout.jsonl"
    _write_jsonl(p, rows)

    instances = mixed_sample(
        swebench_n=20,
        heldout_path=p,
        heldout_n=1,
        seed=42,
        offline=True,
    )
    assert len(instances) == 1
    assert instances[0].source == "heldout"


def test_mixed_sample_offline_no_heldout_returns_empty():
    instances = mixed_sample(
        swebench_n=20, heldout_path=None, heldout_n=10, seed=42, offline=True,
    )
    assert instances == []
