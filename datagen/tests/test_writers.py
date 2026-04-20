from __future__ import annotations

import json
from pathlib import Path

from datagen.writers.swebench_jsonl import InstanceRecord, SWEBenchJSONLWriter
from datagen.writers.harbor_dir import HarborDirWriter


def _rec() -> InstanceRecord:
    return InstanceRecord(
        instance_id="abc__xyz-1-procedural-0",
        repo="abc/xyz",
        base_commit="deadbeef",
        problem_statement="bug X",
        patch="diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n-1\n+2\n",
        test_patch="",
        FAIL_TO_PASS=["tests/test_x.py::test_a"],
        PASS_TO_PASS=[],
        created_at="2026-04-20T00:00:00+00:00",
        metadata={"method": "procedural"},
    )


def test_swebench_jsonl_append(tmp_path: Path):
    out = tmp_path / "p.jsonl"
    w = SWEBenchJSONLWriter(out)
    w.write(_rec())
    w.write(_rec())
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["instance_id"].endswith("procedural-0")
    assert parsed["FAIL_TO_PASS"] == ["tests/test_x.py::test_a"]


def test_harbor_dir_layout(tmp_path: Path):
    w = HarborDirWriter(tmp_path / "harbor")
    dest = w.write(_rec(), ["python", "-m", "pytest", "-x", "tests/test_x.py::test_a"])
    for name in ("instruction.md", "test.sh", "reference.diff", "task.json"):
        assert (dest / name).exists()
    task = json.loads((dest / "task.json").read_text())
    assert task["repository"] == "abc/xyz"
    assert task["test_command"][0] == "python"
