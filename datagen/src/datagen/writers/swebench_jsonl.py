from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(slots=True)
class InstanceRecord:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    test_patch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    created_at: str
    version: str = "0.1.0"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "patch": self.patch,
            "test_patch": self.test_patch,
            "FAIL_TO_PASS": list(self.FAIL_TO_PASS),
            "PASS_TO_PASS": list(self.PASS_TO_PASS),
            "created_at": self.created_at,
            "version": self.version,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class SWEBenchJSONLWriter:
    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, rec: InstanceRecord) -> None:
        line = json.dumps(rec.to_dict(), sort_keys=True) + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line)
