from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass(slots=True)
class MethodYield:
    method: str
    repo: str
    attempted: int
    passed: int
    avg_seconds: float

    @property
    def rate(self) -> float:
        return 0.0 if self.attempted == 0 else self.passed / self.attempted


class YieldLogger:
    HEADER = ("method", "repo", "attempted", "passed", "rate", "avg_seconds")

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._lock = Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADER)

    def append(self, row: MethodYield) -> None:
        with self._lock, self._path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [row.method, row.repo, row.attempted, row.passed, f"{row.rate:.4f}", f"{row.avg_seconds:.2f}"]
            )
