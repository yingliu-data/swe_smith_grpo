from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any


class TraceLogger:
    """Append-only JSONL trace writer. Thread-safe within a single process."""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": time.time(), "event": event, **fields}
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line)
