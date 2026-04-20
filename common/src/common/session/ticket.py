from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..ipc import atomic_write_json
from .manifest import Manifest
from .state import transition


@dataclass(slots=True)
class Ticket:
    ticket_id: str
    operation: str
    path: Path
    state: str = "active"
    created_at: float = 0.0
    completed_at: float | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def start(
        cls,
        *,
        tickets_dir: Path,
        ticket_id: str,
        operation: str,
        inputs: dict[str, Any] | None = None,
    ) -> "Ticket":
        t = cls(
            ticket_id=ticket_id,
            operation=operation,
            path=tickets_dir / f"{ticket_id}.json",
            created_at=time.time(),
            inputs=inputs or {},
        )
        atomic_write_json(t.path, t._payload())
        return t

    def finish(
        self,
        *,
        outputs: dict[str, Any] | None = None,
        state: str = "complete",
        manifest: Manifest | None = None,
        error: str | None = None,
    ) -> None:
        self.state = transition(self.state, state)
        self.completed_at = time.time()
        self.outputs = outputs or {}
        if manifest is not None:
            self.manifest = manifest.to_dict()
        if error is not None:
            self.error = error
        atomic_write_json(self.path, self._payload())

    def _payload(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d
