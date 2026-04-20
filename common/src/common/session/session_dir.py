from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import sessions_root
from ..ids import make_session_id

_SUBDIRS: tuple[str, ...] = ("workspace", "tickets", "logs", "ipc", "checkpoints", "memory")


@dataclass(slots=True)
class SessionDir:
    root: Path
    kind: str
    session_id: str

    @classmethod
    def create(cls, *, kind: str, root: Path | None = None) -> "SessionDir":
        base = root or sessions_root()
        sid = make_session_id(kind)
        r = base / sid
        for sub in _SUBDIRS:
            (r / sub).mkdir(parents=True, exist_ok=True)
        return cls(root=r, kind=kind, session_id=sid)

    @classmethod
    def open(cls, root: Path | str) -> "SessionDir":
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(root)
        kind = root.name.split("-", 1)[0]
        return cls(root=root, kind=kind, session_id=root.name)

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def tickets(self) -> Path:
        return self.root / "tickets"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def ipc(self) -> Path:
        return self.root / "ipc"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def memory(self) -> Path:
        return self.root / "memory"

    @property
    def trace_path(self) -> Path:
        return self.logs / "trace.jsonl"

    @property
    def heartbeat_path(self) -> Path:
        return self.ipc / "heartbeat.json"
