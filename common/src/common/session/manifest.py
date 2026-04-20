from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class Manifest:
    files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def compute(cls, root: Path | str, files: Iterable[str]) -> "Manifest":
        root = Path(root)
        out: dict[str, str] = {}
        for rel in files:
            out[rel] = _sha256(root / rel)
        return cls(files=out)

    def verify(self, root: Path | str) -> list[str]:
        root = Path(root)
        bad: list[str] = []
        for rel, expected in self.files.items():
            p = root / rel
            if not p.exists() or _sha256(p) != expected:
                bad.append(rel)
        return bad

    def to_dict(self) -> dict[str, str]:
        return dict(self.files)

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Manifest":
        return cls(files=dict(d))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
