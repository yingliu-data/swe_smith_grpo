from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass(slots=True)
class MemoryRecord:
    name: str
    description: str
    body: str

    def render(self) -> str:
        return (
            f"---\nname: {self.name}\ndescription: {self.description}\n---\n\n"
            f"{self.body.strip()}\n"
        )


class MemoryStore:
    """Run-scoped procedural facts (per-instance env setup hints, dep versions, etc.).

    File layout: ``<root>/<slug>.md`` with YAML frontmatter (name/description) + body.
    Persistence is directory-scoped; caller decides whether to keep across runs.
    """

    def __init__(self, root: Path | str):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def put(self, slug: str, record: MemoryRecord) -> Path:
        p = self._root / f"{slug}.md"
        p.write_text(record.render(), encoding="utf-8")
        return p

    def get(self, slug: str) -> MemoryRecord | None:
        p = self._root / f"{slug}.md"
        if not p.exists():
            return None
        text = p.read_text(encoding="utf-8")
        m = _FRONTMATTER_RE.match(text)
        if not m:
            return None
        front, body = m.group(1), m.group(2)
        meta: dict[str, str] = {}
        for line in front.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        return MemoryRecord(
            name=meta.get("name", slug),
            description=meta.get("description", ""),
            body=body.strip(),
        )

    def all(self) -> list[MemoryRecord]:
        out: list[MemoryRecord] = []
        for p in sorted(self._root.glob("*.md")):
            r = self.get(p.stem)
            if r is not None:
                out.append(r)
        return out
