from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path | str, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=p.parent, delete=False, suffix=".tmp", encoding="utf-8"
    )
    try:
        json.dump(payload, tmp, indent=2, sort_keys=True, default=str)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    finally:
        tmp.close()
    os.replace(tmp_name, p)


def read_json_once(path: Path | str, *, delete: bool = True) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if delete:
        p.unlink()
    return data
