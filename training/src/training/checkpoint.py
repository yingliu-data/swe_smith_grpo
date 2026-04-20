from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from common.ipc import atomic_write_json
from common.session import Manifest


CHECKPOINT_FILES = ("adapter.safetensors", "optim.pt", "meta.json")


@dataclass(slots=True)
class Checkpoint:
    step: int
    path: Path
    manifest: Manifest

    @classmethod
    def from_dir(cls, step: int, path: Path) -> "Checkpoint":
        m = Manifest.compute(path, CHECKPOINT_FILES)
        return cls(step=step, path=path, manifest=m)

    def write_meta(self, meta: dict) -> None:
        atomic_write_json(self.path / "meta.json", meta)


def list_checkpoints(root: Path) -> list[Checkpoint]:
    """Return every step_* dir under root, sorted ascending by step, with manifests computed."""
    if not root.exists():
        return []
    out: list[Checkpoint] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        try:
            step = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists():
            continue
        m = Manifest.compute(child, CHECKPOINT_FILES) if all(
            (child / f).exists() for f in CHECKPOINT_FILES
        ) else Manifest()
        out.append(Checkpoint(step=step, path=child, manifest=m))
    out.sort(key=lambda c: c.step)
    return out


def latest_valid(root: Path) -> Checkpoint | None:
    for ckpt in reversed(list_checkpoints(root)):
        bad = ckpt.manifest.verify(ckpt.path)
        if not bad and ckpt.manifest.files:
            return ckpt
    return None


def prune_old(root: Path, keep_last: int) -> list[Path]:
    """Delete all but the newest `keep_last` checkpoints. Return pruned paths."""
    ckpts = list_checkpoints(root)
    if len(ckpts) <= keep_last:
        return []
    import shutil

    pruned: list[Path] = []
    for c in ckpts[:-keep_last]:
        shutil.rmtree(c.path)
        pruned.append(c.path)
    return pruned
