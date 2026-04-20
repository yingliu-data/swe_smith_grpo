from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

Source = Literal["swebench_verified", "heldout"]


@dataclass(slots=True)
class EvalInstance:
    instance_id: str
    source: Source
    repo: str
    base_commit: str
    instruction: str
    test_command: list[str]
    reference_patch: str | None = None
    fail_to_pass: list[str] | None = None
    pass_to_pass: list[str] | None = None


def sample_swebench_verified(n: int, seed: int) -> list[EvalInstance]:
    """Deterministic sample of `n` instances from princeton-nlp/SWE-bench_Verified.

    We sort the instance_ids first so the set is reproducible regardless of
    whatever order HF returns.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - only hit locally
        raise RuntimeError(
            "datasets not installed; install evaluation[gpu] or run --offline with --heldout-only"
        ) from exc
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    ids = sorted([row["instance_id"] for row in ds])
    chosen = random.Random(seed).sample(ids, n)
    chosen_set = set(chosen)
    by_id = {row["instance_id"]: row for row in ds if row["instance_id"] in chosen_set}
    return [_row_to_instance(by_id[i], source="swebench_verified") for i in chosen]


def load_heldout_jsonl(path: Path, n: int | None = None) -> list[EvalInstance]:
    """Read the held-out SWE-bench-style JSONL produced by datagen.

    `n` is an upper-bound; if the file has fewer rows we take what we have (no
    padding). The file is expected to contain *only* the heldout split — we
    do not sub-sample here, because datagen already carved out 10 instances
    with seed=42 and a non-overlapping-with-training contract.
    """
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if n is not None:
        lines = lines[:n]
    out: list[EvalInstance] = []
    for row in lines:
        out.append(_row_to_instance(row, source="heldout"))
    return out


def _row_to_instance(row: dict, *, source: Source) -> EvalInstance:
    return EvalInstance(
        instance_id=row["instance_id"],
        source=source,
        repo=row["repo"],
        base_commit=row["base_commit"],
        instruction=row.get("problem_statement") or row.get("instruction", ""),
        test_command=_resolve_test_command(row),
        reference_patch=row.get("patch"),
        fail_to_pass=row.get("FAIL_TO_PASS") or row.get("fail_to_pass"),
        pass_to_pass=row.get("PASS_TO_PASS") or row.get("pass_to_pass"),
    )


def _resolve_test_command(row: dict) -> list[str]:
    cmd = row.get("test_command")
    if isinstance(cmd, list):
        return list(cmd)
    if isinstance(cmd, str) and cmd.strip():
        return cmd.split()
    # Fallback: run the F2P tests individually; SWE-bench Verified rows include
    # FAIL_TO_PASS as JSON list of nodeids.
    f2p = row.get("FAIL_TO_PASS") or row.get("fail_to_pass") or []
    if isinstance(f2p, str):
        try:
            f2p = json.loads(f2p)
        except json.JSONDecodeError:
            f2p = [f2p]
    if f2p:
        return ["pytest", "-x", *f2p]
    return ["pytest", "-x"]


def mixed_sample(
    *,
    swebench_n: int,
    heldout_path: Path | None,
    heldout_n: int,
    seed: int,
    offline: bool = False,
) -> list[EvalInstance]:
    """Returns the merged 30-instance eval set (20 cross-repo + 10 in-dist)."""
    instances: list[EvalInstance] = []
    if not offline and swebench_n > 0:
        instances.extend(sample_swebench_verified(swebench_n, seed))
    if heldout_path is not None and heldout_n > 0:
        instances.extend(load_heldout_jsonl(heldout_path, heldout_n))
    return instances
