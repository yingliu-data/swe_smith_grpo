from __future__ import annotations

from pathlib import Path

import pytest

from common.session import (
    InvalidTransition,
    Manifest,
    MemoryRecord,
    MemoryStore,
    SessionDir,
    Ticket,
    transition,
)


def test_session_dir_create_has_all_subdirs(tmp_path: Path):
    s = SessionDir.create(kind="train", root=tmp_path)
    for sub in ("workspace", "tickets", "logs", "ipc", "checkpoints", "memory"):
        assert (s.root / sub).is_dir(), sub
    assert s.root.name.startswith("train-")


def test_state_transitions():
    assert transition("active", "complete") == "complete"
    assert transition("active", "failed") == "failed"
    with pytest.raises(InvalidTransition):
        transition("complete", "active")
    with pytest.raises(InvalidTransition):
        transition("failed", "complete")


def test_ticket_start_finish_roundtrip(tmp_path: Path):
    s = SessionDir.create(kind="train", root=tmp_path)
    t = Ticket.start(
        tickets_dir=s.tickets,
        ticket_id="tk_0001_train_run",
        operation="train.run",
        inputs={"seed": 42, "profile": "smoke"},
    )
    assert t.path.exists()
    t.finish(outputs={"steps": 30}, state="complete")
    import json

    payload = json.loads(t.path.read_text())
    assert payload["state"] == "complete"
    assert payload["outputs"] == {"steps": 30}


def test_manifest_compute_verify_detects_corruption(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    m = Manifest.compute(tmp_path, ["a.txt", "b.txt"])
    assert m.verify(tmp_path) == []
    (tmp_path / "a.txt").write_text("hello-corrupted")
    assert m.verify(tmp_path) == ["a.txt"]
    round_trip = Manifest.from_dict(m.to_dict())
    assert round_trip.files == m.files


def test_memory_store_roundtrip(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem")
    rec = MemoryRecord(
        name="fastapi_env_setup",
        description="deps for fastapi instance abc123",
        body="install anyio==3.5\ninstall pytest==7.4",
    )
    store.put("fastapi_env_setup", rec)
    got = store.get("fastapi_env_setup")
    assert got is not None
    assert got.body.startswith("install anyio")
    assert len(store.all()) == 1
