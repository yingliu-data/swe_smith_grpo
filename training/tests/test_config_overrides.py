"""Run-time prime-rl config overrides (train.py::_materialize_configs).

Exercised against the real baked configs so the tests fail if the toml
layout drifts from what the materializer expects.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from training.train import _materialize_configs

CFG_ROOT = Path(__file__).parent.parent / "src" / "training" / "configs"


def _load(root: Path, name: str) -> dict:
    return tomllib.loads((root / f"{name}.toml").read_text())


def test_no_overrides_returns_baked_root(tmp_path):
    assert _materialize_configs(CFG_ROOT, tmp_path / "cfg", {}) == CFG_ROOT
    assert not (tmp_path / "cfg").exists()


def test_seq_len_fans_out_to_train_and_orch(tmp_path):
    out = _materialize_configs(CFG_ROOT, tmp_path / "cfg", {"seq_len": 2048})
    assert out == tmp_path / "cfg"
    assert _load(out, "train")["model"]["seq_len"] == 2048
    assert _load(out, "orch")["seq_len"] == 2048
    # Untouched values survive the round-trip.
    assert _load(out, "orch")["batch_size"] == _load(CFG_ROOT, "orch")["batch_size"]
    assert _load(out, "infer") == _load(CFG_ROOT, "infer")


def test_max_steps_fans_out_to_train_and_orch(tmp_path):
    out = _materialize_configs(CFG_ROOT, tmp_path / "cfg", {"max_steps": 3})
    assert _load(out, "train")["max_steps"] == 3
    assert _load(out, "orch")["max_steps"] == 3


def test_orchestrator_only_keys(tmp_path):
    out = _materialize_configs(
        CFG_ROOT, tmp_path / "cfg",
        {"batch_size": 8, "rollouts_per_example": 2,
         "max_async_level": 2, "max_off_policy_steps": 4},
    )
    orch = _load(out, "orch")
    assert (orch["batch_size"], orch["rollouts_per_example"]) == (8, 2)
    assert (orch["max_async_level"], orch["max_off_policy_steps"]) == (2, 4)
    # Trainer file untouched by orch-only keys.
    assert _load(out, "train") == _load(CFG_ROOT, "train")


def test_seq_len_capped_by_infer_max_model_len(tmp_path):
    cap = _load(CFG_ROOT, "infer")["model"]["max_model_len"]
    with pytest.raises(SystemExit, match="max_model_len"):
        _materialize_configs(CFG_ROOT, tmp_path / "cfg", {"seq_len": cap + 1})


def test_batch_size_must_divide_by_group(tmp_path):
    with pytest.raises(SystemExit, match="divisible"):
        _materialize_configs(
            CFG_ROOT, tmp_path / "cfg",
            {"batch_size": 10, "rollouts_per_example": 4},
        )


async def test_run_with_overrides_exercises_full_wrapper(tmp_path):
    """Drive _run end-to-end with a stub prime-rl command.

    Regression: the override-announcement line in _run once called a helper
    that only exists in datagen (_say) and NameError'd on the pod — unit
    tests on _materialize_configs alone can't catch bugs in _run's body.
    """
    import argparse

    from training.train import _run

    args = argparse.Namespace(
        dataset=tmp_path / "fake.jsonl", profile="smoke", resume=None,
        output_dir=tmp_path / "ckpt", sessions_root=tmp_path / "sess",
        prime_rl="python -c pass",  # resolvable stub; exits 0 immediately
        seq_len=2048, batch_size=None, rollouts_per_example=None,
        max_steps=2, max_async_level=None, max_off_policy_steps=None,
    )
    assert await _run(args) == 0
    cfg_dirs = list((tmp_path / "sess").glob("train-*/configs"))
    assert len(cfg_dirs) == 1
    assert tomllib.loads((cfg_dirs[0] / "orch.toml").read_text())["seq_len"] == 2048
