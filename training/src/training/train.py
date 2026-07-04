from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

import tomli_w

from common.config import apply_seed
from common.session import SessionDir

from training.config import load_profile
from training.session_logger import RunLogger
from training.watchdog import StaleHeartbeatError, watchdog_loop


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="train", description="GRPO+ training wrapper around prime-rl")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    p.add_argument("--resume", default=None, help='"latest" or a specific run_id')
    p.add_argument("--output-dir", type=Path, default=Path("/workspace/checkpoints"))
    p.add_argument("--sessions-root", type=Path, default=Path("/workspace/sessions"))
    p.add_argument(
        "--prime-rl",
        default=os.environ.get("PRIME_RL_CMD", "uv run rl"),
        help="command to invoke the prime-rl trainer binary",
    )
    # Run-time overrides for the baked prime-rl tomls, so a `docker run` can
    # retune the loop without rebuilding the image (see _materialize_configs).
    # Defaults of None mean "keep the baked value".
    g = p.add_argument_group("prime-rl config overrides")
    g.add_argument("--seq-len", type=int, default=None,
                   help="context length; kept in sync across train.toml and orch.toml")
    g.add_argument("--batch-size", type=int, default=None,
                   help="total rollouts per step (orch.toml)")
    g.add_argument("--rollouts-per-example", type=int, default=None,
                   help="GRPO group size G; must divide batch-size (orch.toml)")
    g.add_argument("--max-steps", type=int, default=None,
                   help="training steps; kept in sync across train.toml and orch.toml")
    g.add_argument("--max-async-level", type=int, default=None,
                   help="rollout/update overlap depth (orch.toml)")
    g.add_argument("--max-off-policy-steps", type=int, default=None,
                   help="staleness bound for buffered rollouts (orch.toml)")
    return p.parse_args()


_OVERRIDE_KEYS = ("seq_len", "batch_size", "rollouts_per_example",
                  "max_steps", "max_async_level", "max_off_policy_steps")


def _collect_overrides(args: argparse.Namespace) -> dict[str, int]:
    return {k: v for k in _OVERRIDE_KEYS if (v := getattr(args, k)) is not None}


def _profile_geometry(profile) -> dict[str, int]:
    """Loop geometry a --profile contributes to the toml materialization.

    Merged under explicit CLI overrides (CLI flag > profile > baked toml),
    which is what makes `--profile full` actually run the full schedule.
    """
    return {
        "max_steps": profile.max_steps,
        "batch_size": profile.batch_size,
        "rollouts_per_example": profile.rollouts_per_example,
        "seq_len": profile.seq_len,
    }


def _materialize_configs(cfg_root: Path, out_root: Path, overrides: dict[str, int]) -> Path:
    """Apply run-time overrides on top of the baked prime-rl tomls.

    Returns cfg_root untouched when there is nothing to override; otherwise
    writes patched copies of train/orch/infer.toml under out_root and returns
    that. Cross-file invariants are enforced structurally: seq_len and
    max_steps are single flags fanned out to every file that carries them, so
    the CLI cannot desynchronise train.toml from orch.toml.
    """
    if not overrides:
        return cfg_root
    cfgs = {name: tomllib.loads((cfg_root / f"{name}.toml").read_text())
            for name in ("train", "orch", "infer")}
    if (v := overrides.get("seq_len")) is not None:
        max_model_len = cfgs["infer"]["model"]["max_model_len"]
        if v > max_model_len:
            raise SystemExit(
                f"--seq-len {v} exceeds infer.toml max_model_len {max_model_len}")
        cfgs["train"]["model"]["seq_len"] = v
        cfgs["orch"]["seq_len"] = v
    if (v := overrides.get("max_steps")) is not None:
        cfgs["train"]["max_steps"] = v
        cfgs["orch"]["max_steps"] = v
    for key in ("batch_size", "rollouts_per_example", "max_async_level",
                "max_off_policy_steps"):
        if (v := overrides.get(key)) is not None:
            cfgs["orch"][key] = v
    bs, g = cfgs["orch"]["batch_size"], cfgs["orch"]["rollouts_per_example"]
    if bs % g:
        raise SystemExit(
            f"batch_size {bs} is not divisible by rollouts_per_example {g}")
    out_root.mkdir(parents=True, exist_ok=True)
    for name, data in cfgs.items():
        (out_root / f"{name}.toml").write_text(tomli_w.dumps(data))
    return out_root


async def _run(args: argparse.Namespace) -> int:
    apply_seed()
    profile = load_profile(args.profile)
    session = _open_or_create_session(args.sessions_root, args.resume)
    log = RunLogger(session)
    cli_overrides = _collect_overrides(args)
    # Precedence: explicit CLI flag > profile geometry > baked toml.
    geometry = {**_profile_geometry(profile), **cli_overrides}
    ticket = log.next_ticket(
        "train.run",
        inputs={
            "profile": profile.profile,
            "dataset": str(args.dataset),
            "resume": args.resume,
            "output_dir": str(args.output_dir),
            "overrides": cli_overrides,
            "geometry": geometry,
        },
    )
    log.trace.log("train.start", profile=profile.profile, dataset=str(args.dataset),
                  overrides=cli_overrides, geometry=geometry)

    cmd_parts = args.prime_rl.split()
    if _prime_rl_missing(cmd_parts):
        msg = (
            f"prime-rl not resolvable from {args.prime_rl!r}. "
            "prime-rl is Linux+CUDA-only; run via `docker compose -f "
            "infra/docker-compose.train.yml up --build train` on Pod B, or set "
            "PRIME_RL_CMD to a working `rl` entry point."
        )
        print(msg, file=sys.stderr)
        ticket.finish(state="failed", error=msg)
        return 127

    cfg_root = _materialize_configs(
        Path(__file__).parent / "configs", session.root / "configs", geometry)
    print(f"[train] loop geometry {geometry} "
          f"(profile={profile.profile}, cli={cli_overrides}) -> {cfg_root}",
          file=sys.stderr, flush=True)
    cmd = [
        *cmd_parts,
        "--trainer", f"@{cfg_root / 'train.toml'}",
        "--orchestrator", f"@{cfg_root / 'orch.toml'}",
        "--inference", f"@{cfg_root / 'infer.toml'}",
        "--output-dir", str(args.output_dir / session.session_id),
    ]
    # Env + dataset are encoded in orch.toml's [[env]] section (prime-rl's
    # RLConfig has no top-level --env/--dataset flags). The --dataset wrapper
    # arg is retained for wrapper-level bookkeeping only.
    if args.resume:
        cmd += ["--ckpt.resume-step", "-1"]  # -1 = restart from latest checkpoint

    subprocess_env = {**os.environ, "ML_SYSTEMS_SESSION": str(session.root)}
    _pin_coloc_cuda_visible_devices(subprocess_env)
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env=subprocess_env,
    )
    streamer = asyncio.create_task(_stream_to_trace(proc, log))
    watcher = asyncio.create_task(
        watchdog_loop(
            heartbeat_path=session.heartbeat_path,
            stale_after_seconds=profile.heartbeat_stale_seconds,
            on_stall=lambda: _sigterm(proc),
        )
    )
    try:
        rc = await proc.wait()
    except StaleHeartbeatError:
        log.trace.log("train.stall_detected")
        ticket.finish(state="failed", error="heartbeat stall")
        return 42
    finally:
        watcher.cancel()
        await streamer

    if rc == 0:
        ticket.finish(outputs={"exit_code": rc, "session": session.session_id}, state="complete")
        log.trace.log("train.complete")
        return 0
    ticket.finish(state="failed", error=f"prime-rl exit {rc}")
    log.trace.log("train.failed", exit_code=rc)
    return rc


def _open_or_create_session(sessions_root: Path, resume: str | None) -> SessionDir:
    if resume:
        candidate = sessions_root / resume if resume != "latest" else _pick_latest(sessions_root)
        if candidate and candidate.exists():
            return SessionDir.open(candidate)
    return SessionDir.create(kind="train", root=sessions_root)


def _pick_latest(sessions_root: Path) -> Path | None:
    if not sessions_root.exists():
        return None
    runs = sorted(sessions_root.glob("train-*"))
    return runs[-1] if runs else None


def _resume_path(output_dir: Path, resume: str) -> Path:
    if resume == "latest":
        from training.checkpoint import latest_valid

        ckpt = latest_valid(output_dir)
        if ckpt is None:
            raise RuntimeError("no valid checkpoint found for --resume latest")
        return ckpt.path
    return output_dir / resume


def _pin_coloc_cuda_visible_devices(env: dict[str, str]) -> None:
    # RunPod rewrites CUDA_VISIBLE_DEVICES at pod start to match the
    # assigned GPU count, overriding the image-level ENV. On a 1-GPU pod
    # that leaves a single "0", and prime-rl's deployment check rejects
    # num_train_gpus + num_infer_gpus > 1. Re-pin "0,0" in the subprocess
    # env so the launcher sees two entries and co-locates both workers on
    # the one physical card.
    env["CUDA_VISIBLE_DEVICES"] = "0,0"


def _prime_rl_missing(cmd_parts: list[str]) -> bool:
    """Return True if neither `rl` nor the target of `uv run rl` is installed."""
    if shutil.which(cmd_parts[0]) is None:
        return True
    # Typical shape: `uv run rl` or `uv --project X run rl`. Use `uv run` to
    # ask uv whether the `rl` entry point resolves without spawning a trainer.
    try:
        probe = subprocess.run(
            [*cmd_parts, "--help"],
            capture_output=True, timeout=15, text=True,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False  # let the real spawn speak for itself
    return probe.returncode != 0 and "Failed to spawn" in (probe.stderr or "")


def _sigterm(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass


async def _stream_to_trace(proc: asyncio.subprocess.Process, log: RunLogger) -> None:
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        log.trace.log("prime_rl.stdout", line=line.decode(errors="replace").rstrip())


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
