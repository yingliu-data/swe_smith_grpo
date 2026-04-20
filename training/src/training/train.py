from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
from pathlib import Path

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
    return p.parse_args()


async def _run(args: argparse.Namespace) -> int:
    apply_seed()
    profile = load_profile(args.profile)
    session = _open_or_create_session(args.sessions_root, args.resume)
    log = RunLogger(session)
    ticket = log.next_ticket(
        "train.run",
        inputs={
            "profile": profile.profile,
            "dataset": str(args.dataset),
            "resume": args.resume,
            "output_dir": str(args.output_dir),
        },
    )
    log.trace.log("train.start", profile=profile.profile, dataset=str(args.dataset))

    cfg_root = Path(__file__).parent / "src" / "training" / "configs"
    cmd = [
        *args.prime_rl.split(),
        "--trainer", f"@{cfg_root / 'train.toml'}",
        "--orchestrator", f"@{cfg_root / 'orch.toml'}",
        "--inference", f"@{cfg_root / 'infer.toml'}",
        "--env", "training.swe_env:SWEAgentEnv",
        "--dataset", str(args.dataset),
        "--output-dir", str(args.output_dir / session.session_id),
    ]
    if args.resume:
        cmd += ["--resume-from", str(_resume_path(args.output_dir, args.resume))]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "ML_SYSTEMS_SESSION": str(session.root)},
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
