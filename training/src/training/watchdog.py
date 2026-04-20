from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable


class StaleHeartbeatError(RuntimeError):
    pass


async def watchdog_loop(
    *,
    heartbeat_path: Path,
    stale_after_seconds: int,
    on_stall: Callable[[], None],
    poll_seconds: int = 30,
) -> None:
    """Poll a heartbeat JSON file; trigger `on_stall` if the file goes stale.

    The trainer writes ``{"ts": <epoch>, "step": N}`` to ``heartbeat_path`` every ~30s.
    If we don't see a fresh write for ``stale_after_seconds``, we assume the trainer
    has stalled (deadlock, hang, bad CUDA state) and invoke the callback — typically
    sending SIGTERM to the trainer subprocess.
    """
    last_seen_ts: float | None = None
    while True:
        await asyncio.sleep(poll_seconds)
        ts = _read_heartbeat_ts(heartbeat_path)
        now = time.time()
        if ts is None:
            if last_seen_ts is None:
                continue
            if now - last_seen_ts > stale_after_seconds:
                on_stall()
                raise StaleHeartbeatError(f"heartbeat file missing for > {stale_after_seconds}s")
            continue
        if last_seen_ts is None or ts > last_seen_ts:
            last_seen_ts = ts
            continue
        if now - ts > stale_after_seconds:
            on_stall()
            raise StaleHeartbeatError(
                f"heartbeat stale: last ts={ts}, now={now}, delta={now-ts:.1f}s"
            )


def _read_heartbeat_ts(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())["ts"])
    except Exception:
        return None
