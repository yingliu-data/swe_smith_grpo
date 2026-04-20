from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from pathlib import Path

import httpx

from .config import EvalConfig


class VllmLaunchError(RuntimeError):
    pass


async def launch(checkpoint: Path, cfg: EvalConfig) -> asyncio.subprocess.Process:
    """Spawn vLLM server on the given checkpoint with eval-time params.

    Eval uses prefix caching + greedy sampling (temperature=0) for determinism.
    Training uses prefix_caching=false + temp=1.0; the flag flip is the point of
    this module.
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(checkpoint),
        "--port", str(cfg.vllm_port),
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(cfg.vllm_gpu_memory_utilization),
        "--seed", str(cfg.seed),
    ]
    if cfg.vllm_enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    env = {**os.environ, "VLLM_DO_NOT_TRACK": "1"}
    return await asyncio.create_subprocess_exec(*cmd, env=env)


async def wait_ready(cfg: EvalConfig, timeout_seconds: int = 600) -> None:
    url = f"http://127.0.0.1:{cfg.vllm_port}/v1/models"
    async with httpx.AsyncClient(timeout=2.0) as client:
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            await asyncio.sleep(2.0)
    raise VllmLaunchError(f"vLLM did not become ready within {timeout_seconds}s")


def terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
