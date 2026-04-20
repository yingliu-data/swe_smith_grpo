"""Docker-dependent tests; skipped unless DOCKER_TESTS=1 is set."""
from __future__ import annotations

import asyncio
import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DOCKER_TESTS") != "1",
    reason="requires docker daemon; set DOCKER_TESTS=1 to run",
)


@pytest.mark.asyncio
async def test_exec_does_not_block_loop() -> None:
    """Defense #9: aiodocker streaming exec must yield the event loop.

    A 3 s in-container sleep must not starve concurrent asyncio.sleep ticks.
    """
    import aiodocker

    from agent import DockerEnvironment, TaskSpec

    task = TaskSpec(repository="test/x", base_commit="HEAD", instruction="", test_command=["true"])
    env = DockerEnvironment(workspace_root="/tmp", image="alpine:3.19", task=task)
    try:
        env._docker = aiodocker.Docker()
        env._container = await env._docker.containers.create_or_replace(
            name="ml-systems-loop-test",
            config={"Image": "alpine:3.19", "Cmd": ["sleep", "infinity"], "HostConfig": {"NetworkMode": "none"}},
        )
        await env._container.start()

        ticks = 0

        async def tick():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.1)
                ticks += 1

        ticker = asyncio.create_task(tick())
        start = time.monotonic()
        rc, _ = await env._exec(["sleep", "3"])
        elapsed = time.monotonic() - start
        ticker.cancel()

        assert rc == 0
        assert elapsed >= 2.9
        assert ticks >= 20
    finally:
        await env.teardown()
