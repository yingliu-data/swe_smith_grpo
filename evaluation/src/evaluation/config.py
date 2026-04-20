from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class EvalConfig:
    """All eval-time knobs in one place. vLLM is re-launched with prefix caching
    ON and temperature=0 (greedy) for determinism."""

    swebench_n: int = 20
    heldout_n: int = 10
    seed: int = 42

    llm_concurrency: int = 8
    docker_concurrency: int = 4

    max_tool_calls: int = 20
    max_wall_seconds: int = 120
    max_tokens: int = 2048

    vllm_port: int = 8000
    vllm_gpu_memory_utilization: float = 0.85  # no trainer contention → push higher
    vllm_enable_prefix_caching: bool = True
    vllm_temperature: float = 0.0  # greedy
    vllm_top_p: float = 1.0

    swebench_dataset: str = "princeton-nlp/SWE-bench_Verified"
    rollout_image: str = "ml-systems/rollout:latest"
    workspace_root: Path = Path("/workspace")


DEFAULT = EvalConfig()
