from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from common.config import SEED

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
NEBIUS_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@dataclass(slots=True)
class DatagenConfig:
    repo: str = "fastapi/fastapi"
    t_per_method: int = 5
    base: bool = True
    validation_timeout_seconds: int = 120
    llm_concurrency: int = 8
    docker_concurrency: int = 4
    heldout_count: int = 5
    seed: int = SEED
    output_root: Path = field(default_factory=lambda: Path("/workspace/datasets/pilot"))
    repos_root: Path = field(default_factory=lambda: Path("/workspace/repos"))
    sessions_root: Path = field(default_factory=lambda: Path("/workspace/sessions"))
    docker_cache_root: Path = field(default_factory=lambda: Path("/workspace/docker-cache"))
    methods: tuple[str, ...] = ("lm_modify", "lm_rewrite", "procedural", "pr_mirror")
    offline: bool = False
    dry_run: bool = False
    max_prs: int = 15

    @classmethod
    def from_env(cls) -> "DatagenConfig":
        kw: dict = {}
        if v := os.environ.get("DATAGEN_REPO"):
            kw["repo"] = v
        if v := os.environ.get("DATAGEN_T"):
            kw["t_per_method"] = int(v)
        if v := os.environ.get("DATAGEN_OUTPUT"):
            kw["output_root"] = Path(v)
        return cls(**kw)
