from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from common.config import apply_seed
from common.logging import TraceLogger
from common.session import SessionDir

from datagen.config import DatagenConfig
from datagen.pipeline import Pipeline


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="datagen", description="SWE-Smith pilot data generation")
    p.add_argument("--repo", default="fastapi/fastapi")
    p.add_argument("--t", type=int, default=15, dest="t_per_method")
    p.add_argument("--heldout", type=int, default=10)
    p.add_argument("--llm-concurrency", type=int, default=8)
    p.add_argument("--docker-concurrency", type=int, default=4)
    p.add_argument("--max-prs", type=int, default=None)
    p.add_argument("--output-root", type=Path, default=Path("/workspace/datasets/pilot"))
    p.add_argument("--repos-root", type=Path, default=Path("/workspace/repos"))
    p.add_argument("--offline", action="store_true", help="skip Nebius; only procedural+pr_mirror run")
    p.add_argument("--dry-run", action="store_true", help="stub PR enumeration; just verify wiring")
    return p.parse_args()


async def _run(cfg: DatagenConfig) -> int:
    apply_seed(cfg.seed)
    sess = SessionDir.create(kind="datagen", root=cfg.sessions_root)
    trace = TraceLogger(sess.trace_path)
    trace.log("datagen.start", cfg=dict(
        repo=cfg.repo, t=cfg.t_per_method, offline=cfg.offline, dry_run=cfg.dry_run
    ))
    if cfg.dry_run:
        trace.log("datagen.dry_run", note="wiring ok; exiting without cloning or API calls")
        return 0
    methods = tuple(m for m in cfg.methods if not (cfg.offline and m.startswith("lm_")))
    cfg = DatagenConfig(
        repo=cfg.repo, t_per_method=cfg.t_per_method,
        validation_timeout_seconds=cfg.validation_timeout_seconds,
        llm_concurrency=cfg.llm_concurrency, docker_concurrency=cfg.docker_concurrency,
        heldout_count=cfg.heldout_count, seed=cfg.seed, output_root=cfg.output_root,
        repos_root=cfg.repos_root, sessions_root=cfg.sessions_root,
        harbor_root=cfg.harbor_root, docker_cache_root=cfg.docker_cache_root,
        methods=methods, offline=cfg.offline, dry_run=cfg.dry_run, max_prs=cfg.max_prs,
    )
    pipeline = Pipeline(cfg, trace=trace)
    records = await pipeline.run()
    trace.log("datagen.done", n_pilot=len(records))
    return 0


def main() -> None:
    args = _parse_args()
    cfg = DatagenConfig(
        repo=args.repo, t_per_method=args.t_per_method,
        llm_concurrency=args.llm_concurrency, docker_concurrency=args.docker_concurrency,
        heldout_count=args.heldout, output_root=args.output_root,
        repos_root=args.repos_root, harbor_root=args.output_root / "harbor",
        offline=args.offline, dry_run=args.dry_run, max_prs=args.max_prs,
    )
    raise SystemExit(asyncio.run(_run(cfg)))


if __name__ == "__main__":
    main()
