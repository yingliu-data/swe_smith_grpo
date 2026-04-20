from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from common.config import apply_seed
from common.session import SessionDir

from . import vllm_server
from .config import DEFAULT, EvalConfig
from .runner import run_eval
from .sample import mixed_sample


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="evaluate", description="Post-training evaluation CLI")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--swebench-n", type=int, default=DEFAULT.swebench_n)
    p.add_argument("--heldout", type=Path, default=None,
                   help="path to datagen/output/heldout.jsonl")
    p.add_argument("--heldout-n", type=int, default=DEFAULT.heldout_n)
    p.add_argument("--sessions-root", type=Path, default=Path("/workspace/sessions"))
    p.add_argument("--seed", type=int, default=DEFAULT.seed)
    p.add_argument("--dry-run", action="store_true",
                   help="print the sample + config without launching vLLM or any containers")
    p.add_argument("--offline", action="store_true",
                   help="skip HuggingFace SWE-bench fetch; use only the heldout file")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> int:
    apply_seed(args.seed)
    cfg = EvalConfig(
        swebench_n=args.swebench_n,
        heldout_n=args.heldout_n,
        seed=args.seed,
    )
    instances = mixed_sample(
        swebench_n=cfg.swebench_n if not args.offline else 0,
        heldout_path=args.heldout,
        heldout_n=cfg.heldout_n if args.heldout is not None else 0,
        seed=cfg.seed,
        offline=args.offline,
    )
    if args.dry_run:
        summary = [{"instance_id": i.instance_id, "source": i.source, "repo": i.repo}
                   for i in instances]
        print(json.dumps({"n": len(instances), "instances": summary}, indent=2))
        return 0

    session = SessionDir.create(kind="eval", root=args.sessions_root)
    proc = await vllm_server.launch(args.checkpoint, cfg)
    try:
        await vllm_server.wait_ready(cfg)
        summary = await run_eval(
            instances=instances, cfg=cfg, session=session,
            checkpoint=args.checkpoint,
            vllm_base_url=f"http://127.0.0.1:{cfg.vllm_port}",
        )
        print(json.dumps({
            "total": summary.total,
            "per_source": {k: {"n": v.n, "pass_rate": v.pass_rate}
                           for k, v in summary.per_source.items()},
            "session": str(session.root),
        }, indent=2))
        return 0
    finally:
        vllm_server.terminate(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            proc.kill()


def main() -> None:
    args = _parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
