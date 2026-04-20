from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiofiles

from common.logging import TraceLogger
from common.session import SessionDir, Ticket
from common.ids import make_ticket_id

from .config import EvalConfig
from .rollout import RolloutResult, VllmClient, run_single_rollout
from .sample import EvalInstance


@dataclass(slots=True)
class SourceMetrics:
    source: str
    n: int
    n_passed: int
    pass_rate: float


@dataclass(slots=True)
class EvalSummary:
    total: int
    per_source: dict[str, SourceMetrics]
    elapsed_seconds: float
    checkpoint: str
    seed: int


def compute_metrics(results: list[RolloutResult]) -> dict[str, SourceMetrics]:
    """Per-source so memorisation effects (high heldout, low verified) stay visible."""
    by_src: dict[str, list[RolloutResult]] = {}
    for r in results:
        by_src.setdefault(r.source, []).append(r)
    return {
        src: SourceMetrics(
            source=src,
            n=len(rs),
            n_passed=sum(1 for r in rs if r.passed),
            pass_rate=round(sum(1 for r in rs if r.passed) / len(rs), 4) if rs else 0.0,
        )
        for src, rs in by_src.items()
    }


async def run_eval(
    *,
    instances: list[EvalInstance],
    cfg: EvalConfig,
    session: SessionDir,
    checkpoint: Path,
    vllm_base_url: str,
) -> EvalSummary:
    """Orchestrate N rollouts under docker + llm semaphores; write per-instance
    JSONL and an aggregate summary.json. Returns the summary so the CLI can
    print it and non-zero exit if pass_rate < some threshold (caller's choice).
    """
    trace = TraceLogger(session.trace_path)
    ticket = Ticket.start(
        tickets_dir=session.tickets,
        ticket_id=make_ticket_id(1, "eval.run"),
        operation="eval.run",
        inputs={"checkpoint": str(checkpoint), "n": len(instances),
                "seed": cfg.seed, "swebench_n": cfg.swebench_n,
                "heldout_n": cfg.heldout_n},
    )

    llm_sem = asyncio.Semaphore(cfg.llm_concurrency)
    docker_sem = asyncio.Semaphore(cfg.docker_concurrency)
    llm = VllmClient(
        base_url=vllm_base_url,
        model=str(checkpoint),
        llm_sem=llm_sem,
        temperature=cfg.vllm_temperature,
        top_p=cfg.vllm_top_p,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
    )

    results_path = session.logs / "results.jsonl"
    summary_path = session.logs / "summary.json"

    t0 = time.time()
    try:
        async def _one(inst: EvalInstance) -> RolloutResult:
            trace.log("eval.rollout.start", instance_id=inst.instance_id, source=inst.source)
            try:
                res = await run_single_rollout(
                    instance=inst, cfg=cfg, llm=llm, docker_sem=docker_sem,
                )
            except Exception as exc:
                trace.log("eval.rollout.error", instance_id=inst.instance_id, error=str(exc))
                # Treat errors as failed rollouts — the alternative (skip) would bias stats.
                res = RolloutResult(
                    instance_id=inst.instance_id, source=inst.source, reward=0.0,
                    passed=False, n_tool_calls=0, finalise_reason=f"error:{type(exc).__name__}",
                    final_diff="", trajectory=[], defense_log=[],
                )
            async with _results_lock:
                async with aiofiles.open(results_path, "a") as f:
                    await f.write(json.dumps(_serialise_result(res)) + "\n")
            trace.log("eval.rollout.end", instance_id=inst.instance_id,
                      passed=res.passed, reward=res.reward, n_tool_calls=res.n_tool_calls)
            return res

        _results_lock = asyncio.Lock()
        results = await asyncio.gather(*[_one(i) for i in instances])

        per_source = compute_metrics(results)
        summary = EvalSummary(
            total=len(results),
            per_source=per_source,
            elapsed_seconds=round(time.time() - t0, 2),
            checkpoint=str(checkpoint),
            seed=cfg.seed,
        )
        summary_path.write_text(json.dumps(_serialise_summary(summary), indent=2))
        ticket.finish(outputs={"summary_path": str(summary_path),
                               "results_path": str(results_path),
                               "total": summary.total,
                               "per_source": {k: asdict(v) for k, v in per_source.items()}})
        trace.log("eval.complete", **_serialise_summary(summary))
        return summary
    except Exception as exc:
        ticket.finish(state="failed", error=f"{type(exc).__name__}: {exc}")
        trace.log("eval.failed", error=str(exc))
        raise
    finally:
        await llm.close()


def _serialise_result(r: RolloutResult) -> dict[str, Any]:
    return {
        "instance_id": r.instance_id,
        "source": r.source,
        "reward": r.reward,
        "passed": r.passed,
        "n_tool_calls": r.n_tool_calls,
        "finalise_reason": r.finalise_reason,
        "final_diff": r.final_diff,
        "trajectory": r.trajectory,
        "defense_log": [asdict(d) for d in r.defense_log],
    }


def _serialise_summary(s: EvalSummary) -> dict[str, Any]:
    return {
        "total": s.total,
        "elapsed_seconds": s.elapsed_seconds,
        "checkpoint": s.checkpoint,
        "seed": s.seed,
        "per_source": {k: asdict(v) for k, v in s.per_source.items()},
    }
