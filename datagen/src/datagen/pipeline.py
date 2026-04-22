from __future__ import annotations

import asyncio
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.logging import TraceLogger
from common.session import MemoryRecord, MemoryStore

from .config import DatagenConfig
from .methods import Context, get_method, iter_methods
from .methods.base import BaseMutationMethod, Candidate
from .methods.lm import _LMBase
from .nebius_client import NebiusClient
from .repo_manager import PullRequestInfo, RepoManager
from .validator import Validator, split_reference_patch, extract_f2p_nodeids
from .writers import SWEBenchJSONLWriter
from .writers.swebench_jsonl import InstanceRecord
from .yield_logger import MethodYield, YieldLogger


@dataclass(slots=True)
class _PerMethodStats:
    attempted: int = 0
    passed: int = 0
    wall_total: float = 0.0


class Pipeline:
    def __init__(self, cfg: DatagenConfig, *, trace: TraceLogger):
        self._cfg = cfg
        self._trace = trace
        self._llm_sem = asyncio.Semaphore(cfg.llm_concurrency)
        self._docker_sem = asyncio.Semaphore(cfg.docker_concurrency)
        self._repo_manager = RepoManager(repos_root=cfg.repos_root)
        self._validator = Validator(command_timeout_seconds=cfg.validation_timeout_seconds)

    async def run(self) -> list[InstanceRecord]:
        cfg = self._cfg
        print(f"[datagen] ensuring clone of {cfg.repo}", file=sys.stderr, flush=True)
        repo_dir = await self._repo_manager.ensure_clone(cfg.repo)
        print(f"[datagen] clone ready at {repo_dir}; enumerating PRs", file=sys.stderr, flush=True)
        prs = await self._repo_manager.list_bug_prs(cfg.repo, limit=cfg.max_prs or 25)
        self._trace.log("datagen.prs_enumerated", repo=cfg.repo, n_prs=len(prs))
        if not prs:
            # Surface this immediately — silent n_prs=0 is indistinguishable from success
            # in stderr breadcrumbs, and it caused a full pod crash-loop before.
            raise RuntimeError(
                f"PR enumeration returned 0 matches for {cfg.repo}. "
                "Likely causes: bad label filter, GitHub rate limit (set GITHUB_TOKEN), "
                "or the repo has no merged PRs with the configured labels."
            )

        nebius: NebiusClient | None = None
        methods: dict[str, BaseMutationMethod] = {}
        if not cfg.offline:
            nebius = NebiusClient()
        for name in cfg.methods:
            cls = get_method(name)
            methods[name] = cls(nebius) if issubclass(cls, _LMBase) else cls()  # type: ignore[arg-type]

        jsonl_writer = SWEBenchJSONLWriter(cfg.output_root / "pilot.jsonl")
        heldout_writer = SWEBenchJSONLWriter(cfg.output_root / "heldout.jsonl")
        yield_logger = YieldLogger(cfg.output_root / "yield.csv")
        memory = MemoryStore(cfg.output_root / "setup_facts")

        stats: dict[str, _PerMethodStats] = {name: _PerMethodStats() for name in cfg.methods}
        passing: list[tuple[InstanceRecord, list[str]]] = []

        try:
            tasks: list[asyncio.Task[Any]] = []
            if not cfg.base:
                print(f"[datagen] fetching patches for {len(prs)} PRs x {len(methods)} methods x {cfg.t_per_method} trials", file=sys.stderr, flush=True)
            for i, pr in enumerate(prs, 1):
                print(f"[datagen] PR {i}/{len(prs)}: #{pr.number} base={pr.base_commit[:8]}", file=sys.stderr, flush=True)
                patch = await self._repo_manager.fetch_patch(pr)
                await self._repo_manager.checkout(repo_dir, pr.base_commit)
                if cfg.base:
                    if not any(l in {"bug", "fix"} for l in pr.labels):
                        continue
                    src_patch, test_patch = split_reference_patch(patch)
                    f2p = extract_f2p_nodeids(test_patch)
                    if not src_patch.strip() or not test_patch.strip() or not f2p:
                        continue
                    instance_id = f"{cfg.repo.replace('/', '__')}-{pr.number}-base"
                    rec = InstanceRecord(
                        instance_id=instance_id,
                        repo=cfg.repo,
                        base_commit=pr.base_commit,
                        problem_statement=f"# {pr.title}\n\n{(pr.body or '').strip()}",
                        patch=src_patch,
                        test_patch=test_patch,
                        FAIL_TO_PASS=f2p,
                        PASS_TO_PASS=[],
                        created_at=datetime.now(timezone.utc).isoformat(),
                        metadata={"method": "base", "pr": pr.number},
                    )
                    passing.append((rec, f2p))
                else:
                    for name, method in methods.items():
                        for t in range(cfg.t_per_method):
                            tasks.append(asyncio.create_task(
                                self._generate_validate_write(
                                    pr=pr, repo_dir=repo_dir, reference_patch=patch,
                                    method=method, trial=t, stats=stats[name],
                                    jsonl=jsonl_writer, memory=memory,
                                    passing=passing,
                                )
                            ))
            print(f"[datagen] scheduled {len(tasks)} tasks; awaiting gather (LLM+docker)", file=sys.stderr, flush=True)
            await asyncio.gather(*tasks, return_exceptions=True)
            print(f"[datagen] gather complete: passing={len(passing)}", file=sys.stderr, flush=True)
        finally:
            if nebius is not None:
                await nebius.close()

        for name, s in stats.items():
            avg = s.wall_total / max(s.attempted, 1)
            yield_logger.append(MethodYield(
                method=name, repo=cfg.repo,
                attempted=s.attempted, passed=s.passed, avg_seconds=avg,
            ))
            self._trace.log("datagen.method_yield", method=name,
                            attempted=s.attempted, passed=s.passed, rate=s.attempted and s.passed/s.attempted)

        # Held-out split: seeded sample of passing instances, NOT in pilot.jsonl
        rng = random.Random(cfg.seed)
        rng.shuffle(passing)
        held = passing[:cfg.heldout_count]
        pilot = passing[cfg.heldout_count:]
        for rec, _ in pilot:
            jsonl_writer.write(rec)
        for rec, _ in held:
            heldout_writer.write(rec)
        self._trace.log("datagen.split", pilot=len(pilot), heldout=len(held))
        return [rec for rec, _ in pilot]

    async def _generate_validate_write(
        self,
        *,
        pr: PullRequestInfo,
        repo_dir: Path,
        reference_patch: str,
        method: BaseMutationMethod,
        trial: int,
        stats: _PerMethodStats,
        jsonl: SWEBenchJSONLWriter,
        memory: MemoryStore,
        passing: list[tuple[InstanceRecord, list[str]]],
    ) -> None:
        stats.attempted += 1
        ctx = Context(
            repo=self._cfg.repo, repo_dir=repo_dir, pr_number=pr.number,
            base_commit=pr.base_commit, merge_commit=pr.merge_commit,
            pr_title=pr.title, pr_body=pr.body,
            reference_patch=reference_patch, seed=self._cfg.seed, trial_index=trial,
        )
        try:
            if isinstance(method, _LMBase):
                async with self._llm_sem:
                    cand = await method.generate(ctx)
            else:
                cand = await method.generate(ctx)
        except Exception as exc:
            self._trace.log("datagen.generate_error", method=method.name, pr=pr.number,
                            trial=trial, error=str(exc))
            return
        if cand is None:
            return

        async with self._docker_sem:
            result = await self._validator.validate(
                repo_dir=repo_dir, base_commit=pr.base_commit,
                reference_patch=reference_patch, buggy_patch=cand.buggy_patch,
            )
        stats.wall_total += result.wall_seconds
        self._trace.log("datagen.validate", method=method.name, pr=pr.number, trial=trial,
                        passed=result.passed, reason=result.reason, wall=result.wall_seconds,
                        diag=result.diag)
        if not result.passed:
            return
        stats.passed += 1

        _, test_patch = split_reference_patch(reference_patch)
        f2p = extract_f2p_nodeids(test_patch)
        instance_id = f"{self._cfg.repo.replace('/', '__')}-{pr.number}-{method.name}-{trial}"
        rec = InstanceRecord(
            instance_id=instance_id, repo=self._cfg.repo, base_commit=pr.base_commit,
            problem_statement=_build_problem_statement(pr, cand),
            patch=cand.buggy_patch, test_patch=test_patch,
            FAIL_TO_PASS=f2p, PASS_TO_PASS=[],
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={"method": method.name, "trial": trial, "pr": pr.number, "rationale": cand.rationale},
        )
        memory.put(
            f"{instance_id}",
            MemoryRecord(
                name=instance_id,
                description=f"setup hints for {self._cfg.repo} PR {pr.number} ({method.name})",
                body=f"# Base commit\n{pr.base_commit}\n\n# PR\n#{pr.number}: {pr.title}",
            ),
        )
        passing.append((rec, f2p))


def _build_problem_statement(pr: PullRequestInfo, cand: Candidate) -> str:
    return (
        f"# {pr.title}\n\n"
        f"{pr.body.strip()}\n\n"
        f"Observed bug: the referenced commit introduces a regression that causes the "
        f"F2P tests to fail. Your task is to restore the pre-regression behaviour."
    )
