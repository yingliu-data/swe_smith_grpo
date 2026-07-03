from __future__ import annotations

import asyncio
import random
import subprocess
import sys
import traceback
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
from .test_env import ensure_test_env
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
        prs = await self._repo_manager.list_bug_prs(cfg.repo, limit=cfg.max_prs)
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
        try:
            if not cfg.base:
                # Base mode makes no LLM calls; requiring NEBIUS_API_KEY there
                # would abort pure diff-extraction runs for no reason.
                if not cfg.offline:
                    nebius = NebiusClient()
                for name in cfg.methods:
                    cls = get_method(name)
                    methods[name] = cls(nebius) if issubclass(cls, _LMBase) else cls()  # type: ignore[arg-type]

            # The writers append; stale files from a previous run into the same
            # output_root would concatenate datasets and leak an old run's
            # heldout instances into this run's pilot split.
            for stale in (cfg.output_root / "pilot.jsonl", cfg.output_root / "heldout.jsonl"):
                stale.unlink(missing_ok=True)
            jsonl_writer = SWEBenchJSONLWriter(cfg.output_root / "pilot.jsonl")
            heldout_writer = SWEBenchJSONLWriter(cfg.output_root / "heldout.jsonl")
            yield_logger = YieldLogger(cfg.output_root / "yield.csv")
            memory = MemoryStore(cfg.output_root / "setup_facts")

            stats: dict[str, _PerMethodStats] = {name: _PerMethodStats() for name in cfg.methods}
            passing: list[tuple[InstanceRecord, list[str]]] = []

            print(f"[datagen] fetching patches for {len(prs)} PRs", file=sys.stderr, flush=True)
            patches = await self._fetch_patches(prs)

            n_tasks_total = 0
            n_errs_total = 0
            for i, (pr, patch) in enumerate(zip(prs, patches), 1):
                print(f"[datagen] PR {i}/{len(prs)}: #{pr.number} base={pr.base_commit[:8]}", file=sys.stderr, flush=True)
                if patch is None:
                    continue  # fetch failed; already traced in _fetch_patches
                if cfg.base and not any(l in {"bug", "fix"} for l in pr.labels):
                    self._trace.log("datagen.pr_skip", pr=pr.number, reason="label_filter", labels=pr.labels)
                    continue
                # Parse once per PR: split_reference_patch traps unparseable and
                # binary-bearing patches and returns ("", ""), which the check
                # below turns into a traced skip.
                src_patch, test_patch = split_reference_patch(patch)
                f2p = extract_f2p_nodeids(test_patch)
                if not src_patch.strip() or not test_patch.strip() or not f2p:
                    self._trace.log("datagen.pr_skip", pr=pr.number, reason="unusable_reference_patch",
                                    has_src=bool(src_patch.strip()), has_tests=bool(test_patch.strip()),
                                    n_f2p=len(f2p))
                    print(f"[datagen] PR #{pr.number}: skipping (reference patch unparseable, binary, or without F2P tests)", file=sys.stderr, flush=True)
                    continue
                try:
                    await self._repo_manager.checkout(repo_dir, pr.base_commit)
                except Exception as exc:
                    self._trace.log("datagen.pr_skip", pr=pr.number, reason="checkout_failed", error=str(exc))
                    print(f"[datagen] PR #{pr.number}: skipping (checkout failed: {exc})", file=sys.stderr, flush=True)
                    continue

                if cfg.base:
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
                    continue

                # Mutation mode needs the target repo's deps importable when
                # pytest runs; build (or reuse) the per-commit test venv from
                # the clone we just checked out.
                env_key = f"{cfg.repo.replace('/', '__')}-{pr.base_commit[:12]}"
                print(f"[datagen] PR #{pr.number}: ensuring test env {env_key} (first build takes minutes)", file=sys.stderr, flush=True)
                try:
                    python_bin = await asyncio.get_running_loop().run_in_executor(
                        None, ensure_test_env, repo_dir, cfg.repos_root / "_envs", env_key,
                    )
                except Exception as exc:
                    detail = exc.stderr[-500:] if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
                    self._trace.log("datagen.pr_skip", pr=pr.number, reason="test_env_build_failed",
                                    error_type=type(exc).__name__, error=detail)
                    print(f"[datagen] PR #{pr.number}: skipping (test env build failed: {detail})", file=sys.stderr, flush=True)
                    continue

                # Schedule this PR's trials and drain them before the next
                # iteration checks out a different commit: generate() and the
                # validator's copytree read the shared clone, so the working
                # tree must stay at this PR's base commit until they finish.
                tasks: list[asyncio.Task[Any]] = []
                meta: list[tuple[str, int]] = []
                for name, method in methods.items():
                    for t in range(cfg.t_per_method):
                        tasks.append(asyncio.create_task(
                            self._generate_validate_write(
                                pr=pr, repo_dir=repo_dir, reference_patch=patch,
                                src_patch=src_patch, test_patch=test_patch, f2p=f2p,
                                python_bin=str(python_bin),
                                method=method, trial=t, stats=stats[name],
                                memory=memory, passing=passing,
                            )
                        ))
                        meta.append((name, t))
                print(f"[datagen] PR #{pr.number}: awaiting {len(tasks)} tasks (LLM+docker)", file=sys.stderr, flush=True)
                results = await asyncio.gather(*tasks, return_exceptions=True)
                n_tasks_total += len(tasks)
                n_errs = 0
                for (name, t), r in zip(meta, results):
                    if isinstance(r, BaseException):
                        n_errs += 1
                        self._trace.log("datagen.task_error", pr=pr.number, method=name, trial=t,
                                        error_type=type(r).__name__, error=repr(r),
                                        traceback="".join(traceback.format_exception(r)))
                n_errs_total += n_errs
                if n_errs:
                    print(f"[datagen] PR #{pr.number}: {n_errs}/{len(tasks)} tasks raised (see trace.jsonl datagen.task_error)", file=sys.stderr, flush=True)

            if n_tasks_total and n_errs_total == n_tasks_total:
                raise RuntimeError(
                    f"all {n_tasks_total} generation tasks raised — validation infrastructure "
                    "is likely broken; see datagen.task_error entries in trace.jsonl"
                )
            print(f"[datagen] PR loop complete: passing={len(passing)}", file=sys.stderr, flush=True)
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

        # Held-out split: seeded sample of passing instances, NOT in pilot.jsonl.
        # Sort first — in mutation mode `passing` fills in task-completion order,
        # which would make the seeded shuffle non-reproducible across runs.
        passing.sort(key=lambda item: item[0].instance_id)
        rng = random.Random(cfg.seed)
        rng.shuffle(passing)
        n_held = cfg.heldout_count
        if passing and len(passing) <= n_held:
            n_held = len(passing) // 2
            self._trace.log("datagen.heldout_capped", requested=cfg.heldout_count,
                            actual=n_held, passing=len(passing))
            print(f"[datagen] WARNING: only {len(passing)} passing instances; capping heldout at {n_held} so pilot.jsonl is not starved", file=sys.stderr, flush=True)
        if not passing:
            print("[datagen] WARNING: 0 passing instances — pilot.jsonl and heldout.jsonl are empty; see datagen.pr_skip/datagen.task_error in trace.jsonl", file=sys.stderr, flush=True)
        held = passing[:n_held]
        pilot = passing[n_held:]
        for rec, _ in pilot:
            jsonl_writer.write(rec)
        for rec, _ in held:
            heldout_writer.write(rec)
        self._trace.log("datagen.split", pilot=len(pilot), heldout=len(held))
        return [rec for rec, _ in pilot]

    async def _fetch_patches(self, prs: list[PullRequestInfo]) -> list[str | None]:
        """Fetch all PR patches concurrently; a failed fetch skips that PR only."""
        sem = asyncio.Semaphore(4)

        async def fetch(pr: PullRequestInfo) -> str | None:
            async with sem:
                try:
                    return await self._repo_manager.fetch_patch(pr)
                except Exception as exc:
                    self._trace.log("datagen.pr_skip", pr=pr.number, reason="patch_fetch_failed",
                                    error_type=type(exc).__name__, error=str(exc))
                    print(f"[datagen] PR #{pr.number}: skipping (patch fetch failed: {exc})", file=sys.stderr, flush=True)
                    return None

        return list(await asyncio.gather(*(fetch(pr) for pr in prs)))

    async def _generate_validate_write(
        self,
        *,
        pr: PullRequestInfo,
        repo_dir: Path,
        reference_patch: str,
        src_patch: str,
        test_patch: str,
        f2p: list[str],
        python_bin: str,
        method: BaseMutationMethod,
        trial: int,
        stats: _PerMethodStats,
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
            self._trace.log("datagen.generate_skip", method=method.name, pr=pr.number,
                            trial=trial, reason="generate_returned_none")
            return

        async with self._docker_sem:
            result = await self._validator.validate(
                repo_dir=repo_dir, base_commit=pr.base_commit,
                src_patch=src_patch, test_patch=test_patch, f2p=f2p,
                buggy_patch=cand.buggy_patch, python_bin=python_bin,
            )
        stats.wall_total += result.wall_seconds
        self._trace.log("datagen.validate", method=method.name, pr=pr.number, trial=trial,
                        passed=result.passed, reason=result.reason, wall=result.wall_seconds,
                        f2p_with_buggy_failed=result.f2p_with_buggy_failed,
                        f2p_with_reference_passed=result.f2p_with_reference_passed)
        if not result.passed:
            return
        stats.passed += 1

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
