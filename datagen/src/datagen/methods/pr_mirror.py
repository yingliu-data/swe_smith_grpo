from __future__ import annotations

import random

from unidiff import PatchSet

from .base import BaseMutationMethod, Candidate, Context


def _is_test_path(p: str) -> bool:
    parts = p.split("/")
    name = parts[-1] if parts else ""
    return (
        any(seg in ("tests", "test") for seg in parts)
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _render_file_section(path: str, hunks) -> str:
    out = [
        f"diff --git a/{path} b/{path}\n",
        f"--- a/{path}\n",
        f"+++ b/{path}\n",
    ]
    for hunk in hunks:
        out.append(str(hunk))
    return "".join(out)


class PRMirrorMethod(BaseMutationMethod):
    """Forward-patch subset: drop one random hunk from the merged PR's src diff.

    The remaining hunks are a VALID SUBSET of the forward fix — they apply cleanly
    at base_commit because each hunk's context lines reflect the state before the
    fix. The dropped hunk leaves part of the bug in place so the F2P tests should
    still fail (probability depends on whether the dropped hunk was load-bearing
    for the specific failure the test probes).

    The previous implementation inverted the patch by swapping +/- prefixes, which
    produced a diff whose context lines referenced POST-fix code that does not
    exist at base_commit — every candidate was rejected by ``git apply --check``.
    """

    name = "pr_mirror"

    async def generate(self, ctx: Context) -> Candidate | None:
        rng = random.Random(ctx.seed * 17 + ctx.trial_index + 5_000_003)
        ps = PatchSet(ctx.reference_patch)
        src_files = [
            pf for pf in ps
            if pf.path.endswith(".py") and not _is_test_path(pf.path)
        ]
        if not src_files:
            return None

        all_hunks: list[tuple[str, object]] = []
        for pf in src_files:
            for hunk in pf:
                all_hunks.append((pf.path, hunk))
        if len(all_hunks) < 2:
            return None  # can't drop one and still have a non-trivial diff

        drop_idx = rng.randrange(len(all_hunks))
        kept_by_path: dict[str, list] = {}
        for i, (path, hunk) in enumerate(all_hunks):
            if i == drop_idx:
                continue
            kept_by_path.setdefault(path, []).append(hunk)
        if not kept_by_path:
            return None

        sections = [_render_file_section(path, hunks) for path, hunks in kept_by_path.items()]
        return Candidate(
            method=self.name,
            buggy_patch="".join(sections),
            rationale=f"forward src_patch minus hunk #{drop_idx} of {len(all_hunks)} (trial {ctx.trial_index})",
        )
