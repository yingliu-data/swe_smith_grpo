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

    Each ``PatchedFile`` section of the reference_patch is treated as its own
    sequential commit (fastapi's .patch URL returns git-format-patch output for
    multi-commit PRs, so a single file can appear across several independent
    ``diff --git`` sections whose hunk line-numbers are relative to the state
    AFTER earlier sections have been applied). Consolidating those into a single
    section scrambles hunk ordering and yields "patch fragment without header"
    from ``git apply``. We preserve the original section structure: each file
    section keeps its own ``diff --git`` header, and we drop a single hunk
    across the whole patch.

    Safety rule: drops are restricted to the LAST section so earlier sections'
    state setup remains intact for any subsequent sections.

    The previous implementation inverted the patch by swapping +/- prefixes,
    producing a diff whose context lines referenced POST-fix code that does not
    exist at base_commit — every candidate was rejected by ``git apply --check``.
    """

    name = "pr_mirror"

    async def generate(self, ctx: Context) -> Candidate | None:
        rng = random.Random(ctx.seed * 17 + ctx.trial_index + 5_000_003)
        ps = PatchSet(ctx.reference_patch)
        sections: list[tuple[str, list]] = []
        for pf in ps:
            if not pf.path.endswith(".py") or _is_test_path(pf.path):
                continue
            hunks = list(pf)
            if hunks:
                sections.append((pf.path, hunks))
        if not sections:
            return None

        total_hunks = sum(len(hs) for _, hs in sections)
        if total_hunks < 2:
            return None  # dropping the only hunk leaves nothing

        # Restrict drops to the LAST section so earlier sections still set up
        # state correctly for any hunks that depend on them.
        last_path, last_hunks = sections[-1]
        if len(last_hunks) >= 2:
            drop_local = rng.randrange(len(last_hunks))
            last_hunks.pop(drop_local)
            drop_desc = f"hunk #{drop_local} of last section ({last_path})"
        else:
            # Last section has only one hunk — drop the entire section
            sections.pop()
            drop_desc = f"entire last section ({last_path})"
            if not sections:
                return None

        rendered = [
            _render_file_section(path, hunks)
            for path, hunks in sections
            if hunks
        ]
        if not rendered:
            return None

        return Candidate(
            method=self.name,
            buggy_patch="".join(rendered),
            rationale=f"forward src_patch minus {drop_desc} (trial {ctx.trial_index})",
        )
