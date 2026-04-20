from __future__ import annotations

from unidiff import PatchSet

from .base import BaseMutationMethod, Candidate, Context


def _is_test_path(p: str) -> bool:
    parts = p.split("/")
    return (
        any(seg in ("tests", "test") for seg in parts)
        or parts[-1].startswith("test_")
        or parts[-1].endswith("_test.py")
    )


def _invert_file_patch(patched_file) -> str | None:
    """Flip every hunk's additions and deletions.

    unidiff's PatchedFile rendering preserves header + hunks; we reconstruct with the
    @@ line and swapped +/- prefixes.
    """
    lines: list[str] = []
    header = f"diff --git a/{patched_file.path} b/{patched_file.path}\n"
    lines.append(header)
    lines.append(f"--- a/{patched_file.path}\n")
    lines.append(f"+++ b/{patched_file.path}\n")
    for hunk in patched_file:
        new_start = hunk.target_start
        new_length = hunk.target_length
        old_start = hunk.source_start
        old_length = hunk.source_length
        lines.append(f"@@ -{new_start},{new_length} +{old_start},{old_length} @@\n")
        for line in hunk:
            if line.is_added:
                lines.append(f"-{line.value}")
            elif line.is_removed:
                lines.append(f"+{line.value}")
            else:
                lines.append(f" {line.value}")
    return "".join(lines)


class PRMirrorMethod(BaseMutationMethod):
    name = "pr_mirror"

    async def generate(self, ctx: Context) -> Candidate | None:
        ps = PatchSet(ctx.reference_patch)
        kept: list[str] = []
        for pf in ps:
            if _is_test_path(pf.path):
                continue
            inverted = _invert_file_patch(pf)
            if inverted is not None:
                kept.append(inverted)
        if not kept:
            return None
        return Candidate(
            method=self.name,
            buggy_patch="".join(kept),
            rationale="inverted merged PR diff (non-test files only)",
        )
