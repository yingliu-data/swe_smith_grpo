from __future__ import annotations

import ast
import difflib
import re

from unidiff import PatchSet

from ..nebius_client import NebiusClient
from .base import BaseMutationMethod, Candidate, Context

_SYSTEM_MODIFY = (
    "You are a mutation tester generating synthetic bugs for RL training data.\n"
    "\n"
    "You will receive a Python source file AT ITS BASE COMMIT (i.e. BEFORE the reference fix\n"
    "is applied, meaning the bug is still present). The F2P tests already FAIL at this\n"
    "commit. Your job: produce a small mutation (1-3 lines) that keeps the file parseable\n"
    "and does NOT coincidentally reintroduce the reference fix. Any non-fix mutation keeps\n"
    "the F2P failing, which is the valid training signal.\n"
    "\n"
    "Output RULES (strictly):\n"
    "- Reply with EXACTLY ONE <edit> block in this shape and nothing else:\n"
    "  <edit file=\"<relative/path.py>\">\n"
    "  <old>\n"
    "  <verbatim text from the file, copy byte-for-byte>\n"
    "  </old>\n"
    "  <new>\n"
    "  <replacement text>\n"
    "  </new>\n"
    "  </edit>\n"
    "- The <old> block MUST match a contiguous region of the given file exactly (spaces,\n"
    "  indentation, punctuation — byte-for-byte). Include enough surrounding lines so the\n"
    "  match is unique in the file.\n"
    "- Never touch any test file (paths under tests/ or matching test_*.py / *_test.py).\n"
)

_SYSTEM_REWRITE = (
    "You are a mutation tester generating synthetic bugs. You will receive a Python source\n"
    "file AT ITS BASE COMMIT (bug still present) and the reference fix. Rewrite a LARGER\n"
    "region (5-15 lines) with a different plausible-but-buggy implementation. The F2P tests\n"
    "already fail at base; any non-fix rewrite preserves the signal.\n"
    "\n"
    "Output RULES (strictly):\n"
    "- Reply with EXACTLY ONE <edit> block in this shape and nothing else:\n"
    "  <edit file=\"<relative/path.py>\">\n"
    "  <old>\n"
    "  <verbatim text from the file, copy byte-for-byte, 5-15 lines>\n"
    "  </old>\n"
    "  <new>\n"
    "  <replacement text>\n"
    "  </new>\n"
    "  </edit>\n"
    "- The <old> block MUST match a contiguous region of the given file exactly.\n"
    "- Never touch any test file.\n"
)

_EDIT_RE = re.compile(
    r'<edit\s+file="(?P<file>[^"]+)"\s*>'
    r'\s*<old>\s*\n(?P<old>.*?)\n?\s*</old>'
    r'\s*<new>\s*\n(?P<new>.*?)\n?\s*</new>'
    r'\s*</edit>',
    re.DOTALL,
)


def _is_test_path(p: str) -> bool:
    parts = p.split("/")
    name = parts[-1] if parts else ""
    return (
        any(seg in ("tests", "test") for seg in parts)
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _relevant_files(reference_patch: str) -> list[str]:
    return [
        pf.path
        for pf in PatchSet(reference_patch)
        if pf.path.endswith(".py") and not _is_test_path(pf.path)
    ]


def _format_file_with_line_numbers(content: str, *, max_chars: int = 20_000) -> str:
    lines = content.splitlines()
    width = max(3, len(str(len(lines))))
    rendered = "\n".join(f"{i + 1:>{width}} {ln}" for i, ln in enumerate(lines))
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "\n... [file truncated for prompt budget]"
    return rendered


def _unified_diff(path: str, before: str, after: str) -> str:
    body = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    return f"diff --git a/{path} b/{path}\n" + body


def _find_unique_replacement(original: str, old: str, new: str) -> tuple[str, str] | None:
    """Return an (old_variant, new_variant) that matches in original exactly once.

    The LM sometimes trims trailing newlines or adds them; we try a few variants.
    """
    variants = [
        (old, new),
        (old + "\n", new + "\n"),
        (old.rstrip("\n"), new.rstrip("\n")),
    ]
    for o, n in variants:
        if o and original.count(o) == 1:
            return o, n
    return None


class _LMBase(BaseMutationMethod):
    _system: str

    def __init__(self, client: NebiusClient):
        self._client = client

    async def generate(self, ctx: Context) -> Candidate | None:
        files = _relevant_files(ctx.reference_patch)
        if not files:
            return None
        rel = files[ctx.trial_index % len(files)]
        src_path = ctx.repo_dir / rel
        if not src_path.exists():
            return None
        original = src_path.read_text()
        numbered = _format_file_with_line_numbers(original)

        user = (
            f"# Repository: {ctx.repo}\n"
            f"# PR #{ctx.pr_number}: {ctx.pr_title}\n"
            f"# Trial: {ctx.trial_index}\n"
            f"\n"
            f"## Target file (at base commit, buggy): {rel}\n"
            f"```python\n{numbered}\n```\n"
            f"\n"
            f"## Reference fix (the change that would correctly fix the bug — do NOT reproduce it)\n"
            f"```diff\n{ctx.reference_patch[:6000]}\n```\n"
            f"\n"
            f"Produce exactly ONE <edit> block now."
        )
        resp = await self._client.complete(
            system=self._system, user=user, seed=ctx.seed + ctx.trial_index
        )
        m = _EDIT_RE.search(resp.text)
        if not m:
            return None

        file_rel = m.group("file").strip().lstrip("./")
        old = m.group("old").replace("\r\n", "\n").replace("\r", "\n")
        new = m.group("new").replace("\r\n", "\n").replace("\r", "\n")

        # Resolve file path; LM may drop a leading directory component.
        target_rel = rel
        target_path = src_path
        if file_rel != rel:
            if _is_test_path(file_rel):
                return None
            alt = ctx.repo_dir / file_rel
            if alt.exists() and alt.is_file():
                target_rel = file_rel
                target_path = alt
                original = target_path.read_text()
            # else: trust our chosen `rel` — the LM just got the path wrong

        match = _find_unique_replacement(original, old, new)
        if match is None:
            return None
        old_variant, new_variant = match
        new_content = original.replace(old_variant, new_variant, 1)
        if new_content == original:
            return None
        try:
            ast.parse(new_content)
        except SyntaxError:
            return None

        diff = _unified_diff(target_rel, original, new_content)
        return Candidate(
            method=self.name,
            buggy_patch=diff,
            rationale=f"LM {self.name} {target_rel} trial {ctx.trial_index}, tokens={resp.completion_tokens}",
        )


class LMModifyMethod(_LMBase):
    name = "lm_modify"
    _system = _SYSTEM_MODIFY


class LMRewriteMethod(_LMBase):
    name = "lm_rewrite"
    _system = _SYSTEM_REWRITE
