from __future__ import annotations

import re

from ..nebius_client import NebiusClient
from .base import BaseMutationMethod, Candidate, Context

_SYSTEM_MODIFY = (
    "You are a mutation tester. Given a buggy-patch target, produce a UNIFIED DIFF that "
    "reintroduces a realistic bug into code that the reference patch fixed. The diff must "
    "NOT touch any file under tests/ or matching test_*.py. Output ONLY the diff between "
    "<diff>...</diff> tags. No commentary."
)

_SYSTEM_REWRITE = (
    "You are a mutation tester. Given the repository fix in a merged PR, rewrite the fix "
    "into a DIFFERENT realistic bug (not a literal inversion) that would cause the same "
    "F2P tests to fail. Output ONLY a unified diff inside <diff>...</diff> tags. No test "
    "files may be modified."
)

_DIFF_RE = re.compile(r"<diff>(.*?)</diff>", re.DOTALL | re.IGNORECASE)


def _extract(text: str) -> str | None:
    m = _DIFF_RE.search(text)
    if not m:
        return None
    diff = m.group(1).strip()
    return diff or None


class _LMBase(BaseMutationMethod):
    _system: str

    def __init__(self, client: NebiusClient):
        self._client = client

    async def generate(self, ctx: Context) -> Candidate | None:
        user = (
            f"# Repository: {ctx.repo}\n"
            f"# PR #{ctx.pr_number}: {ctx.pr_title}\n\n"
            f"## PR body\n{ctx.pr_body[:2000]}\n\n"
            f"## Reference fix patch (the code you must INVERT into a bug)\n"
            f"```\n{ctx.reference_patch[:12000]}\n```\n"
            f"# Trial {ctx.trial_index}\n"
            f"Produce your diff now."
        )
        resp = await self._client.complete(
            system=self._system, user=user, seed=ctx.seed + ctx.trial_index
        )
        diff = _extract(resp.text)
        if diff is None:
            return None
        return Candidate(
            method=self.name,
            buggy_patch=diff,
            rationale=f"LM ({self.name}) trial {ctx.trial_index}, tokens={resp.completion_tokens}",
        )


class LMModifyMethod(_LMBase):
    name = "lm_modify"
    _system = _SYSTEM_MODIFY


class LMRewriteMethod(_LMBase):
    name = "lm_rewrite"
    _system = _SYSTEM_REWRITE
