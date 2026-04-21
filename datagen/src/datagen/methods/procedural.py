from __future__ import annotations

import ast
import difflib
import random
from dataclasses import dataclass

from unidiff import PatchSet

from .base import BaseMutationMethod, Candidate, Context

# Ordered longer-first so " is not " binds before " is ", etc.
_COMPARISON_FLIPS: list[tuple[str, str]] = [
    (" is not ", " is "),
    (" not in ", " in "),
    (" == ", " != "),
    (" != ", " == "),
    (" <= ", " >= "),
    (" >= ", " <= "),
    (" is ", " is not "),
    (" in ", " not in "),
    (" < ", " > "),
    (" > ", " < "),
    ("==", "!="),
    ("!=", "=="),
    ("<=", ">="),
    (">=", "<="),
]

_BINOP_FLIPS: list[tuple[str, str]] = [
    (" + ", " - "),
    (" - ", " + "),
    (" * ", " / "),
    (" / ", " * "),
]


@dataclass
class _MutationSite:
    lineno: int  # 0-indexed
    kind: str    # "cmp" | "bin"


def _is_test_path(p: str) -> bool:
    parts = p.split("/")
    name = parts[-1] if parts else ""
    return (
        any(seg in ("tests", "test") for seg in parts)
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _candidate_files(reference_patch: str) -> list[str]:
    ps = PatchSet(reference_patch)
    return [p.path for p in ps if p.path.endswith(".py") and not _is_test_path(p.path)]


def _collect_sites(src: str) -> list[_MutationSite]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    sites: list[_MutationSite] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            sites.append(_MutationSite(lineno=node.lineno - 1, kind="cmp"))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            sites.append(_MutationSite(lineno=node.lineno - 1, kind="bin"))
    return sites


def _try_mutate_line(line: str, kind: str) -> str | None:
    """Swap the first matching operator on the line. Preserves all other formatting."""
    table = _COMPARISON_FLIPS if kind == "cmp" else _BINOP_FLIPS
    for old, new in table:
        idx = line.find(old)
        if idx != -1:
            return line[:idx] + new + line[idx + len(old):]
    return None


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


class ProceduralMethod(BaseMutationMethod):
    """AST-located, text-surgery operator flip.

    Uses AST solely to *locate* comparison/binop statements. The actual edit is a
    string replacement on that line — preserving all other formatting, comments,
    whitespace and string-literal quote styles. The original implementation used
    ``ast.unparse`` which reformatted the entire file and stripped comments,
    producing multi-hundred-line diffs that ``git apply`` rejected as corrupt.
    """

    name = "procedural"

    async def generate(self, ctx: Context) -> Candidate | None:
        rng = random.Random(ctx.seed * 1_000_003 + ctx.trial_index)
        files = _candidate_files(ctx.reference_patch)
        if not files:
            return None
        rng.shuffle(files)
        for rel in files:
            src_path = ctx.repo_dir / rel
            if not src_path.exists():
                continue
            before = src_path.read_text()
            sites = _collect_sites(before)
            if not sites:
                continue
            rng.shuffle(sites)
            lines = before.splitlines(keepends=True)
            for site in sites:
                if site.lineno >= len(lines):
                    continue
                original_line = lines[site.lineno]
                mutated_line = _try_mutate_line(original_line, site.kind)
                if mutated_line is None or mutated_line == original_line:
                    continue
                after_lines = lines[:]
                after_lines[site.lineno] = mutated_line
                after = "".join(after_lines)
                # Sanity-check that the mutated source still parses.
                try:
                    ast.parse(after)
                except SyntaxError:
                    continue
                diff = _unified_diff(rel, before, after)
                return Candidate(
                    method=self.name,
                    buggy_patch=diff,
                    rationale=f"{site.kind}-flip {rel}:{site.lineno + 1} trial {ctx.trial_index}",
                )
        return None
