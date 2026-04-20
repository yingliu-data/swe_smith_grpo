from __future__ import annotations

import ast
import difflib
import random
from pathlib import Path
from typing import Iterable

from unidiff import PatchSet

from .base import BaseMutationMethod, Candidate, Context

_COMPARE_FLIPS: dict[type, type] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.Gt,
    ast.Gt: ast.Lt,
    ast.LtE: ast.GtE,
    ast.GtE: ast.LtE,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


class _CollectMutations(ast.NodeVisitor):
    """Walk an AST and collect every candidate node we know how to mutate."""

    def __init__(self) -> None:
        self.compares: list[ast.Compare] = []
        self.returns: list[ast.Return] = []
        self.binops: list[ast.BinOp] = []

    def visit_Compare(self, node: ast.Compare) -> None:
        if node.ops and type(node.ops[0]) in _COMPARE_FLIPS:
            self.compares.append(node)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self.returns.append(node)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self.binops.append(node)
        self.generic_visit(node)


def _candidate_files(reference_patch: str) -> list[str]:
    ps = PatchSet(reference_patch)
    return [p.path for p in ps if p.path.endswith(".py") and not _is_test_path(p.path)]


def _is_test_path(p: str) -> bool:
    parts = p.split("/")
    return (
        any(seg in ("tests", "test") for seg in parts)
        or parts[-1].startswith("test_")
        or parts[-1].endswith("_test.py")
    )


def _mutate_source(src: str, rng: random.Random) -> str | None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    collector = _CollectMutations()
    collector.visit(tree)
    pool: list[tuple[str, ast.AST]] = (
        [("cmp", n) for n in collector.compares]
        + [("ret", n) for n in collector.returns]
        + [("bin", n) for n in collector.binops]
    )
    if not pool:
        return None
    kind, node = rng.choice(pool)
    if kind == "cmp":
        old = type(node.ops[0])  # type: ignore[attr-defined]
        node.ops[0] = _COMPARE_FLIPS[old]()  # type: ignore[attr-defined]
    elif kind == "ret":
        val = node.value  # type: ignore[attr-defined]
        node.value = ast.UnaryOp(op=ast.Not(), operand=val)  # type: ignore[attr-defined]
    elif kind == "bin":
        flips = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
        new_op = flips.get(type(node.op))  # type: ignore[attr-defined]
        if new_op is None:
            return None
        node.op = new_op()  # type: ignore[attr-defined]
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def _unified_diff(path: str, before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    )
    header = f"diff --git a/{path} b/{path}\n"
    return header + "".join(diff)


class ProceduralMethod(BaseMutationMethod):
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
            after = _mutate_source(before, rng)
            if after is None or after == before:
                continue
            diff = _unified_diff(rel, before, after)
            return Candidate(
                method=self.name,
                buggy_patch=diff,
                rationale=f"AST mutation on {rel} trial {ctx.trial_index}",
            )
        return None
