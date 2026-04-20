from __future__ import annotations

from typing import Iterator

from .base import BaseMutationMethod, Candidate, Context
from .lm import LMModifyMethod, LMRewriteMethod
from .pr_mirror import PRMirrorMethod
from .procedural import ProceduralMethod

_REGISTRY: dict[str, type[BaseMutationMethod]] = {
    "lm_modify": LMModifyMethod,
    "lm_rewrite": LMRewriteMethod,
    "procedural": ProceduralMethod,
    "pr_mirror": PRMirrorMethod,
}


def get_method(name: str) -> type[BaseMutationMethod]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown method {name!r}; known={sorted(_REGISTRY)}") from exc


def iter_methods() -> Iterator[tuple[str, type[BaseMutationMethod]]]:
    yield from _REGISTRY.items()


__all__ = [
    "BaseMutationMethod",
    "Candidate",
    "Context",
    "LMModifyMethod",
    "LMRewriteMethod",
    "PRMirrorMethod",
    "ProceduralMethod",
    "get_method",
    "iter_methods",
]
