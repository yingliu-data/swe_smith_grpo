from __future__ import annotations

ALLOWED: dict[str, set[str]] = {
    "active": {"complete", "failed", "escalated"},
    "complete": set(),
    "failed": set(),
    "escalated": set(),
}


class InvalidTransition(ValueError):
    pass


def transition(current: str, target: str) -> str:
    if target not in ALLOWED.get(current, set()):
        raise InvalidTransition(f"{current!r} -> {target!r} not allowed")
    return target
