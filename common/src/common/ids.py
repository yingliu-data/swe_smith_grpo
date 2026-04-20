from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone


def make_session_id(kind: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{kind}-{ts}"


def make_ticket_id(seq: int, op: str) -> str:
    return f"tk_{seq:04d}_{safe_key(op)}"


def safe_key(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s)
    return (s[:maxlen].strip("_")) or "_"


def short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]
