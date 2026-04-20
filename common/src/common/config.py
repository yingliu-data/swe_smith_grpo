from __future__ import annotations

import os
import random
from pathlib import Path

SEED = 42


def workspace_root() -> Path:
    return Path(os.environ.get("ML_SYSTEMS_WORKSPACE", "/workspace"))


def sessions_root() -> Path:
    return workspace_root() / "sessions"


def apply_seed(seed: int = SEED) -> None:
    """Apply the project-wide seed to every RNG touched by the codebase."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
