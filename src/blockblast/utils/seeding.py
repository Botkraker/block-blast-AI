"""Global seeding and seed ranges (training < 1_000_000 ≤ evaluation)."""

from __future__ import annotations

import random
from typing import Final

import numpy as np

TRAIN_SEED_MAX: Final[int] = 1_000_000
EVAL_SEED_START: Final[int] = 1_000_000
VALIDATION_SEED_START: Final[int] = 900_000  # in-training eval callback; never the test range


def set_global_seeds(seed: int, deterministic_torch: bool = False) -> None:
    """Seed python, numpy and torch. GPU runs are still not bitwise reproducible."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)


def spawn_seeds(seed: int, n: int) -> list[int]:
    """``n`` independent 31-bit seeds derived from ``seed`` via ``SeedSequence``."""
    children = np.random.SeedSequence(seed).spawn(n)
    return [int(c.generate_state(1)[0] % (2**31 - 1)) for c in children]
