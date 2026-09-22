"""Baseline: uniform choice among legal actions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from blockblast.engine.game import GameState


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(
        self,
        obs: npt.NDArray[np.float32],
        action_mask: npt.NDArray[np.bool_],
        state: GameState,
    ) -> int:
        legal = np.flatnonzero(action_mask)
        if legal.size == 0:
            raise ValueError("no legal action")
        return int(legal[self._rng.integers(legal.size)])
