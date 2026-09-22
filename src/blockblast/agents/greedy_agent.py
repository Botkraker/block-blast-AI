"""Baseline: legal action maximizing immediate Δscore, ties → lowest action index."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from blockblast.engine.game import GameState, apply_placement
from blockblast.engine.scoring import ScoreConfig


class GreedyAgent:
    name = "greedy"

    def __init__(self, score_config: ScoreConfig = ScoreConfig()) -> None:
        self._cfg = score_config

    def reset(self, seed: int | None = None) -> None:
        """Stateless and deterministic; nothing to reset."""

    def act(
        self,
        obs: npt.NDArray[np.float32],
        action_mask: npt.NDArray[np.bool_],
        state: GameState,
    ) -> int:
        best_a, best_delta = -1, -1
        actions: list[int] = np.flatnonzero(action_mask).tolist()
        for a in actions:
            delta = apply_placement(state, a // 64, (a // 8) % 8, a % 8, self._cfg).score_delta
            if delta > best_delta:
                best_a, best_delta = a, delta
        if best_a < 0:
            raise ValueError("no legal action")
        return best_a
