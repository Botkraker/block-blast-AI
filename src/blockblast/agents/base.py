"""Common agent protocol used by evaluation and baselines."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from blockblast.engine.game import GameState


class Agent(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def act(
        self,
        obs: npt.NDArray[np.float32],
        action_mask: npt.NDArray[np.bool_],
        state: GameState,
    ) -> int: ...
