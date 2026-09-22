"""Hand dealing: uniform i.i.d. with replacement from the catalogue (PRD Q2)."""

from __future__ import annotations

import numpy as np

from blockblast.engine.pieces import PIECE_CATALOGUE


class PieceDealer:
    """Draws 3-piece hands. The injected generator is the engine's only randomness."""

    def __init__(self, rng: np.random.Generator, n_pieces: int = len(PIECE_CATALOGUE)) -> None:
        if n_pieces <= 0:
            raise ValueError("n_pieces must be positive")
        self._rng = rng
        self._n_pieces = n_pieces

    def deal(self) -> tuple[int, int, int]:
        a, b, c = (int(x) for x in self._rng.integers(0, self._n_pieces, size=3))
        return a, b, c
