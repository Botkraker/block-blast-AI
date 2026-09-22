"""Hand dealing: uniform i.i.d. with replacement from the catalogue (PRD Q2).

``hard_weight < 1`` is a training curriculum: the hardest pieces (O3, I5h, I5v) are drawn
that much less often than the others. 1.0 is the real game and uses the uniform draw.
"""

from __future__ import annotations

import numpy as np

from blockblast.engine.pieces import PIECE_CATALOGUE

HARD_PIECES = frozenset(p.piece_id for p in PIECE_CATALOGUE if p.name in ("O3", "I5h", "I5v"))


class PieceDealer:
    """Draws 3-piece hands. The injected generator is the engine's only randomness."""

    def __init__(
        self,
        rng: np.random.Generator,
        n_pieces: int = len(PIECE_CATALOGUE),
        hard_weight: float = 1.0,
    ) -> None:
        if n_pieces <= 0:
            raise ValueError("n_pieces must be positive")
        if not 0.0 <= hard_weight <= 1.0:
            raise ValueError("hard_weight must be in [0, 1]")
        self._rng = rng
        self._n_pieces = n_pieces
        self._p: np.ndarray | None = None
        if hard_weight < 1.0:
            w = np.array([hard_weight if i in HARD_PIECES else 1.0 for i in range(n_pieces)])
            self._p = w / w.sum()

    def deal(self) -> tuple[int, int, int]:
        if self._p is None:  # the real game; keeps seeded deals identical to before
            draw = self._rng.integers(0, self._n_pieces, size=3)
        else:
            draw = self._rng.choice(self._n_pieces, size=3, p=self._p)
        a, b, c = (int(x) for x in draw)
        return a, b, c
