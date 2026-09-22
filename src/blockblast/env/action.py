"""Action space: ``a = slot·64 + row·8 + col`` over ``Discrete(192)``."""

from __future__ import annotations

from typing import Final

import gymnasium
import numpy as np
import numpy.typing as npt

from blockblast.engine.game import GameState, legal_moves

N_SLOTS: Final[int] = 3
N_ACTIONS: Final[int] = 192


def encode_action(slot: int, row: int, col: int) -> int:
    if not (0 <= slot < N_SLOTS and 0 <= row < 8 and 0 <= col < 8):
        raise ValueError(f"out of range: slot={slot} row={row} col={col}")
    return slot * 64 + row * 8 + col


def decode_action(action: int) -> tuple[int, int, int]:
    a = int(action)
    if not 0 <= a < N_ACTIONS:
        raise ValueError(f"action {action} not in [0, {N_ACTIONS})")
    return a // 64, (a // 8) % 8, a % 8


def action_mask(state: GameState) -> npt.NDArray[np.bool_]:
    """(192,) bool legal-action mask, C-order flattening of ``legal_moves``."""
    return legal_moves(state).reshape(-1)


def action_space() -> gymnasium.spaces.Discrete:  # type: ignore[type-arg]
    return gymnasium.spaces.Discrete(N_ACTIONS)
