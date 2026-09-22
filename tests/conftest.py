"""Shared fixtures: seeded RNGs and handcrafted boards."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from blockblast.engine.board import BOARD_SIZE, LINE_MASKS
from blockblast.engine.game import GameState, Hand


def cell(row: int, col: int) -> int:
    """Bitmask of a single cell."""
    return 1 << (row * BOARD_SIZE + col)


def make_state(
    board: int = 0,
    hand: Hand = (0, 0, 0),
    score: int = 0,
    combo_streak: int = 0,
    round_index: int = 1,
    moves: int = 0,
    game_over: bool = False,
) -> GameState:
    return GameState(board, hand, score, combo_streak, round_index, moves, game_over)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def state_factory() -> Callable[..., GameState]:
    return make_state


@pytest.fixture
def row_minus_last() -> int:
    """Row 0 filled except (0, 7)."""
    return LINE_MASKS[0] & ~cell(0, 7)
