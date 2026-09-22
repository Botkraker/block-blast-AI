from __future__ import annotations

import numpy as np
import pytest

from blockblast.agents import GreedyAgent
from blockblast.engine.board import LINE_MASKS
from blockblast.env.action import action_mask, encode_action
from blockblast.env.observation import encode_observation
from tests.conftest import cell, make_state

OBS = np.zeros((5, 8, 8), np.float32)


def act(state):  # type: ignore[no-untyped-def]
    return GreedyAgent().act(encode_observation(state), action_mask(state), state)


def test_prefers_line_clear() -> None:
    # Row 4 missing (4, 6); a mono there clears a line (11 pts) vs O3 anywhere (9 pts).
    s = make_state(board=LINE_MASKS[4] & ~cell(4, 6), hand=(10, 0, None))
    assert act(s) == encode_action(1, 4, 6)


def test_prefers_bigger_piece_without_clears() -> None:
    s = make_state(hand=(0, 10, 7))  # mono(1), O3(9), I5h(5)
    assert act(s) == encode_action(1, 0, 0)  # lowest index among max-Δscore actions


def test_tie_break_lowest_index() -> None:
    s = make_state(hand=(0, 0, 0))
    assert act(s) == 0


def test_no_legal_action_raises() -> None:
    with pytest.raises(ValueError):
        GreedyAgent().act(OBS, np.zeros(192, bool), make_state())


def test_reset_is_noop() -> None:
    GreedyAgent().reset(3)
