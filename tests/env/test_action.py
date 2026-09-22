from __future__ import annotations

import numpy as np
import pytest

from blockblast.engine.game import Game, apply_placement
from blockblast.engine.scoring import ScoreConfig
from blockblast.env.action import (
    N_ACTIONS,
    action_mask,
    action_space,
    decode_action,
    encode_action,
)


def test_bijection() -> None:
    assert N_ACTIONS == 192 and action_space().n == 192
    seen = set()
    for slot in range(3):
        for row in range(8):
            for col in range(8):
                a = encode_action(slot, row, col)
                assert a == slot * 64 + row * 8 + col
                assert decode_action(a) == (slot, row, col)
                seen.add(a)
    assert seen == set(range(192))


@pytest.mark.parametrize("bad", [(-1, 0, 0), (3, 0, 0), (0, 8, 0), (0, 0, -1)])
def test_encode_rejects_out_of_range(bad: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        encode_action(*bad)


@pytest.mark.parametrize("bad", [-1, 192])
def test_decode_rejects_out_of_range(bad: int) -> None:
    with pytest.raises(ValueError):
        decode_action(bad)


def test_mask_equals_engine_legality_over_random_states() -> None:
    """Property: mask[a] is True iff apply_placement accepts the decoded move."""
    cfg = ScoreConfig()
    rng = np.random.default_rng(0)
    for seed in range(10):
        g = Game()
        g.reset(seed)
        while not g.state.game_over:
            mask = action_mask(g.state)
            np.testing.assert_array_equal(mask, g.legal_moves().reshape(-1))
            for a in range(192):
                slot, row, col = decode_action(a)
                try:
                    apply_placement(g.state, slot, row, col, cfg)
                    ok = True
                except ValueError:
                    ok = False
                assert ok == bool(mask[a])
            legal = np.flatnonzero(mask)
            g.step(*decode_action(int(legal[rng.integers(legal.size)])))
