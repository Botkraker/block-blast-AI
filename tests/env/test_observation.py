from __future__ import annotations

import numpy as np

from blockblast.engine.board import to_array
from blockblast.engine.pieces import PIECE_CATALOGUE
from blockblast.env.observation import (
    COMBO_NORM,
    OBS_SHAPE,
    encode_observation,
    observation_space,
    piece_plane,
)
from tests.conftest import cell, make_state


def test_space() -> None:
    space = observation_space()
    assert space.shape == OBS_SHAPE == (5, 8, 8)
    assert space.dtype == np.float32
    assert float(space.low.min()) == 0.0 and float(space.high.max()) == 1.0


def test_channels() -> None:
    board = cell(0, 0) | cell(7, 3)
    s = make_state(board=board, hand=(10, None, 7), combo_streak=2)
    obs = encode_observation(s)
    assert obs.shape == OBS_SHAPE and obs.dtype == np.float32
    np.testing.assert_array_equal(obs[0], to_array(board))
    np.testing.assert_array_equal(obs[1], piece_plane(10))
    assert not obs[2].any()  # used slot
    np.testing.assert_array_equal(obs[3], piece_plane(7))
    assert np.all(obs[4] == 2 / COMBO_NORM)
    assert observation_space().contains(obs)


def test_combo_plane_is_capped() -> None:
    assert np.all(encode_observation(make_state(combo_streak=50))[4] == 1.0)
    assert not encode_observation(make_state(combo_streak=0))[4].any()


def test_piece_planes_anchor_top_left() -> None:
    for p in PIECE_CATALOGUE:
        plane = piece_plane(p.piece_id)
        assert plane.sum() == p.size
        rows, cols = np.nonzero(plane)
        assert rows.min() == 0 and cols.min() == 0
        assert not plane.flags.writeable
