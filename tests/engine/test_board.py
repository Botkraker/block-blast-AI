from __future__ import annotations

import numpy as np
import pytest

from blockblast.engine.board import (
    FULL_BOARD,
    LINE_MASKS,
    can_place,
    clear_lines,
    from_array,
    full_lines,
    place,
    popcount,
    to_array,
)
from tests.conftest import cell


def test_line_masks() -> None:
    assert len(LINE_MASKS) == 16
    assert all(popcount(m) == 8 for m in LINE_MASKS)
    assert LINE_MASKS[0] == 0xFF
    assert LINE_MASKS[8] == sum(cell(r, 0) for r in range(8))
    union = 0
    for m in LINE_MASKS[:8]:
        union |= m
    assert union == FULL_BOARD


def test_can_place_and_place() -> None:
    board = cell(3, 3)
    assert can_place(board, cell(3, 4))
    assert not can_place(board, cell(3, 3))
    assert not can_place(board, 0)  # out-of-bounds sentinel
    assert place(board, cell(3, 4)) == cell(3, 3) | cell(3, 4)


def test_full_lines_row_col_and_cross() -> None:
    assert full_lines(0) == ()
    assert full_lines(LINE_MASKS[2]) == (2,)
    assert full_lines(LINE_MASKS[13]) == (13,)
    cross = LINE_MASKS[4] | LINE_MASKS[8 + 4]
    assert full_lines(cross) == (4, 12)
    assert full_lines(FULL_BOARD) == tuple(range(16))


def test_clear_lines_is_simultaneous() -> None:
    cross = LINE_MASKS[4] | LINE_MASKS[12] | cell(0, 0)
    # Clearing row first then re-detecting would miss the column; union clears both.
    assert clear_lines(cross, full_lines(cross)) == cell(0, 0)
    assert clear_lines(FULL_BOARD, full_lines(FULL_BOARD)) == 0
    assert clear_lines(cell(1, 1), ()) == cell(1, 1)


def test_array_round_trip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        board = int(rng.integers(0, 2**63)) | (int(rng.integers(0, 2)) << 63)
        arr = to_array(board)
        assert arr.shape == (8, 8) and arr.dtype == np.uint8
        assert from_array(arr) == board
    assert to_array(cell(2, 5))[2, 5] == 1
    assert to_array(cell(2, 5)).sum() == 1


def test_from_array_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        from_array(np.zeros((7, 8), dtype=np.uint8))
