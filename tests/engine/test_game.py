from __future__ import annotations

import numpy as np
import pytest

from blockblast.engine.board import FULL_BOARD, LINE_MASKS
from blockblast.engine.game import (
    Game,
    InvalidMoveError,
    apply_placement,
    is_game_over,
    legal_moves,
)
from blockblast.engine.pieces import PIECE_CATALOGUE, PLACEMENT_MASKS
from blockblast.engine.scoring import ScoreConfig
from tests.conftest import cell, make_state

CFG = ScoreConfig()
MONO, I2H, O3 = 0, 1, 10


def test_reset_initial_state() -> None:
    s = Game().reset(0)
    assert s.board == 0 and s.score == 0 and s.combo_streak == 0
    assert s.round_index == 1 and s.moves == 0 and not s.game_over
    assert all(p is not None for p in s.hand)


def test_reset_accepts_generator_int_and_none() -> None:
    a = Game().reset(np.random.default_rng(5))
    b = Game().reset(5)
    assert a == b
    assert Game().reset(None).round_index == 1


def test_state_before_reset_raises() -> None:
    with pytest.raises(RuntimeError):
        _ = Game().state


def test_score_config_property() -> None:
    cfg = ScoreConfig(2, 3)
    assert Game(cfg).score_config is cfg


def test_placement_scores_cells_and_consumes_slot() -> None:
    r = apply_placement(make_state(hand=(O3, MONO, MONO)), 0, 2, 3, CFG)
    assert r.n_cells == 9 and r.n_lines == 0 and r.score_delta == 9
    assert r.state.hand == (None, MONO, MONO)
    assert r.state.moves == 1 and r.state.score == 9
    assert r.state.board == PLACEMENT_MASKS[O3][2 * 8 + 3]
    assert not r.new_hand_dealt


def test_line_clear_and_combo_progression(row_minus_last: int) -> None:
    s = make_state(board=row_minus_last | (LINE_MASKS[1] & ~cell(1, 7)), hand=(MONO, MONO, MONO))
    r1 = apply_placement(s, 0, 0, 7, CFG)
    assert r1.n_lines == 1 and r1.cleared_lines == (0,)
    assert r1.score_delta == 1 + 10 * 1 * (1 + 0)
    assert r1.state.combo_streak == 1
    assert r1.state.board == LINE_MASKS[1] & ~cell(1, 7)
    r2 = apply_placement(r1.state, 1, 1, 7, CFG)
    assert r2.score_delta == 1 + 10 * 1 * (1 + 1)
    assert r2.state.combo_streak == 2
    r3 = apply_placement(r2.state, 2, 5, 5, CFG)
    assert r3.score_delta == 1 and r3.state.combo_streak == 0


def test_simultaneous_row_and_column_clear() -> None:
    board = (LINE_MASKS[0] | LINE_MASKS[8]) & ~cell(0, 0)
    r = apply_placement(make_state(board=board, hand=(MONO, MONO, MONO)), 0, 0, 0, CFG)
    assert r.n_lines == 2 and r.cleared_lines == (0, 8)
    assert r.state.board == 0
    assert r.score_delta == 1 + 20


def test_apply_placement_is_pure() -> None:
    s = make_state(hand=(MONO, MONO, MONO))
    apply_placement(s, 0, 0, 0, CFG)
    assert s == make_state(hand=(MONO, MONO, MONO))


@pytest.mark.parametrize(
    ("slot", "row", "col"),
    [(-1, 0, 0), (3, 0, 0), (0, -1, 0), (0, 8, 0), (0, 0, -1), (0, 0, 8)],
)
def test_out_of_range_raises(slot: int, row: int, col: int) -> None:
    with pytest.raises(InvalidMoveError):
        apply_placement(make_state(), slot, row, col, CFG)


def test_used_slot_overlap_and_out_of_board_raise() -> None:
    with pytest.raises(InvalidMoveError, match="already used"):
        apply_placement(make_state(hand=(None, MONO, MONO)), 0, 0, 0, CFG)
    with pytest.raises(InvalidMoveError, match="overlaps"):
        apply_placement(make_state(board=cell(4, 4), hand=(MONO, 0, 0)), 0, 4, 4, CFG)
    with pytest.raises(InvalidMoveError, match="leaves the board"):
        apply_placement(make_state(hand=(O3, 0, 0)), 0, 6, 0, CFG)
    with pytest.raises(InvalidMoveError, match="leaves the board"):
        apply_placement(make_state(hand=(O3, 0, 0)), 0, 0, 6, CFG)


def test_game_over_state_rejects_moves() -> None:
    with pytest.raises(InvalidMoveError, match="over"):
        apply_placement(make_state(game_over=True), 0, 0, 0, CFG)


def test_is_game_over() -> None:
    assert not is_game_over(0, (MONO, None, None))
    assert is_game_over(FULL_BOARD, (MONO, MONO, MONO))
    assert is_game_over(0, (None, None, None))
    one_hole = FULL_BOARD & ~cell(3, 3)
    assert not is_game_over(one_hole, (O3, None, MONO))
    assert is_game_over(one_hole, (O3, I2H, None))


def test_placement_sets_game_over_when_remaining_hand_cannot_fit() -> None:
    # Holes: a 2x2 block at (0,0) plus the diagonal (r, r) for r >= 2, so no placement in
    # the block completes a line. The O2 fits only in the block.
    holes = [(0, 0), (0, 1), (1, 0), (1, 1)] + [(r, r) for r in range(2, 8)]
    board = FULL_BOARD
    for r, c in holes:
        board &= ~cell(r, c)
    O2 = 9
    r = apply_placement(make_state(board=board, hand=(MONO, O2, None)), 0, 0, 0, CFG)
    assert r.n_lines == 0 and r.state.game_over
    # Filling the diagonal hole (2,2) clears row 2 and col 2 instead: O2 still fits.
    r2 = apply_placement(make_state(board=board, hand=(MONO, O2, None)), 0, 2, 2, CFG)
    assert r2.n_lines == 2 and not r2.state.game_over


def test_empty_hand_after_placement_is_not_game_over() -> None:
    r = apply_placement(make_state(hand=(MONO, None, None)), 0, 0, 0, CFG)
    assert r.state.hand == (None, None, None) and not r.state.game_over


def test_legal_moves_matches_is_legal() -> None:
    g = Game()
    g.reset(3)
    for _ in range(30):
        lm = g.legal_moves()
        assert lm.shape == (3, 8, 8) and lm.dtype == np.bool_
        for slot in range(3):
            for row in range(8):
                for col in range(8):
                    assert lm[slot, row, col] == g.is_legal(slot, row, col)
        if g.state.game_over:
            break
        slot, row, col = np.argwhere(lm)[0]
        g.step(int(slot), int(row), int(col))


def test_legal_moves_empty_when_game_over() -> None:
    assert not legal_moves(make_state(game_over=True)).any()
    assert not legal_moves(make_state(hand=(None, None, None))).any()


def test_round_dealing_after_three_placements() -> None:
    g = Game()
    s0 = g.reset(11)
    moves = []
    for _ in range(3):
        slot, row, col = np.argwhere(g.legal_moves())[0]
        moves.append(g.step(int(slot), int(row), int(col)))
    assert [m.new_hand_dealt for m in moves] == [False, False, True]
    s = g.state
    assert s.round_index == s0.round_index + 1
    assert all(p is not None for p in s.hand) and s.moves == 3


def test_step_raises_and_keeps_state() -> None:
    g = Game()
    s = g.reset(1)
    with pytest.raises(InvalidMoveError):
        g.step(0, 8, 8)
    assert g.state == s


def test_from_state_resumes() -> None:
    s = make_state(board=cell(0, 0), hand=(MONO, None, None), score=42)
    g = Game.from_state(s, np.random.default_rng(0))
    assert g.state == s
    r = g.step(0, 1, 1)
    assert r.new_hand_dealt and r.state.round_index == 2 and r.state.score == 43


def test_game_over_after_deal() -> None:
    # Four free corners; placing a mono in one clears nothing and leaves 3 isolated holes.
    board = FULL_BOARD & ~cell(0, 0) & ~cell(7, 7) & ~cell(0, 7) & ~cell(7, 0)
    s = make_state(board=board, hand=(MONO, None, None))
    g = Game.from_state(s, np.random.default_rng(0))
    r = g.step(0, 0, 0)
    assert r.new_hand_dealt
    # Result depends on the dealt hand; verify consistency with the rule.
    assert r.state.game_over == is_game_over(r.state.board, r.state.hand)


def test_full_game_until_over_is_consistent() -> None:
    g = Game()
    g.reset(2024)
    rng = np.random.default_rng(0)
    total = 0
    while not g.state.game_over:
        legal = np.argwhere(g.legal_moves())
        assert len(legal) > 0
        slot, row, col = legal[rng.integers(len(legal))]
        r = g.step(int(slot), int(row), int(col))
        total += r.score_delta
        assert r.state.score == total
    assert not g.legal_moves().any()
    assert len(PIECE_CATALOGUE) == 27
