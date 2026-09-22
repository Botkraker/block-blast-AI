from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blockblast.agents import GreedyAgent
from blockblast.app import PlaySession
from blockblast.engine.board import to_array
from blockblast.engine.game import InvalidMoveError


def test_colors_track_board_through_a_full_game(tmp_path: Path) -> None:
    s = PlaySession(seed=3, best_path=tmp_path / "best.json")
    agent = GreedyAgent()
    while not s.state.game_over:
        slot, row, col = s.agent_action(agent)
        pid = s.state.hand[slot]
        preview = s.preview(slot, row, col)
        result = s.place(slot, row, col)
        assert preview is not None and preview.score_delta == result.score_delta
        np.testing.assert_array_equal(s.colors >= 0, to_array(s.state.board) == 1)
        assert pid is not None and s.last is result
    assert s.best == s.state.score > 0
    assert PlaySession(best_path=tmp_path / "best.json").best == s.best  # persisted


def test_illegal_moves() -> None:
    s = PlaySession(seed=0)
    assert s.preview(0, 8, 8) is None and not s.is_legal(0, 8, 8)
    with pytest.raises(InvalidMoveError):
        s.place(0, 8, 8)


def test_reset_and_corrupt_best_file(tmp_path: Path) -> None:
    bad = tmp_path / "best.json"
    bad.write_text("not json", encoding="utf-8")
    s = PlaySession(seed=1, best_path=bad)
    assert s.best == 0
    slot, row, col = s.agent_action(GreedyAgent())
    s.place(slot, row, col)
    s.reset(1)
    assert s.state.moves == 0 and (s.colors == -1).all() and s.last is None
