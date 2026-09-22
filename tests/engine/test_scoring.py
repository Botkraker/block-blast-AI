from __future__ import annotations

import pytest

from blockblast.engine.scoring import ScoreConfig, next_combo_streak, score_move

CFG = ScoreConfig()


@pytest.mark.parametrize(
    ("n_cells", "n_lines", "streak", "expected"),
    [
        (1, 0, 0, 1),
        (4, 0, 5, 4),  # streak irrelevant without clears
        (4, 1, 0, 14),
        (4, 2, 0, 24),
        (3, 1, 1, 23),
        (5, 2, 3, 85),
    ],
)
def test_score_move(n_cells: int, n_lines: int, streak: int, expected: int) -> None:
    assert score_move(n_cells, n_lines, streak, CFG) == expected


def test_score_config_is_applied() -> None:
    assert score_move(2, 1, 1, ScoreConfig(points_per_cell=3, points_per_line=7)) == 6 + 14


def test_next_combo_streak() -> None:
    assert next_combo_streak(1, 0) == 1
    assert next_combo_streak(3, 4) == 5
    assert next_combo_streak(0, 4) == 0
    assert next_combo_streak(0, 0) == 0
