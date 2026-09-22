"""Score rule: Δscore = n_cells·ppc + ppl · n_lines · (1 + s), s = streak before the move."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    points_per_cell: int = 1
    points_per_line: int = 10


def score_move(n_cells: int, n_lines: int, combo_streak: int, cfg: ScoreConfig) -> int:
    """Points earned by one placement given the combo streak *before* it."""
    return n_cells * cfg.points_per_cell + cfg.points_per_line * n_lines * (1 + combo_streak)


def next_combo_streak(n_lines: int, combo_streak: int) -> int:
    """Streak grows on any clearing placement and resets on a non-clearing one."""
    return combo_streak + 1 if n_lines >= 1 else 0
