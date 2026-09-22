"""Reward modes. Default: r_t = Δscore_t / 10 − 5 · 1[terminated_t].

``safe`` plays for survival: every move earns ``alive_bonus`` minus ``board_weight`` times
the fraction of the 64 cells still filled after the move, so an emptier board is worth
more than points. r_t = alive_bonus − board_weight · filled_t / 64 − penalty · 1[terminated_t].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from blockblast.engine.game import MoveResult

RewardMode = Literal["score", "survival", "sparse", "safe"]


@dataclass(frozen=True, slots=True)
class RewardConfig:
    mode: RewardMode = "score"
    score_scale: float = 10.0
    game_over_penalty: float = 5.0
    sparse_scale: float = 100.0
    alive_bonus: float = 1.0
    board_weight: float = 1.0


def compute_reward(result: MoveResult, terminated: bool, cfg: RewardConfig) -> float:
    """Truncation is never penalized; only ``terminated`` (true game over) is."""
    penalty = cfg.game_over_penalty if terminated else 0.0
    if cfg.mode == "score":
        return result.score_delta / cfg.score_scale - penalty
    if cfg.mode == "survival":
        return 1.0 - penalty
    if cfg.mode == "sparse":
        return result.state.score / cfg.sparse_scale if terminated else 0.0
    if cfg.mode == "safe":
        filled = result.state.board.bit_count()
        return cfg.alive_bonus - cfg.board_weight * filled / 64 - penalty
    raise ValueError(f"unknown reward mode {cfg.mode!r}")
