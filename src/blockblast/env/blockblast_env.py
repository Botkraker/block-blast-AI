"""Gymnasium environment wrapping ``engine.Game`` with invalid-action masking.

Two training-only options, both off by default (evaluation always plays the real game):

- ``mid_start_prob``: each env remembers crowded boards (``>= mid_start_min_cells`` filled)
  from its own past games, and starts that share of new games from one of them.
- ``hard_piece_weight`` (attribute, set by the curriculum callback): < 1 deals the hardest
  pieces less often in the games that start afterwards.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium
import numpy as np
import numpy.typing as npt

from blockblast.engine.game import Game, GameState
from blockblast.engine.scoring import ScoreConfig
from blockblast.env.action import action_mask, action_space, decode_action
from blockblast.env.observation import encode_observation, observation_space
from blockblast.env.reward import RewardConfig, compute_reward
from blockblast.visualization.render import render_ansi, render_rgb

Obs = npt.NDArray[np.float32]


class BlockBlastEnv(gymnasium.Env[Obs, int]):
    metadata: dict[str, Any] = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}  # noqa: RUF012

    def __init__(
        self,
        reward_config: RewardConfig = RewardConfig(),
        score_config: ScoreConfig = ScoreConfig(),
        max_steps: int = 10_000,
        render_mode: str | None = None,
        mid_start_prob: float = 0.0,
        mid_start_min_cells: int = 24,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.reward_config = reward_config
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.observation_space = observation_space()
        self.action_space = action_space()
        self._game = Game(score_config)
        self._mask: npt.NDArray[np.bool_] = np.zeros(192, dtype=np.bool_)
        self._lines_total = 0
        self._max_combo = 0
        self.mid_start_prob = mid_start_prob
        self.mid_start_min_cells = mid_start_min_cells
        self.hard_piece_weight = 1.0
        # ponytail: per-env buffer of this env's own boards; share one pool if diversity matters
        self._crowded: deque[int] = deque(maxlen=1000)

    @property
    def game(self) -> Game:
        return self._game

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        super().reset(seed=seed)
        state = self._game.reset(self.np_random, hard_weight=self.hard_piece_weight)
        # Rng is only consumed when mid starts are on, so default seeded games are unchanged.
        if (
            self.mid_start_prob > 0
            and self._crowded
            and self.np_random.random() < self.mid_start_prob
        ):
            board = self._crowded[int(self.np_random.integers(len(self._crowded)))]
            mid = self._game.reset(self.np_random, board=board, hard_weight=self.hard_piece_weight)
            if not mid.game_over:
                state = mid
            else:  # the dealt hand fits nowhere on that board: play a normal game
                state = self._game.reset(self.np_random, hard_weight=self.hard_piece_weight)
        self._lines_total = 0
        self._max_combo = 0
        self._mask = action_mask(state)
        return encode_observation(state), self._info(state, score_delta=0, n_lines=0)

    def step(self, action: int) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        slot, row, col = decode_action(action)
        result = self._game.step(slot, row, col)  # raises InvalidMoveError if illegal
        state = result.state
        self._lines_total += result.n_lines
        self._max_combo = max(self._max_combo, state.combo_streak)
        terminated = state.game_over
        if (
            self.mid_start_prob > 0
            and not terminated
            and state.board.bit_count() >= self.mid_start_min_cells
        ):
            self._crowded.append(state.board)
        truncated = not terminated and state.moves >= self.max_steps
        reward = compute_reward(result, terminated, self.reward_config)
        self._mask = action_mask(state)
        info = self._info(state, result.score_delta, result.n_lines)
        return encode_observation(state), reward, terminated, truncated, info

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """Legal-action mask for MaskablePPO (discovered via ``env_method``)."""
        return self._mask.copy()

    def render(self) -> str | npt.NDArray[np.uint8] | None:
        if self.render_mode == "ansi":
            return render_ansi(self._game.state)
        if self.render_mode == "rgb_array":
            return render_rgb(self._game.state)
        return None

    def _info(self, state: GameState, score_delta: int, n_lines: int) -> dict[str, Any]:
        return {
            "score": state.score,
            "score_delta": score_delta,
            "round": state.round_index,
            "moves": state.moves,
            "n_lines": n_lines,
            "combo_streak": state.combo_streak,
            "lines_total": self._lines_total,
            "max_combo": self._max_combo,
        }
