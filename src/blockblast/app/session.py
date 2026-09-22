"""Display-free game session for the GUI: human moves, AI moves, colors, best score."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from blockblast.agents.base import Agent
from blockblast.engine.board import to_array
from blockblast.engine.game import Game, GameState, InvalidMoveError, MoveResult, apply_placement
from blockblast.engine.pieces import get_piece
from blockblast.engine.scoring import ScoreConfig
from blockblast.env.action import action_mask, decode_action
from blockblast.env.observation import encode_observation


class PlaySession:
    """Wraps ``engine.Game`` with per-cell colors (piece ids) and a persistent best score.

    ``colors[r, c]`` is the piece id that filled the cell, or -1 if empty. It is purely
    cosmetic: the engine board stays the single source of truth.
    """

    def __init__(
        self,
        seed: int | None = None,
        score_config: ScoreConfig = ScoreConfig(),
        best_path: Path | None = None,
    ) -> None:
        self._cfg = score_config
        self._best_path = best_path
        self.best = self._load_best()
        self.game = Game(score_config)
        self.colors: npt.NDArray[np.int8] = np.full((8, 8), -1, dtype=np.int8)
        self.last: MoveResult | None = None
        self.reset(seed)

    @property
    def state(self) -> GameState:
        return self.game.state

    def reset(self, seed: int | None = None) -> GameState:
        self.colors[:] = -1
        self.last = None
        return self.game.reset(seed)

    def is_legal(self, slot: int, row: int, col: int) -> bool:
        return self.game.is_legal(slot, row, col)

    def preview(self, slot: int, row: int, col: int) -> MoveResult | None:
        """Result of a placement without applying it, or None if illegal."""
        try:
            return apply_placement(self.state, slot, row, col, self._cfg)
        except InvalidMoveError:
            return None

    def place(self, slot: int, row: int, col: int) -> MoveResult:
        """Apply a legal placement (raises ``InvalidMoveError`` otherwise)."""
        pid = self.state.hand[slot]
        result = self.game.step(slot, row, col)
        assert pid is not None
        for dr, dc in get_piece(pid).cells:
            self.colors[row + dr, col + dc] = pid
        self.colors[to_array(result.state.board) == 0] = -1
        self.last = result
        if result.state.score > self.best:
            self.best = result.state.score
            self._save_best()
        return result

    def agent_action(self, agent: Agent) -> tuple[int, int, int]:
        """Move the agent would play in the current state, as (slot, row, col)."""
        s = self.state
        return decode_action(agent.act(encode_observation(s), action_mask(s), s))

    def _load_best(self) -> int:
        if self._best_path is None or not self._best_path.exists():
            return 0
        try:
            return int(json.loads(self._best_path.read_text(encoding="utf-8"))["best"])
        except (ValueError, KeyError, OSError):
            return 0

    def _save_best(self) -> None:
        if self._best_path is None:
            return
        self._best_path.parent.mkdir(parents=True, exist_ok=True)
        self._best_path.write_text(json.dumps({"best": self.best}), encoding="utf-8")
