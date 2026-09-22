"""Observation: (5, 8, 8) float32 — board, 3 slot piece masks, combo plane."""

from __future__ import annotations

from typing import Final

import gymnasium
import numpy as np
import numpy.typing as npt

from blockblast.engine.board import to_array
from blockblast.engine.game import GameState
from blockblast.engine.pieces import PIECE_CATALOGUE

OBS_SHAPE: Final[tuple[int, int, int]] = (5, 8, 8)
COMBO_NORM: Final[int] = 8


def _build_planes() -> tuple[npt.NDArray[np.float32], ...]:
    planes = []
    for piece in PIECE_CATALOGUE:
        plane = np.zeros((8, 8), dtype=np.float32)
        for dr, dc in piece.cells:
            plane[dr, dc] = 1.0
        plane.setflags(write=False)
        planes.append(plane)
    return tuple(planes)


_PLANES: Final[tuple[npt.NDArray[np.float32], ...]] = _build_planes()


def observation_space() -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=0.0, high=1.0, shape=OBS_SHAPE, dtype=np.float32)


def piece_plane(piece_id: int) -> npt.NDArray[np.float32]:
    """Read-only (8, 8) mask of the piece anchored at (0, 0)."""
    return _PLANES[piece_id]


def encode_observation(state: GameState) -> npt.NDArray[np.float32]:
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    obs[0] = to_array(state.board)
    for slot, pid in enumerate(state.hand):
        if pid is not None:
            obs[1 + slot] = _PLANES[pid]
    obs[4] = min(state.combo_streak, COMBO_NORM) / COMBO_NORM
    return obs
