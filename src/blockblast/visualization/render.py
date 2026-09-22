"""ANSI and RGB rendering of a ``GameState`` (NumPy only, no GUI dependency)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from blockblast.engine.board import to_array
from blockblast.engine.game import GameState
from blockblast.engine.pieces import get_piece

_BG = np.array([24, 24, 32], dtype=np.uint8)
_EMPTY = np.array([48, 50, 64], dtype=np.uint8)
_FILLED = np.array([86, 156, 255], dtype=np.uint8)
_PIECE = np.array([255, 190, 70], dtype=np.uint8)
_OVER = np.array([200, 60, 60], dtype=np.uint8)


def render_ansi(state: GameState) -> str:
    grid = to_array(state.board)
    lines = ["  " + " ".join(str(c) for c in range(8))]
    lines += [f"{r} " + " ".join("#" if v else "." for v in grid[r]) for r in range(8)]
    lines.append(
        f"score={state.score} round={state.round_index} moves={state.moves} "
        f"combo={state.combo_streak}{' GAME OVER' if state.game_over else ''}"
    )
    hand_rows = [""] * 5
    labels = []
    for slot, pid in enumerate(state.hand):
        block = [["."] * 5 for _ in range(5)]
        name = "used"
        if pid is not None:
            piece = get_piece(pid)
            for dr, dc in piece.cells:
                block[dr][dc] = "#"
            name = piece.name
        labels.append(f"[{slot}] {name}".ljust(8))
        for r in range(5):
            hand_rows[r] += "".join(block[r]) + "   "
    lines.append("".join(labels).rstrip())
    lines += [row.rstrip() for row in hand_rows]
    return "\n".join(lines)


def render_rgb(state: GameState, cell_px: int = 32) -> npt.NDArray[np.uint8]:
    """(H, W, 3) uint8 image: board on top, the 3 hand slots below."""
    gap = max(cell_px // 16, 1)
    small = max(cell_px // 2, 2)
    width = 8 * cell_px
    hand_h = 5 * small + 2 * gap
    img = np.empty((8 * cell_px + hand_h, width, 3), dtype=np.uint8)
    img[:] = _BG
    grid = to_array(state.board)
    fill = _OVER if state.game_over else _FILLED
    for r in range(8):
        for c in range(8):
            y, x = r * cell_px, c * cell_px
            img[y + gap : y + cell_px - gap, x + gap : x + cell_px - gap] = (
                fill if grid[r, c] else _EMPTY
            )
    slot_w = width // 3
    for slot, pid in enumerate(state.hand):
        if pid is None:
            continue
        piece = get_piece(pid)
        x0 = slot * slot_w + (slot_w - piece.width * small) // 2
        y0 = 8 * cell_px + gap
        for dr, dc in piece.cells:
            y, x = y0 + dr * small, x0 + dc * small
            img[y + 1 : y + small - 1, x + 1 : x + small - 1] = _PIECE
    return img
