"""Bitboard primitives. Bit ``r*8 + c`` set means cell (r, c) is occupied."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

BOARD_SIZE: Final[int] = 8
FULL_BOARD: Final[int] = (1 << (BOARD_SIZE * BOARD_SIZE)) - 1

_ROW: Final[int] = (1 << BOARD_SIZE) - 1
_COL: Final[int] = sum(1 << (r * BOARD_SIZE) for r in range(BOARD_SIZE))

# Rows 0..7 then columns 0..7.
LINE_MASKS: Final[tuple[int, ...]] = tuple(_ROW << (r * BOARD_SIZE) for r in range(BOARD_SIZE)) + (
    tuple(_COL << c for c in range(BOARD_SIZE))
)


def can_place(board: int, mask: int) -> bool:
    """True if ``mask`` is an in-bounds placement that overlaps no occupied cell."""
    return mask != 0 and board & mask == 0


def place(board: int, mask: int) -> int:
    """Return the board with ``mask`` cells filled. Caller guarantees legality."""
    return board | mask


def full_lines(board: int) -> tuple[int, ...]:
    """Indices into ``LINE_MASKS`` of every completely filled row/column."""
    return tuple(i for i, line in enumerate(LINE_MASKS) if board & line == line)


def clear_lines(board: int, lines: Sequence[int]) -> int:
    """Clear all given lines simultaneously (union of masks)."""
    cleared = 0
    for i in lines:
        cleared |= LINE_MASKS[i]
    return board & ~cleared & FULL_BOARD


def popcount(board: int) -> int:
    """Number of occupied cells."""
    return board.bit_count()


def to_array(board: int) -> npt.NDArray[np.uint8]:
    """Board as an (8, 8) uint8 array of 0/1."""
    raw = np.frombuffer(board.to_bytes(8, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little").reshape(BOARD_SIZE, BOARD_SIZE)


def from_array(grid: npt.NDArray[np.integer]) -> int:
    """Inverse of :func:`to_array`. Any non-zero entry counts as occupied."""
    if grid.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"expected shape (8, 8), got {grid.shape}")
    bits = np.packbits((grid != 0).reshape(-1).astype(np.uint8), bitorder="little")
    return int.from_bytes(bits.tobytes(), "little")
