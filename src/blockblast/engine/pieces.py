"""Piece catalogue v1 and precomputed placement bitmasks.

Pieces never rotate, so every orientation is its own catalogue entry. The order of
``PIECE_CATALOGUE`` defines ``piece_id`` and is frozen per ``CATALOGUE_VERSION``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

CATALOGUE_VERSION: Final[int] = 1
_BOARD: Final[int] = 8


@dataclass(frozen=True, slots=True)
class Piece:
    """An immutable polyomino. ``cells`` are (dr, dc) offsets from the bbox top-left."""

    piece_id: int
    name: str
    cells: tuple[tuple[int, int], ...]
    height: int
    width: int

    @property
    def size(self) -> int:
        """Number of cells the piece occupies."""
        return len(self.cells)


# (name, rows). "X" = filled cell. Order == piece_id.
_SHAPES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("mono", ("X",)),
    ("I2h", ("XX",)),
    ("I2v", ("X", "X")),
    ("I3h", ("XXX",)),
    ("I3v", ("X", "X", "X")),
    ("I4h", ("XXXX",)),
    ("I4v", ("X", "X", "X", "X")),
    ("I5h", ("XXXXX",)),
    ("I5v", ("X", "X", "X", "X", "X")),
    ("O2", ("XX", "XX")),
    ("O3", ("XXX", "XXX", "XXX")),
    ("L0", ("X.", "X.", "XX")),
    ("L1", ("XXX", "X..")),
    ("L2", ("XX", ".X", ".X")),
    ("L3", ("..X", "XXX")),
    ("J0", (".X", ".X", "XX")),
    ("J1", ("X..", "XXX")),
    ("J2", ("XX", "X.", "X.")),
    ("J3", ("XXX", "..X")),
    ("T0", ("XXX", ".X.")),
    ("T1", (".X", "XX", ".X")),
    ("T2", (".X.", "XXX")),
    ("T3", ("X.", "XX", "X.")),
    ("S0", (".XX", "XX.")),
    ("S1", ("X.", "XX", ".X")),
    ("Z0", ("XX.", ".XX")),
    ("Z1", (".X", "XX", "X.")),
)


def _build_piece(piece_id: int, name: str, rows: tuple[str, ...]) -> Piece:
    cells = tuple((r, c) for r, line in enumerate(rows) for c, ch in enumerate(line) if ch == "X")
    return Piece(piece_id, name, cells, height=len(rows), width=len(rows[0]))


PIECE_CATALOGUE: Final[tuple[Piece, ...]] = tuple(
    _build_piece(i, name, rows) for i, (name, rows) in enumerate(_SHAPES)
)


def _placement_masks(piece: Piece) -> tuple[int, ...]:
    masks: list[int] = []
    for row in range(_BOARD):
        for col in range(_BOARD):
            if row + piece.height > _BOARD or col + piece.width > _BOARD:
                masks.append(0)
                continue
            mask = 0
            for dr, dc in piece.cells:
                mask |= 1 << ((row + dr) * _BOARD + (col + dc))
            masks.append(mask)
    return tuple(masks)


# [piece_id][row*8+col] -> bitmask of covered cells, 0 if the piece leaves the board.
PLACEMENT_MASKS: Final[tuple[tuple[int, ...], ...]] = tuple(
    _placement_masks(p) for p in PIECE_CATALOGUE
)

# [piece_id] -> ((cell_index, mask), ...) for in-bounds placements only; hot path helper.
VALID_PLACEMENTS: Final[tuple[tuple[tuple[int, int], ...], ...]] = tuple(
    tuple((idx, m) for idx, m in enumerate(masks) if m) for masks in PLACEMENT_MASKS
)


def get_piece(piece_id: int) -> Piece:
    """Return the catalogue piece for ``piece_id`` (raises ``IndexError`` if unknown)."""
    if not 0 <= piece_id < len(PIECE_CATALOGUE):
        raise IndexError(f"unknown piece_id {piece_id}")
    return PIECE_CATALOGUE[piece_id]
