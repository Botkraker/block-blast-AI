"""Colors and layout constants for the graphical game (portrait, mobile-like)."""

from __future__ import annotations

from typing import Final

from blockblast.engine.pieces import PIECE_CATALOGUE

Color = tuple[int, int, int]

WIDTH: Final[int] = 480
HEIGHT: Final[int] = 820
FPS: Final[int] = 60

CELL: Final[int] = 50
BOARD_X: Final[int] = (WIDTH - 8 * CELL) // 2
BOARD_Y: Final[int] = 170
BOARD_PAD: Final[int] = 8

TRAY_Y: Final[int] = BOARD_Y + 8 * CELL + 36
TRAY_H: Final[int] = 170
TRAY_CELL: Final[int] = 26

BG_TOP: Final[Color] = (40, 62, 132)
BG_BOTTOM: Final[Color] = (22, 30, 74)
BOARD_BG: Final[Color] = (24, 30, 64)
EMPTY_CELL: Final[Color] = (36, 44, 88)
TEXT: Final[Color] = (245, 245, 255)
TEXT_DIM: Final[Color] = (170, 180, 220)
GOLD: Final[Color] = (255, 206, 84)
LEGAL: Final[Color] = (120, 230, 140)
ILLEGAL: Final[Color] = (240, 90, 90)
HINT: Final[Color] = (255, 255, 255)

_FAMILY_COLORS: Final[dict[str, Color]] = {
    "mono": (255, 214, 64),
    "I2": (72, 214, 238),
    "I3": (60, 190, 240),
    "I4": (66, 150, 245),
    "I5": (92, 120, 250),
    "O2": (255, 150, 60),
    "O3": (240, 86, 86),
    "L": (255, 128, 48),
    "J": (70, 110, 240),
    "T": (176, 96, 236),
    "S": (90, 210, 110),
    "Z": (236, 76, 120),
}


def _family(name: str) -> str:
    if name.startswith(("I", "O")):
        return name[:2]
    return name if name == "mono" else name[0]


PIECE_COLORS: Final[tuple[Color, ...]] = tuple(
    _FAMILY_COLORS[_family(p.name)] for p in PIECE_CATALOGUE
)


def shade(color: Color, factor: float) -> Color:
    """Lighten (factor > 1) or darken (factor < 1) a color."""
    if factor >= 1.0:
        return (
            min(255, int(color[0] + (255 - color[0]) * (factor - 1.0))),
            min(255, int(color[1] + (255 - color[1]) * (factor - 1.0))),
            min(255, int(color[2] + (255 - color[2]) * (factor - 1.0))),
        )
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))
