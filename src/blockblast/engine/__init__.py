"""Pure, deterministic Block Blast rules. Depends only on numpy and the stdlib."""

from blockblast.engine.game import (
    HAND_SIZE,
    Game,
    GameState,
    InvalidMoveError,
    MoveResult,
    apply_placement,
    is_game_over,
    legal_moves,
)
from blockblast.engine.pieces import CATALOGUE_VERSION, PIECE_CATALOGUE, Piece, get_piece
from blockblast.engine.scoring import ScoreConfig

__all__ = [
    "CATALOGUE_VERSION",
    "HAND_SIZE",
    "PIECE_CATALOGUE",
    "Game",
    "GameState",
    "InvalidMoveError",
    "MoveResult",
    "Piece",
    "ScoreConfig",
    "apply_placement",
    "get_piece",
    "is_game_over",
    "legal_moves",
]
