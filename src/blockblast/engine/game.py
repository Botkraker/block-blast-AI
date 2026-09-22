"""Game state, rules and the stateful ``Game`` driver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import numpy as np
import numpy.typing as npt

from blockblast.engine.board import BOARD_SIZE, clear_lines, full_lines
from blockblast.engine.dealer import PieceDealer
from blockblast.engine.pieces import PIECE_CATALOGUE, VALID_PLACEMENTS, get_piece
from blockblast.engine.scoring import ScoreConfig, next_combo_streak, score_move

HAND_SIZE: Final[int] = 3

Hand = tuple[int | None, int | None, int | None]


@dataclass(frozen=True, slots=True)
class GameState:
    board: int
    hand: Hand
    score: int
    combo_streak: int
    round_index: int
    moves: int
    game_over: bool


@dataclass(frozen=True, slots=True)
class MoveResult:
    state: GameState
    score_delta: int
    n_cells: int
    n_lines: int
    cleared_lines: tuple[int, ...]
    new_hand_dealt: bool


class InvalidMoveError(ValueError):
    """Raised for any illegal placement. Illegal actions are never silently ignored."""


def _piece_fits(board: int, piece_id: int) -> bool:
    return any(board & m == 0 for _, m in VALID_PLACEMENTS[piece_id])


def is_game_over(board: int, hand: tuple[int | None, ...]) -> bool:
    """True iff no remaining piece in ``hand`` fits anywhere on ``board``."""
    return not any(pid is not None and _piece_fits(board, pid) for pid in hand)


def legal_moves(state: GameState) -> npt.NDArray[np.bool_]:
    """(3, 8, 8) bool: ``[slot, row, col]`` is True iff that placement is legal."""
    flat = np.zeros(HAND_SIZE * BOARD_SIZE * BOARD_SIZE, dtype=np.bool_)
    if state.game_over:
        return flat.reshape(HAND_SIZE, BOARD_SIZE, BOARD_SIZE)
    board = state.board
    for slot, pid in enumerate(state.hand):
        if pid is None:
            continue
        base = slot * BOARD_SIZE * BOARD_SIZE
        for idx, m in VALID_PLACEMENTS[pid]:
            if board & m == 0:
                flat[base + idx] = True
    return flat.reshape(HAND_SIZE, BOARD_SIZE, BOARD_SIZE)


def _validate(state: GameState, slot: int, row: int, col: int) -> int:
    """Return the placement mask or raise ``InvalidMoveError``."""
    if state.game_over:
        raise InvalidMoveError("game is over")
    if not (0 <= slot < HAND_SIZE and 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise InvalidMoveError(f"out of range: slot={slot} row={row} col={col}")
    pid = state.hand[slot]
    if pid is None:
        raise InvalidMoveError(f"slot {slot} is already used")
    piece = get_piece(pid)
    if row + piece.height > BOARD_SIZE or col + piece.width > BOARD_SIZE:
        raise InvalidMoveError(f"piece {piece.name} leaves the board at ({row}, {col})")
    mask = 0
    for dr, dc in piece.cells:
        mask |= 1 << ((row + dr) * BOARD_SIZE + col + dc)
    if state.board & mask:
        raise InvalidMoveError(f"piece {piece.name} overlaps at ({row}, {col})")
    return mask


def apply_placement(
    state: GameState, slot: int, row: int, col: int, cfg: ScoreConfig
) -> MoveResult:
    """Pure placement: place, clear, score, update combo. Never deals a new hand.

    ``game_over`` in the returned state reflects only the remaining hand; when the hand
    is empty it is False because the outcome depends on the next deal.
    """
    mask = _validate(state, slot, row, col)
    n_cells = mask.bit_count()
    board = state.board | mask
    lines = full_lines(board)
    if lines:
        board = clear_lines(board, lines)
    n_lines = len(lines)
    delta = score_move(n_cells, n_lines, state.combo_streak, cfg)
    hand_list = list(state.hand)
    hand_list[slot] = None
    hand: Hand = (hand_list[0], hand_list[1], hand_list[2])
    empty = all(p is None for p in hand)
    new_state = GameState(
        board=board,
        hand=hand,
        score=state.score + delta,
        combo_streak=next_combo_streak(n_lines, state.combo_streak),
        round_index=state.round_index,
        moves=state.moves + 1,
        game_over=False if empty else is_game_over(board, hand),
    )
    return MoveResult(new_state, delta, n_cells, n_lines, lines, new_hand_dealt=False)


class Game:
    """Stateful driver: owns the dealer and advances rounds."""

    def __init__(self, score_config: ScoreConfig = ScoreConfig()) -> None:
        self._cfg = score_config
        self._dealer: PieceDealer | None = None
        self._state: GameState | None = None

    @property
    def score_config(self) -> ScoreConfig:
        return self._cfg

    def reset(
        self,
        rng: np.random.Generator | int | None = None,
        board: int = 0,
        hard_weight: float = 1.0,
    ) -> GameState:
        """Start a new game. An int or None seeds a fresh PCG64 generator.

        ``board`` starts from a non-empty board (training mid-game starts) and
        ``hard_weight`` makes the hardest pieces rarer (curriculum); see ``PieceDealer``.
        """
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self._dealer = PieceDealer(gen, len(PIECE_CATALOGUE), hard_weight)
        hand = self._dealer.deal()
        self._state = GameState(
            board=board,
            hand=hand,
            score=0,
            combo_streak=0,
            round_index=1,
            moves=0,
            game_over=is_game_over(board, hand),
        )
        return self._state

    @classmethod
    def from_state(
        cls,
        state: GameState,
        rng: np.random.Generator,
        score_config: ScoreConfig = ScoreConfig(),
    ) -> Game:
        """Resume from an arbitrary state (e.g. a parsed real-game frame)."""
        game = cls(score_config)
        game._dealer = PieceDealer(rng, len(PIECE_CATALOGUE))
        game._state = state
        return game

    @property
    def state(self) -> GameState:
        if self._state is None:
            raise RuntimeError("call reset() first")
        return self._state

    def legal_moves(self) -> npt.NDArray[np.bool_]:
        return legal_moves(self.state)

    def is_legal(self, slot: int, row: int, col: int) -> bool:
        try:
            _validate(self.state, slot, row, col)
        except InvalidMoveError:
            return False
        return True

    def step(self, slot: int, row: int, col: int) -> MoveResult:
        """Apply a placement, deal a new hand if the hand is empty, update game over."""
        result = apply_placement(self.state, slot, row, col, self._cfg)
        state = result.state
        dealt = False
        if all(p is None for p in state.hand):
            assert self._dealer is not None
            hand = self._dealer.deal()
            state = replace(
                state,
                hand=hand,
                round_index=state.round_index + 1,
                game_over=is_game_over(state.board, hand),
            )
            dealt = True
        self._state = state
        return replace(result, state=state, new_hand_dealt=dealt)
