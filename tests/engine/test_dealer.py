from __future__ import annotations

import numpy as np
import pytest

from blockblast.engine.dealer import PieceDealer
from blockblast.engine.pieces import PIECE_CATALOGUE


def test_same_seed_same_hands() -> None:
    a = PieceDealer(np.random.default_rng(7))
    b = PieceDealer(np.random.default_rng(7))
    assert [a.deal() for _ in range(50)] == [b.deal() for _ in range(50)]


def test_hand_values_are_python_ints_in_range() -> None:
    d = PieceDealer(np.random.default_rng(0))
    for _ in range(100):
        hand = d.deal()
        assert len(hand) == 3
        assert all(type(p) is int and 0 <= p < len(PIECE_CATALOGUE) for p in hand)


def test_distribution_is_uniform() -> None:
    d = PieceDealer(np.random.default_rng(1))
    n = len(PIECE_CATALOGUE)
    counts = np.zeros(n)
    draws = 34_000  # ~100k pieces
    for _ in range(draws):
        for p in d.deal():
            counts[p] += 1
    expected = draws * 3 / n
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 60.0  # df=26, p ≈ 0.0002 critical value ≈ 60


def test_rejects_empty_catalogue() -> None:
    with pytest.raises(ValueError):
        PieceDealer(np.random.default_rng(0), n_pieces=0)
