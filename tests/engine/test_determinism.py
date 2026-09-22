"""PRD S2: seeded games are bitwise identical across runs and processes."""

from __future__ import annotations

import hashlib
import subprocess
import sys

import numpy as np

from blockblast.engine.game import Game

N_GAMES = 1000


def play(seed: int) -> list[tuple[int, tuple[int | None, ...], int, int, int]]:
    """Play one game with actions chosen by a seed-derived RNG; return the trajectory."""
    g = Game()
    g.reset(seed)
    pick = np.random.default_rng(seed + 10**9)
    traj = []
    while not g.state.game_over:
        legal = np.flatnonzero(g.legal_moves())
        a = int(legal[pick.integers(len(legal))])
        s = g.step(a // 64, (a // 8) % 8, a % 8).state
        traj.append((s.board, s.hand, s.score, s.combo_streak, s.round_index))
    return traj


def digest(n: int) -> str:
    h = hashlib.sha256()
    for seed in range(n):
        h.update(repr(play(seed)).encode())
    return h.hexdigest()


def test_same_seed_same_trajectory() -> None:
    for seed in range(20):
        assert play(seed) == play(seed)


def test_different_seeds_differ() -> None:
    assert play(0) != play(1)


def test_1000_games_identical_across_runs_and_processes() -> None:
    local = digest(N_GAMES)
    assert local == digest(N_GAMES)
    code = (
        "import sys; sys.path[:0] = sys.argv[1:];"
        "from tests.engine.test_determinism import digest;"
        f"print(digest({N_GAMES}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, *sys.path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == local
