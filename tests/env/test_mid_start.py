"""Training-only env options: mid-game starts from crowded boards, hard-piece curriculum."""

from __future__ import annotations

import numpy as np

from blockblast.env import BlockBlastEnv


def _play_until_done(env: BlockBlastEnv, rng: np.random.Generator) -> None:
    while True:
        legal = np.flatnonzero(env.action_masks())
        _, _, terminated, truncated, _ = env.step(int(rng.choice(legal)))
        if terminated or truncated:
            return


def test_mid_start_reuses_crowded_boards() -> None:
    env = BlockBlastEnv(mid_start_prob=1.0, mid_start_min_cells=20)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    assert env.game.state.board == 0  # nothing recorded yet: a normal start
    for _ in range(5):
        _play_until_done(env, rng)
        env.reset()
    crowded = set(env._crowded)
    assert crowded and all(b.bit_count() >= 20 for b in crowded)
    starts = []
    for _ in range(20):
        env.reset()
        starts.append(env.game.state)
    assert all(s.score == 0 and s.moves == 0 and not s.game_over for s in starts)
    assert sum(s.board in crowded for s in starts) >= 10  # the rest fell back to empty boards


def test_defaults_keep_seeded_games_unchanged() -> None:
    a, b = BlockBlastEnv(), BlockBlastEnv(mid_start_prob=0.0)
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    np.testing.assert_array_equal(oa, ob)
    assert not b._crowded


def test_hard_piece_weight_applies_to_next_game() -> None:
    from blockblast.engine.dealer import HARD_PIECES

    env = BlockBlastEnv()
    env.hard_piece_weight = 0.0
    rng = np.random.default_rng(1)
    for seed in range(30):
        env.reset(seed=seed)
        assert not set(env.game.state.hand) & HARD_PIECES
        env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
