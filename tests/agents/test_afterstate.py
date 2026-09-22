"""The afterstate policy must reproduce engine rules exactly for every legal action."""

from __future__ import annotations

import numpy as np
import torch

from blockblast.agents.networks import AfterstateExtractor, AfterstatePolicy
from blockblast.engine.board import to_array
from blockblast.engine.game import Game, apply_placement
from blockblast.env.action import action_mask, decode_action
from blockblast.env.observation import encode_observation, observation_space


def test_afterstates_match_engine() -> None:
    ext = AfterstateExtractor(observation_space(), hidden=8, value_dim=8)
    rng = np.random.default_rng(0)
    checked = 0
    for seed in range(15):
        g = Game()
        g.reset(seed)
        while not g.state.game_over:
            s = g.state
            obs = torch.as_tensor(encode_observation(s)).unsqueeze(0)
            after, n_lines, delta, new_combo = (t[0].numpy() for t in ext.afterstates(obs))
            mask = action_mask(s)
            for a in np.flatnonzero(mask):
                r = apply_placement(s, *decode_action(int(a)), g.score_config)
                np.testing.assert_array_equal(after[a].reshape(8, 8), to_array(r.state.board))
                assert n_lines[a] == r.n_lines
                if s.combo_streak < 8:  # obs caps the streak at 8
                    assert np.isclose(delta[a] * 10, r.score_delta)
                    assert new_combo[a] == min(r.state.combo_streak, 8)
                checked += 1
            legal = np.flatnonzero(mask)
            g.step(*decode_action(int(legal[rng.integers(legal.size)])))
    assert checked > 1000


def test_initial_policy_leans_greedy() -> None:
    """At init the logits are ≈ Δscore/10, so line clears dominate plain placements."""
    ext = AfterstateExtractor(observation_space(), hidden=16, value_dim=8)
    torch.nn.init.zeros_(ext.head_out.weight)
    torch.nn.init.zeros_(ext.head_out.bias)
    obs = torch.zeros(1, 5, 8, 8)
    obs[0, 0, 0, :7] = 1.0  # row 0 missing (0, 7)
    obs[0, 1, 0, 0] = 1.0  # slot 0: mono
    logits = ext.action_logits(obs)[0]
    assert torch.isclose(logits[7], torch.tensor(1.1))  # mono at (0, 7): 1 + 10 points
    assert torch.isclose(logits[8], torch.tensor(0.1))


def test_policy_outputs_192_logits() -> None:
    policy = AfterstatePolicy(
        observation_space(),
        __import__("gymnasium").spaces.Discrete(192),
        lambda _: 1e-3,
    )
    obs = torch.zeros(4, 5, 8, 8)
    dist = policy.get_distribution(obs)
    assert dist.distribution.logits.shape == (4, 192)
    assert policy.predict_values(obs).shape == (4, 1)
