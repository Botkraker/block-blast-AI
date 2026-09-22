"""The afterstate policy must reproduce engine rules exactly for every legal action."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium
import numpy as np
import pytest
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from blockblast.agents.networks import ILLEGAL_LOGIT, AfterstateExtractor, AfterstatePolicy
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
            np.testing.assert_array_equal(ext.legal(obs)[0].numpy(), mask)
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


def _dense_logits(ext: AfterstateExtractor, obs: torch.Tensor) -> torch.Tensor:
    """Reference: the MLP scored on all 192 actions (the pre-compaction implementation)."""
    after, n_lines, delta, new_combo = ext.afterstates(obs)
    b = obs.shape[0]
    rest = obs[:, 1:4].reshape(b, 3, 64)[:, ext.other_slots].reshape(b, 3, 128)
    rest_h = ext.rest_in(rest).repeat_interleave(64, dim=1)
    scalars = torch.stack([n_lines / 4.0, new_combo / 8.0, delta], dim=-1)
    h = torch.relu(ext.after_in(after) + rest_h + ext.scalar_in(scalars))
    h = torch.relu(ext.hidden(h))
    return ext.head_out(h).squeeze(-1) + ext.alpha * delta


def _game_batch(n_games: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(1)
    obs, masks = [], []
    for seed in range(n_games):
        g = Game()
        g.reset(seed)
        while not g.state.game_over:
            obs.append(encode_observation(g.state))
            masks.append(action_mask(g.state))
            legal = np.flatnonzero(masks[-1])
            g.step(*decode_action(int(legal[rng.integers(legal.size)])))
    return torch.as_tensor(np.stack(obs)), torch.as_tensor(np.stack(masks))


def test_compact_logits_match_dense_scoring() -> None:
    """Scoring only legal actions gives the same masked distribution and gradients."""
    torch.manual_seed(0)
    ext = AfterstateExtractor(observation_space(), hidden=32, value_dim=8)
    torch.nn.init.normal_(ext.head_out.weight)
    obs, masks = _game_batch(6)
    obs[0, 1:4] = 0.0  # a state with an empty hand: no legal action at all
    masks[0] = False
    logits = ext.action_logits(obs)
    dense = _dense_logits(ext, obs)
    torch.testing.assert_close(logits[masks], dense[masks], rtol=1e-5, atol=1e-5)
    assert torch.all(logits[~masks] == ILLEGAL_LOGIT)

    def grads(fn: Callable[[], torch.Tensor]) -> list[torch.Tensor]:
        ext.zero_grad()
        # PPO-like loss on the masked distribution: log-prob of a legal action + entropy.
        m = masks[1:]
        logp = torch.log_softmax(torch.where(m, fn()[1:], torch.tensor(ILLEGAL_LOGIT)), dim=-1)
        chosen = logp.gather(1, m.float().argmax(1, keepdim=True)).sum()
        entropy = -(logp.exp() * logp)[m].sum()
        (-chosen - 0.01 * entropy).backward()
        return [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in ext.parameters()
        ]

    for g_compact, g_dense in zip(
        grads(lambda: ext.action_logits(obs)), grads(lambda: _dense_logits(ext, obs)), strict=True
    ):
        torch.testing.assert_close(g_compact, g_dense, rtol=1e-4, atol=1e-5)


def _policy(device: str) -> AfterstatePolicy:
    torch.manual_seed(0)
    policy = AfterstatePolicy(observation_space(), gymnasium.spaces.Discrete(192), lambda _: 1e-3)
    torch.nn.init.normal_(policy.features_extractor.head_out.weight)  # type: ignore[union-attr, arg-type]
    return policy.to(device)


def test_policy_forward_matches_base_class() -> None:
    """The overridden forward/predict_values equal the generic MaskableActorCriticPolicy path."""
    policy = _policy("cpu")
    obs, masks = _game_batch(2)
    with torch.no_grad():
        torch.manual_seed(1)
        actions, values, log_prob = policy(obs, action_masks=masks.numpy())
        torch.manual_seed(1)
        ref = MaskableActorCriticPolicy.forward(policy, obs, action_masks=masks.numpy())
        ref_values = MaskableActorCriticPolicy.predict_values(policy, obs)
    assert torch.equal(actions, ref[0])
    torch.testing.assert_close(values, ref[1])
    torch.testing.assert_close(log_prob, ref[2])
    torch.testing.assert_close(policy.predict_values(obs), ref_values)
    assert masks[torch.arange(len(actions)), actions].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_graph_rollout_matches_eager() -> None:
    policy = _policy("cuda")
    obs, masks = _game_batch(2)
    obs, masks = obs[:64].cuda(), masks[:64].cuda()
    with torch.no_grad():
        eager = policy._logits_and_values(obs, compact=True)
        for _ in range(2):  # capture, then replay
            graphed = policy.logits_and_values(obs)
            torch.testing.assert_close(graphed[0][masks], eager[0][masks], rtol=1e-5, atol=1e-5)
            assert torch.all(graphed[0][~masks] == ILLEGAL_LOGIT)
            torch.testing.assert_close(graphed[1], eager[1], rtol=1e-5, atol=1e-5)
        # the graph reads the live weights: an in-place update must show up on replay
        policy.features_extractor.alpha.add_(1.0)  # type: ignore[union-attr, operator]
        eager = policy._logits_and_values(obs, compact=True)
        torch.testing.assert_close(policy.logits_and_values(obs)[0][masks], eager[0][masks])
    assert len(policy._graphs) == 1


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
