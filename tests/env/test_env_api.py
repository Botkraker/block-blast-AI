from __future__ import annotations

import warnings

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from blockblast.engine.game import InvalidMoveError
from blockblast.env import BlockBlastEnv, action_mask

INFO_KEYS = {
    "score",
    "score_delta",
    "round",
    "moves",
    "n_lines",
    "combo_streak",
    "lines_total",
    "max_combo",
}


class _LegalDiscrete(gymnasium.spaces.Discrete):
    """check_env samples random actions; route them through the mask (API check only)."""

    def __init__(self, env: BlockBlastEnv) -> None:
        super().__init__(192)
        self._env = env

    def sample(self, mask=None, probability=None):  # type: ignore[no-untyped-def,override]
        return super().sample(mask=self._env.action_masks().astype(np.int8))


def test_check_env_passes_without_warnings() -> None:
    env = gymnasium.make("BlockBlast-v0").unwrapped  # has a spec for the render-mode check
    assert isinstance(env, BlockBlastEnv)
    env.action_space = _LegalDiscrete(env)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env, skip_render_check=False)


def test_registered_id() -> None:
    env = gymnasium.make("BlockBlast-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5, 8, 8)


def test_reset_and_step_contract() -> None:
    env = BlockBlastEnv()
    obs, info = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    assert set(info) == INFO_KEYS
    assert info["score"] == 0 and info["round"] == 1
    mask = env.action_masks()
    assert mask.shape == (192,) and mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, action_mask(env.game.state))
    a = int(np.flatnonzero(mask)[0])
    obs, reward, terminated, truncated, info = env.step(a)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and not terminated and not truncated
    assert info["moves"] == 1 and set(info) == INFO_KEYS


def test_action_masks_returns_copy() -> None:
    env = BlockBlastEnv()
    env.reset(seed=0)
    m = env.action_masks()
    m[:] = False
    assert env.action_masks().any()


def test_illegal_action_raises() -> None:
    env = BlockBlastEnv()
    env.reset(seed=0)
    illegal = np.flatnonzero(~env.action_masks())
    assert illegal.size > 0
    with pytest.raises(InvalidMoveError):
        env.step(int(illegal[0]))
    with pytest.raises(ValueError):
        env.step(192)


def test_episode_terminates_and_info_is_consistent() -> None:
    env = BlockBlastEnv()
    env.reset(seed=5)
    rng = np.random.default_rng(0)
    total, lines, max_combo, terminated = 0, 0, 0, False
    while not terminated:
        legal = np.flatnonzero(env.action_masks())
        _, _, terminated, truncated, info = env.step(int(legal[rng.integers(legal.size)]))
        assert not truncated
        total += info["score_delta"]
        lines += info["n_lines"]
        max_combo = max(max_combo, info["combo_streak"])
    assert info["score"] == total and info["lines_total"] == lines
    assert info["max_combo"] == max_combo
    assert not env.action_masks().any()


def test_truncation() -> None:
    env = BlockBlastEnv(max_steps=3)
    env.reset(seed=0)
    flags = []
    for _ in range(3):
        a = int(np.flatnonzero(env.action_masks())[0])
        _, _, terminated, truncated, _ = env.step(a)
        flags.append((terminated, truncated))
    assert flags == [(False, False), (False, False), (False, True)]


def test_seeded_reset_reproducible_and_unseeded_continues() -> None:
    env = BlockBlastEnv()
    env.reset(seed=9)
    h1 = env.game.state.hand
    env.reset()
    h2 = env.game.state.hand
    env.reset(seed=9)
    assert env.game.state.hand == h1
    env.reset()
    assert env.game.state.hand == h2


def test_render_modes() -> None:
    env = BlockBlastEnv(render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    assert isinstance(text, str) and "score=0" in text
    env = BlockBlastEnv(render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.shape[2] == 3
    env = BlockBlastEnv()
    env.reset(seed=0)
    assert env.render() is None
    with pytest.raises(ValueError):
        BlockBlastEnv(render_mode="human")
