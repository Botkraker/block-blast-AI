"""Seeded (vectorized) environment construction from config."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from blockblast.engine.scoring import ScoreConfig
from blockblast.env.blockblast_env import BlockBlastEnv
from blockblast.env.reward import RewardConfig
from blockblast.utils.config import EnvConfig, RewardSection

MONITOR_INFO_KEYS = ("score", "round", "lines_total", "max_combo")


def env_kwargs_from_config(env: EnvConfig, reward: RewardSection) -> dict[str, Any]:
    return {
        "reward_config": RewardConfig(
            mode=reward.mode,  # type: ignore[arg-type]
            score_scale=reward.score_scale,
            game_over_penalty=reward.game_over_penalty,
            sparse_scale=reward.sparse_scale,
        ),
        "score_config": ScoreConfig(env.points_per_cell, env.points_per_line),
        "max_steps": env.max_steps,
    }


def make_env_fn(
    env_kwargs: dict[str, Any], seed: int, rank: int
) -> Callable[[], gymnasium.Env[Any, Any]]:
    """Factory for subprocesses; seed ``seed + rank`` is applied on the first reset."""

    def _init() -> gymnasium.Env[Any, Any]:
        env = BlockBlastEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vec_env(
    env_kwargs: dict[str, Any], n_envs: int, seed: int, use_subproc: bool = True
) -> VecEnv:
    fns = [make_env_fn(env_kwargs, seed, i) for i in range(n_envs)]
    vec: VecEnv = SubprocVecEnv(fns) if use_subproc and n_envs > 1 else DummyVecEnv(fns)
    vec.seed(seed)  # env i gets seed + i on the next reset
    return VecMonitor(vec, info_keywords=MONITOR_INFO_KEYS)
