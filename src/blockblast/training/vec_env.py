"""Seeded (vectorized) environment construction from config."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import gymnasium
import numpy as np
import numpy.typing as npt
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn

from blockblast.engine.scoring import ScoreConfig
from blockblast.env.action import N_ACTIONS
from blockblast.env.blockblast_env import BlockBlastEnv
from blockblast.env.reward import RewardConfig
from blockblast.training.env_worker import worker
from blockblast.utils.config import EnvConfig, RewardSection

MONITOR_INFO_KEYS = ("score", "round", "lines_total", "max_combo")

log = logging.getLogger(__name__)


def env_kwargs_from_config(env: EnvConfig, reward: RewardSection) -> dict[str, Any]:
    return {
        "reward_config": RewardConfig(
            mode=reward.mode,  # type: ignore[arg-type]
            score_scale=reward.score_scale,
            game_over_penalty=reward.game_over_penalty,
            sparse_scale=reward.sparse_scale,
            alive_bonus=reward.alive_bonus,
            board_weight=reward.board_weight,
        ),
        "score_config": ScoreConfig(env.points_per_cell, env.points_per_line),
        "max_steps": env.max_steps,
        "mid_start_prob": env.mid_start_prob,
        "mid_start_min_cells": env.mid_start_min_cells,
    }


def make_env_fn(
    env_kwargs: dict[str, Any], seed: int, rank: int
) -> Callable[[], gymnasium.Env[Any, Any]]:
    """Factory for in-process envs; seed ``seed + rank`` is applied on the first reset."""

    def _init() -> gymnasium.Env[Any, Any]:
        env = BlockBlastEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        return env

    return _init


class ParallelVecEnv(VecEnv):
    """Steps ``n_envs`` games across ``n_workers`` torch-free subprocesses.

    Each worker owns a contiguous chunk of envs, so one step is one message per worker
    (not per env), and the legal-action masks come back with the step: ``env_method(
    "action_masks")`` is answered from that cache without another round trip.
    """

    def __init__(self, env_kwargs: dict[str, Any], n_envs: int, n_workers: int) -> None:
        n_workers = max(1, min(n_workers, n_envs))
        sizes = [
            n_envs // n_workers + (1 if i < n_envs % n_workers else 0) for i in range(n_workers)
        ]
        ctx = mp.get_context("spawn")
        self.remotes = []
        self.processes = []
        self.slices: list[tuple[int, int]] = []
        start = 0
        for size in sizes:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(child, parent, env_kwargs, size), daemon=True)
            proc.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(proc)
            self.slices.append((start, start + size))
            start += size
        self.closed = False
        self._masks: npt.NDArray[np.bool_] = np.ones((n_envs, N_ACTIONS), dtype=np.bool_)
        probe = BlockBlastEnv(**env_kwargs)
        super().__init__(n_envs, probe.observation_space, probe.action_space)
        if any(self.workers_import_torch()):
            log.warning(
                "env workers imported torch (~0.5 GB each): the launching script imports the "
                "training stack at module level; move those imports inside main()"
            )

    def reset(self) -> npt.NDArray[np.float32]:
        for remote, (a, b) in zip(self.remotes, self.slices, strict=True):
            remote.send(("reset", self._seeds[a:b]))
        results = [remote.recv() for remote in self.remotes]
        self.reset_infos = [info for _, infos, _ in results for info in infos]
        self._masks = np.concatenate([m for _, _, m in results])
        self._reset_seeds()
        self._reset_options()
        return np.concatenate([o for o, _, _ in results])

    def step_async(self, actions: np.ndarray) -> None:
        for remote, (a, b) in zip(self.remotes, self.slices, strict=True):
            remote.send(("step", actions[a:b].tolist()))

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        obs = np.concatenate([r[0] for r in results])
        rews = np.concatenate([r[1] for r in results])
        dones = np.concatenate([r[2] for r in results])
        infos = [info for r in results for info in r[3]]
        self._masks = np.concatenate([r[4] for r in results])
        return obs, rews, dones, infos

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            with contextlib.suppress(BrokenPipeError, OSError):
                remote.send(("close", None))
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self.closed = True

    def _route(self, indices: VecEnvIndices) -> list[tuple[int, list[int]]]:
        """(worker, local indices) groups, in the order of ``indices``."""
        wanted = self._get_indices(indices)
        groups: list[tuple[int, list[int]]] = []
        for i in wanted:
            w = next(k for k, (a, b) in enumerate(self.slices) if a <= i < b)
            if groups and groups[-1][0] == w:
                groups[-1][1].append(i - self.slices[w][0])
            else:
                groups.append((w, [i - self.slices[w][0]]))
        return groups

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        out: list[Any] = []
        for w, local in self._route(indices):
            self.remotes[w].send(("get_attr", (attr_name, local)))
            out.extend(self.remotes[w].recv())
        return out

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        for w, local in self._route(indices):
            self.remotes[w].send(("set_attr", (attr_name, value, local)))
            self.remotes[w].recv()

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        if method_name == "action_masks":  # cached from the last step/reset
            return list(self._masks[self._get_indices(indices)])
        out: list[Any] = []
        for w, local in self._route(indices):
            self.remotes[w].send(("env_method", (method_name, method_args, method_kwargs, local)))
            out.extend(self.remotes[w].recv())
        return out

    def has_attr(self, attr_name: str) -> bool:
        return attr_name == "action_masks" or super().has_attr(attr_name)

    def env_is_wrapped(
        self,
        wrapper_class: type[gymnasium.Wrapper[Any, Any, Any, Any]],
        indices: VecEnvIndices = None,
    ) -> list[bool]:
        return [False] * len(list(self._get_indices(indices)))

    def workers_import_torch(self) -> list[bool]:
        """Diagnostics: True for any worker that ended up importing torch (should never)."""
        for remote in self.remotes:
            remote.send(("torch_loaded", None))
        return [remote.recv() for remote in self.remotes]

    def _get_indices(self, indices: VecEnvIndices) -> Sequence[int]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return list(indices) if isinstance(indices, Iterable) else [indices]


def make_vec_env(
    env_kwargs: dict[str, Any],
    n_envs: int,
    seed: int,
    use_subproc: bool = True,
    n_workers: int = 8,
) -> VecEnv:
    """Seeded vec env (env i gets ``seed + i``), wrapped in ``VecMonitor``.

    ``use_subproc`` runs the games in ``n_workers`` torch-free processes (``ParallelVecEnv``);
    otherwise all envs step in this process (``DummyVecEnv``).
    """
    vec: VecEnv
    if use_subproc and n_envs > 1 and n_workers > 1:
        vec = ParallelVecEnv(env_kwargs, n_envs, n_workers)
    else:
        vec = DummyVecEnv([make_env_fn(env_kwargs, seed, i) for i in range(n_envs)])
    vec.seed(seed)
    return VecMonitor(vec, info_keywords=MONITOR_INFO_KEYS)
