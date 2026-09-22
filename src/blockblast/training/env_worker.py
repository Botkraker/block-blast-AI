"""Subprocess worker that steps a chunk of envs. Deliberately imports no torch/SB3.

Spawned children (Windows) re-import only this module and ``blockblast.env``, so each
worker costs ~60 MB instead of ~0.5 GB for a stock SB3 ``SubprocVecEnv`` worker.
"""

from __future__ import annotations

import sys
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from blockblast.env.blockblast_env import BlockBlastEnv


def worker(
    remote: Connection, parent_remote: Connection, env_kwargs: dict[str, Any], n_envs: int
) -> None:
    parent_remote.close()
    envs = [BlockBlastEnv(**env_kwargs) for _ in range(n_envs)]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, rews, dones, infos, masks = [], [], [], [], []
                for env, action in zip(envs, data, strict=True):
                    o, r, terminated, truncated, info = env.step(int(action))
                    done = terminated or truncated
                    if done:
                        info["terminal_observation"] = o
                        info["TimeLimit.truncated"] = truncated and not terminated
                        o, _ = env.reset()
                    obs.append(o)
                    rews.append(r)
                    dones.append(done)
                    infos.append(info)
                    masks.append(env.action_masks())
                remote.send(
                    (
                        np.stack(obs),
                        np.array(rews, np.float32),
                        np.array(dones),
                        infos,
                        np.stack(masks),
                    )
                )
            elif cmd == "reset":
                obs, infos = [], []
                for env, seed in zip(envs, data, strict=True):
                    o, info = env.reset(seed=seed)
                    obs.append(o)
                    infos.append(info)
                remote.send((np.stack(obs), infos, np.stack([e.action_masks() for e in envs])))
            elif cmd == "get_attr":
                remote.send([getattr(envs[i], data[0]) for i in data[1]])
            elif cmd == "set_attr":
                name, value, idx = data
                for i in idx:
                    setattr(envs[i], name, value)
                remote.send(None)
            elif cmd == "env_method":
                name, args, kwargs, idx = data
                remote.send([getattr(envs[i], name)(*args, **kwargs) for i in idx])
            elif cmd == "torch_loaded":
                remote.send("torch" in sys.modules)
            elif cmd == "close":
                break
            else:
                raise ValueError(f"unknown command {cmd!r}")
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        pass
    finally:
        remote.close()
