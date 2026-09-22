"""MaskablePPO construction/loading and its ``Agent`` wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecEnv

from blockblast.agents.networks import AfterstatePolicy, BlockBlastCNN
from blockblast.engine.game import GameState

# Distribution argument validation re-checks every logits tensor on each action sample
# and forces a GPU sync; the action mask already guarantees valid inputs. Profiling
# showed it as the largest single cost of a PPO iteration.
torch.distributions.Distribution.set_default_validate_args(False)


def linear_schedule(initial: float) -> Callable[[float], float]:
    """SB3 schedule: ``progress_remaining`` goes 1 → 0."""

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial

    return schedule


def build_model(
    env: VecEnv,
    ppo_kwargs: Mapping[str, Any],
    features_dim: int = 256,
    tensorboard_log: Path | None = None,
    seed: int = 0,
    device: str = "auto",
    policy: str = "afterstate",
    prior: str = "score",
) -> MaskablePPO:
    """Create a MaskablePPO.

    ``policy="afterstate"`` (default) uses ``AfterstatePolicy``: logits from the exact
    board each action produces; ``net_arch["vf"]`` sizes the critic MLP and ``prior``
    ("score" | "board") picks its greedy starting bias.
    ``policy="cnn"`` uses the plain CNN extractor with ``net_arch`` pi/vf MLPs (run ppo_v1).
    ``ppo_kwargs`` holds MaskablePPO arguments; ``learning_rate`` decays linearly to 0.
    """
    kwargs = dict(ppo_kwargs)
    net_arch = dict(kwargs.pop("net_arch", {"pi": [256, 256], "vf": [256, 256]}))
    lr = float(kwargs.pop("learning_rate", 3e-4))
    policy_cls: Any
    if policy == "afterstate":
        policy_cls = AfterstatePolicy
        policy_kwargs: dict[str, Any] = {
            "features_extractor_kwargs": {"value_dim": features_dim, "prior": prior},
            "vf_arch": list(net_arch.get("vf", [256])),
        }
    elif policy == "cnn":
        policy_cls = "CnnPolicy"
        policy_kwargs = {
            "features_extractor_class": BlockBlastCNN,
            "features_extractor_kwargs": {"features_dim": features_dim},
            "net_arch": net_arch,
            "normalize_images": False,
        }
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return MaskablePPO(
        policy_cls,
        env,
        learning_rate=linear_schedule(lr),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        seed=seed,
        device=device,
        verbose=0,
        **kwargs,
    )


class PPOAgent:
    def __init__(self, model: MaskablePPO, deterministic: bool = True, name: str = "ppo") -> None:
        self.model = model
        self.deterministic = deterministic
        self.name = name

    @classmethod
    def load(cls, path: Path, device: str = "auto", deterministic: bool = True) -> PPOAgent:
        return cls(MaskablePPO.load(str(path), device=device), deterministic, name="ppo")

    def reset(self, seed: int | None = None) -> None:
        """Policy is stateless across steps; nothing to reset."""

    def act(
        self,
        obs: npt.NDArray[np.float32],
        action_mask: npt.NDArray[np.bool_],
        state: GameState,
    ) -> int:
        action, _ = self.model.predict(
            obs, action_masks=action_mask, deterministic=self.deterministic
        )
        return int(action)
