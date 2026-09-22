"""Gymnasium env for Block Blast. Registers ``BlockBlast-v0``."""

import gymnasium

from blockblast.env.action import N_ACTIONS, action_mask, decode_action, encode_action
from blockblast.env.blockblast_env import BlockBlastEnv
from blockblast.env.observation import OBS_SHAPE, encode_observation
from blockblast.env.reward import RewardConfig, compute_reward

if "BlockBlast-v0" not in gymnasium.registry:
    gymnasium.register(id="BlockBlast-v0", entry_point="blockblast.env:BlockBlastEnv")

__all__ = [
    "N_ACTIONS",
    "OBS_SHAPE",
    "BlockBlastEnv",
    "RewardConfig",
    "action_mask",
    "compute_reward",
    "decode_action",
    "encode_action",
    "encode_observation",
]
