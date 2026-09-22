"""Agents: baselines and the MaskablePPO wrapper.

``PPOAgent`` is imported lazily (from ``blockblast.agents.ppo_agent``) so baselines work
without loading torch.
"""

from blockblast.agents.base import Agent
from blockblast.agents.greedy_agent import GreedyAgent
from blockblast.agents.random_agent import RandomAgent

__all__ = ["Agent", "GreedyAgent", "RandomAgent"]
