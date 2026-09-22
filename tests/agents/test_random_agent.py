from __future__ import annotations

import numpy as np
import pytest

from blockblast.agents import RandomAgent
from blockblast.env import BlockBlastEnv
from blockblast.evaluation import run_episode
from tests.conftest import make_state


def test_only_legal_actions_and_seeded() -> None:
    env = BlockBlastEnv()
    a1, _ = run_episode(RandomAgent(), env, seed=1_000_001)
    a2, _ = run_episode(RandomAgent(), env, seed=1_000_001)
    assert a1 == a2  # reset(seed) reseeds the agent too; env raises on illegal actions


def test_picks_from_mask() -> None:
    agent = RandomAgent(0)
    mask = np.zeros(192, dtype=np.bool_)
    mask[[5, 70]] = True
    picks = {agent.act(np.zeros((5, 8, 8), np.float32), mask, make_state()) for _ in range(50)}
    assert picks == {5, 70}


def test_no_legal_action_raises() -> None:
    with pytest.raises(ValueError):
        RandomAgent(0).act(np.zeros((5, 8, 8), np.float32), np.zeros(192, bool), make_state())
