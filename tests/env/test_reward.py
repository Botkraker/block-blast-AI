from __future__ import annotations

import pytest

from blockblast.engine.game import MoveResult
from blockblast.env.reward import RewardConfig, compute_reward
from tests.conftest import make_state


def result(delta: int, score: int = 100) -> MoveResult:
    return MoveResult(make_state(score=score), delta, 4, 1, (0,), False)


def test_default_score_mode() -> None:
    cfg = RewardConfig()
    assert compute_reward(result(14), False, cfg) == pytest.approx(1.4)
    assert compute_reward(result(14), True, cfg) == pytest.approx(1.4 - 5.0)


def test_survival_mode() -> None:
    cfg = RewardConfig(mode="survival")
    assert compute_reward(result(99), False, cfg) == 1.0
    assert compute_reward(result(99), True, cfg) == -4.0


def test_sparse_mode() -> None:
    cfg = RewardConfig(mode="sparse")
    assert compute_reward(result(10, score=250), False, cfg) == 0.0
    assert compute_reward(result(10, score=250), True, cfg) == pytest.approx(2.5)


def test_unknown_mode() -> None:
    with pytest.raises(ValueError):
        compute_reward(result(1), False, RewardConfig(mode="nope"))  # type: ignore[arg-type]


def test_truncation_not_penalized_in_env() -> None:
    import numpy as np

    from blockblast.env import BlockBlastEnv

    env = BlockBlastEnv(max_steps=1)
    env.reset(seed=0)
    a = int(np.flatnonzero(env.action_masks())[0])
    _, reward, terminated, truncated, info = env.step(a)
    assert truncated and not terminated
    assert reward == pytest.approx(info["score_delta"] / 10)
