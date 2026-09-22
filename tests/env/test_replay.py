"""PRD S11: a recorded episode replays to exactly the same states and score."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blockblast.agents import GreedyAgent, RandomAgent
from blockblast.env import BlockBlastEnv
from blockblast.evaluation import run_episode
from blockblast.visualization import (
    EpisodeRecord,
    export_gif,
    load_record,
    render_ansi,
    replay_states,
    save_record,
)


@pytest.mark.parametrize("agent", [RandomAgent(0), GreedyAgent()], ids=lambda a: a.name)
def test_record_replays_exactly(agent, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    env = BlockBlastEnv()
    stats, rec = run_episode(agent, env, seed=1_000_123, record=True)
    assert rec is not None and rec.final_score == stats.score
    path = tmp_path / "r.json"
    save_record(rec, path)
    loaded = load_record(path)
    assert loaded == rec
    states = list(replay_states(loaded))
    assert len(states) == len(rec.actions) + 1
    assert states[-1].score == stats.score and states[-1].game_over
    assert states[-1].round_index == stats.rounds


def test_replay_matches_env_state_by_state() -> None:
    env = BlockBlastEnv()
    env.reset(seed=77)
    rng = np.random.default_rng(1)
    env_states = [env.game.state]
    actions = []
    for _ in range(40):
        legal = np.flatnonzero(env.action_masks())
        if legal.size == 0:
            break
        a = int(legal[rng.integers(legal.size)])
        actions.append(a)
        env.step(a)
        env_states.append(env.game.state)
    rec = EpisodeRecord(
        77,
        tuple(actions),
        env_states[-1].score,
        "t",
        1,
        {"points_per_cell": 1, "points_per_line": 10},
    )
    assert list(replay_states(rec)) == env_states


def test_catalogue_version_mismatch() -> None:
    rec = EpisodeRecord(0, (), 0, "t", 999, {"points_per_cell": 1, "points_per_line": 10})
    with pytest.raises(ValueError):
        list(replay_states(rec))


def test_gif_and_ansi(tmp_path: Path) -> None:
    _, rec = run_episode(RandomAgent(0), BlockBlastEnv(), seed=1_000_000, record=True)
    assert rec is not None
    out = export_gif(rec, tmp_path / "g.gif", fps=10, cell_px=8)
    assert out.exists() and out.stat().st_size > 0
    last = list(replay_states(rec))[-1]
    assert "GAME OVER" in render_ansi(last)
