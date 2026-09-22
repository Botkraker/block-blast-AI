"""Run agents on fixed seeds and collect per-episode game statistics."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from blockblast.agents.base import Agent
from blockblast.engine.pieces import CATALOGUE_VERSION
from blockblast.env.blockblast_env import BlockBlastEnv
from blockblast.evaluation.metrics import bootstrap_ci, summarize
from blockblast.visualization.replay import EpisodeRecord, save_record


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    seed: int
    score: int
    rounds: int
    moves: int
    lines_cleared: int
    max_combo: int
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class EvalResult:
    agent_name: str
    episodes: tuple[EpisodeStats, ...]

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {"n_episodes": float(len(self.episodes))}
        for metric in ("score", "rounds", "moves", "lines_cleared", "max_combo"):
            values = [float(getattr(e, metric)) for e in self.episodes]
            for k, v in summarize(values).items():
                out[f"{metric}_{k}"] = v
        lo, hi = bootstrap_ci([float(e.score) for e in self.episodes])
        out["score_ci_lo"], out["score_ci_hi"] = lo, hi
        out["truncated_frac"] = sum(e.truncated for e in self.episodes) / len(self.episodes)
        return out

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "agent_name": self.agent_name,
            "summary": self.summary(),
            "episodes": [asdict(e) for e in self.episodes],
        }
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> EvalResult:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["agent_name"], tuple(EpisodeStats(**e) for e in data["episodes"]))


def run_episode(
    agent: Agent, env: BlockBlastEnv, seed: int, record: bool = False
) -> tuple[EpisodeStats, EpisodeRecord | None]:
    obs, info = env.reset(seed=seed)
    agent.reset(seed)
    actions: list[int] = []
    terminated = truncated = False
    while not (terminated or truncated):
        a = agent.act(obs, env.action_masks(), env.game.state)
        actions.append(a)
        obs, _, terminated, truncated, info = env.step(a)
    stats = EpisodeStats(
        seed=seed,
        score=int(info["score"]),
        rounds=int(info["round"]),
        moves=int(info["moves"]),
        lines_cleared=int(info["lines_total"]),
        max_combo=int(info["max_combo"]),
        truncated=bool(truncated),
    )
    rec = None
    if record:
        cfg = env.game.score_config
        rec = EpisodeRecord(
            seed=seed,
            actions=tuple(actions),
            final_score=stats.score,
            agent_name=agent.name,
            catalogue_version=CATALOGUE_VERSION,
            score_config={
                "points_per_cell": cfg.points_per_cell,
                "points_per_line": cfg.points_per_line,
            },
        )
    return stats, rec


def evaluate_agent(
    agent: Agent,
    seeds: Sequence[int],
    env_kwargs: dict[str, Any],
    record_dir: Path | None = None,
    record_n: int = 0,
) -> EvalResult:
    """Evaluate on ``seeds``; the first ``record_n`` episodes are saved to ``record_dir``."""
    env = BlockBlastEnv(**env_kwargs)
    episodes = []
    for i, seed in enumerate(seeds):
        record = record_dir is not None and i < record_n
        stats, rec = run_episode(agent, env, seed, record=record)
        episodes.append(stats)
        if rec is not None and record_dir is not None:
            save_record(rec, record_dir / f"{agent.name}_seed{seed}.json")
    return EvalResult(agent.name, tuple(episodes))
