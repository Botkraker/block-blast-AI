"""Episode records ``(seed, actions)`` and deterministic replay."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from blockblast.engine.game import Game, GameState
from blockblast.engine.pieces import CATALOGUE_VERSION
from blockblast.engine.scoring import ScoreConfig
from blockblast.visualization.render import render_rgb


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    seed: int
    actions: tuple[int, ...]
    final_score: int
    agent_name: str
    catalogue_version: int
    score_config: dict[str, int]


def save_record(record: EpisodeRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(record)
    data["actions"] = list(record.actions)
    path.write_text(json.dumps(data), encoding="utf-8")


def load_record(path: Path) -> EpisodeRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["actions"] = tuple(data["actions"])
    return EpisodeRecord(**data)


def replay_states(record: EpisodeRecord) -> Iterator[GameState]:
    """Yield the initial state, then the state after every recorded action."""
    if record.catalogue_version != CATALOGUE_VERSION:
        raise ValueError(
            f"record uses catalogue v{record.catalogue_version}, engine is v{CATALOGUE_VERSION}"
        )
    game = Game(ScoreConfig(**record.score_config))
    yield game.reset(np.random.default_rng(record.seed))
    for a in record.actions:
        yield game.step(a // 64, (a // 8) % 8, a % 8).state


def export_gif(record: EpisodeRecord, path: Path, fps: int = 4, cell_px: int = 32) -> Path:
    import imageio.v3 as iio

    frames = np.stack([render_rgb(s, cell_px) for s in replay_states(record)])
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, duration=int(1000 / fps), loop=0)
    return path
