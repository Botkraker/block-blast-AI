"""Structured config schemas mirroring ``configs/*.yaml`` (primitives only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


@dataclass
class EnvConfig:
    max_steps: int = 10_000
    points_per_cell: int = 1
    points_per_line: int = 10


@dataclass
class RewardSection:
    mode: str = "score"
    score_scale: float = 10.0
    game_over_penalty: float = 5.0
    sparse_scale: float = 100.0
    alive_bonus: float = 1.0
    board_weight: float = 1.0


@dataclass
class AgentConfig:
    name: str = "maskable_ppo"
    seed: int = 0
    policy: str = "afterstate"  # afterstate | cnn
    features_dim: int = 256
    ppo: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    total_timesteps: int = 50_000_000
    n_envs: int = 64
    use_subproc: bool = True
    n_workers: int = 8
    checkpoint_freq: int = 2_000_000
    eval_freq: int = 1_000_000
    n_eval_episodes: int = 50
    device: str = "auto"
    deterministic_torch: bool = False
    resume: str | None = None  # "auto" = latest models/<run_name>/ckpt_*, or a .zip path


@dataclass
class EvalConfig:
    seed_start: int = 1_000_000
    n_episodes: int = 1000
    checkpoint: str | None = None
    record_n: int = 5
    deterministic: bool = True
    tag: str = "default"
    compare_to: str | None = None


@dataclass
class PathsConfig:
    models: str = "models"
    logs: str = "logs"
    data: str = "data"


@dataclass
class AppConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardSection = field(default_factory=RewardSection)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0
    run_name: str = "run"
    paths: PathsConfig = field(default_factory=PathsConfig)


def to_app_config(cfg: DictConfig) -> AppConfig:
    """Validate a Hydra config against the schema (unknown keys raise)."""
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    plain.pop("hydra", None)
    merged = OmegaConf.merge(OmegaConf.structured(AppConfig), plain)
    obj = OmegaConf.to_object(merged)
    assert isinstance(obj, AppConfig)
    return obj


def load_config(overrides: list[str] | None = None) -> AppConfig:
    """Compose ``configs/config.yaml`` outside a Hydra app (notebooks, tests)."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides or [])
    return to_app_config(cfg)
