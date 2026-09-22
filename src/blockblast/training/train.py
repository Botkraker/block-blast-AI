"""End-to-end MaskablePPO training from an ``AppConfig``."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from blockblast.agents.ppo_agent import build_model
from blockblast.training.callbacks import make_callbacks
from blockblast.training.vec_env import env_kwargs_from_config, make_vec_env
from blockblast.utils.config import AppConfig
from blockblast.utils.logger import make_run_dir
from blockblast.utils.seeding import TRAIN_SEED_MAX, set_global_seeds

log = logging.getLogger(__name__)


def train(cfg: AppConfig) -> Path:
    """Train, checkpoint and return the path of ``models/<run_name>/final.zip``."""
    if not 0 <= cfg.seed < TRAIN_SEED_MAX - cfg.train.n_envs:
        raise ValueError(f"training seed must be in [0, {TRAIN_SEED_MAX - cfg.train.n_envs})")
    set_global_seeds(cfg.seed, cfg.train.deterministic_torch)
    run_dir = make_run_dir(Path(cfg.paths.logs), cfg.run_name)
    model_dir = make_run_dir(Path(cfg.paths.models), cfg.run_name)
    (model_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")

    env_kwargs = env_kwargs_from_config(cfg.env, cfg.reward)
    vec_env = make_vec_env(env_kwargs, cfg.train.n_envs, cfg.seed, cfg.train.use_subproc)
    model = build_model(
        vec_env,
        cfg.agent.ppo,
        features_dim=cfg.agent.features_dim,
        tensorboard_log=run_dir,
        seed=cfg.seed,
        device=cfg.train.device,
        policy=cfg.agent.policy,
    )
    log.info(
        "training %s on %s for %d steps", cfg.run_name, model.device, cfg.train.total_timesteps
    )
    callbacks = make_callbacks(cfg, env_kwargs, run_dir, model_dir)
    try:
        model.learn(
            total_timesteps=cfg.train.total_timesteps,
            callback=callbacks,
            tb_log_name="tb",
            progress_bar=False,
        )
    finally:
        final = model_dir / "final.zip"
        model.save(str(final))
        vec_env.close()
    log.info("saved %s", final)
    return final
