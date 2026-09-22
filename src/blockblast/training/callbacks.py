"""Training callbacks: game metrics to TensorBoard, validation eval, checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from blockblast.training.vec_env import make_vec_env
from blockblast.utils.config import AppConfig
from blockblast.utils.seeding import VALIDATION_SEED_START


class GameMetricsCallback(BaseCallback):
    """Logs raw game metrics of finished episodes: game/score, rounds, lines, max_combo."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones, strict=False):
            if done:
                self.logger.record_mean("game/score", info["score"])
                self.logger.record_mean("game/rounds", info["round"])
                self.logger.record_mean("game/lines", info["lines_total"])
                self.logger.record_mean("game/max_combo", info["max_combo"])
        return True


def make_callbacks(
    cfg: AppConfig, env_kwargs: dict[str, Any], run_dir: Path, model_dir: Path
) -> CallbackList:
    n_envs = cfg.train.n_envs
    eval_env = make_vec_env(env_kwargs, 1, VALIDATION_SEED_START, use_subproc=False)
    return CallbackList(
        [
            GameMetricsCallback(),
            MaskableEvalCallback(
                eval_env,
                n_eval_episodes=cfg.train.n_eval_episodes,
                eval_freq=max(cfg.train.eval_freq // n_envs, 1),
                best_model_save_path=str(model_dir),
                log_path=str(run_dir),
                deterministic=True,
            ),
            CheckpointCallback(
                save_freq=max(cfg.train.checkpoint_freq // n_envs, 1),
                save_path=str(model_dir),
                name_prefix="ckpt",
            ),
        ]
    )
