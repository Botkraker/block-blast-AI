"""Training callbacks: game metrics to TensorBoard, validation eval, checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from blockblast.training.vec_env import make_vec_env
from blockblast.utils.config import AppConfig
from blockblast.utils.seeding import VALIDATION_SEED_START

log = logging.getLogger(__name__)


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


class CurriculumCallback(BaseCallback):
    """Raises the hard-piece weight linearly from ``start`` (step 0) to 1.0 at ``steps``.

    The schedule uses the global step count, so a resumed run continues on it. It is sent to
    the envs before each rollout and applies to games that start afterwards.
    """

    def __init__(self, start: float, steps: int) -> None:
        super().__init__()
        self.start = start
        self.steps = steps
        self.current = -1.0

    def weight(self, num_timesteps: int) -> float:
        return min(1.0, self.start + (1.0 - self.start) * num_timesteps / self.steps)

    def _on_rollout_start(self) -> None:
        w = self.weight(self.num_timesteps)
        if abs(w - self.current) > 1e-3:
            self.training_env.set_attr("hard_piece_weight", w)
            self.current = w
        self.logger.record("curriculum/hard_piece_weight", w)

    def _on_step(self) -> bool:
        return True


class PauseFileCallback(BaseCallback):
    """Stops training cleanly when ``pause_file`` appears (checked every ``check_every`` steps).

    Lets a background run be paused from outside, which Ctrl+C cannot do on Windows.
    """

    def __init__(self, pause_file: Path, check_every: int = 256) -> None:
        super().__init__()
        self.pause_file = pause_file
        self.check_every = check_every
        self.paused = False

    def _on_step(self) -> bool:
        if self.n_calls % self.check_every == 0 and self.pause_file.exists():
            log.info("pause requested (%s)", self.pause_file)
            self.paused = True
            return False
        return True


def restore_eval_history(callback: MaskableEvalCallback, run_dir: Path) -> None:
    """On resume, keep the validation history so best_model.zip is only replaced by a better one."""
    path = run_dir / "evaluations.npz"
    if not path.exists():
        return
    data = np.load(path)
    callback.evaluations_timesteps = [int(x) for x in data["timesteps"]]
    callback.evaluations_results = [list(r) for r in data["results"]]
    callback.evaluations_length = [list(r) for r in data["ep_lengths"]]
    if callback.evaluations_results:
        callback.best_mean_reward = float(max(np.mean(r) for r in data["results"]))
    log.info("restored %d validation evaluations", len(callback.evaluations_timesteps))


def make_callbacks(
    cfg: AppConfig,
    env_kwargs: dict[str, Any],
    run_dir: Path,
    model_dir: Path,
    resumed: bool = False,
) -> tuple[CallbackList, PauseFileCallback]:
    n_envs = cfg.train.n_envs
    # validation plays the real game: no mid-game starts (and the curriculum never reaches it)
    eval_kwargs = {**env_kwargs, "mid_start_prob": 0.0}
    eval_env = make_vec_env(eval_kwargs, 1, VALIDATION_SEED_START, use_subproc=False)
    eval_cb = MaskableEvalCallback(
        eval_env,
        n_eval_episodes=cfg.train.n_eval_episodes,
        eval_freq=max(cfg.train.eval_freq // n_envs, 1),
        best_model_save_path=str(model_dir),
        log_path=str(run_dir),
        deterministic=True,
    )
    if resumed:
        restore_eval_history(eval_cb, run_dir)
    pause_cb = PauseFileCallback(model_dir / "PAUSE")
    curriculum: list[BaseCallback] = []
    if cfg.train.curriculum_steps > 0:
        curriculum.append(
            CurriculumCallback(cfg.train.curriculum_hard_start, cfg.train.curriculum_steps)
        )
    callbacks = CallbackList(
        [
            *curriculum,
            GameMetricsCallback(),
            eval_cb,
            CheckpointCallback(
                save_freq=max(cfg.train.checkpoint_freq // n_envs, 1),
                save_path=str(model_dir),
                name_prefix="ckpt",
            ),
            pause_cb,
        ]
    )
    return callbacks, pause_cb
