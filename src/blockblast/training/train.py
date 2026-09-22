"""End-to-end MaskablePPO training from an ``AppConfig``, with pause and resume."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from sb3_contrib import MaskablePPO

from blockblast.agents.ppo_agent import build_model
from blockblast.training.callbacks import make_callbacks
from blockblast.training.vec_env import env_kwargs_from_config, make_vec_env
from blockblast.utils.config import AppConfig
from blockblast.utils.logger import make_run_dir
from blockblast.utils.seeding import TRAIN_SEED_MAX, set_global_seeds

log = logging.getLogger(__name__)

_CKPT = re.compile(r"ckpt_(\d+)_steps\.zip$")


def latest_checkpoint(model_dir: Path) -> Path | None:
    """Checkpoint with the most steps in ``model_dir`` (``ckpt_<steps>_steps.zip``)."""
    found = []
    for path in model_dir.glob("ckpt_*_steps.zip"):
        match = _CKPT.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return max(found)[1] if found else None


def _resume_path(resume: str | None, model_dir: Path) -> Path | None:
    if not resume:
        return None
    if resume == "auto":
        path = latest_checkpoint(model_dir)
        if path is None:
            raise FileNotFoundError(f"train.resume=auto but no ckpt_*_steps.zip in {model_dir}")
        return path
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def train(cfg: AppConfig) -> Path:
    """Train (or resume) and return the saved model path.

    Returns ``models/<run_name>/final.zip`` when ``train.total_timesteps`` is reached. If the
    run is paused (a ``PAUSE`` file in the model directory, or Ctrl+C) it saves and returns
    ``ckpt_<steps>_steps.zip``; continue later with ``train.resume=auto``.
    """
    if not 0 <= cfg.seed < TRAIN_SEED_MAX - cfg.train.n_envs:
        raise ValueError(f"training seed must be in [0, {TRAIN_SEED_MAX - cfg.train.n_envs})")
    set_global_seeds(cfg.seed, cfg.train.deterministic_torch)
    run_dir = make_run_dir(Path(cfg.paths.logs), cfg.run_name)
    model_dir = make_run_dir(Path(cfg.paths.models), cfg.run_name)
    resume = _resume_path(cfg.train.resume, model_dir)
    (model_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")
    pause_file = model_dir / "PAUSE"
    pause_file.unlink(missing_ok=True)  # a stale request must not stop the new session

    env_kwargs = env_kwargs_from_config(cfg.env, cfg.reward)
    vec_env = make_vec_env(
        env_kwargs, cfg.train.n_envs, cfg.seed, cfg.train.use_subproc, cfg.train.n_workers
    )
    if resume is not None:
        model = MaskablePPO.load(
            str(resume), env=vec_env, device=cfg.train.device, tensorboard_log=str(run_dir)
        )
        log.info("resumed %s from %s at %d steps", cfg.run_name, resume, model.num_timesteps)
    else:
        model = build_model(
            vec_env,
            cfg.agent.ppo,
            features_dim=cfg.agent.features_dim,
            tensorboard_log=run_dir,
            seed=cfg.seed,
            device=cfg.train.device,
            policy=cfg.agent.policy,
        )
    remaining = cfg.train.total_timesteps - model.num_timesteps
    log.info("training %s on %s: %d steps to go", cfg.run_name, model.device, max(remaining, 0))
    callbacks, pause_cb = make_callbacks(
        cfg, env_kwargs, run_dir, model_dir, resumed=resume is not None
    )
    interrupted = False
    try:
        if remaining > 0:
            # reset_num_timesteps=False keeps the step count, the linear LR schedule
            # (progress = steps / total) and the TensorBoard curve continuous on resume.
            model.learn(
                total_timesteps=remaining,
                callback=callbacks,
                tb_log_name="tb",
                reset_num_timesteps=resume is None,
                progress_bar=False,
            )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        vec_env.close()
    if interrupted or pause_cb.paused:
        ckpt = model_dir / f"ckpt_{model.num_timesteps}_steps.zip"
        model.save(str(ckpt))
        pause_file.unlink(missing_ok=True)
        log.info(
            "paused at %d steps -> %s. Resume: scripts/train.py run_name=%s train.resume=auto",
            model.num_timesteps,
            ckpt,
            cfg.run_name,
        )
        return ckpt
    final = model_dir / "final.zip"
    model.save(str(final))
    log.info("saved %s", final)
    return final
