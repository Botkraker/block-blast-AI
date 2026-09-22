"""Pause (PAUSE file / Ctrl+C) and resume (train.resume=auto) keep training continuous."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sb3_contrib import MaskablePPO

from blockblast.training.callbacks import PauseFileCallback
from blockblast.training.train import latest_checkpoint, train
from blockblast.utils.config import AppConfig, load_config


def tiny(tmp_path: Path, total: int, *extra: str) -> AppConfig:
    return load_config(
        [
            f"train.total_timesteps={total}",
            "train.n_envs=2",
            "train.eval_freq=256",
            "train.n_eval_episodes=1",
            "train.checkpoint_freq=256",
            "train.device=cpu",
            "agent.features_dim=16",
            "agent.ppo.n_steps=64",
            "agent.ppo.batch_size=64",
            "agent.ppo.n_epochs=1",
            "run_name=pr",
            f"paths.models={tmp_path.as_posix()}/models",
            f"paths.logs={tmp_path.as_posix()}/logs",
            *extra,
        ]
    )


def test_pause_file_callback(tmp_path: Path) -> None:
    flag = tmp_path / "PAUSE"
    cb = PauseFileCallback(flag, check_every=2)
    cb.n_calls = 2
    assert cb._on_step() and not cb.paused
    flag.touch()
    cb.n_calls = 3
    assert cb._on_step()  # only checked every 2 calls
    cb.n_calls = 4
    assert not cb._on_step() and cb.paused


def test_ctrl_c_saves_checkpoint_then_resume_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "models" / "pr"
    # 1) first session reaches 256 steps (checkpoint), then "Ctrl+C" on the next learn call
    train(tiny(tmp_path, 256))
    assert latest_checkpoint(run) == run / "ckpt_256_steps.zip"

    def interrupted(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(MaskablePPO, "learn", interrupted)
        paused = train(tiny(tmp_path, 1024, "train.resume=auto"))
    assert paused == run / "ckpt_256_steps.zip" and not (run / "PAUSE").exists()

    # 2) resume to the full budget: steps continue from 256, not from 0
    final = train(tiny(tmp_path, 1024, "train.resume=auto"))
    assert final == run / "final.zip"
    assert MaskablePPO.load(str(final), device="cpu").num_timesteps == 1024
    evals = np.load(tmp_path / "logs" / "pr" / "evaluations.npz")
    assert list(evals["timesteps"]) == sorted(evals["timesteps"])  # history kept, in order
    assert evals["timesteps"][0] <= 256 and evals["timesteps"][-1] >= 768


def test_stale_pause_file_is_ignored(tmp_path: Path) -> None:
    run = tmp_path / "models" / "pr"
    run.mkdir(parents=True)
    (run / "PAUSE").touch()
    assert train(tiny(tmp_path, 256)) == run / "final.zip"


def test_resume_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        train(tiny(tmp_path, 256, "train.resume=auto"))
    with pytest.raises(FileNotFoundError):
        train(tiny(tmp_path, 256, f"train.resume={tmp_path.as_posix()}/missing.zip"))
