"""MaskablePPO smoke test: learn a little, save/load, act legally, reproducible eval (S12)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from blockblast.agents.ppo_agent import PPOAgent, build_model
from blockblast.evaluation import evaluate_agent
from blockblast.training.vec_env import env_kwargs_from_config, make_vec_env
from blockblast.utils.config import EnvConfig, RewardSection, load_config
from blockblast.utils.seeding import set_global_seeds

SMALL_PPO = {"n_steps": 64, "batch_size": 64, "n_epochs": 1, "net_arch": {"pi": [32], "vf": [32]}}


@pytest.mark.parametrize("policy", ["afterstate", "cnn"])
def test_train_save_load_eval(tmp_path: Path, policy: str) -> None:
    set_global_seeds(0)
    kwargs = env_kwargs_from_config(EnvConfig(), RewardSection())
    vec = make_vec_env(kwargs, n_envs=2, seed=0, use_subproc=False)
    model = build_model(vec, SMALL_PPO, features_dim=32, seed=0, device="cpu", policy=policy)
    model.learn(total_timesteps=256)
    path = tmp_path / "m.zip"
    model.save(str(path))
    vec.close()

    agent = PPOAgent.load(path, device="cpu")
    seeds = range(1_000_000, 1_000_003)
    r1 = evaluate_agent(agent, seeds, kwargs)  # env raises on any illegal action
    r2 = evaluate_agent(PPOAgent.load(path, device="cpu"), seeds, kwargs)
    assert r1 == r2
    assert all(e.moves > 0 for e in r1.episodes)


def test_cnn_forward_shape() -> None:
    from blockblast.agents.networks import BlockBlastCNN
    from blockblast.env.observation import observation_space

    net = BlockBlastCNN(observation_space(), features_dim=16)
    assert net(torch.zeros(4, 5, 8, 8)).shape == (4, 16)


def test_config_composes() -> None:
    cfg = load_config(["agent=greedy", "train.n_envs=2"])
    assert cfg.agent.name == "greedy" and cfg.train.n_envs == 2
    assert cfg.eval.seed_start == 1_000_000
    ppo = load_config([])
    assert ppo.agent.ppo["gamma"] == 0.995


def test_train_end_to_end(tmp_path: Path) -> None:
    """training.train() with a tiny budget writes final, best and checkpoint models."""
    from blockblast.training.train import train

    cfg = load_config(
        [
            "train.total_timesteps=512",
            "train.n_envs=2",
            "train.use_subproc=false",
            "train.eval_freq=256",
            "train.n_eval_episodes=1",
            "train.checkpoint_freq=256",
            "train.device=cpu",
            "agent.features_dim=16",
            "agent.ppo.n_steps=64",
            "agent.ppo.batch_size=64",
            "agent.ppo.n_epochs=1",
            "run_name=smoke",
            f"paths.models={tmp_path.as_posix()}/models",
            f"paths.logs={tmp_path.as_posix()}/logs",
        ]
    )
    final = train(cfg)
    run = tmp_path / "models" / "smoke"
    assert final == run / "final.zip" and final.exists()
    assert (run / "best_model.zip").exists() and (run / "config.json").exists()
    assert list(run.glob("ckpt_*_steps.zip"))
    assert list((tmp_path / "logs" / "smoke").rglob("events.out.tfevents.*"))


def test_train_rejects_eval_range_seed() -> None:
    import pytest

    from blockblast.training.train import train

    with pytest.raises(ValueError):
        train(load_config(["seed=1000000"]))


def test_unknown_policy() -> None:
    vec = make_vec_env(env_kwargs_from_config(EnvConfig(), RewardSection()), 1, 0, False)
    with pytest.raises(ValueError):
        build_model(vec, SMALL_PPO, policy="nope", device="cpu")
