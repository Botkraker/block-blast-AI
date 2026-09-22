"""Evaluate an agent on held-out seeds and write data/eval/<agent>_<tag>.json.

Examples:
    uv run python scripts/evaluate.py agent=greedy
    uv run python scripts/evaluate.py agent=maskable_ppo
        eval.checkpoint=models/<run>/best_model.zip
        eval.compare_to=data/eval/greedy_default.json
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from blockblast.agents import Agent, GreedyAgent, RandomAgent
from blockblast.evaluation import EvalResult, compare, evaluate_agent
from blockblast.training.vec_env import env_kwargs_from_config
from blockblast.utils.config import AppConfig, to_app_config
from blockblast.utils.logger import setup_logging
from blockblast.utils.seeding import EVAL_SEED_START


def build_agent(cfg: AppConfig) -> Agent:
    name = cfg.agent.name
    if name == "random":
        return RandomAgent(cfg.agent.seed)
    if name == "greedy":
        return GreedyAgent()
    if name == "maskable_ppo":
        if not cfg.eval.checkpoint:
            raise SystemExit("eval.checkpoint=<path to .zip> is required for maskable_ppo")
        from blockblast.agents.ppo_agent import PPOAgent

        return PPOAgent.load(Path(cfg.eval.checkpoint), deterministic=cfg.eval.deterministic)
    raise SystemExit(f"unknown agent {name!r}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    app = to_app_config(cfg)
    if app.eval.seed_start < EVAL_SEED_START:
        raise SystemExit(f"eval seeds must be >= {EVAL_SEED_START} (disjoint from training)")
    agent = build_agent(app)
    seeds = range(app.eval.seed_start, app.eval.seed_start + app.eval.n_episodes)
    data = Path(app.paths.data)
    t0 = time.perf_counter()
    result = evaluate_agent(
        agent,
        seeds,
        env_kwargs_from_config(app.env, app.reward),
        record_dir=data / "replays",
        record_n=app.eval.record_n,
    )
    out = data / "eval" / f"{app.agent.name}_{app.eval.tag}.json"
    result.to_json(out)
    s = result.summary()
    print(f"{app.agent.name}: {len(seeds)} episodes in {time.perf_counter() - t0:.1f}s -> {out}")
    print(
        f"  score  mean={s['score_mean']:.1f}  95% CI=[{s['score_ci_lo']:.1f}, "
        f"{s['score_ci_hi']:.1f}]  median={s['score_median']:.0f}  max={s['score_max']:.0f}"
    )
    print(f"  rounds mean={s['rounds_mean']:.2f}  moves mean={s['moves_mean']:.1f}")
    print(f"  lines  mean={s['lines_cleared_mean']:.2f}  max_combo mean={s['max_combo_mean']:.2f}")
    if app.eval.compare_to:
        other = EvalResult.from_json(Path(app.eval.compare_to))
        for metric in ("score", "rounds"):
            c = compare(result, other, metric)
            print(
                f"  vs {other.agent_name} [{metric}]: ratio={c.ratio:.2f}x  "
                f"diff 95% CI=[{c.diff_ci[0]:.1f}, {c.diff_ci[1]:.1f}]"
            )


if __name__ == "__main__":
    main()
