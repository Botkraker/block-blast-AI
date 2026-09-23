"""Play Block Blast in a window, get AI hints, or watch the AI play.

uv run python scripts/play.py                 # you play; H = hint, A = let the AI play
uv run python scripts/play.py --watch         # the AI plays (Space pause, +/- speed)
uv run python scripts/play.py --watch --loop --agent greedy
uv run python scripts/play.py --agent ppo --checkpoint models/ppo_safe/final.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

from blockblast.agents import Agent, GreedyAgent, RandomAgent
from blockblast.app import PlaySession

DEFAULT_CHECKPOINTS = (
    Path("models/ppo_safe/final.zip"),  # ships with the repo
    Path("models/ppo_v2/best_model.zip"),
    Path("models/ppo_v1/best_model.zip"),
)


def build_agent(name: str, checkpoint: Path | None) -> tuple[Agent | None, str]:
    if name == "none":
        return None, ""
    if name == "greedy":
        return GreedyAgent(), "GREEDY"
    if name == "random":
        return RandomAgent(), "RANDOM"
    candidates = [checkpoint] if checkpoint else list(DEFAULT_CHECKPOINTS)
    found = next((p for p in candidates if p is not None and p.exists()), None)
    if found is None:
        if name == "ppo":
            raise SystemExit(f"no checkpoint found (tried {', '.join(map(str, candidates))})")
        return GreedyAgent(), "GREEDY"  # auto: fall back when no model is trained yet
    from blockblast.agents.ppo_agent import PPOAgent

    print(f"loaded {found}")
    return PPOAgent.load(found, device="cpu"), "PPO"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--agent", choices=["auto", "ppo", "greedy", "random", "none"], default="auto"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--watch", action="store_true", help="start with the AI playing")
    parser.add_argument("--loop", action="store_true", help="auto-restart after game over")
    parser.add_argument("--seed", type=int, default=None, help="fixed deal sequence")
    parser.add_argument("--speed", type=int, default=1, help="AI speed 0 (slow) .. 4 (fast)")
    args = parser.parse_args()

    agent, label = build_agent(args.agent, args.checkpoint)
    from blockblast.app.game_app import AppOptions, BlockBlastApp

    best = Path("data/app") / ("best_ai.json" if args.watch else "best_human.json")
    session = PlaySession(seed=args.seed, best_path=best)
    options = AppOptions(
        autoplay=args.watch, loop=args.loop, seed=args.seed, agent_label=label, speed=args.speed
    )
    BlockBlastApp(session, agent, options).run()


if __name__ == "__main__":
    main()
