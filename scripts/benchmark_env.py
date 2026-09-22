"""PRD S4: single-process env throughput with random masked actions (target ≥ 5,000 steps/s)."""

from __future__ import annotations

import argparse
import time

import numpy as np

from blockblast.env import BlockBlastEnv

TARGET = 5_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = BlockBlastEnv()
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)
    episodes = 0
    t0 = time.perf_counter()
    for _ in range(args.steps):
        legal = np.flatnonzero(env.action_masks())
        _, _, terminated, truncated, _ = env.step(int(legal[rng.integers(legal.size)]))
        if terminated or truncated:
            episodes += 1
            env.reset()
    dt = time.perf_counter() - t0
    sps = args.steps / dt
    print(f"{args.steps} steps, {episodes} episodes, {dt:.2f}s -> {sps:,.0f} steps/s")
    print("PASS" if sps >= TARGET else f"FAIL (target {TARGET:,} steps/s)")


if __name__ == "__main__":
    main()
