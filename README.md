# Block Blast AI

A reinforcement-learning agent (MaskablePPO + CNN) that plays Block Blast in a deterministic Python simulator. It is compared against random and greedy baselines on 1,000 held-out games.

Design docs live in [`.claude/`](.claude/): [PRD](.claude/PRD.md), [architecture](.claude/ARCHITECTURE.md), [essentials](.claude/ARCHITECTURE-ESSENTIALS.md), [agent rules](.claude/AGENTS.md).

## Rules (simulator)
- 8x8 board. Each round you get 3 pieces (27 fixed shapes, no rotation) and place all 3 in any order.
- Full rows and columns clear simultaneously.
- Score: 1 per placed cell + 10 per cleared line × (1 + combo streak).
- The game ends when no piece in hand fits.

## Play
```bash
uv run python scripts/play.py                 # you play: drag pieces, H = AI hint, A = let the AI play
uv run python scripts/play.py --watch --loop  # watch the AI (Space pause, +/- speed, A take over)
```
By default the AI is the best trained PPO model (`models/ppo_v2`, then `models/ppo_v1`). If none is trained it falls back to greedy. Pick one with `--agent greedy|random|ppo --checkpoint <zip>`.

## Quick start
```bash
uv sync
uv run pytest
uv run python scripts/benchmark_env.py
uv run python scripts/evaluate.py agent=random
uv run python scripts/evaluate.py agent=greedy
uv run python scripts/train.py train.total_timesteps=50000000
uv run tensorboard --logdir logs
uv run python scripts/evaluate.py agent=maskable_ppo eval.checkpoint=models/<run>/best_model.zip eval.compare_to=data/eval/greedy_default.json
uv run python scripts/replay.py data/replays/greedy_seed1000000.json --gif greedy.gif
```

To train on Google Colab, use [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).

## Layout
| Path | Content |
|---|---|
| `src/blockblast/engine` | Pure game rules (bitboard) |
| `src/blockblast/env` | Gymnasium env, observation `(5,8,8)`, 192 masked actions, reward |
| `src/blockblast/agents` | Random, greedy, MaskablePPO + CNN |
| `src/blockblast/training` | Vectorized training loop and callbacks |
| `src/blockblast/evaluation` | Fixed-seed evaluation and bootstrap statistics |
| `src/blockblast/visualization` | ANSI/RGB rendering, replays, GIF |
| `src/blockblast/app` | Graphical game (pygame) |
| `configs/` | Hydra configs (`agent=…`, `train.…=…` overrides) |
