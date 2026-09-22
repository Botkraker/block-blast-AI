Read and follow [.claude/AGENTS.md](.claude/AGENTS.md) before doing any work in this repo.

# Project status (updated 2026-09-22, training complete)

## Done
| Milestone | State | Evidence |
|---|---|---|
| Docs | PRD, ARCHITECTURE, ESSENTIALS, AGENTS. Open questions Q1–Q7 resolved (PRD §8) | `.claude/` |
| M1 Simulator | `engine/` (bitboard, 27-piece catalogue v1, scoring, dealer, game) | 100 % line + branch coverage. 1,000-game determinism test incl. subprocess |
| M2 Env + baselines | `env/`, `agents/{random,greedy}`, `evaluation/`, `visualization/` | `check_env` clean. 16,040 env steps/s. Baselines in PRD §9 |
| M3 RL training | `agents/{networks,ppo_agent}.py`, `training/` (pause/resume, parallel env workers), Hydra configs, `scripts/`, Colab notebook | Smoke, end-to-end and pause/resume tests |
| **M4 Evaluation** | **PPO v2 passes S6–S12** (PRD §9 final status) | `data/eval/maskable_ppo_ppo_v2_best.json` |
| Graphical game | `app/` (pygame-ce): drag and drop, snap ghost, line-clear preview and animation, popups, AI hint (H), AI autoplay (A) | `scripts/play.py`, `tests/app/` |
| Training speed | 2,360 → ~4,350 steps/s on the GTX 1650: torch-free env workers, legal-only afterstate MLP, CUDA-graph rollout inference, no distribution arg validation | ARCHITECTURE §11.1, §19 |
| README | GitHub landing page with GIF (2,389-point game), charts and final results | `docs/images/` via `scripts/make_readme_assets.py` |
| Quality | 179 tests pass. `ruff` + `mypy --strict` clean | `uv run pytest` |

Held-out results (seeds 1_000_000…1_000_999, 1,000 games):
| Agent | Mean score | Mean rounds |
|---|---|---|
| Random | 60.8 | 4.60 |
| Greedy | 295.7 | 11.96 |
| PPO v1 (CNN policy, 50M steps) | 164.0 (0.55× greedy) | 8.18 |
| **PPO v2 (afterstate, 40M, `models/ppo_v2/best_model.zip`)** | **1,299.4 (4.39× greedy, 21.4× random; CI of the difference vs greedy [935, 1,073])** | **40.39 (3.38×)** |

PPO v2 best game: 9,539. Median: 1,002 points, 31 rounds. `final.zip` gives the same result (1,296.9 / 40.30).

## Left to do
1. Git: stage and commit the final results (README, PRD, CLAUDE.md, docs/images, scripts/make_readme_assets.py). The earlier speed-up and pause/resume work is staged but uncommitted.
2. **Next challenge: play forever.** Metric `truncated_frac` (games reaching 10,000 moves, still 0 % for v2). Ideas are in the README section "Next challenge": whole-hand lookahead, survival reward + gamma 0.999, worst-case-piece feature, longer training (the v2 curve was still rising at 40M).
3. Optional faster-training settings the optimizer measured (they change learning, so use them for new runs only): `agent.ppo.n_epochs=2` (~5,800 steps/s), `train.n_envs=128 agent.ppo.n_steps=128` (~5,000).
4. M5 real game: deferred (PRD Q6).

## Play
`uv run python scripts/play.py` (you play) · `uv run python scripts/play.py --watch --loop` (the AI plays; uses `models/ppo_v2` if present, then `ppo_v1`, then greedy).

## Pause / resume
- Pause: Ctrl+C, or create `models/<run>/PAUSE` (the model is saved as `ckpt_<steps>_steps.zip`).
- Resume: the same command + `train.resume=auto`. For ppo_v2: `uv run python scripts/train.py run_name=ppo_v2 train.total_timesteps=40000000 train.resume=auto`.
- A process started before a code change keeps running the old code: pause it (PAUSE file) and resume to pick up changes.

## Gotchas
- Windows: `train.use_subproc=true` (default) runs the games in 8 torch-free `ParallelVecEnv` workers (~36 MB each). Scripts that start workers must import torch/SB3/the training stack inside `main()`, never at module level: spawned workers re-run the main module's imports (a warning is logged if a worker loads torch). Stock `SubprocVecEnv` imports torch in every worker (~0.5 GB) and stalled on 12 GB RAM.
- `AfterstatePolicy` mirrors the engine rules in torch. `tests/agents/test_afterstate.py` must stay green if the rules change.
- The afterstate MLP only runs on legal actions (~19 % of `batch × 192` rows). Peak CUDA memory is ~250 MB at `batch_size=1024` and ~435 MB at 2048.
- Rollout inference replays a CUDA graph (`AfterstatePolicy.logits_and_values`). If you debug policy outputs during collection, set `policy.use_cuda_graphs = False`.
