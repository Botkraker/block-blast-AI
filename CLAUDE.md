Read and follow [.claude/AGENTS.md](.claude/AGENTS.md) before doing any work in this repo.

# Project status (updated 2026-09-22 10:45)

## Done
| Milestone | State | Evidence |
|---|---|---|
| Docs | PRD, ARCHITECTURE, ESSENTIALS, AGENTS. Open questions Q1–Q7 resolved (PRD §8) | `.claude/` |
| M1 Simulator | `engine/` (bitboard, 27-piece catalogue v1, scoring, dealer, game) | 100 % line + branch coverage. 1,000-game determinism test incl. subprocess |
| M2 Env + baselines | `env/`, `agents/{random,greedy}`, `evaluation/`, `visualization/` | `check_env` clean. 16,040 env steps/s. Baselines in PRD §9 |
| M3 RL training | `agents/{networks,ppo_agent}.py`, `training/`, Hydra configs, `scripts/`, Colab notebook | Smoke + end-to-end `train()` tests |
| Graphical game | `app/` (pygame-ce): drag and drop, snap ghost, line-clear preview and animation, combo/score popups, AI hint (H), AI autoplay (A, speed, pause), best scores in `data/app/` | `scripts/play.py`. Headless tests in `tests/app/` |
| Quality | 179 tests pass. `ruff` + `mypy --strict` clean | `uv run pytest` |
| Training speed | 2,360 → ~4,350 steps/s on the GTX 1650 (afterstate, defaults): torch-free env workers, legal-only afterstate MLP, CUDA-graph rollout inference, no distribution arg validation. Same checkpoints, same learned function | ARCHITECTURE §11.1, §19 |

Held-out results (seeds 1_000_000…1_000_999):
| Agent | Mean score | Mean rounds |
|---|---|---|
| Random | 60.8 | 4.60 |
| Greedy | 295.7 | 11.96 |
| PPO v1 (CNN policy, 50M steps) | 164.0 (0.55× greedy, fails) | 8.18 |
| PPO v2 snapshot at 6M/40M (`models/ppo_v2/snapshot_6M.zip`) | 416.2 (1.41× greedy, CI of the difference [102, 139]) | 15.37 |

PPO targets: score ≥ 443.6 (S6), rounds ≥ 17.9 (S7), score ≥ 608 (S9).

## Paused
- `ppo_v2`: **afterstate policy** (ARCHITECTURE §11.1), 40M-step run. **Paused 2026-09-22 10:10 at 15,262,464 / 40M steps**. Latest checkpoint: `models/ppo_v2/ckpt_15262464_steps.zip` (verified it resumes with the new code on a copy in a temp dir).
  - Resume (about 1 h 35 min left at ~4,300 steps/s; it was ~2,050 before the speed-ups): `uv run python scripts/train.py run_name=ppo_v2 train.total_timesteps=40000000 train.resume=auto`
  - Rollout score was 519 at 8M steps. The held-out eval of the 6M snapshot gave 416.
  - Output: `logs/ppo_v2.out`, `logs/ppo_v2/` (TensorBoard, evaluations.npz), `models/ppo_v2/`.

## Left to do
1. Evaluate `ppo_v2` when it finishes:
   `uv run python scripts/evaluate.py agent=maskable_ppo eval.checkpoint=models/ppo_v2/best_model.zip eval.tag=ppo_v2 eval.compare_to=data/eval/greedy_default.json`
   Then write S6–S10 into PRD §9 and this file.
2. Refresh the GitHub README: fill in the PPO v2 row in the Results table and the roadmap checkbox, then regenerate the images with `uv run --with matplotlib python scripts/make_readme_assets.py` (the GIF and screenshots use `models/ppo_v2/best_model.zip`).
3. Export a replay GIF of the best agent: `scripts/replay.py data/replays/maskable_ppo_seed1000000.json --gif ppo.gif`.
4. If targets are missed: train longer (S10 allows 50M), tune `ent_coef` and `hidden`, and consider adding mobility (legal moves after the placement) as an afterstate feature.
5. Git: the repo exists (latest commit "updating readme"). The pause/resume, parallel-env and speed-up work is uncommitted, and `.gitignore`, `.claude/`, `docs/` and `CLAUDE.md` are still untracked.
6. M5 real game: deferred (PRD Q6).
7. **Next challenge: play forever.** Metric `truncated_frac` (games reaching 10,000 moves, currently 0 %). Ideas are in the README section "Next challenge": whole-hand lookahead, survival reward + gamma 0.999, worst-case-piece feature, longer training.

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
