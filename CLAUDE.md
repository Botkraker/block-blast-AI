Read and follow [.claude/AGENTS.md](.claude/AGENTS.md) before doing any work in this repo.

# Project status (updated 2026-09-22, v3 survival agent trained and evaluated)

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
| README | GitHub landing page with GIF (2,258-point v3 game), charts and v3 results | `docs/images/` via `scripts/make_readme_assets.py` |
| Quality | 190 tests pass. `ruff` + `mypy --strict` clean | `uv run pytest` |

Held-out results (seeds 1_000_000…1_000_999, 1,000 games):
| Agent | Mean score | Mean rounds |
|---|---|---|
| Random | 60.8 | 4.60 |
| Greedy | 295.7 | 11.96 |
| PPO v1 (CNN policy, 50M steps) | 164.0 (0.55× greedy) | 8.18 |
| PPO v2 (afterstate, points reward, 40M, `models/ppo_v2/best_model.zip`) | 1,299.4 (4.39× greedy) | 40.39 |
| **PPO v3 (survival reward, 40M, `models/ppo_safe/final.zip`)** | **1,436.8 (4.86× greedy, 23.6× random)** | **45.74 (3.82×)** |

PPO v3 vs v2 (same 1,000 seeds): score +11 % (CI of the difference [35.6, 237.2]), rounds +13 % ([2.4, 8.3]), median 1,063 / 35 rounds, max 8,661, longest 260 rounds, lines 62.2, `truncated_frac` 0. `best_model.zip` (34M) is worse: 1,372.3 / 43.77. Eval JSONs: `data/eval/maskable_ppo_ppo_safe40_{best,final}.json`.
Trained as: 0→22.1M plain safe reward, then resumed with `agent.ppo.gamma=0.999 agent.prior=lookahead env.mid_start_prob=0.25` to 40M (~3,530 steps/s). `snapshot_21M.zip` = the pre-change best (973.4 / 32.4).

PPO v2 best game: 9,539. Median: 1,002 points, 31 rounds. `final.zip` gives the same result (1,296.9 / 40.30).

## Left to do
1. Git: everything since commit 494c619 is uncommitted (v3 training options, docs, regenerated images). User said: do not commit or push.
2. **Play forever** is still open: `truncated_frac` 0 for every agent. Untried ideas in README "What I want to do next": whole-hand lookahead (6 orders), worst-case-piece penalty, longer training (both curves still rising at 40M), curriculum on a fresh run (`train.curriculum_steps`, schedule counts from step 0 so it does nothing on a resumed run).
3. Optional faster-training settings (they change learning, so new runs only): `agent.ppo.n_epochs=2` (~5,800 steps/s), `train.n_envs=128 agent.ppo.n_steps=128` (~5,000).
4. M5 real game: deferred (PRD Q6).

## Training options (all off by default, never applied to eval/validation)
- `reward=safe`: r = alive_bonus − board_weight·filled/64 − penalty·1[terminated] (`configs/reward/safe.yaml`: penalty 20).
- `env.mid_start_prob` / `env.mid_start_min_cells`: each env keeps its own deque of crowded boards and restarts from one. `make_callbacks` forces `mid_start_prob=0` for validation.
- `train.curriculum_steps` + `train.curriculum_hard_start`: `CurriculumCallback` ramps `hard_piece_weight` (O3, I5h, I5v) on global steps and `set_attr`s it to the envs; applies to games started after.
- `agent.prior`: auto | score | board | lookahead. `lookahead` = −filled/8 − 4·(hand pieces that fit nowhere on the afterstate), exact bitboard check in torch (`AfterstateExtractor.n_dead`, ~11 % slower).
- On resume, `gamma` and `agent.prior` from the config override the checkpoint (`train.py`); the prior is written into `model.policy_kwargs` so it survives saving.

## User preferences
- Do not commit or push unless asked explicitly.
- README is "unclaudified": no `.claude/` links, no emoji headings, no AI-writing tells (see the humanizer skill). Keep it that way.

## Play
`uv run python scripts/play.py` (you play) · `uv run python scripts/play.py --watch --loop` (the AI plays).
Assets: `scripts/make_readme_assets.py` (CHECKPOINTS[0] = `models/ppo_safe/final.zip`; GIF = seed 1000034, 2,258 points).
`models/` now keeps only best/final/snapshot zips: intermediate `ckpt_*.zip` were deleted (models/ is gitignored).

## Pause / resume
- Pause: Ctrl+C, or create `models/<run>/PAUSE` (the model is saved as `ckpt_<steps>_steps.zip`).
- Resume: the same command + `train.resume=auto`. For ppo_v2: `uv run python scripts/train.py run_name=ppo_v2 train.total_timesteps=40000000 train.resume=auto`.
- A process started before a code change keeps running the old code: pause it (PAUSE file) and resume to pick up changes.

## Gotchas
- Windows: `train.use_subproc=true` (default) runs the games in 8 torch-free `ParallelVecEnv` workers (~36 MB each). Scripts that start workers must import torch/SB3/the training stack inside `main()`, never at module level: spawned workers re-run the main module's imports (a warning is logged if a worker loads torch). Stock `SubprocVecEnv` imports torch in every worker (~0.5 GB) and stalled on 12 GB RAM.
- `AfterstatePolicy` mirrors the engine rules in torch. `tests/agents/test_afterstate.py` must stay green if the rules change.
- The afterstate MLP only runs on legal actions (~19 % of `batch × 192` rows). Peak CUDA memory is ~250 MB at `batch_size=1024` and ~435 MB at 2048.
- Rollout inference replays a CUDA graph (`AfterstatePolicy.logits_and_values`). If you debug policy outputs during collection, set `policy.use_cuda_graphs = False`.
