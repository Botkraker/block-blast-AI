# PRD: Block Blast RL Agent

## 1. Problem statement

Block Blast is a single-player placement puzzle: an 8x8 grid, a hand of 3 polyomino pieces per round, line clears on full rows/columns, and a combo multiplier for consecutive clearing moves. The game has a large branching factor (up to 192 placements per decision), a stochastic piece supply, and long-horizon consequences: a greedy placement can leave the board unable to fit future pieces. Hand-written heuristics plateau quickly. We want an agent that **learns** a placement policy through reinforcement learning and outperforms simple baselines by a measurable margin.

## 2. End-product goal

An agent that plays complete Block Blast games autonomously in a faithful simulator and beats both baselines:

| Baseline | Definition |
|---|---|
| Random | Uniform choice among legal actions (action mask). |
| Greedy | Legal action maximizing immediate `Δscore`. Ties go to the lowest action index. |

The final agent is a MaskablePPO policy with a CNN feature extractor. It is evaluated on a fixed held-out seed set and reports mean score and mean rounds survived.

## 3. Users

| User | Need |
|---|---|
| Project owner (ML engineer) | Train, evaluate and compare agents reproducibly. |
| AI coding agents | Unambiguous invariants and module boundaries so they can implement modules independently. |
| Viewers / learners | Watch replays (ANSI, GIF) of agent games. |

## 4. Scope

**In scope**
- Deterministic, seedable game engine implementing the rules in `ARCHITECTURE-ESSENTIALS.md`.
- Gymnasium environment with invalid-action masking.
- Random and greedy baselines.
- MaskablePPO training with vectorized environments, TensorBoard logging and checkpoints.
- Evaluation harness with fixed seeds, summary statistics and confidence intervals.
- Replay recording (seed + action list) and rendering (ANSI, RGB frames, GIF).
- Optional M5: play the real game through screen capture and simulated input.

**Out of scope**
- Rotating pieces (the rules forbid it).
- Multiplayer, monetization, power-ups, or any mechanic not listed in the rules.
- Search-based agents (MCTS, beam search) as the deliverable. They may be added later as extra baselines.
- Mobile app packaging and cloud training infrastructure.
- Human-vs-agent UI beyond replay viewing.

## 5. Success metrics

All evaluations use 1,000 episodes with seeds `1_000_000 … 1_000_999` (disjoint from training seeds), a deterministic policy, and default configs.

| # | Metric | Target | How tested |
|---|---|---|---|
| S1 | Engine line + branch coverage | 100 % | `pytest --cov=blockblast.engine --cov-branch --cov-fail-under=100` |
| S2 | Engine determinism | 1,000 seeded random-action games are bitwise identical across 2 runs and across processes | `tests/engine/test_determinism.py` |
| S3 | Env API compliance | `gymnasium.utils.env_checker.check_env` passes with no warnings | `tests/env/test_env_api.py` |
| S4 | Env throughput | ≥ 5,000 steps/s single process, random masked actions, including observation and mask | `scripts/benchmark_env.py` |
| S5 | Baseline ordering | Greedy mean score > Random mean score, 95 % bootstrap CI of the difference excludes 0 | `scripts/evaluate.py` |
| S6 | PPO vs greedy (score) | PPO mean score ≥ 1.5 × greedy mean score | `scripts/evaluate.py` |
| S7 | PPO vs greedy (survival) | PPO mean rounds survived ≥ 1.5 × greedy mean rounds survived | `scripts/evaluate.py` |
| S8 | Statistical significance | Lower bound of the 95 % bootstrap CI of (PPO − greedy) mean score > 0 | `evaluation/metrics.py::compare` |
| S9 | PPO vs random | PPO mean score ≥ 10 × random mean score | `scripts/evaluate.py` |
| S10 | Training budget | S6–S9 reached within 50 M environment steps | TensorBoard + eval checkpoints |
| S11 | Replay fidelity | Replaying any recorded episode reproduces its final score exactly | `tests/env/test_replay.py` |
| S12 | Eval reproducibility | Two evaluations of the same checkpoint with the same seeds give identical metrics | `tests/agents/test_ppo_smoke.py` |

"Rounds survived" = number of hands dealt in the episode (`GameState.round_index` at game end).

## 6. Milestones

| ID | Name | Deliverables | Exit criteria |
|---|---|---|---|
| M1 | Simulator | `engine/` (pieces, board, scoring, dealer, game), full unit tests | S1, S2 |
| M2 | Env + baselines | `env/`, `agents/{random,greedy}`, `evaluation/`, `visualization/render.py`, benchmark | S3, S4, S5; baseline numbers recorded in this PRD |
| M3 | RL training | `agents/networks.py`, `agents/ppo_agent.py`, `training/`, configs, TensorBoard | Training runs end-to-end; learning curve rises above greedy mean score |
| M4 | Evaluation + visualization | Final eval report, replays, GIF export | S6–S12 |
| M5 (optional) | Real game | `capture/` (screen grab, parser, input controller), `scripts/play_real.py` | Agent completes 10 real games without a parser error; parsed board matches screenshot in ≥ 99 % of 500 labelled frames |

RL work (M3) does not start until M1 exit criteria pass.

## 7. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Simulator differs from the real game (piece set, deal distribution, scoring constants) | Policy learned for the wrong game; M5 transfer fails | Catalogue and score constants are data/config (§8); M5 validates on real frames |
| Long-horizon credit assignment | PPO does not beat greedy | Dense score reward + game-over penalty; γ = 0.995; reward alternatives configurable |
| Env throughput too low | Training budget exceeded | Bitboards, precomputed placement masks, SubprocVecEnv; benchmark gate S4 |
| Action-mask bug lets illegal actions through | Corrupted training | Engine raises on illegal moves; mask/legality equivalence property test |
| Reward hacking via shaping | Agent optimizes the proxy rather than score | Evaluation always reports raw game score, never shaped reward |
| Overfitting to seeds | Inflated metrics | Disjoint held-out eval seed range |
| Real-game automation violates terms of service | M5 blocked | M5 is optional and deferred (Q6); personal use only |

## 8. Resolved questions

All questions were answered by the project owner on 2026-09-21. The earlier assumptions are now decisions.

| # | Question | Decision |
|---|---|---|
| Q1 | Piece catalogue | Catalogue v1 in `ARCHITECTURE.md` §6.1 (27 shapes, each orientation a distinct piece). |
| Q2 | Deal distribution | Uniform i.i.d. with replacement. |
| Q3 | Score constants | 1 per cell, 10 per line. |
| Q4 | Combo rule | Multiplier `1 + s`. The streak resets on any non-clearing placement and persists across rounds. |
| Q5 | Simultaneous-clear bonus | Linear `10 · n_lines · (1 + s)`. The owner has no preference, so the simplest rule stays. |
| Q6 | M5 target platform | None. This is a personal project, so M5 stays optional and deferred. `capture/` and `scripts/play_real.py` remain documented but unimplemented until a platform is chosen. |
| Q7 | Compute | Local NVIDIA GTX 1650 (4 GB) or Google Colab GPU (`notebooks/train_colab.ipynb`). The S10 budget of 50 M steps is unchanged. |

## 9. Measured results

Held-out seeds `1_000_000 … 1_000_999`, default configs, measured 2026-09-21 on an i5-10500H + GTX 1650.

| Metric | Result | Status |
|---|---|---|
| S1 engine coverage | 100 % lines and branches | Pass |
| S2 determinism | 1,000 games identical across runs and a subprocess | Pass |
| S3 `check_env` | No warnings | Pass |
| S4 env throughput | 16,040 steps/s | Pass |
| S5 greedy vs random | 295.7 vs 60.8 mean score (4.87×), difference CI [224.7, 245.5] | Pass |

| Agent | Mean score (95 % CI) | Median | Max | Mean rounds | Mean lines |
|---|---|---|---|---|---|
| Random | 60.8 [58.7, 62.9] | 52 | 292 | 4.60 | 1.20 |
| Greedy | 295.7 [285.8, 306.2] | 260 | 990 | 11.96 | 12.09 |
| PPO v1 (CNN policy, 50M steps) | 164.0 [159.1, 168.9] | 151 | 571 | 8.18 | 6.11 |
| PPO v2 (afterstate, snapshot at 6M of 40M) | 416.2 [401.5, 431.1] | 366 | 2033 | 15.37 | 17.10 |
| **PPO v2 (afterstate, 40M, best_model)** | **1,299.4 [1,232.2, 1,367.4]** | 1,002 | 9,539 | **40.39** | 54.28 |

Derived PPO targets: S6 mean score ≥ 443.6. S7 mean rounds ≥ 17.9. S9 mean score ≥ 608.

Final status (2026-09-22, v2 at 40M steps, `models/ppo_v2/best_model.zip`; `final.zip` gives 1,296.9 / 40.30, within noise):

| Metric | Target | Result | Status |
|---|---|---|---|
| S6 score vs greedy | ≥ 1.5× | 4.39× | Pass |
| S7 rounds vs greedy | ≥ 1.5× | 3.38× | Pass |
| S8 difference CI | lower bound > 0 | [935.4, 1,072.5] | Pass |
| S9 score vs random | ≥ 10× | 21.4× | Pass |
| S10 training budget | ≤ 50M steps | 40M | Pass |
| S11 replay fidelity | exact | `tests/env/test_replay.py` | Pass |
| S12 eval reproducibility | identical | `tests/agents/test_ppo_smoke.py` | Pass |

M4 is complete.

Next challenge (post-M4, not yet a milestone): near-infinite play, measured as the share of evaluation games reaching `max_steps = 10,000` (`truncated_frac`, 0 % for v2 at 40M; median game 31 rounds).
