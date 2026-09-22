# AGENTS: Rules for AI coding agents

## 1. Read order
1. `.claude/AGENTS.md` (this file)
2. `.claude/ARCHITECTURE-ESSENTIALS.md` (invariants, always load)
3. `.claude/PRD.md` (scope, milestones, metrics). Read it when a task touches scope or metrics.
4. `.claude/ARCHITECTURE.md`. Read the section for the module you are touching.

If code and docs disagree, the docs win. Stop and report the conflict. Do not change code to match a guess.

## 2. Invariants (copy of ARCHITECTURE-ESSENTIALS)
| Item | Value |
|---|---|
| Action space | `Discrete(192)`, `a = slot·64 + row·8 + col`. `(row, col)` = bounding-box top-left of the piece |
| Observation | `Box(0.0, 1.0, shape=(5, 8, 8), dtype=float32)`: ch0 board · ch1–3 slot piece masks anchored at `(0,0)` · ch4 `min(combo_streak, 8)/8` |
| Score | `Δscore_t = n_cells + 10 · n_lines · (1 + s)`. `s ← s + 1 if n_lines ≥ 1 else 0` |
| Reward | `r_t = Δscore_t / 10 − 5 · 1[terminated_t]`. Truncation not penalized |
| Dependencies | `engine ← visualization ← env ← agents ← {training, evaluation, app} ← scripts`. `engine ← capture`. `utils ← {training, evaluation, scripts}` |

## 3. Coding conventions
- Python 3.11. Type hints on every public function. `mypy --strict` clean on `src/`. `ruff check` and `ruff format` clean.
- Use `@dataclass(frozen=True, slots=True)` for value types and `Final` for constants. No mutable module-level state.
- Google-style docstrings on public APIs. Comments explain *why*, not *what*.
- Library code uses the `logging` module and never `print`. Only `scripts/` may print.
- Library code never uses global RNGs (`random`, `np.random.*` module functions). Pass `np.random.Generator` explicitly.
- Signatures must match `ARCHITECTURE.md`. Adding a public symbol means adding it to `ARCHITECTURE.md` in the same commit.
- `scripts/` stay thin: parse config, call package functions, write outputs. No logic.
- `notebooks/` are never imported by `src/` or `tests/`.

## 4. Module ownership boundaries
| Package | Owns | Allowed imports from `blockblast.*` |
|---|---|---|
| `engine` | Rules, pieces, scoring, dealing, `GameState` | none |
| `visualization` | Rendering, replay records, GIF | `engine` |
| `env` | Gym API, observation, action/mask, reward | `engine`, `visualization` |
| `agents` | Baselines, PPO wrapper, CNN extractor | `env`, `engine` |
| `training` | Vec envs, callbacks, `train()` | `agents`, `env`, `utils` |
| `evaluation` | Episode runner, metrics, comparisons | `agents`, `env`, `visualization`, `utils` |
| `utils` | Config schemas, seeding, logging | none |
| `app` | Graphical game (pygame), session, theme | `agents`, `env`, `engine` |
| `capture` (M5) | Screen grab, parsing, input | `engine` |

One task touches one package unless the task explicitly says otherwise.

## 5. Test requirements
- **Engine gate:** `blockblast.engine` must be 100 % deterministic with 100 % line + branch coverage **before any RL work** (`agents/networks.py`, `agents/ppo_agent.py`, `training/`).
- Every bug fix adds a regression test that fails before the fix.
- New public functions ship with unit tests in the mirrored `tests/<package>/` directory.
- Env changes must keep `check_env` passing and the mask/legality equivalence test green.
- Tests are seeded and run in under 60 s total, except tests marked `@pytest.mark.slow`.

## 6. Commands
```bash
uv sync                                                    # install (add --extra capture for M5)
uv run pytest                                              # all tests
uv run pytest tests/engine --cov=blockblast.engine --cov-branch --cov-fail-under=100
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run python scripts/benchmark_env.py                     # throughput gate
uv run python scripts/train.py agent=maskable_ppo train.total_timesteps=50000000
uv run python scripts/train.py run_name=<run> train.resume=auto   # resume after Ctrl+C or a PAUSE file
uv run python scripts/evaluate.py agent=greedy
uv run python scripts/evaluate.py agent=maskable_ppo eval.checkpoint=models/<run>/best_model.zip eval.compare_to=data/eval/greedy_default.json
uv run python scripts/replay.py data/replays/<file>.json --gif out.gif
uv run tensorboard --logdir logs
uv run python scripts/play.py                              # play; H hint, A AI plays
uv run python scripts/play.py --watch --loop               # watch the AI
```

## 7. Commit rules
- Conventional Commits: `feat(engine): …`, `fix(env): …`, `test(…)`, `docs(…)`, `refactor(…)`, `chore(…)`.
- One logical change per commit. Tests, ruff and mypy pass before committing.
- Never commit `models/*`, `logs/*`, `data/*` contents (only `.gitkeep`), `.venv/`, or secrets.
- Changes to invariants: update `ARCHITECTURE-ESSENTIALS.md`, `ARCHITECTURE.md` and this file §2 in the same commit, and mark the commit `!` (breaking).

## 8. Definition of done
- [ ] Code matches the signatures in `ARCHITECTURE.md` and the dependency direction.
- [ ] Unit tests added. `uv run pytest` is green. The engine coverage gate still holds.
- [ ] `ruff` and `mypy --strict` are clean.
- [ ] Determinism preserved: same seed + actions ⇒ same result.
- [ ] Docs updated if any public interface, config key or command changed.
- [ ] PRD metric affected by the change is re-measured and reported.

## 9. Forbidden actions
- Changing the observation space, action space, score formula or reward default without updating `ARCHITECTURE-ESSENTIALS.md` (and §2 of this file and `ARCHITECTURE.md`).
- Adding game mechanics not in the PRD rules. Raise an open question in `PRD.md` instead.
- Importing against the dependency direction or introducing import cycles.
- Silently ignoring illegal actions, or clipping them to a legal one.
- Using unseeded randomness, wall-clock time or hash ordering in `engine/`.
- Starting RL work before the engine gate passes.
- Reporting shaped reward as a success metric. Metrics use raw score and rounds.
- Deleting or rewriting files in `models/`, `logs/` or `data/` that belong to other runs.
- Adding dependencies without listing them, with a justification, in `ARCHITECTURE.md` §1.
