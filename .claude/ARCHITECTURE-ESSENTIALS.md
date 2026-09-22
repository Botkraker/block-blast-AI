# ARCHITECTURE-ESSENTIALS

These are the non-negotiable invariants. Changing any of them requires updating this file, `ARCHITECTURE.md` and `AGENTS.md` in the same commit. Full design: `ARCHITECTURE.md`.

## Game rules (engine)
- 8x8 board, bitboard `int` with bit `r*8+c`. The hand has 3 slots and each slot holds a `piece_id` or `None` (used). No rotation.
- After each placement, full rows and columns are detected and then cleared **simultaneously**.
- When all 3 slots are used, a new hand of 3 is dealt and `round_index += 1`.
- Game over ⇔ no remaining hand piece fits anywhere.
- Only randomness: `PieceDealer` with an injected `np.random.Generator`. Same seed + same actions ⇒ identical game.

## Action space
- `Discrete(192)`, `a = slot·64 + row·8 + col`, `slot ∈ {0,1,2}`, `row, col ∈ {0..7}`.
- `(row, col)` = board cell for the piece's bounding-box top-left.
- Mask `(192,)` bool via `BlockBlastEnv.action_masks()`. An illegal action raises `InvalidMoveError`.

## Observation
- `Box(0.0, 1.0, shape=(5, 8, 8), dtype=float32)`, channel-first.
- ch0 board occupancy · ch1–3 piece mask of slots 0–2 anchored at `(0,0)` (zeros if used) · ch4 constant `min(combo_streak, 8)/8`.

## Score and reward
```
Δscore_t = n_cells + 10 · n_lines · (1 + s)      # s = combo streak before the move
s ← s + 1 if n_lines ≥ 1 else 0
r_t = Δscore_t / 10 − 5 · 1[terminated_t]         # truncation not penalized
```
Evaluation metrics use raw game score and rounds survived, never reward.

## Module boundaries and dependency direction
```
engine ← visualization ← env ← agents ← {training, evaluation, app} ← scripts
engine ← capture            utils ← {training, evaluation, scripts}
```
| Package | Owns | Must not |
|---|---|---|
| `engine` | Rules, scoring, dealing, state | Import anything outside numpy/stdlib |
| `env` | Gym API, obs, action, mask, reward | Implement game rules |
| `agents` | Policies and baselines. `AfterstatePolicy` mirrors the placement, clear and score rules in torch, locked to the engine by `tests/agents/test_afterstate.py` | Mutate env or engine state. Change the rule mirror without that test passing |
| `app` | Graphical game | Change game rules (the engine is the source of truth) |
| `training` / `evaluation` | Loops, metrics, I/O | Redefine spaces or rewards |
| `utils` | Config, seeding, logging | Import `blockblast.*` |

## Seeds
Training base seeds `< 1_000_000`. Evaluation seeds `1_000_000 … 1_000_999` (1,000 episodes, deterministic policy).
