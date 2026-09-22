"""Regenerate the README images in docs/images/ (screenshots, gameplay GIF, charts).

    uv run --with matplotlib python scripts/make_readme_assets.py

matplotlib is only needed here, so it is not a project dependency.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio.v3 as iio
import numpy as np
import pygame

from blockblast.agents import Agent, GreedyAgent
from blockblast.app import PlaySession
from blockblast.app import theme as t
from blockblast.app.game_app import AppOptions, BlockBlastApp
from blockblast.engine.board import to_array
from blockblast.engine.pieces import get_piece
from blockblast.env.observation import encode_observation

OUT = Path("docs/images")
CHECKPOINTS = (
    Path("models/ppo_safe/final.zip"),
    Path("models/ppo_v2/best_model.zip"),
    Path("models/ppo_v1/best_model.zip"),
)

# Chart palette: reference categorical slots 1-2 (validated pair) + neutral baselines.
MODES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "s1": "#2a78d6",
        "s2": "#eb6834",
        "s3": "#1a9e6f",
        "base": "#a8a7a2",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a37",
        "s1": "#3987e5",
        "s2": "#d95926",
        "s3": "#2bb98a",
        "base": "#6f6e69",
    },
}


def _publish(tmp: Path, dest: Path) -> None:
    """Move ``tmp`` onto ``dest``; retries because Windows viewers briefly lock open images."""
    import time

    for _ in range(20):
        try:
            os.replace(tmp, dest)
            return
        except OSError:
            time.sleep(0.5)
    raise OSError(f"{dest} is locked (close any program showing it) - new image left at {tmp}")


def save_image(dest: Path, img: np.ndarray, **kwargs: object) -> None:
    tmp = dest.with_name(f"_tmp_{dest.name}")
    iio.imwrite(tmp, img, **kwargs)
    _publish(tmp, dest)


def save_fig(fig, dest: Path, facecolor: str) -> None:  # type: ignore[no-untyped-def]
    tmp = dest.with_name(f"_tmp_{dest.name}")
    fig.savefig(tmp, facecolor=facecolor)
    _publish(tmp, dest)


def load_agent() -> tuple[Agent, str]:
    for path in CHECKPOINTS:
        if path.exists():
            from blockblast.agents.ppo_agent import PPOAgent

            return PPOAgent.load(path, device="cpu"), "PPO"
    return GreedyAgent(), "GREEDY"


def frame(app: BlockBlastApp) -> np.ndarray:
    return np.transpose(pygame.surfarray.array3d(app.screen), (1, 0, 2))


def downscale(img: np.ndarray, width: int) -> np.ndarray:
    surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    h = int(img.shape[0] * width / img.shape[1])
    small = pygame.transform.smoothscale(surf, (width, h))
    return np.transpose(pygame.surfarray.array3d(small), (1, 0, 2))


def pick_showcase_seed(agent: Agent, max_moves: int = 220) -> int:
    """Held-out seed where the agent plays its best game that still fits in the GIF."""
    best_seed, best_score = 1_000_000, -1
    for seed in range(1_000_000, 1_000_040):
        s = PlaySession(seed=seed)
        while not s.state.game_over and s.state.moves <= max_moves:
            s.place(*s.agent_action(agent))
        if s.state.game_over and s.state.score > best_score:
            best_seed, best_score = seed, s.state.score
    print(f"showcase seed {best_seed}: score {best_score}")
    return best_seed


def gameplay_gif(agent: Agent, label: str) -> None:
    seed = pick_showcase_seed(agent)
    app = BlockBlastApp(PlaySession(seed=seed), agent, AppOptions(True, False, seed, label, 3))
    frames, now = [], 0
    while len(frames) < 600 and not app.session.state.game_over:
        for _ in range(6):  # 100 ms per GIF frame
            now += 16
            app.update(now)
        app.draw()
        frames.append(downscale(frame(app), 270))
    for _ in range(30):  # hold the game-over screen
        now += 100
        app.update(now)
        app.draw()
        frames.append(downscale(frame(app), 270))
    save_image(OUT / "gameplay.gif", np.stack(frames), duration=100, loop=0)


def screenshots(agent: Agent, label: str) -> list[np.ndarray]:
    shots = []
    # 1) You play: drag a piece over a move that clears a line (glow preview).
    s = PlaySession(seed=21)
    app = BlockBlastApp(s, GreedyAgent(), AppOptions(True, False, None, label, 1))
    now = 1000
    while True:
        now += 16
        app.update(now)
        move = s.agent_action(GreedyAgent())
        preview = s.preview(*move)
        if (
            preview is not None and preview.n_lines > 0 and s.state.moves > 8
        ) or s.state.moves > 40:
            break
    app.ai_on, app.pending = False, None
    now += 2000
    app.update(now)
    slot, r, c = move
    app._on_press(app._slot_rect(slot).center)
    d = app.drag
    assert d is not None
    d.pos = (
        int(t.BOARD_X + c * t.CELL + d.grab[0]) + 8,
        int(t.BOARD_Y + r * t.CELL + d.grab[1]) - 6,
    )
    app.draw()
    shots.append(frame(app))
    # 2) AI hint.
    s = PlaySession(seed=33)
    app = BlockBlastApp(s, agent, AppOptions(True, False, None, label, 1))
    now = 1000
    while s.state.moves < 10:
        now += 16
        app.update(now)
    app.ai_on, app.pending = False, None
    now += 2000
    app.update(now)
    app.hint = s.agent_action(agent)
    app.now += 800 - app.now % 800  # (now // 8) % 100 == 0: brightest point of the hint pulse
    app.draw()
    shots.append(frame(app))
    # 3) Game over.
    s = PlaySession(seed=5)
    app = BlockBlastApp(s, GreedyAgent(), AppOptions(True, False, None, "GREEDY", 4))
    now = 0
    while not s.state.game_over:
        now += 16
        app.update(now)
    app.update(now + 2000)
    app.draw()
    shots.append(frame(app))
    for name, img in zip(("play", "hint", "game_over"), shots, strict=True):
        save_image(OUT / f"screenshot_{name}.png", img)
    return shots


def feature_strip(shots: list[np.ndarray]) -> None:
    small = [downscale(img, 280) for img in shots]
    gap = np.full((small[0].shape[0], 16, 3), 255, dtype=np.uint8)
    save_image(
        OUT / "features.png", np.concatenate([small[0], gap, small[1], gap, small[2]], axis=1)
    )


def _style(ax, m: dict[str, str]) -> None:  # type: ignore[no-untyped-def]
    ax.set_facecolor(m["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(m["grid"])
    ax.tick_params(colors=m["muted"], length=0)
    ax.yaxis.grid(True, color=m["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def results_chart(plt) -> None:  # type: ignore[no-untyped-def]
    rows = []
    for label, file in [
        ("Random", "random_default"),
        ("Greedy", "greedy_default"),
        ("PPO v1\n(CNN policy)", "maskable_ppo_ppo_v1_best_model"),
        ("PPO v2\n(afterstate)", "maskable_ppo_ppo_v2_best"),
        ("PPO v3\n(survival)", "maskable_ppo_ppo_safe40_final"),
    ]:
        p = Path(f"data/eval/{file}.json")
        if p.exists():
            summary = json.loads(p.read_text(encoding="utf-8"))["summary"]
            rows.append(
                (label, summary["score_mean"], summary["score_ci_lo"], summary["score_ci_hi"])
            )
    greedy = next(v for lab, v, _, _ in rows if lab == "Greedy")
    for mode, m in MODES.items():
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        fig.patch.set_facecolor(m["surface"])
        _style(ax, m)
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colors = [
            m["s3"]
            if n.startswith("PPO v3")
            else m["s1"]
            if n.startswith("PPO v2")
            else m["s2"]
            if n.startswith("PPO v1")
            else m["base"]
            for n in names
        ]
        bars = ax.bar(names, vals, width=0.56, color=colors, edgecolor=m["surface"], linewidth=2)
        err = np.array([[v - lo for _, v, lo, _ in rows], [hi - v for _, v, _, hi in rows]])
        ax.errorbar(
            range(len(rows)), vals, yerr=err, fmt="none", ecolor=m["muted"], capsize=4, lw=1
        )
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + max(vals) * 0.04,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                color=m["text"],
                fontsize=11,
                fontweight="bold",
            )
        target = 1.5 * greedy
        ax.axhline(target, color=m["muted"], lw=1.2, ls=(0, (4, 3)))
        ax.text(
            -0.45,
            target,
            f"target 1.5× greedy = {target:.0f}",
            va="bottom",
            ha="left",
            color=m["muted"],
            fontsize=9,
        )
        ax.set_ylim(0, max(max(vals), target) * 1.25)
        ax.set_ylabel("Mean score (1,000 held-out games)", color=m["muted"], fontsize=9)
        ax.tick_params(axis="x", colors=m["text"], labelsize=10)
        fig.tight_layout()
        save_fig(fig, OUT / f"results_{mode}.png", m["surface"])
        plt.close(fig)


def _scalars(run: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # One event file per training session (pause/resume). Stitch them in time order; a
    # resumed session restarts from its checkpoint, so it replaces any later points of
    # the previous session.
    files = sorted(
        glob.glob(f"logs/{run}/**/events.out.tfevents.*", recursive=True), key=os.path.getmtime
    )
    points: list[tuple[int, float]] = []
    for f in files:
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in acc.Tags()["scalars"]:
            continue
        new = [(p.step, p.value) for p in acc.Scalars(tag)]
        if new:
            points = [p for p in points if p[0] < new[0][0]] + new
    if not points:
        return None
    return np.array([s for s, _ in points]) / 1e6, np.array([v for _, v in points])


def training_chart(plt) -> None:  # type: ignore[no-untyped-def]
    greedy = json.loads(Path("data/eval/greedy_default.json").read_text(encoding="utf-8"))
    greedy_mean = greedy["summary"]["score_mean"]
    series = [
        ("PPO v1 · CNN policy", "ppo_v1", "s2"),
        ("PPO v2 · afterstate policy", "ppo_v2", "s1"),
        ("PPO v3 · survival reward", "ppo_safe", "s3"),
    ]
    for mode, m in MODES.items():
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        fig.patch.set_facecolor(m["surface"])
        _style(ax, m)
        for label, run, key in series:
            data = _scalars(run, "game/score")
            if data is None:
                continue
            x, y = data
            y_smooth = np.empty_like(y)  # exponential moving average
            acc = y[0]
            for i, v in enumerate(y):
                acc = 0.9 * acc + 0.1 * v
                y_smooth[i] = acc
            x_smooth = x
            ax.plot(x_smooth, y_smooth, color=m[key], lw=2, label=label)
            ax.text(
                x_smooth[-1],
                y_smooth[-1],
                f"  {label.split(' · ')[0]}",
                color=m["text"],
                va="center",
                fontsize=9,
            )
        ax.axhline(greedy_mean, color=m["muted"], lw=1.2, ls=(0, (4, 3)))
        ax.text(
            ax.get_xlim()[1] * 0.97,
            greedy_mean,
            f"greedy baseline ({greedy_mean:.0f})",
            va="bottom",
            ha="right",
            color=m["muted"],
            fontsize=9,
        )
        ax.set_xlabel("Environment steps (millions)", color=m["muted"], fontsize=9)
        ax.set_ylabel("Training game score (rolling mean)", color=m["muted"], fontsize=9)
        legend = ax.legend(frameon=False, loc="upper left", fontsize=9)
        for text in legend.get_texts():
            text.set_color(m["text"])
        ax.margins(x=0.12)
        fig.tight_layout()
        save_fig(fig, OUT / f"training_{mode}.png", m["surface"])
        plt.close(fig)


def how_it_sees(plt) -> None:  # type: ignore[no-untyped-def]
    """Observation channels and one afterstate, drawn with the game's own colors."""
    s = PlaySession(seed=21)
    agent = GreedyAgent()
    while s.state.moves < 12:
        s.place(*s.agent_action(agent))
    state = s.state
    obs = encode_observation(state)
    for mode, m in MODES.items():
        fig, axes = plt.subplots(1, 5, figsize=(9.6, 2.5), dpi=150)
        fig.patch.set_facecolor(m["surface"])
        titles = ["board", "piece slot 1", "piece slot 2", "piece slot 3", "combo"]
        empty = np.array([0.16, 0.19, 0.35]) if mode == "dark" else np.array([0.88, 0.89, 0.93])
        for i, ax in enumerate(axes):
            img = np.tile(empty, (8, 8, 1))
            blue = np.array([0.34, 0.61, 1.0])
            if i == 4:  # constant plane: shade by the normalized combo value
                v = float(obs[4, 0, 0])
                img[:] = empty * (1 - v) + blue * v
                titles[4] = f"combo streak ({state.combo_streak})"
            else:
                pid_i = state.hand[i - 1] if i >= 1 else None
                fill = np.array(t.PIECE_COLORS[pid_i]) / 255 if pid_i is not None else blue
                img[obs[i] > 0] = fill if i >= 1 else np.array([0.45, 0.50, 0.64])
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
            ax.grid(which="minor", color=m["surface"], linewidth=1.5)
            ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(titles[i], color=m["text"], fontsize=10)
        fig.tight_layout()
        save_fig(fig, OUT / f"observation_{mode}.png", m["surface"])
        plt.close(fig)

        # afterstate: before, placed, after clears (pick the best-scoring placement)
        best = max(
            (s.preview(sl, r, c) for sl in range(3) for r in range(8) for c in range(8)),
            key=lambda res: -1 if res is None else res.score_delta,
        )
        assert best is not None
        move = next(
            (sl, r, c)
            for sl in range(3)
            for r in range(8)
            for c in range(8)
            if (p := s.preview(sl, r, c)) is not None and p.score_delta == best.score_delta
        )
        sl, r, c = move
        piece = get_piece(state.hand[sl])  # type: ignore[arg-type]
        before = to_array(state.board).astype(float)
        placed = before.copy()
        for dr, dc in piece.cells:
            placed[r + dr, c + dc] = 2
        after = to_array(best.state.board).astype(float)
        slate = np.array([0.45, 0.50, 0.64]) if mode == "light" else np.array([0.52, 0.57, 0.72])
        accent = np.array([0.92, 0.41, 0.20])  # palette slot 2: the new piece
        gold = np.array([0.93, 0.63, 0.0])
        cleared = np.zeros((8, 8), dtype=bool)
        for line in best.cleared_lines:
            if line < 8:
                cleared[line, :] = True
            else:
                cleared[:, line - 8] = True
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), dpi=150)
        fig.patch.set_facecolor(m["surface"])
        cap = [
            "1 · current board",
            f"2 · place {piece.name} at ({r},{c})",
            f"3 · after clear: +{best.score_delta} pts",
        ]
        for k, (ax, grid, title) in enumerate(zip(axes, (before, placed, after), cap, strict=True)):
            img = np.tile(empty, (8, 8, 1))
            img[grid == 1] = slate
            img[grid == 2] = accent
            if k == 1:  # lines about to clear
                img[cleared] = img[cleared] * 0.45 + gold * 0.55
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
            ax.grid(which="minor", color=m["surface"], linewidth=1.5)
            ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(title, color=m["text"], fontsize=10)
        fig.tight_layout()
        save_fig(fig, OUT / f"afterstate_{mode}.png", m["surface"])
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    agent, label = load_agent()
    print(f"agent for screenshots: {label}")
    shots = screenshots(agent, label)
    feature_strip(shots)
    gameplay_gif(agent, label)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_chart(plt)
    training_chart(plt)
    how_it_sees(plt)
    for f in sorted(OUT.iterdir()):
        print(f"{f}  {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
