"""Headless GUI tests (SDL dummy video driver)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from blockblast.agents import GreedyAgent  # noqa: E402
from blockblast.app import PlaySession  # noqa: E402
from blockblast.app import theme as t  # noqa: E402
from blockblast.app.game_app import AppOptions, BlockBlastApp  # noqa: E402


def make_app(**opts: object) -> BlockBlastApp:
    return BlockBlastApp(PlaySession(seed=5), GreedyAgent(), AppOptions(**opts))  # type: ignore[arg-type]


def mouse(kind: int, pos: tuple[int, int]) -> pygame.event.Event:
    if kind == pygame.MOUSEMOTION:
        return pygame.event.Event(kind, pos=pos, rel=(0, 0), buttons=(1, 0, 0))
    return pygame.event.Event(kind, pos=pos, button=1)


def test_drag_and_drop_places_piece(tmp_path: Path) -> None:
    app = make_app()
    s = app.session
    slot = 0
    r, c = map(int, np.argwhere(s.game.legal_moves()[slot])[0])
    app.handle_event(mouse(pygame.MOUSEBUTTONDOWN, app._slot_rect(slot).center))
    assert app.drag is not None
    gx, gy = app.drag.grab
    target = (int(t.BOARD_X + c * t.CELL + gx), int(t.BOARD_Y + r * t.CELL + gy))
    app.handle_event(mouse(pygame.MOUSEMOTION, target))
    assert app._drag_target() == (r, c)
    app.draw()
    app.handle_event(mouse(pygame.MOUSEBUTTONUP, target))
    assert s.state.moves == 1 and s.state.hand[slot] is None and app.drag is None
    app.save_screenshot(tmp_path / "shot.png")
    assert (tmp_path / "shot.png").stat().st_size > 0


def test_drop_outside_board_cancels() -> None:
    app = make_app()
    app.handle_event(mouse(pygame.MOUSEBUTTONDOWN, app._slot_rect(1).center))
    app.handle_event(mouse(pygame.MOUSEMOTION, (5, 5)))
    app.handle_event(mouse(pygame.MOUSEBUTTONUP, (5, 5)))
    assert app.session.state.moves == 0 and app.drag is None


def test_autoplay_until_game_over_then_restart() -> None:
    app = make_app(autoplay=True, speed=4)
    now = 0
    while not app.session.state.game_over and now < 600_000:
        now += 16
        app.update(now)
        app.draw()
    assert app.session.state.game_over and app.session.state.moves > 5
    app.update(now + 1000)
    app.draw()
    assert app._overlay_visible()
    app.handle_event(mouse(pygame.MOUSEBUTTONDOWN, (240, 400)))  # click restarts
    assert app.session.state.moves == 0


def test_loop_mode_restarts_automatically() -> None:
    app = make_app(autoplay=True, loop=True, speed=4)
    now = 0
    while not app.session.state.game_over:
        now += 16
        app.update(now)
    app.update(now + 3000)
    assert not app.session.state.game_over and app.session.state.moves == 0


def test_keys() -> None:
    app = make_app()
    key = lambda k: app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k))  # noqa: E731
    key(pygame.K_h)
    assert app.hint is not None
    app.draw()
    key(pygame.K_a)
    assert app.ai_on
    key(pygame.K_EQUALS)
    key(pygame.K_MINUS)
    key(pygame.K_MINUS)
    assert app.speed == 0
    key(pygame.K_SPACE)
    app.draw()
    assert app.paused
    key(pygame.K_r)
    assert app.session.state.moves == 0
    key(pygame.K_ESCAPE)
    assert not app.running
