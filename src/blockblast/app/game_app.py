"""Pygame front end: play Block Blast with the mouse, ask the AI for hints, or watch it play."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame

from blockblast.agents.base import Agent
from blockblast.app import theme as t
from blockblast.app.session import PlaySession
from blockblast.engine.board import LINE_MASKS
from blockblast.engine.pieces import get_piece

AI_DELAYS_S = (1.2, 0.6, 0.3, 0.12, 0.04)
CLEAR_MS = 320
POPUP_MS = 900
POP_IN_MS = 220
GAME_OVER_DELAY_MS = 500
LOOP_RESTART_MS = 2500


@dataclass
class _Drag:
    slot: int
    grab: tuple[float, float]  # cursor offset from the piece's top-left, board-cell pixels
    pos: tuple[int, int]


@dataclass
class _ClearAnim:
    start: int
    cells: list[tuple[int, int, t.Color]]


@dataclass
class _Popup:
    start: int
    text: str
    color: t.Color
    pos: tuple[int, int]
    size: str = "mid"


@dataclass
class _Pending:
    move: tuple[int, int, int]
    at: int


@dataclass
class AppOptions:
    autoplay: bool = False
    loop: bool = False
    seed: int | None = None
    agent_label: str = "AI"
    speed: int = 1


class BlockBlastApp:
    def __init__(self, session: PlaySession, agent: Agent | None, options: AppOptions) -> None:
        pygame.init()
        pygame.display.set_caption("Block Blast AI")
        self.screen = pygame.display.set_mode((t.WIDTH, t.HEIGHT))
        self.clock = pygame.time.Clock()
        self.session = session
        self.agent = agent
        self.opt = options
        self.ai_on = options.autoplay and agent is not None
        self.paused = False
        self.speed = max(0, min(options.speed, len(AI_DELAYS_S) - 1))
        self.running = True
        self.now = 0
        self.drag: _Drag | None = None
        self.hint: tuple[int, int, int] | None = None
        self.pending: _Pending | None = None
        self.ai_next = 0
        self.anims: list[_ClearAnim] = []
        self.popups: list[_Popup] = []
        self.hand_pop_at = 0
        self.over_at: int | None = None
        self._blocks: dict[tuple[t.Color, int, int], pygame.Surface] = {}
        self._fonts = {
            "big": pygame.font.SysFont("segoeui,arial", 58, bold=True),
            "mid": pygame.font.SysFont("segoeui,arial", 30, bold=True),
            "small": pygame.font.SysFont("segoeui,arial", 18, bold=True),
            "tiny": pygame.font.SysFont("segoeui,arial", 15),
        }
        self._bg = self._gradient()

    # ------------------------------------------------------------------ loop

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self.update(pygame.time.get_ticks())
            self.draw()
            pygame.display.flip()
            self.clock.tick(t.FPS)
        pygame.quit()

    def restart(self) -> None:
        self.session.reset(self.opt.seed)
        self.drag = self.hint = self.pending = None
        self.anims.clear()
        self.popups.clear()
        self.over_at = None
        self.hand_pop_at = self.now
        self.ai_next = self.now + 400

    # ---------------------------------------------------------------- input

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            self._on_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._on_press(event.pos)
        elif event.type == pygame.MOUSEMOTION and self.drag is not None:
            self.drag.pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.drag is not None:
            self._on_release()

    def _on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_r:
            self.restart()
        elif key == pygame.K_h and self.agent is not None and not self.session.state.game_over:
            self.hint = self.session.agent_action(self.agent)
        elif key == pygame.K_a and self.agent is not None:
            self.ai_on = not self.ai_on
            self.drag = self.pending = None
            self.ai_next = self.now + 300
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(self.speed + 1, len(AI_DELAYS_S) - 1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(self.speed - 1, 0)

    def _on_press(self, pos: tuple[int, int]) -> None:
        if self._overlay_visible():
            self.restart()
            return
        if self.ai_on or self.session.state.game_over:
            return
        for slot in range(3):
            pid = self.session.state.hand[slot]
            if pid is None or not self._slot_rect(slot).collidepoint(pos):
                continue
            piece = get_piece(pid)
            x0, y0 = self._tray_origin(slot, piece.width, piece.height, t.TRAY_CELL)
            gx = min(max((pos[0] - x0) / t.TRAY_CELL, 0.0), float(piece.width))
            gy = min(max((pos[1] - y0) / t.TRAY_CELL, 0.0), float(piece.height))
            self.drag = _Drag(slot, (gx * t.CELL, gy * t.CELL), pos)
            self.hint = None
            return

    def _on_release(self) -> None:
        assert self.drag is not None
        slot = self.drag.slot
        target = self._drag_target()
        self.drag = None
        if target is not None and self.session.is_legal(slot, *target):
            self.apply_move(slot, *target)

    def _drag_target(self) -> tuple[int, int] | None:
        if self.drag is None:
            return None
        x = self.drag.pos[0] - self.drag.grab[0]
        y = self.drag.pos[1] - self.drag.grab[1]
        col = round((x - t.BOARD_X) / t.CELL)
        row = round((y - t.BOARD_Y) / t.CELL)
        return (row, col) if 0 <= row < 8 and 0 <= col < 8 else None

    # ---------------------------------------------------------------- logic

    def apply_move(self, slot: int, row: int, col: int) -> None:
        """Place a piece, queueing clear animation and score popups."""
        preview = self.session.preview(slot, row, col)
        assert preview is not None
        pid = self.session.state.hand[slot]
        assert pid is not None
        piece = get_piece(pid)
        colors = self.session.colors.copy()
        for dr, dc in piece.cells:
            colors[row + dr, col + dc] = pid
        cleared: set[tuple[int, int]] = set()
        for line in preview.cleared_lines:
            mask = LINE_MASKS[line]
            cleared |= {(i // 8, i % 8) for i in range(64) if mask >> i & 1}
        result = self.session.place(slot, row, col)
        if cleared:
            cells = [(r, c, t.PIECE_COLORS[int(colors[r, c])]) for r, c in sorted(cleared)]
            self.anims.append(_ClearAnim(self.now, cells))
        cx = t.BOARD_X + int((col + piece.width / 2) * t.CELL)
        cy = t.BOARD_Y + int((row + piece.height / 2) * t.CELL)
        if not (self.ai_on and self.speed >= 3):  # fast autoplay: keep the board readable
            self.popups.append(
                _Popup(self.now, f"+{result.score_delta}", t.TEXT, (cx, cy), "small")
            )
        if result.n_lines >= 2:
            self.popups.append(
                _Popup(self.now, f"{result.n_lines} LINES!", t.GOLD, (t.WIDTH // 2, 340))
            )
        if result.n_lines and result.state.combo_streak >= 2:
            self.popups.append(
                _Popup(self.now, f"COMBO x{result.state.combo_streak}", t.GOLD, (t.WIDTH // 2, 400))
            )
        if result.new_hand_dealt:
            self.hand_pop_at = self.now
        self.hint = None
        if result.state.game_over:
            self.over_at = self.now

    def update(self, now: int) -> None:
        self.now = now
        self.anims = [a for a in self.anims if now - a.start < CLEAR_MS]
        self.popups = [p for p in self.popups if now - p.start < POPUP_MS]
        state = self.session.state
        if state.game_over:
            if self.opt.loop and self.over_at is not None and now - self.over_at > LOOP_RESTART_MS:
                self.restart()
            return
        if not self.ai_on or self.paused or self.agent is None:
            return
        delay = AI_DELAYS_S[self.speed] * 1000
        if self.pending is None and now >= self.ai_next:
            self.pending = _Pending(self.session.agent_action(self.agent), now + int(delay * 0.55))
        elif self.pending is not None and now >= self.pending.at:
            self.apply_move(*self.pending.move)
            self.pending = None
            self.ai_next = now + int(delay * 0.45)

    def _overlay_visible(self) -> bool:
        return (
            self.session.state.game_over
            and self.over_at is not None
            and self.now - self.over_at >= GAME_OVER_DELAY_MS
            and not self.anims
        )

    # --------------------------------------------------------------- drawing

    def draw(self) -> None:
        s = self.screen
        s.blit(self._bg, (0, 0))
        self._draw_header()
        self._draw_board()
        if self.hint is not None:
            self._draw_ghost(*self.hint, t.HINT, pulse=True)
        if self.pending is not None:
            self._draw_ghost(*self.pending.move, None, pulse=False)
        if self.drag is not None:
            target = self._drag_target()
            if target is not None and self.session.is_legal(self.drag.slot, *target):
                self._draw_ghost(self.drag.slot, *target, None, pulse=False)
        self._draw_clear_anims()
        self._draw_tray()
        if self.drag is not None:
            pid = self.session.state.hand[self.drag.slot]
            assert pid is not None
            x = int(self.drag.pos[0] - self.drag.grab[0])
            y = int(self.drag.pos[1] - self.drag.grab[1])
            self._draw_piece(pid, x, y, t.CELL)
        self._draw_popups()
        self._draw_footer()
        if self._overlay_visible():
            self._draw_game_over()
        elif self.ai_on and self.paused:
            self._banner("PAUSED", "Space to resume")

    def save_screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(path))

    def _text(self, text: str, font: str, color: t.Color, center: tuple[int, int]) -> None:
        surf = self._fonts[font].render(text, True, color)
        self.screen.blit(surf, surf.get_rect(center=center))

    def _draw_header(self) -> None:
        st = self.session.state
        best = self._fonts["small"].render(
            f"BEST  {max(self.session.best, st.score)}", True, t.GOLD
        )
        self.screen.blit(best, (20, 20))
        mode = f"{self.opt.agent_label} PLAYING" if self.ai_on else "YOU"
        label = self._fonts["small"].render(mode, True, t.TEXT_DIM)
        self.screen.blit(label, label.get_rect(topright=(t.WIDTH - 20, 20)))
        self._text(f"{st.score}", "big", t.TEXT, (t.WIDTH // 2, 88))
        if st.combo_streak >= 1:
            self._text(f"Combo x{st.combo_streak}", "small", t.GOLD, (t.WIDTH // 2, 136))
        else:
            self._text(f"Round {st.round_index}", "small", t.TEXT_DIM, (t.WIDTH // 2, 136))

    def _draw_board(self) -> None:
        frame = pygame.Rect(
            t.BOARD_X - t.BOARD_PAD,
            t.BOARD_Y - t.BOARD_PAD,
            8 * t.CELL + 2 * t.BOARD_PAD,
            8 * t.CELL + 2 * t.BOARD_PAD,
        )
        pygame.draw.rect(self.screen, t.BOARD_BG, frame, border_radius=14)
        colors = self.session.colors
        for r in range(8):
            for c in range(8):
                x, y = t.BOARD_X + c * t.CELL, t.BOARD_Y + r * t.CELL
                pid = int(colors[r, c])
                if pid >= 0:
                    self._blit_block(x, y, t.CELL, t.PIECE_COLORS[pid])
                else:
                    cell = pygame.Rect(x + 2, y + 2, t.CELL - 4, t.CELL - 4)
                    pygame.draw.rect(self.screen, t.EMPTY_CELL, cell, border_radius=6)

    def _draw_ghost(
        self, slot: int, row: int, col: int, color: t.Color | None, pulse: bool
    ) -> None:
        """Translucent piece at the target plus a glow on every line it would clear."""
        preview = self.session.preview(slot, row, col)
        pid = self.session.state.hand[slot]
        if preview is None or pid is None:
            return
        base = color or t.PIECE_COLORS[pid]
        alpha = 150
        if pulse:
            alpha = 90 + int(80 * abs(((self.now // 8) % 100) - 50) / 50)
        glow = pygame.Surface((t.CELL, t.CELL), pygame.SRCALPHA)
        glow.fill((*t.shade(t.PIECE_COLORS[pid], 1.3), 110))
        for line in preview.cleared_lines:
            for i in range(64):
                if LINE_MASKS[line] >> i & 1:
                    self.screen.blit(
                        glow, (t.BOARD_X + (i % 8) * t.CELL, t.BOARD_Y + (i // 8) * t.CELL)
                    )
        for dr, dc in get_piece(pid).cells:
            x = t.BOARD_X + (col + dc) * t.CELL
            y = t.BOARD_Y + (row + dr) * t.CELL
            self._blit_block(x, y, t.CELL, base, alpha)
            if pulse:
                outline = pygame.Rect(x + 2, y + 2, t.CELL - 4, t.CELL - 4)
                pygame.draw.rect(self.screen, t.HINT, outline, width=3, border_radius=8)

    def _draw_clear_anims(self) -> None:
        for anim in self.anims:
            p = (self.now - anim.start) / CLEAR_MS
            size = max(4, int(t.CELL * (1.0 - 0.7 * p)))
            alpha = max(0, int(255 * (1.0 - p)))
            for r, c, color in anim.cells:
                x = t.BOARD_X + c * t.CELL + (t.CELL - size) // 2
                y = t.BOARD_Y + r * t.CELL + (t.CELL - size) // 2
                self._blit_block(x, y, size, t.shade(color, 1.0 + 0.6 * (1.0 - p)), alpha)

    def _slot_rect(self, slot: int) -> pygame.Rect:
        w = t.WIDTH // 3
        return pygame.Rect(slot * w, t.TRAY_Y, w, t.TRAY_H)

    def _tray_origin(self, slot: int, width: int, height: int, cell: int) -> tuple[int, int]:
        rect = self._slot_rect(slot)
        return rect.centerx - width * cell // 2, rect.centery - height * cell // 2

    def _draw_tray(self) -> None:
        tray = pygame.Rect(12, t.TRAY_Y, t.WIDTH - 24, t.TRAY_H)
        pygame.draw.rect(self.screen, (*t.BOARD_BG,), tray, border_radius=16)
        grow = min(1.0, (self.now - self.hand_pop_at) / POP_IN_MS)
        cell = max(4, int(t.TRAY_CELL * (0.55 + 0.45 * grow)))
        for slot, pid in enumerate(self.session.state.hand):
            if pid is None or (self.drag is not None and self.drag.slot == slot):
                continue
            piece = get_piece(pid)
            x, y = self._tray_origin(slot, piece.width, piece.height, cell)
            playable = bool(self.session.game.legal_moves()[slot].any())
            self._draw_piece(pid, x, y, cell, alpha=255 if playable else 90)

    def _draw_piece(self, pid: int, x: int, y: int, cell: int, alpha: int = 255) -> None:
        color = t.PIECE_COLORS[pid]
        for dr, dc in get_piece(pid).cells:
            self._blit_block(x + dc * cell, y + dr * cell, cell, color, alpha)

    def _draw_popups(self) -> None:
        for p in self.popups:
            k = (self.now - p.start) / POPUP_MS
            surf = self._fonts[p.size].render(p.text, True, p.color)
            surf.set_alpha(int(255 * (1.0 - k * k)))
            self.screen.blit(surf, surf.get_rect(center=(p.pos[0], p.pos[1] - int(50 * k))))

    def _draw_footer(self) -> None:
        if self.ai_on:
            speed = f"speed {self.speed + 1}/{len(AI_DELAYS_S)}"
            help_text = f"Space pause · +/- {speed} · A take over · R restart"
        elif self.agent is not None:
            help_text = "Drag pieces · H hint · A let the AI play · R restart"
        else:
            help_text = "Drag pieces onto the board · R restart · Esc quit"
        self._text(help_text, "tiny", t.TEXT_DIM, (t.WIDTH // 2, t.HEIGHT - 22))

    def _banner(self, title: str, subtitle: str) -> None:
        shade = pygame.Surface((t.WIDTH, t.HEIGHT), pygame.SRCALPHA)
        shade.fill((8, 10, 30, 170))
        self.screen.blit(shade, (0, 0))
        self._text(title, "big", t.TEXT, (t.WIDTH // 2, t.HEIGHT // 2 - 60))
        self._text(subtitle, "small", t.TEXT_DIM, (t.WIDTH // 2, t.HEIGHT // 2 + 70))

    def _draw_game_over(self) -> None:
        st = self.session.state
        self._banner("GAME OVER", "Click or press R to play again")
        self._text(f"Score {st.score}", "mid", t.TEXT, (t.WIDTH // 2, t.HEIGHT // 2 + 2))
        self._text(f"Best {self.session.best}", "small", t.GOLD, (t.WIDTH // 2, t.HEIGHT // 2 + 36))

    def _blit_block(self, x: int, y: int, size: int, color: t.Color, alpha: int = 255) -> None:
        key = (color, size, alpha // 8 * 8)
        surf = self._blocks.get(key)
        if surf is None:
            if len(self._blocks) > 4000:
                self._blocks.clear()
            surf = self._render_block(size, color)
            surf.set_alpha(key[2] if alpha < 255 else 255)
            self._blocks[key] = surf
        self.screen.blit(surf, (x, y))

    @staticmethod
    def _render_block(size: int, color: t.Color) -> pygame.Surface:
        """Beveled block: dark rim at the bottom, main face, light gloss stripe."""
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        radius = max(2, size // 6)
        body = pygame.Rect(1, 1, size - 2, size - 2)
        pygame.draw.rect(surf, t.shade(color, 0.62), body, border_radius=radius)
        face = pygame.Rect(1, 1, size - 2, size - 2 - max(2, size // 9))
        pygame.draw.rect(surf, color, face, border_radius=radius)
        inner = face.inflate(-max(2, size // 5), -max(2, size // 5))
        pygame.draw.rect(surf, t.shade(color, 1.12), inner, border_radius=max(1, radius - 2))
        gloss = pygame.Rect(int(size * 0.2), int(size * 0.14), int(size * 0.45), max(2, size // 10))
        pygame.draw.rect(surf, t.shade(color, 1.6), gloss, border_radius=gloss.h // 2)
        return surf

    @staticmethod
    def _gradient() -> pygame.Surface:
        surf = pygame.Surface((t.WIDTH, t.HEIGHT))
        for y in range(t.HEIGHT):
            k = y / t.HEIGHT
            color = tuple(int(a + (b - a) * k) for a, b in zip(t.BG_TOP, t.BG_BOTTOM, strict=True))
            pygame.draw.line(surf, color, (0, y), (t.WIDTH, y))
        return surf
