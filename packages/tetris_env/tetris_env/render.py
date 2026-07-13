"""Human-viewable frame renderer for the custom Tetris engine.

The engine's own ``render()`` returns ANSI text, which is fine for debugging but
not watchable. This module draws an RGB frame (board + ghost piece + HUD) that
both the pure-RL track and the tool-assisted track render into mp4.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .engine import COLS, HIDDEN_ROWS, PIECES, ROWS, PieceState, TetrisGame


CELL = 24
PANEL = 280  # wide enough for the two-line track titles at 14px
WIDTH = COLS * CELL + PANEL
HEIGHT = (ROWS - HIDDEN_ROWS) * CELL

BG = (10, 10, 12)
EMPTY = (26, 26, 31)
# The engine board is binary uint8, so piece identity is not retained once a
# piece locks. The stack gets one neutral colour; only the live piece is tinted.
STACK = (150, 156, 172)
STACK_EDGE = (186, 192, 208)
TEXT = (236, 238, 242)
DIM = (150, 152, 160)

PIECE_COLORS = {
    "I": (64, 200, 232),
    "O": (240, 208, 76),
    "T": (176, 112, 224),
    "S": (96, 208, 120),
    "Z": (232, 92, 100),
    "J": (84, 132, 232),
    "L": (240, 152, 68),
}

_FONTS: dict[int, ImageFont.ImageFont] = {}


def font(size: int):
    if size not in _FONTS:
        try:
            _FONTS[size] = ImageFont.truetype("arial.ttf", size)
        except OSError:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def ghost_state(game: TetrisGame) -> PieceState | None:
    """Where the live piece would land under a hard drop."""
    state = game.current
    if state is None or game.game_over:
        return None
    while True:
        nxt = PieceState(state.name, state.row + 1, state.col, state.rotation)
        if game.collides(nxt):
            return state
        state = nxt


def _cell_box(row: int, col: int) -> tuple[int, int, int, int]:
    x = col * CELL
    y = (row - HIDDEN_ROWS) * CELL
    return x, y, x + CELL - 2, y + CELL - 2


def _draw_next(draw: ImageDraw.ImageDraw, piece: str, x0: int, y0: int) -> None:
    draw.text((x0, y0), "NEXT", fill=DIM, font=font(14))
    color = PIECE_COLORS[piece]
    for row, col in PIECES[piece][0]:
        x = x0 + 4 + col * 18
        y = y0 + 26 + row * 18
        draw.rectangle((x, y, x + 16, y + 16), fill=color)


def render_frame(
    game: TetrisGame,
    *,
    title: str | None = None,
    seed: int | None = None,
    score: int | None = None,
    flash: bool = False,
) -> np.ndarray:
    """Draw one RGB frame of the current game state.

    ``score`` overrides ``game.score`` for the HUD. The tool-assisted track
    scores line clears only (no drop points), so it passes its own running total
    to keep the video consistent with the reported result.
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    for row in range(HIDDEN_ROWS, ROWS):
        for col in range(COLS):
            box = _cell_box(row, col)
            if game.board[row, col]:
                draw.rectangle(box, fill=STACK)
                draw.rectangle((box[0], box[1], box[2], box[1] + 2), fill=STACK_EDGE)
            else:
                draw.rectangle(box, fill=EMPTY)

    ghost = ghost_state(game)
    live = game.current
    if ghost is not None and live is not None:
        color = PIECE_COLORS[live.name]
        faded = tuple(c // 3 for c in color)
        for row, col in game.cells(ghost):
            if row >= HIDDEN_ROWS:
                draw.rectangle(_cell_box(row, col), outline=color, fill=faded, width=1)
        for row, col in game.cells(live):
            if row >= HIDDEN_ROWS:
                box = _cell_box(row, col)
                draw.rectangle(box, fill=color)
                draw.rectangle((box[0], box[1], box[2], box[1] + 3), fill=(255, 255, 255))

    if flash:
        overlay = Image.new("RGBA", (COLS * CELL, HEIGHT), (255, 255, 255, 90))
        img.paste(Image.alpha_composite(img.crop((0, 0, COLS * CELL, HEIGHT)).convert("RGBA"), overlay).convert("RGB"))

    x0 = COLS * CELL + 16
    y = 16
    if title:
        for line in title.split("\n"):
            draw.text((x0, y), line, fill=TEXT, font=font(14))
            y += 19
        y += 6
    if seed is not None:
        draw.text((x0, y), f"seed {seed}", fill=DIM, font=font(13))
        y += 24
    shown_score = game.score if score is None else score
    draw.text((x0, y), f"score {shown_score}", fill=TEXT, font=font(17))
    draw.text((x0, y + 26), f"lines {game.lines}", fill=TEXT, font=font(17))
    draw.text((x0, y + 52), f"pieces {game.pieces}", fill=TEXT, font=font(17))
    _draw_next(draw, game.next_piece, x0, y + 92)
    if game.game_over:
        draw.text((x0, HEIGHT - 34), "GAME OVER", fill=(232, 92, 100), font=font(17))
    return np.asarray(img)
