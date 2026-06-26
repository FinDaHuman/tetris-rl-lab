from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tetris_env import TetrisGame
from tetris_env.engine import COLS, HIDDEN_ROWS, PIECES, ROWS
from agents.custom.tetris_custom_agent import choose_placement, load_weights


CELL = 24
PANEL = 176
COLORS = {
    0: (18, 18, 20),
    1: (64, 190, 230),
}


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_next(draw: ImageDraw.ImageDraw, piece: str, x0: int, y0: int) -> None:
    draw.text((x0, y0), "NEXT", fill=(230, 230, 230), font=font(16))
    shape = PIECES[piece][0]
    for row, col in shape:
        x = x0 + 16 + col * CELL
        y = y0 + 32 + row * CELL
        draw.rectangle((x, y, x + CELL - 2, y + CELL - 2), fill=(245, 204, 80))


def render_frame(game: TetrisGame) -> np.ndarray:
    width = COLS * CELL + PANEL
    height = (ROWS - HIDDEN_ROWS) * CELL
    img = Image.new("RGB", (width, height), (10, 10, 12))
    draw = ImageDraw.Draw(img)
    active = set(game.cells())
    for row in range(HIDDEN_ROWS, ROWS):
        for col in range(COLS):
            x = col * CELL
            y = (row - HIDDEN_ROWS) * CELL
            filled = bool(game.board[row, col]) or (row, col) in active
            fill = COLORS[1] if filled else COLORS[0]
            draw.rectangle((x, y, x + CELL - 2, y + CELL - 2), fill=fill)
    x0 = COLS * CELL + 18
    draw.text((x0, 18), f"score {game.score}", fill=(255, 255, 255), font=font(16))
    draw.text((x0, 42), f"lines {game.lines}", fill=(255, 255, 255), font=font(16))
    draw.text((x0, 66), f"pieces {game.pieces}", fill=(255, 255, 255), font=font(16))
    draw_next(draw, game.next_piece, x0, 104)
    return np.asarray(img)


def render(args: argparse.Namespace) -> None:
    weights = load_weights(args.weights)
    game = TetrisGame(seed=args.seed)
    frames = [render_frame(game)]
    while not game.game_over and game.pieces < args.max_pieces:
        placement = choose_placement(game, weights)
        if placement is None:
            break
        game.board = placement.board.copy()
        game.score += placement.score_delta
        game.lines += placement.lines
        game.level = game.lines // 10
        game.pieces += 1
        game.spawn()
        if game.pieces == 1 or game.pieces % args.frame_every == 0:
            frames.append(render_frame(game))
    frames.append(render_frame(game))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps)
    stats = {"seed": args.seed, "score": game.score, "lines": game.lines, "pieces": game.pieces, "frames": len(frames)}
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))
    print(f"wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/custom_best/best_weights.npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pieces", type=int, default=5000)
    parser.add_argument("--frame-every", type=int, default=25)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out", default="artifacts/custom_best/custom_episode.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
