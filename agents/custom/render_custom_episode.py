from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

from tetris_env import TetrisGame
from tetris_env.render import render_frame
from tetris_env.replay import placement_actions, placement_drop_states

from agents.custom.tetris_custom_agent import choose_placement, load_weights
from agents.video import VideoWriter


TITLE = "Track 4 - custom engine\ntool-assisted (CEM + lookahead)"


def play_and_render(
    weights,
    *,
    seed: int,
    max_pieces: int,
    writer: VideoWriter | None = None,
    title: str = TITLE,
    lookahead_weight: float = 0.35,
    lookahead_candidates: int = 4,
    lookahead_depth: int = 2,
    future_source: str = "queue",
) -> dict:
    """Replay the placement planner, animating each piece into position.

    The planner chooses a final ``Placement``; ``agents/custom/tetris_custom_agent.play_episode``
    commits it by overwriting the board, which is what produced the recorded
    result. To stay bit-identical to that result while still animating, each
    placement is first replayed as primitive actions on a clone
    (``placement_actions``); if that lands on exactly the planner's board we step
    the real game through those actions, otherwise we animate a synthetic fly-in
    and commit the placement directly. Either way the board after each piece is
    the planner's board.

    Score is accumulated from ``placement.score_delta`` (line score only, no drop
    points), matching how the track is evaluated and reported.
    """
    game = TetrisGame(seed=seed)
    score = 0
    fallbacks = 0

    def draw(flash: bool = False, hold: int = 1) -> None:
        if writer is not None:
            writer.append(render_frame(game, title=title, seed=seed, score=score, flash=flash), hold=hold)

    draw(hold=8)
    while not game.game_over and game.pieces < max_pieces:
        placement = choose_placement(
            game,
            weights,
            lookahead_weight=lookahead_weight,
            lookahead_candidates=lookahead_candidates,
            lookahead_depth=lookahead_depth,
            future_source=future_source,
        )
        if placement is None:
            break

        lines_before = game.lines
        actions = placement_actions(game, placement) if writer is not None else None
        if actions is not None:
            for action in actions:
                game.step(action)
                draw()
            # The engine's own scoring includes hard-drop points; keep the
            # reported score on the line-clear-only convention used by the eval.
            score += placement.score_delta
        else:
            if writer is not None:
                fallbacks += 1
                for state in placement_drop_states(game, placement):
                    game.current = state
                    draw()
            game.board = placement.board.copy()
            score += placement.score_delta
            game.lines += placement.lines
            game.level = game.lines // 10
            game.pieces += 1
            game.spawn()

        if game.lines > lines_before:
            draw(flash=True, hold=2)

    draw(hold=20)
    return {
        "seed": seed,
        "score": int(score),
        "lines": int(game.lines),
        "pieces": int(game.pieces),
        "fallbacks": fallbacks,
        "frames": 0 if writer is None else writer.frames,
    }


def main() -> None:
    args = parse_args()
    start = time.time()
    weights = load_weights(args.weights)
    out = Path(args.out)
    with VideoWriter(out, fps=args.fps) as writer:
        stats = play_and_render(weights, seed=args.seed, max_pieces=args.max_pieces, writer=writer)
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))
    print(f"wrote {out} elapsed={time.time() - start:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/custom_best/best_weights.npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pieces", type=int, default=500)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out", default="artifacts/best_plays/track4_custom_tool.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    main()
