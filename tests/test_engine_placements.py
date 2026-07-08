from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

from tetris_env.engine import COLS, PIECES, PieceState, TetrisGame, enumerate_placements
from tetris_env.features import FEATURE_SCALE, placement_features
from tetris_env import features


def _reference_enumerate(game: TetrisGame, piece: str | None = None):
    """Original clone-and-drop enumeration, kept as the correctness oracle."""
    if game.current is None or game.game_over:
        return []
    placements = []
    piece_name = game.current.name if piece is None else piece
    for rotation in range(4):
        shape = PIECES[piece_name][rotation]
        min_col = -min(c for _, c in shape)
        max_col = COLS - max(c for _, c in shape)
        for col in range(min_col, max_col):
            trial = game.clone()
            trial.current = PieceState(piece_name, game.current.row, col, rotation)
            if trial.collides(trial.current):
                continue
            while trial._try_shift(1, 0):
                pass
            lines_before = trial.lines
            score_before = trial.score
            trial._lock_piece()
            placements.append(
                (
                    trial.board.tobytes(),
                    trial.lines - lines_before,
                    trial.score - score_before,
                )
            )
    return placements


def _play_random_states(pieces: int, seed: int) -> list[TetrisGame]:
    """Collect game states from a random-drop rollout so boards are realistic."""
    rng = np.random.default_rng(seed)
    game = TetrisGame(seed=seed)
    states = []
    while not game.game_over and game.pieces < pieces:
        states.append(game.clone())
        placements = enumerate_placements(game)
        if not placements:
            break
        choice = placements[int(rng.integers(len(placements)))]
        game.board = choice.board.copy()
        game.lines += choice.lines
        game.score += choice.score_delta
        game.level = game.lines // 10
        game.pieces += 1
        game.spawn()
    return states


def test_fast_enumeration_matches_reference_modulo_duplicates() -> None:
    for seed in (0, 1, 2):
        for game in _play_random_states(40, seed):
            fast = {
                (p.board.tobytes(), p.lines, p.score_delta)
                for p in enumerate_placements(game)
            }
            reference = set(_reference_enumerate(game))
            assert fast == reference


def test_placement_features_matches_standalone_functions() -> None:
    game = TetrisGame(seed=3)
    for _ in range(10):
        placements = enumerate_placements(game)
        assert placements
        for placement in placements:
            board = placement.board
            heights = features.column_heights(board)
            expected = np.array(
                [
                    placement.score_delta,
                    placement.lines,
                    heights.sum(),
                    features.count_holes(board),
                    np.abs(np.diff(heights)).sum(),
                    features.wells(board),
                    features.row_transitions(board),
                    features.col_transitions(board),
                    heights.max(),
                    placement.landing_height,
                ],
                dtype=np.float64,
            ) / FEATURE_SCALE
            assert np.allclose(placement_features(placement), expected)
        chosen = placements[0]
        game.board = chosen.board.copy()
        game.lines += chosen.lines
        game.score += chosen.score_delta
        game.level = game.lines // 10
        game.pieces += 1
        game.spawn()
