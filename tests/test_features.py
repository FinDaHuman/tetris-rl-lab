from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

from tetris_env.engine import COLS, HIDDEN_ROWS, ROWS
from tetris_env import features


# Reference implementations: the original pure-Python loops, kept verbatim so the
# vectorized versions in features.py can be checked against them exactly.


def _ref_column_heights(board: np.ndarray) -> np.ndarray:
    heights = np.zeros(COLS, dtype=np.int64)
    for col in range(COLS):
        filled = np.flatnonzero(board[:, col])
        heights[col] = 0 if len(filled) == 0 else ROWS - int(filled[0])
    return heights


def _ref_count_holes(board: np.ndarray) -> int:
    total = 0
    for col in range(COLS):
        seen = False
        for row in range(HIDDEN_ROWS, ROWS):
            if board[row, col]:
                seen = True
            elif seen:
                total += 1
    return total


def _ref_row_transitions(board: np.ndarray) -> int:
    total = 0
    for row in range(HIDDEN_ROWS, ROWS):
        prev = 1
        for col in range(COLS):
            cur = int(board[row, col])
            total += int(cur != prev)
            prev = cur
        total += int(prev != 1)
    return total


def _ref_col_transitions(board: np.ndarray) -> int:
    total = 0
    for col in range(COLS):
        prev = 1
        for row in range(HIDDEN_ROWS, ROWS):
            cur = int(board[row, col])
            total += int(cur != prev)
            prev = cur
        total += int(prev != 1)
    return total


def _ref_wells(board: np.ndarray) -> int:
    heights = _ref_column_heights(board)
    total = 0
    for col in range(COLS):
        left = heights[col - 1] if col > 0 else ROWS
        right = heights[col + 1] if col < COLS - 1 else ROWS
        depth = min(left, right) - heights[col]
        if depth > 0:
            total += int(depth * (depth + 1) // 2)
    return total


def _random_boards(count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    boards = []
    for density in np.linspace(0.05, 0.95, count):
        board = (rng.random((ROWS, COLS)) < density).astype(np.uint8)
        boards.append(board)
    boards.append(np.zeros((ROWS, COLS), dtype=np.uint8))
    boards.append(np.ones((ROWS, COLS), dtype=np.uint8))
    return boards


def test_vectorized_features_match_reference_loops() -> None:
    for board in _random_boards(50):
        assert np.array_equal(features.column_heights(board), _ref_column_heights(board))
        assert features.count_holes(board) == _ref_count_holes(board)
        assert features.row_transitions(board) == _ref_row_transitions(board)
        assert features.col_transitions(board) == _ref_col_transitions(board)
        assert features.wells(board) == _ref_wells(board)
