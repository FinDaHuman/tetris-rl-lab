from __future__ import annotations

import numpy as np

from .engine import COLS, HIDDEN_ROWS, ROWS, Placement


FEATURE_NAMES = (
    "score_delta",
    "lines",
    "aggregate_height",
    "holes",
    "bumpiness",
    "wells",
    "row_transitions",
    "col_transitions",
    "max_height",
    "landing_height",
)

DEFAULT_WEIGHTS = np.array([1.0, 500.0, -6.0, -35.0, -5.0, -8.0, -3.0, -2.5, -12.0, -2.0], dtype=np.float64)
FEATURE_SCALE = np.array([2000, 4, 220, 200, 120, 120, 240, 240, 22, 22], dtype=np.float64)


def column_heights(board: np.ndarray) -> np.ndarray:
    heights = np.zeros(COLS, dtype=np.int64)
    for col in range(COLS):
        filled = np.flatnonzero(board[:, col])
        heights[col] = 0 if len(filled) == 0 else ROWS - int(filled[0])
    return heights


def count_holes(board: np.ndarray) -> int:
    total = 0
    for col in range(COLS):
        seen = False
        for row in range(HIDDEN_ROWS, ROWS):
            if board[row, col]:
                seen = True
            elif seen:
                total += 1
    return total


def row_transitions(board: np.ndarray) -> int:
    total = 0
    for row in range(HIDDEN_ROWS, ROWS):
        prev = 1
        for col in range(COLS):
            cur = int(board[row, col])
            total += int(cur != prev)
            prev = cur
        total += int(prev != 1)
    return total


def col_transitions(board: np.ndarray) -> int:
    total = 0
    for col in range(COLS):
        prev = 1
        for row in range(HIDDEN_ROWS, ROWS):
            cur = int(board[row, col])
            total += int(cur != prev)
            prev = cur
        total += int(prev != 1)
    return total


def wells(board: np.ndarray) -> int:
    heights = column_heights(board)
    total = 0
    for col in range(COLS):
        left = heights[col - 1] if col > 0 else ROWS
        right = heights[col + 1] if col < COLS - 1 else ROWS
        depth = min(left, right) - heights[col]
        if depth > 0:
            total += int(depth * (depth + 1) // 2)
    return total


def placement_features(placement: Placement) -> np.ndarray:
    board = placement.board
    heights = column_heights(board)
    raw = np.array(
        [
            placement.score_delta,
            placement.lines,
            heights.sum(),
            count_holes(board),
            np.abs(np.diff(heights)).sum(),
            wells(board),
            row_transitions(board),
            col_transitions(board),
            heights.max(),
            placement.landing_height,
        ],
        dtype=np.float64,
    )
    return raw / FEATURE_SCALE
