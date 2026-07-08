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
    filled = board != 0
    has_any = filled.any(axis=0)
    first_filled = filled.argmax(axis=0)
    return np.where(has_any, ROWS - first_filled, 0).astype(np.int64)


def count_holes(board: np.ndarray) -> int:
    visible = board[HIDDEN_ROWS:] != 0
    covered = np.cumsum(visible, axis=0) > 0
    return int((~visible & covered).sum())


def row_transitions(board: np.ndarray) -> int:
    # Walls on the left and right count as filled cells.
    visible = board[HIDDEN_ROWS:] != 0
    interior = (visible[:, 1:] != visible[:, :-1]).sum()
    edges = (~visible[:, 0]).sum() + (~visible[:, -1]).sum()
    return int(interior + edges)


def col_transitions(board: np.ndarray) -> int:
    # The visible top edge and the floor count as filled cells.
    visible = board[HIDDEN_ROWS:] != 0
    interior = (visible[1:, :] != visible[:-1, :]).sum()
    edges = (~visible[0, :]).sum() + (~visible[-1, :]).sum()
    return int(interior + edges)


def wells(board: np.ndarray) -> int:
    heights = column_heights(board)
    left = np.concatenate(((ROWS,), heights[:-1]))
    right = np.concatenate((heights[1:], (ROWS,)))
    depth = np.minimum(left, right) - heights
    depth = np.maximum(depth, 0)
    return int((depth * (depth + 1) // 2).sum())


def placement_features(placement: Placement) -> np.ndarray:
    # Fused computation sharing the boolean board and column heights; each term
    # matches the standalone functions above exactly.
    board = placement.board
    filled = board != 0
    visible = filled[HIDDEN_ROWS:]

    has_any = filled.any(axis=0)
    heights = np.where(has_any, ROWS - filled.argmax(axis=0), 0)

    holes = int((~visible & (np.cumsum(visible, axis=0) > 0)).sum())

    row_trans = int(
        (visible[:, 1:] != visible[:, :-1]).sum() + (~visible[:, 0]).sum() + (~visible[:, -1]).sum()
    )
    col_trans = int(
        (visible[1:, :] != visible[:-1, :]).sum() + (~visible[0, :]).sum() + (~visible[-1, :]).sum()
    )

    left = np.concatenate(((ROWS,), heights[:-1]))
    right = np.concatenate((heights[1:], (ROWS,)))
    depth = np.maximum(np.minimum(left, right) - heights, 0)
    well_sum = int((depth * (depth + 1) // 2).sum())

    raw = np.array(
        [
            placement.score_delta,
            placement.lines,
            heights.sum(),
            holes,
            np.abs(np.diff(heights)).sum(),
            well_sum,
            row_trans,
            col_trans,
            heights.max(),
            placement.landing_height,
        ],
        dtype=np.float64,
    )
    return raw / FEATURE_SCALE
