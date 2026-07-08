from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import random

import numpy as np


VISIBLE_ROWS = 20
HIDDEN_ROWS = 2
ROWS = VISIBLE_ROWS + HIDDEN_ROWS
COLS = 10


class Action(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    ROTATE_CW = 3
    ROTATE_CCW = 4
    SOFT_DROP = 5
    HARD_DROP = 6


PIECES: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    "I": (
        ((1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 1), (1, 1), (2, 1), (3, 1)),
    ),
    "O": (
        ((0, 1), (0, 2), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (1, 2)),
    ),
    "T": (
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
    ),
    "S": (
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 1), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
    ),
    "Z": (
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 2), (1, 1), (1, 2), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ),
    "J": (
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
    ),
    "L": (
        ((0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (1, 2), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
    ),
}

JLSTZ_KICKS = {
    (0, 1): ((0, 0), (0, -1), (-1, -1), (2, 0), (2, -1)),
    (1, 0): ((0, 0), (0, 1), (1, 1), (-2, 0), (-2, 1)),
    (1, 2): ((0, 0), (0, 1), (1, 1), (-2, 0), (-2, 1)),
    (2, 1): ((0, 0), (0, -1), (-1, -1), (2, 0), (2, -1)),
    (2, 3): ((0, 0), (0, 1), (-1, 1), (2, 0), (2, 1)),
    (3, 2): ((0, 0), (0, -1), (1, -1), (-2, 0), (-2, -1)),
    (3, 0): ((0, 0), (0, -1), (1, -1), (-2, 0), (-2, -1)),
    (0, 3): ((0, 0), (0, 1), (-1, 1), (2, 0), (2, 1)),
}

I_KICKS = {
    (0, 1): ((0, 0), (0, -2), (0, 1), (1, -2), (-2, 1)),
    (1, 0): ((0, 0), (0, 2), (0, -1), (-1, 2), (2, -1)),
    (1, 2): ((0, 0), (0, -1), (0, 2), (-2, -1), (1, 2)),
    (2, 1): ((0, 0), (0, 1), (0, -2), (2, 1), (-1, -2)),
    (2, 3): ((0, 0), (0, 2), (0, -1), (-1, 2), (2, -1)),
    (3, 2): ((0, 0), (0, -2), (0, 1), (1, -2), (-2, 1)),
    (3, 0): ((0, 0), (0, 1), (0, -2), (2, 1), (-1, -2)),
    (0, 3): ((0, 0), (0, -1), (0, 2), (-2, -1), (1, 2)),
}

LINE_SCORES = (0, 100, 300, 500, 800)


@dataclass(frozen=True)
class PieceState:
    name: str
    row: int
    col: int
    rotation: int = 0


@dataclass(frozen=True)
class Placement:
    piece: str
    rotation: int
    col: int
    board: np.ndarray
    lines: int
    score_delta: int
    drop_distance: int
    landing_height: int


@dataclass(frozen=True)
class StepResult:
    reward: float
    terminated: bool
    lines: int
    score_delta: int


class TetrisGame:
    """Guideline-style Tetris engine tuned for fast RL score-mode rollouts."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.board = np.zeros((ROWS, COLS), dtype=np.uint8)
        self.queue: list[str] = []
        self.current: PieceState | None = None
        self.score = 0
        self.lines = 0
        self.pieces = 0
        self.level = 0
        self.game_over = False
        self._fill_queue()
        self.spawn()

    def clone(self) -> "TetrisGame":
        other = object.__new__(TetrisGame)
        other.rng = random.Random()
        other.rng.setstate(self.rng.getstate())
        other.board = self.board.copy()
        other.queue = list(self.queue)
        other.current = self.current
        other.score = self.score
        other.lines = self.lines
        other.pieces = self.pieces
        other.level = self.level
        other.game_over = self.game_over
        return other

    @property
    def visible_board(self) -> np.ndarray:
        return self.board[HIDDEN_ROWS:].copy()

    @property
    def next_piece(self) -> str:
        self._fill_queue()
        return self.queue[0]

    def cells(self, state: PieceState | None = None) -> tuple[tuple[int, int], ...]:
        state = self.current if state is None else state
        if state is None:
            return ()
        return tuple((state.row + r, state.col + c) for r, c in PIECES[state.name][state.rotation])

    def collides(self, state: PieceState) -> bool:
        for row, col in self.cells(state):
            if col < 0 or col >= COLS or row >= ROWS:
                return True
            if row >= 0 and self.board[row, col]:
                return True
        return False

    def spawn(self) -> None:
        self._fill_queue()
        name = self.queue.pop(0)
        self.current = PieceState(name=name, row=0, col=3, rotation=0)
        if self.collides(self.current):
            self.game_over = True

    def step(self, action: Action | int) -> StepResult:
        if self.game_over or self.current is None:
            return StepResult(0.0, True, self.lines, 0)
        action = Action(action)
        reward = 0.0
        score_before = self.score
        if action == Action.LEFT:
            self._try_shift(0, -1)
        elif action == Action.RIGHT:
            self._try_shift(0, 1)
        elif action == Action.ROTATE_CW:
            self._try_rotate(1)
        elif action == Action.ROTATE_CCW:
            self._try_rotate(-1)
        elif action == Action.SOFT_DROP:
            if self._try_shift(1, 0):
                self.score += 1
                reward += 1.0
            else:
                reward += self._lock_piece()
        elif action == Action.HARD_DROP:
            reward += self.hard_drop()

        if action not in (Action.SOFT_DROP, Action.HARD_DROP) and not self.game_over:
            if not self._try_shift(1, 0):
                reward += self._lock_piece()
        return StepResult(reward, self.game_over, self.lines, self.score - score_before)

    def hard_drop(self) -> float:
        distance = 0
        while self._try_shift(1, 0):
            distance += 1
        self.score += distance * 2
        return float(distance * 2 + self._lock_piece())

    def _try_shift(self, drow: int, dcol: int) -> bool:
        assert self.current is not None
        shifted = PieceState(self.current.name, self.current.row + drow, self.current.col + dcol, self.current.rotation)
        if self.collides(shifted):
            return False
        self.current = shifted
        return True

    def _try_rotate(self, direction: int) -> bool:
        assert self.current is not None
        old = self.current.rotation
        new = (old + direction) % 4
        if self.current.name == "O":
            return True
        kicks = I_KICKS if self.current.name == "I" else JLSTZ_KICKS
        for drow, dcol in kicks[(old, new)]:
            rotated = PieceState(self.current.name, self.current.row + drow, self.current.col + dcol, new)
            if not self.collides(rotated):
                self.current = rotated
                return True
        return False

    def _lock_piece(self) -> float:
        assert self.current is not None
        for row, col in self.cells():
            if row < 0:
                self.game_over = True
                return -1000.0
            self.board[row, col] = 1
        self.pieces += 1
        cleared = self._clear_lines()
        line_score = LINE_SCORES[cleared] * (self.level + 1)
        self.score += line_score
        self.lines += cleared
        self.level = self.lines // 10
        self.spawn()
        return float(line_score)

    def _clear_lines(self) -> int:
        full = np.all(self.board == 1, axis=1)
        cleared = int(full.sum())
        if cleared:
            kept = self.board[~full]
            self.board[:] = 0
            self.board[ROWS - len(kept) :] = kept
        return cleared

    def _fill_queue(self) -> None:
        while len(self.queue) < 7:
            bag = list(PIECES)
            self.rng.shuffle(bag)
            self.queue.extend(bag)


def _shape_drop_data(piece_name: str) -> tuple[tuple, ...]:
    """Per-rotation static data for fast placement enumeration."""
    data = []
    for rotation in range(4):
        shape = PIECES[piece_name][rotation]
        min_col = -min(c for _, c in shape)
        max_col = COLS - max(c for _, c in shape)
        min_row = min(r for r, _ in shape)
        bottoms: dict[int, int] = {}
        for r, c in shape:
            bottoms[c] = max(bottoms.get(c, -1), r)
        data.append((shape, min_col, max_col, min_row, tuple(bottoms.items())))
    return tuple(data)


_SHAPE_DROP_CACHE = {name: _shape_drop_data(name) for name in PIECES}


def enumerate_placements(game: TetrisGame, piece: str | None = None) -> list[Placement]:
    if game.current is None or game.game_over:
        return []
    placements: list[Placement] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    piece_name = game.current.name if piece is None else piece
    board = game.board
    filled = board != 0
    has_any = filled.any(axis=0)
    tops = np.where(has_any, filled.argmax(axis=0), ROWS)
    start_row = game.current.row
    level_multiplier = game.level + 1
    for rotation, (shape, min_col, max_col, min_row, bottoms) in enumerate(_SHAPE_DROP_CACHE[piece_name]):
        for col in range(min_col, max_col):
            # Symmetric rotations (O all four, I/S/Z opposite pairs) produce the
            # same final footprint; enumerate each footprint once.
            footprint = tuple(sorted((r - min_row, c + col) for r, c in shape))
            if footprint in seen:
                continue
            blocked = False
            for r, c in shape:
                rr = r + start_row
                if rr >= ROWS or filled[rr, c + col]:
                    blocked = True
                    break
            if blocked:
                continue
            seen.add(footprint)
            landing_row = min(int(tops[c + col]) - 1 - b for c, b in bottoms)
            if landing_row < start_row:
                # The spawn position sits inside a covered pocket below a column
                # top; fall back to the exact cell-wise drop.
                landing_row = start_row
                while True:
                    nxt = landing_row + 1
                    if any(r + nxt >= ROWS or filled[r + nxt, c + col] for r, c in shape):
                        break
                    landing_row = nxt
            new_board = board.copy()
            max_row = 0
            for r, c in shape:
                rr = r + landing_row
                new_board[rr, c + col] = 1
                if rr > max_row:
                    max_row = rr
            full = np.all(new_board != 0, axis=1)
            cleared = int(full.sum())
            if cleared:
                kept = new_board[~full]
                new_board = np.zeros_like(board)
                new_board[ROWS - len(kept):] = kept
            placements.append(
                Placement(
                    piece=piece_name,
                    rotation=rotation,
                    col=col,
                    board=new_board,
                    lines=cleared,
                    score_delta=LINE_SCORES[cleared] * level_multiplier,
                    drop_distance=max(0, landing_row - start_row),
                    landing_height=ROWS - max_row,
                )
            )
    return placements
