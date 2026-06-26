from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agents.ale.env import ENV_ID, estimate_atari_score, make_env, reward_to_lines

NOOP, FIRE, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4
ACTIONS = (NOOP, FIRE, RIGHT, LEFT, DOWN)

FIELD_X0, FIELD_X1 = 22, 63
FIELD_Y0, FIELD_Y1 = 27, 202
N_ROWS, N_COLS = 20, 10
BG_GRAY = np.array([111, 111, 111], dtype=np.int16)
GRAY_TOL = 16

PIECES: dict[str, list[tuple[tuple[int, int], ...]]] = {
    "I": [((0, 0), (0, 1), (0, 2), (0, 3)), ((0, 0), (1, 0), (2, 0), (3, 0))],
    "O": [((0, 0), (0, 1), (1, 0), (1, 1))],
    "T": [
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (1, 1), (2, 0)),
    ],
    "S": [((0, 1), (0, 2), (1, 0), (1, 1)), ((0, 0), (1, 0), (1, 1), (2, 1))],
    "Z": [((0, 0), (0, 1), (1, 1), (1, 2)), ((0, 1), (1, 0), (1, 1), (2, 0))],
    "L": [
        ((0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
    ],
    "J": [
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
    ],
}

PATTERN_LUT: dict[frozenset[tuple[int, int]], tuple[str, int]] = {}
for piece_name, rotations in PIECES.items():
    for idx, cells in enumerate(rotations):
        PATTERN_LUT[frozenset(cells)] = (piece_name, idx)

FEATURE_NAMES = (
    "lines",
    "aggregate_height",
    "holes",
    "bumpiness",
    "wells",
    "row_transitions",
    "col_transitions",
    "max_height",
    "side_imbalance",
)
FEATURE_SCALE = np.array([4, 200, 200, 100, 100, 220, 220, 20, 100], dtype=np.float64)
DEFAULT_WEIGHTS = np.array([8.0, -0.55, -4.0, -0.25, -0.8, -0.25, -0.25, -1.5, -3.0], dtype=np.float64)


_xs = (FIELD_X0 + (np.arange(N_COLS) + 0.5) * (FIELD_X1 - FIELD_X0 + 1) / N_COLS).round().astype(int)
_ys = (FIELD_Y0 + (np.arange(N_ROWS) + 0.5) * (FIELD_Y1 - FIELD_Y0 + 1) / N_ROWS).round().astype(int)
_off = np.array([-1, 0, 1])
_dy, _dx = np.meshgrid(_off, _off, indexing="ij")
_dy = _dy.reshape(-1)
_dx = _dx.reshape(-1)
_row_idx = _ys[:, None, None] + _dy[None, None, :]
_col_idx = _xs[None, :, None] + _dx[None, None, :]


def decode_board(frame: np.ndarray) -> np.ndarray:
    patches = frame[_row_idx, _col_idx]
    med = np.median(patches, axis=2).astype(np.int16)
    is_gray = np.all(np.abs(med - BG_GRAY) <= GRAY_TOL, axis=-1)
    is_black = np.all(med <= 12, axis=-1)
    return (~(is_gray | is_black)).astype(np.uint8)


def connected_top_piece(board: np.ndarray) -> np.ndarray | None:
    comps = connected_components(board)
    if not comps:
        return None
    return min(comps, key=lambda cells: (int(cells[:, 0].min()), int(cells[:, 1].min())))


def connected_components(board: np.ndarray) -> list[np.ndarray]:
    filled = np.argwhere(board == 1)
    if filled.size == 0:
        return []
    remaining = {tuple(map(int, rc)) for rc in filled}
    comps: list[np.ndarray] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo = [start]
        comp: list[tuple[int, int]] = []
        while todo:
            r, c = todo.pop()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    todo.append(nxt)
        comps.append(np.asarray(comp, dtype=np.int64))
    return comps


def find_falling_piece(board: np.ndarray, *, max_top_row: int | None = None) -> np.ndarray | None:
    matches = []
    for comp in connected_components(board):
        if len(comp) == 4 and identify_piece(comp)[0] is not None:
            if max_top_row is not None and int(comp[:, 0].min()) > max_top_row:
                continue
            matches.append(comp)
    if not matches:
        return None
    return min(matches, key=lambda cells: (int(cells[:, 0].min()), int(cells[:, 1].min())))


def identify_piece(cells: np.ndarray | None) -> tuple[str | None, int | None]:
    if cells is None or len(cells) != 4:
        return None, None
    r0, c0 = cells[:, 0].min(), cells[:, 1].min()
    pat = frozenset((int(r - r0), int(c - c0)) for r, c in cells)
    return PATTERN_LUT.get(pat, (None, None))


def wait_for_piece(env: gym.Env, obs: np.ndarray, max_wait: int = 40, max_top_row: int = 3):
    for _ in range(max_wait):
        board = decode_board(obs)
        comp = find_falling_piece(board, max_top_row=max_top_row)
        if comp is not None:
            name, rot = identify_piece(comp)
            if name is not None:
                return obs, board, comp, name, int(rot), False, False
        obs, _, term, trunc, _ = env.step(NOOP)
        if term or trunc:
            return obs, decode_board(obs), None, None, None, term, trunc
    return obs, decode_board(obs), None, None, None, False, False


def locked_board(board: np.ndarray, comp: np.ndarray | None) -> np.ndarray:
    out = board.copy()
    if comp is not None:
        for r, c in comp:
            out[int(r), int(c)] = 0
    return out


def clear_lines(board: np.ndarray) -> tuple[np.ndarray, int]:
    full = np.all(board == 1, axis=1)
    n = int(full.sum())
    if n == 0:
        return board, 0
    kept = board[~full]
    out = np.zeros_like(board)
    out[N_ROWS - len(kept) :] = kept
    return out, n


def col_heights(board: np.ndarray) -> np.ndarray:
    heights = np.zeros(N_COLS, dtype=np.int64)
    for c in range(N_COLS):
        filled = np.flatnonzero(board[:, c])
        heights[c] = 0 if len(filled) == 0 else N_ROWS - int(filled[0])
    return heights


def count_holes(board: np.ndarray) -> int:
    total = 0
    for c in range(N_COLS):
        seen = False
        for r in range(N_ROWS):
            if board[r, c]:
                seen = True
            elif seen:
                total += 1
    return total


def row_transitions(board: np.ndarray) -> int:
    total = 0
    for r in range(N_ROWS):
        prev = 1
        for c in range(N_COLS):
            cur = int(board[r, c])
            total += int(cur != prev)
            prev = cur
        total += int(prev != 1)
    return total


def col_transitions(board: np.ndarray) -> int:
    total = 0
    for c in range(N_COLS):
        prev = 1
        for r in range(N_ROWS):
            cur = int(board[r, c])
            total += int(cur != prev)
            prev = cur
        total += int(prev != 1)
    return total


def wells(board: np.ndarray) -> int:
    heights = col_heights(board)
    total = 0
    for c in range(N_COLS):
        left = heights[c - 1] if c > 0 else N_ROWS
        right = heights[c + 1] if c < N_COLS - 1 else N_ROWS
        d = min(left, right) - heights[c]
        if d > 0:
            total += int(d * (d + 1) // 2)
    return total


def board_features(board: np.ndarray, immediate_reward: float = 0.0) -> np.ndarray:
    heights = col_heights(board)
    feats = np.array(
        [
            immediate_reward,
            heights.sum(),
            count_holes(board),
            np.abs(np.diff(heights)).sum(),
            wells(board),
            row_transitions(board),
            col_transitions(board),
            heights.max(),
            abs(int(heights[:5].sum()) - int(heights[5:].sum())),
        ],
        dtype=np.float64,
    )
    return feats / FEATURE_SCALE


def piece_width(cells: tuple[tuple[int, int], ...]) -> int:
    return max(c for _, c in cells) + 1


def drop_piece(board: np.ndarray, cells: tuple[tuple[int, int], ...], col: int):
    width = piece_width(cells)
    if col < 0 or col + width > N_COLS:
        return None
    offset = 0
    while True:
        nxt = offset + 1
        blocked = False
        for r, c in cells:
            rr = r + nxt
            cc = c + col
            if rr >= N_ROWS or board[rr, cc]:
                blocked = True
                break
        if blocked:
            break
        offset = nxt
    placed = board.copy()
    landed = []
    for r, c in cells:
        rr = r + offset
        cc = c + col
        if rr < 0 or rr >= N_ROWS:
            return None
        placed[rr, cc] = 1
        landed.append((rr, cc))
    full = np.all(placed == 1, axis=1)
    piece_cells_cleared = sum(1 for rr, _ in landed if full[rr])
    cleared, lines = clear_lines(placed)
    return cleared, lines, min(rr for rr, _ in landed), lines * piece_cells_cleared


def enumerate_model_placements(board: np.ndarray, name: str):
    for rot_idx, cells in enumerate(PIECES[name]):
        for col in range(0, N_COLS - piece_width(cells) + 1):
            result = drop_piece(board, cells, col)
            if result is None:
                continue
            after, lines, landing_top, eroded = result
            yield rot_idx, col, after, lines, landing_top, eroded


def model_features(board: np.ndarray, lines: int, landing_top: int, eroded: int) -> np.ndarray:
    base = board_features(board, float(lines)).copy()
    base[0] = float(lines) / 4.0
    return np.concatenate([base, np.array([float(N_ROWS - landing_top) / N_ROWS, float(eroded) / 16.0])])


MODEL_FEATURE_NAMES = FEATURE_NAMES + ("landing_height", "eroded_cells")
MODEL_FEATURE_SCALE = np.ones(len(MODEL_FEATURE_NAMES), dtype=np.float64)
MODEL_DEFAULT_WEIGHTS = np.array(
    [18.0, -1.2, -9.0, -1.6, -1.4, -0.35, -0.25, -2.2, -1.0, -0.8, 8.0],
    dtype=np.float64,
)

CLASSIC_FEATURE_NAMES = (
    "landing_height",
    "eroded_cells",
    "row_transitions",
    "column_transitions",
    "holes",
    "wells",
    "aggregate_height",
    "bumpiness",
    "parity",
    "max_well_depth",
)
CLASSIC_DEFAULT_WEIGHTS = np.array(
    [-0.06173701, 0.23375136, -0.49856368, -0.49586722, -0.32304839,
     0.06480258, -0.50981690, -0.02259093, -0.00913214, 0.27934233],
    dtype=np.float64,
)


def classic_features(board: np.ndarray, lines: int, landing_height: int, eroded: int) -> np.ndarray:
    heights = col_heights(board)
    holes = count_holes(board)
    row_trans = 0
    for y in range(N_ROWS):
        prev = 1
        for x in range(N_COLS):
            cur = int(board[y, x])
            if prev != cur:
                row_trans += 1
            prev = cur
        if prev != 1:
            row_trans += 1

    col_trans = 0
    for x in range(N_COLS):
        if board[N_ROWS - 1, x] == 0:
            col_trans += 1
        prev = int(board[0, x])
        for y in range(1, N_ROWS):
            cur = int(board[y, x])
            if prev != cur:
                col_trans += 1
            prev = cur

    wells_total = 0
    max_well_depth = 0
    for x in range(N_COLS):
        depth = 0
        for y in range(N_ROWS):
            if board[y, x]:
                if depth:
                    wells_total += depth * (depth + 1) // 2
                    max_well_depth = max(max_well_depth, depth)
                    depth = 0
            else:
                left_filled = x == 0 or bool(board[y, x - 1])
                right_filled = x == N_COLS - 1 or bool(board[y, x + 1])
                if left_filled and right_filled:
                    depth += 1
                elif depth:
                    wells_total += depth * (depth + 1) // 2
                    max_well_depth = max(max_well_depth, depth)
                    depth = 0
        if depth:
            wells_total += depth * (depth + 1) // 2
            max_well_depth = max(max_well_depth, depth)

    black = 0
    white = 0
    for y in range(N_ROWS):
        for x in range(N_COLS):
            if board[y, x]:
                if (x + y) & 1:
                    white += 1
                else:
                    black += 1

    return np.array(
        [
            landing_height,
            lines * eroded,
            row_trans,
            col_trans,
            holes,
            wells_total,
            int(heights.sum()),
            int(np.abs(np.diff(heights)).sum()),
            abs(black - white),
            max_well_depth,
        ],
        dtype=np.float64,
    )


@dataclass
class Candidate:
    score: float
    actions: list[int]
    reward: float
    after_board: np.ndarray
    survived: bool
    state: object | None = None
    obs: np.ndarray | None = None
    target_rot: int | None = None
    target_col: int | None = None


class ClonePlacementAgent:
    def __init__(self, weights: np.ndarray, *, max_drop_steps: int = 90, depth: int = 2, beam: int = 6):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.max_drop_steps = max_drop_steps
        self.depth = depth
        self.beam = beam

    def choose(self, env: gym.Env, obs: np.ndarray, comp: np.ndarray, name: str, rot: int) -> Candidate | None:
        candidates = self._candidates(env, obs, comp, name, rot)
        if not candidates:
            return None
        start_state = env.unwrapped.ale.cloneState()
        scored: list[Candidate] = []
        for cand in sorted(candidates, key=lambda c: c.score, reverse=True)[: self.beam]:
            future = 0.0
            if self.depth > 1 and cand.state is not None and cand.obs is not None:
                env.unwrapped.ale.restoreState(cand.state)
                future = self._lookahead(env, cand.obs, self.depth - 1)
            cand.score += 0.9 * future
            scored.append(cand)
        env.unwrapped.ale.restoreState(start_state)
        return max(scored, key=lambda c: c.score)

    def _lookahead(self, env: gym.Env, obs: np.ndarray, depth: int) -> float:
        obs, _board, comp, name, rot, term, trunc = wait_for_piece(env, obs)
        if term or trunc or comp is None or name is None or rot is None:
            return -10.0
        candidates = self._candidates(env, obs, comp, name, rot)
        if not candidates:
            return -10.0
        start_state = env.unwrapped.ale.cloneState()
        best = -math.inf
        for cand in sorted(candidates, key=lambda c: c.score, reverse=True)[: self.beam]:
            value = cand.score
            if depth > 1 and cand.state is not None and cand.obs is not None:
                env.unwrapped.ale.restoreState(cand.state)
                value += 0.9 * self._lookahead(env, cand.obs, depth - 1)
            best = max(best, value)
        env.unwrapped.ale.restoreState(start_state)
        return float(best)

    def _candidates(self, env: gym.Env, obs: np.ndarray, comp: np.ndarray, name: str, rot: int) -> list[Candidate]:
        start_state = env.unwrapped.ale.cloneState()
        start_left = int(comp[:, 1].min())
        n_rots = len(PIECES[name])
        candidates: list[Candidate] = []
        for target_rot in range(n_rots):
            fires = (target_rot - rot) % n_rots
            for target_left in range(-1, N_COLS):
                move = RIGHT if target_left > start_left else LEFT
                moves = abs(target_left - start_left)
                prefix = [FIRE] * fires + [move] * moves
                actions = prefix + [DOWN] * self.max_drop_steps
                env.unwrapped.ale.restoreState(start_state)
                trial_obs = obs
                total_reward = 0.0
                term = trunc = False
                last_board = decode_board(trial_obs)
                descended = False
                locked = False
                for step_i, action in enumerate(actions):
                    trial_obs, reward, term, trunc, _ = env.step(action)
                    total_reward += float(reward)
                    last_board = decode_board(trial_obs)
                    if term or trunc:
                        break
                    if step_i >= len(prefix) + 3:
                        nxt = find_falling_piece(last_board, max_top_row=8)
                        if nxt is not None:
                            nxt_name, _ = identify_piece(nxt)
                            if nxt_name is not None and descended:
                                locked = True
                                break
                        falling = connected_top_piece(last_board)
                        if falling is not None and len(falling) >= 4 and int(falling[:, 0].max()) >= 6:
                            descended = True
                after = locked_board(last_board, find_falling_piece(last_board, max_top_row=8))
                features = board_features(after, total_reward)
                value = float(np.dot(self.weights, features))
                if term:
                    value -= 10.0
                if not locked:
                    value -= 10.0
                state_after = env.unwrapped.ale.cloneState() if locked and not term and not trunc else None
                candidates.append(
                    Candidate(
                        value,
                        actions[: len(prefix) + step_i + 1],
                        total_reward,
                        after,
                        not term and locked,
                        state_after,
                        trial_obs.copy(),
                    )
                )
        env.unwrapped.ale.restoreState(start_state)
        return [cand for cand in candidates if cand.survived]


class ModelPlacementAgent:
    def __init__(self, weights: np.ndarray, *, classic: bool = False):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.classic = classic
        default = CLASSIC_DEFAULT_WEIGHTS if classic else MODEL_DEFAULT_WEIGHTS
        if len(self.weights) < len(default):
            padded = default.copy()
            padded[: len(self.weights)] = self.weights
            self.weights = padded

    def choose(self, _env: gym.Env, board: np.ndarray, comp: np.ndarray, name: str, rot: int) -> Candidate | None:
        locked = locked_board(board, comp)
        start_left = int(comp[:, 1].min())
        return self.choose_locked(locked, start_left, int(comp[:, 0].min()), name, rot)

    def choose_locked(
        self,
        locked: np.ndarray,
        start_left: int,
        start_top: int,
        name: str,
        rot: int,
    ) -> Candidate | None:
        filled_rows = np.flatnonzero(np.any(locked, axis=1))
        top_stack = int(filled_rows[0]) if len(filled_rows) else N_ROWS
        available_rows = max(1, top_stack - start_top)
        best: Candidate | None = None
        for target_rot, target_col, after, lines, landing_top, eroded in enumerate_model_placements(locked, name):
            n_rots = len(PIECES[name])
            fires = (target_rot - rot) % n_rots
            move = RIGHT if target_col > start_left else LEFT
            moves = abs(target_col - start_left)
            if fires + moves > available_rows + 2:
                continue
            actions = [FIRE] * fires + [move] * moves + [DOWN] * 90
            if self.classic:
                value = float(np.dot(self.weights, classic_features(after, lines, N_ROWS - landing_top, eroded)))
            else:
                value = float(np.dot(self.weights, model_features(after, lines, landing_top, eroded)))
            cand = Candidate(value, actions, float(lines), after, True, target_rot=target_rot, target_col=target_col)
            if best is None or cand.score > best.score:
                best = cand
        return best


class LookaheadPlacementAgent(ModelPlacementAgent):
    def __init__(self, weights: np.ndarray, *, classic: bool = False, gamma: float = 0.75):
        super().__init__(weights, classic=classic)
        self.gamma = gamma

    def _value(self, board: np.ndarray, lines: int, landing_top: int, eroded: int) -> float:
        if self.classic:
            return float(np.dot(self.weights, classic_features(board, lines, N_ROWS - landing_top, eroded)))
        return float(np.dot(self.weights, model_features(board, lines, landing_top, eroded)))

    def _best_next_value(self, board: np.ndarray, name: str) -> float:
        best = -math.inf
        for _rot, _col, after, lines, landing_top, eroded in enumerate_model_placements(board, name):
            best = max(best, self._value(after, lines, landing_top, eroded))
        return 0.0 if best == -math.inf else float(best)

    def choose_locked(
        self,
        locked: np.ndarray,
        start_left: int,
        start_top: int,
        name: str,
        rot: int,
    ) -> Candidate | None:
        filled_rows = np.flatnonzero(np.any(locked, axis=1))
        top_stack = int(filled_rows[0]) if len(filled_rows) else N_ROWS
        available_rows = max(1, top_stack - start_top)
        best: Candidate | None = None
        next_names = tuple(PIECES.keys())
        for target_rot, target_col, after, lines, landing_top, eroded in enumerate_model_placements(locked, name):
            n_rots = len(PIECES[name])
            fires = (target_rot - rot) % n_rots
            moves = abs(target_col - start_left)
            if fires + moves > available_rows + 2:
                continue
            immediate = self._value(after, lines, landing_top, eroded)
            future = sum(self._best_next_value(after, nxt) for nxt in next_names) / len(next_names)
            value = immediate + self.gamma * future
            move = RIGHT if target_col > start_left else LEFT
            actions = [FIRE] * fires + [move] * moves + [DOWN] * 90
            cand = Candidate(value, actions, float(lines), after, True, target_rot=target_rot, target_col=target_col)
            if best is None or cand.score > best.score:
                best = cand
        return best


def actual_board_value(weights: np.ndarray, board: np.ndarray, actual_lines: int) -> float:
    features = board_features(board, float(actual_lines))
    value = actual_lines * 10000.0
    value += 100.0 * float(np.dot(weights[: len(features)], features))
    heights = col_heights(board)
    # This planner is allowed to be heuristic-heavy. Line clears dominate; the
    # extra terms only separate similarly productive placements.
    value -= 0.02 * float(heights.sum())
    value -= 0.2 * float(count_holes(board))
    value -= 1.0 * float(max(0, int(heights.max()) - 14))
    return value


def run_model_actions(
    env: gym.Env,
    obs: np.ndarray,
    board: np.ndarray,
    comp: np.ndarray,
    name: str,
    rot: int,
    target_rot: int,
    target_col: int,
    move_frames: int,
    *,
    grab=None,
    total_reward: float = 0.0,
):
    locked = locked_board(board, comp).astype(bool)
    term = trunc = False
    actions: list[int] = []

    for _ in range((target_rot - int(rot)) % len(PIECES[name])):
        obs, reward, term, trunc, _ = env.step(FIRE)
        total_reward += float(reward)
        actions.append(FIRE)
        if grab:
            grab()
        if term or trunc:
            return obs, total_reward, term, trunc, False, actions, locked_board(decode_board(obs), None)

    start_left = int(comp[:, 1].min())
    if target_col > start_left:
        move = RIGHT
    elif target_col < start_left:
        move = LEFT
    else:
        move = NOOP
    for _ in range(max(0, move_frames)):
        obs, reward, term, trunc, _ = env.step(move)
        total_reward += float(reward)
        actions.append(move)
        if grab:
            grab()
        if term or trunc:
            return obs, total_reward, term, trunc, False, actions, locked_board(decode_board(obs), None)

    descended = False
    for _ in range(500):
        cur_board = decode_board(obs)
        cur = cur_board.astype(bool)
        falling = np.argwhere(cur & ~locked)
        if len(falling) and int(falling[:, 0].max()) >= 6:
            descended = True
        if descended:
            spawned = find_falling_piece(cur_board, max_top_row=8)
            if spawned is not None and int(spawned[:, 0].max()) <= 3:
                after = locked_board(cur_board, spawned)
                return obs, total_reward, term, trunc, True, actions, after

        obs, reward, term, trunc, _ = env.step(DOWN)
        total_reward += float(reward)
        actions.append(DOWN)
        if grab:
            grab()
        if term or trunc:
            break
    return obs, total_reward, term, trunc, False, actions, locked_board(decode_board(obs), None)


def run_legacy_actions(
    env: gym.Env,
    obs: np.ndarray,
    cand: Candidate,
    *,
    grab=None,
    total_reward: float = 0.0,
    strict_spawn: bool = False,
):
    term = trunc = False
    descended = False
    actions: list[int] = []
    last_board = decode_board(obs)
    after = locked_board(last_board, None)
    locked_ok = False
    for action in cand.actions:
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += float(reward)
        actions.append(action)
        if grab:
            grab()
        cur_board = decode_board(obs)
        cur_piece = find_falling_piece(cur_board, max_top_row=8)
        if cur_piece is not None:
            after = locked_board(cur_board, cur_piece)
        else:
            after = locked_board(cur_board, None)
        if term or trunc:
            break
        if cur_piece is not None and descended:
            if not strict_spawn or int(cur_piece[:, 0].max()) <= 3:
                locked_ok = True
                break
        top_piece = connected_top_piece(cur_board)
        if top_piece is not None and int(top_piece[:, 0].max()) >= 6:
            descended = True
    return obs, total_reward, term, trunc, locked_ok, actions, after


class LegacyCalibratedPlacementAgent(ModelPlacementAgent):
    def __init__(self, weights: np.ndarray, *, top_k: int = 48):
        super().__init__(weights, classic=False)
        self.top_k = top_k

    def choose(self, env: gym.Env, obs: np.ndarray, board: np.ndarray, comp: np.ndarray, name: str, rot: int) -> Candidate | None:
        locked = locked_board(board, comp)
        start_left = int(comp[:, 1].min())
        start_top = int(comp[:, 0].min())
        filled_rows = np.flatnonzero(np.any(locked, axis=1))
        top_stack = int(filled_rows[0]) if len(filled_rows) else N_ROWS
        available_rows = max(1, top_stack - start_top)
        model_candidates: list[Candidate] = []
        for target_rot, target_col, after, lines, landing_top, eroded in enumerate_model_placements(locked, name):
            n_rots = len(PIECES[name])
            fires = (target_rot - rot) % n_rots
            move = RIGHT if target_col > start_left else LEFT
            moves = abs(target_col - start_left)
            if fires + moves > available_rows + 2:
                continue
            actions = [FIRE] * fires + [move] * moves + [DOWN] * 90
            value = float(np.dot(self.weights, model_features(after, lines, landing_top, eroded)))
            model_candidates.append(
                Candidate(value, actions, float(lines), after, True, target_rot=target_rot, target_col=target_col)
            )
        if not model_candidates:
            return None

        start_state = env.unwrapped.ale.cloneState()
        best: Candidate | None = None
        for cand in sorted(model_candidates, key=lambda c: c.score, reverse=True)[: self.top_k]:
            env.unwrapped.ale.restoreState(start_state)
            _obs, reward, term, trunc, ok, actions, actual_after = run_legacy_actions(
                env,
                obs,
                cand,
                strict_spawn=True,
            )
            if term or trunc or not ok:
                continue
            actual_lines = max(0, reward_to_lines(reward))
            value = actual_board_value(self.weights, actual_after, actual_lines)
            if cand.after_board.shape == actual_after.shape:
                value -= 0.05 * float(np.abs(cand.after_board.astype(int) - actual_after.astype(int)).sum())
            calibrated = Candidate(
                value,
                actions,
                float(actual_lines),
                actual_after,
                True,
                target_rot=cand.target_rot,
                target_col=cand.target_col,
            )
            if best is None or calibrated.score > best.score:
                best = calibrated
        env.unwrapped.ale.restoreState(start_state)
        return best


class CalibratedPlacementAgent(ModelPlacementAgent):
    def __init__(self, weights: np.ndarray, *, classic: bool = False, top_k: int = 64):
        super().__init__(weights, classic=classic)
        self.top_k = top_k

    def choose(
        self,
        env: gym.Env,
        obs: np.ndarray,
        board: np.ndarray,
        comp: np.ndarray,
        name: str,
        rot: int,
    ) -> Candidate | None:
        locked = locked_board(board, comp)
        start_left = int(comp[:, 1].min())
        scored: list[Candidate] = []
        for target_rot, target_col, after, lines, landing_top, eroded in enumerate_model_placements(locked, name):
            if self.classic:
                value = float(np.dot(self.weights, classic_features(after, lines, N_ROWS - landing_top, eroded)))
            else:
                value = float(np.dot(self.weights, model_features(after, lines, landing_top, eroded)))
            scored.append(Candidate(value, [], float(lines), after, True, target_rot=target_rot, target_col=target_col))
        if not scored:
            return None

        start_state = env.unwrapped.ale.cloneState()
        best: Candidate | None = None
        for cand in sorted(scored, key=lambda c: c.score, reverse=True)[: self.top_k]:
            target_col = int(cand.target_col or 0)
            delta_cols = abs(target_col - start_left)
            base_frames = 0 if delta_cols == 0 else 3 + 4 * (delta_cols - 1)
            for frames in range(max(0, base_frames - 3), base_frames + 5):
                env.unwrapped.ale.restoreState(start_state)
                trial_obs, reward, term, trunc, ok, actions, actual_after = run_model_actions(
                    env, obs, board, comp, name, rot, int(cand.target_rot or 0), target_col, frames
                )
                if not ok or term or trunc:
                    continue
                actual_lines = max(0, int(round(reward)))
                if self.classic:
                    features = classic_features(actual_after, actual_lines, 0, 0)
                else:
                    features = board_features(actual_after, float(actual_lines))
                value = float(np.dot(self.weights[: len(features)], features))
                value += 100.0 * actual_lines
                if cand.after_board.shape == actual_after.shape:
                    value -= 0.2 * float(np.abs(cand.after_board.astype(int) - actual_after.astype(int)).sum())
                calibrated = Candidate(
                    value,
                    actions,
                    float(actual_lines),
                    actual_after,
                    True,
                    target_rot=cand.target_rot,
                    target_col=cand.target_col,
                )
                if best is None or calibrated.score > best.score:
                    best = calibrated
        env.unwrapped.ale.restoreState(start_state)
        return best


def execute_model_candidate(
    env: gym.Env,
    obs: np.ndarray,
    board: np.ndarray,
    comp: np.ndarray,
    name: str,
    rot: int,
    cand: Candidate,
    grab,
    total_reward: float,
):
    target_rot = int(cand.target_rot or 0)
    target_col = int(cand.target_col or 0)
    start_left = int(comp[:, 1].min())
    delta_cols = abs(target_col - start_left)
    move_frames = 0 if delta_cols == 0 else 2 + 4 * (delta_cols - 1)
    if cand.actions:
        fire_count = (target_rot - int(rot)) % len(PIECES[name])
        move_frames = max(0, len(cand.actions) - fire_count - cand.actions.count(DOWN))
    return run_model_actions(
        env, obs, board, comp, name, rot, target_rot, target_col, move_frames, grab=grab, total_reward=total_reward
    )[:5]


def execute_legacy_actions(env: gym.Env, obs: np.ndarray, cand: Candidate, grab, total_reward: float):
    obs, total_reward, term, trunc, locked_ok, _actions, _after = run_legacy_actions(
        env,
        obs,
        cand,
        grab=grab,
        total_reward=total_reward,
    )
    return obs, total_reward, term, trunc, locked_ok


def play_episode(
    weights: np.ndarray,
    *,
    seed: int,
    sticky: float = 0.0,
    max_pieces: int = 1000,
    capture: bool = False,
    max_frames: int = 120000,
    depth: int = 2,
    beam: int = 6,
    top_k: int = 48,
    planner: str = "clone",
):
    env = make_env(render_mode="rgb_array" if capture else None, sticky=sticky)
    obs, _ = env.reset(seed=seed)
    if planner == "legacy_calibrated":
        agent = LegacyCalibratedPlacementAgent(weights, top_k=top_k)
    elif planner in ("lookahead_model", "lookahead_classic"):
        agent = LookaheadPlacementAgent(weights, classic=planner == "lookahead_classic")
    elif planner in ("model_calibrated", "classic_calibrated"):
        agent = CalibratedPlacementAgent(weights, classic=planner == "classic_calibrated")
    elif planner in ("model", "classic", "classic_internal", "legacy_model"):
        agent = ModelPlacementAgent(weights, classic=planner in ("classic", "classic_internal"))
    else:
        agent = ClonePlacementAgent(weights, depth=depth, beam=beam)
    total_reward = 0.0
    pieces = 0
    frames: list[np.ndarray] = []
    term = trunc = False
    failures = 0
    internal_board = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)

    def grab():
        if capture and len(frames) < max_frames:
            frame = env.render()
            if frame is not None:
                lines = reward_to_lines(total_reward)
                frames.append(annotate(frame, pieces, total_reward, estimate_atari_score(lines), seed))

    grab()
    while not (term or trunc) and pieces < max_pieces:
        wait_top = 8 if planner in ("legacy_model", "legacy_calibrated") else 3
        obs, board, comp, name, rot, term, trunc = wait_for_piece(env, obs, max_top_row=wait_top)
        grab()
        if term or trunc:
            break
        if name is None or comp is None or rot is None:
            failures += 1
            obs, reward, term, trunc, _ = env.step(DOWN)
            total_reward += float(reward)
            grab()
            if failures >= 10:
                break
            continue
        failures = 0
        if isinstance(agent, LegacyCalibratedPlacementAgent):
            cand = agent.choose(env, obs, board, comp, name, rot)
        elif planner == "classic_internal" and isinstance(agent, ModelPlacementAgent):
            cand = agent.choose_locked(internal_board, int(comp[:, 1].min()), int(comp[:, 0].min()), name, rot)
        elif isinstance(agent, CalibratedPlacementAgent):
            cand = agent.choose(env, obs, board, comp, name, rot)
        elif isinstance(agent, ModelPlacementAgent):
            cand = agent.choose(env, board, comp, name, rot)
        else:
            cand = agent.choose(env, obs, comp, name, rot)
        if cand is None:
            break
        if planner in ("legacy_model", "legacy_calibrated", "lookahead_model", "lookahead_classic"):
            obs, total_reward, term, trunc, locked_ok = execute_legacy_actions(env, obs, cand, grab, total_reward)
        elif isinstance(agent, ModelPlacementAgent):
            obs, total_reward, term, trunc, locked_ok = execute_model_candidate(
                env, obs, board, comp, name, rot, cand, grab, total_reward
            )
            if planner == "classic_internal":
                if locked_ok:
                    internal_board = cand.after_board.copy()
                else:
                    failures += 1
                    if failures >= 10:
                        break
        else:
            for action in cand.actions:
                obs, reward, term, trunc, _ = env.step(action)
                total_reward += float(reward)
                grab()
                if term or trunc:
                    break
        pieces += 1
    lines = reward_to_lines(total_reward)
    score = estimate_atari_score(lines)
    env.close()
    return {
        "seed": seed,
        "score": score,
        "lines": lines,
        "native_reward": float(total_reward),
        "decisions": pieces,
        "pieces": pieces,
        "frames": frames,
        "frame_count": len(frames),
    }


def evaluate_population(weights: np.ndarray, seeds: list[int], max_pieces: int, sticky: float, planner: str, top_k: int) -> float:
    scores = []
    for seed in seeds:
        stats = play_episode(
            weights,
            seed=seed,
            max_pieces=max_pieces,
            sticky=sticky,
            depth=1,
            beam=4,
            top_k=top_k,
            planner=planner,
        )
        scores.append(stats["lines"] * 10000.0 + stats["pieces"])
    return float(np.mean(scores))


def train(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    mean = load_weights(args.warm_start, planner=args.planner) if args.warm_start else default_weights(args.planner)
    std = np.full(len(mean), args.init_std, dtype=np.float64)
    best_w = mean.copy()
    best_fit = -math.inf
    history = []
    for gen in range(1, args.generations + 1):
        pop = mean[None, :] + rng.standard_normal((args.population, len(mean))) * std[None, :]
        seeds = [args.seed * 10000 + gen * 100 + i for i in range(args.rollouts)]
        fits = np.array([evaluate_population(w, seeds, args.max_pieces, args.sticky, args.planner, args.top_k) for w in pop])
        elite_count = max(2, int(args.population * args.elite_frac))
        elite = pop[np.argsort(fits)[-elite_count:]]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + args.noise_floor
        gen_best_idx = int(np.argmax(fits))
        if float(fits[gen_best_idx]) > best_fit:
            best_fit = float(fits[gen_best_idx])
            best_w = pop[gen_best_idx].copy()
            np.save(outdir / "best_weights.npy", best_w)
        row = {"generation": gen, "best_fit": best_fit, "gen_best": float(fits.max()), "mean_fit": float(fits.mean())}
        history.append(row)
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"gen {gen:03d}: gen_best={fits.max():.1f} mean={fits.mean():.1f} best={best_fit:.1f}")
    np.save(outdir / "mean_weights.npy", mean)
    (outdir / "meta.json").write_text(
        json.dumps(
            {
                "feature_names": feature_names(args.planner),
                "best_fit": best_fit,
                "weights": best_w.tolist(),
                "planner": args.planner,
                "top_k": args.top_k,
                "max_pieces": args.max_pieces,
                "rollouts": args.rollouts,
                "generations": args.generations,
                "population": args.population,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


_FONT = None


def font():
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype("arial.ttf", 13)
        except OSError:
            _FONT = ImageFont.load_default()
    return _FONT


def annotate(frame: np.ndarray, pieces: int, reward: float, score: int, seed: int) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    text = f"seed {seed}  pieces {pieces}  reward {reward:.0f}  score {score}"
    draw.rectangle((0, 0, 159, 16), fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 0), font=font())
    return np.asarray(img)


def render(args: argparse.Namespace) -> None:
    weights = load_weights(args.weights, planner=args.planner)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = None
    for i in range(args.tries):
        seed = args.seed + i
        stats = play_episode(
            weights,
            seed=seed,
            sticky=args.sticky,
            max_pieces=args.max_pieces,
            capture=True,
            max_frames=args.max_frames,
            depth=args.depth,
            beam=args.beam,
            top_k=args.top_k,
            planner=args.planner,
        )
        no_frames = {k: v for k, v in stats.items() if k != "frames"}
        print(json.dumps(no_frames))
        if best is None or stats["score"] > best["score"]:
            best = stats
        target_lines = args.min_lines if args.min_lines is not None else int(math.ceil(args.min_score / 100.0))
        if stats["lines"] >= target_lines:
            imageio.mimsave(out, stats["frames"], fps=args.fps)
            sidecar = out.with_suffix(".json")
            sidecar.write_text(json.dumps(no_frames, indent=2), encoding="utf-8")
            print(f"wrote {out} with lines={stats['lines']} score={stats['score']}")
            return
    if best is not None and best["frames"]:
        imageio.mimsave(out, best["frames"], fps=args.fps)
        no_frames = {k: v for k, v in best.items() if k != "frames"}
        out.with_suffix(".json").write_text(json.dumps(no_frames, indent=2), encoding="utf-8")
        target_lines = args.min_lines if args.min_lines is not None else int(math.ceil(args.min_score / 100.0))
        raise SystemExit(
            f"best lines {best['lines']} did not reach min-lines {target_lines}; wrote best attempt"
        )
    raise SystemExit("no renderable episode produced")


def draw_board(board: np.ndarray, score: int, pieces: int, seed: int, cell: int = 12) -> np.ndarray:
    img = Image.new("RGB", (N_COLS * cell + 180, N_ROWS * cell), (12, 12, 12))
    draw = ImageDraw.Draw(img)
    for y in range(N_ROWS):
        for x in range(N_COLS):
            fill = (230, 220, 80) if board[y, x] else (30, 30, 30)
            draw.rectangle((x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1), fill=fill)
    ox = N_COLS * cell + 12
    draw.text((ox, 10), f"seed {seed}", fill=(255, 255, 255), font=font())
    draw.text((ox, 30), f"pieces {pieces}", fill=(255, 255, 255), font=font())
    draw.text((ox, 50), f"score {score}", fill=(255, 255, 0), font=font())
    return np.asarray(img)


def simulate(args: argparse.Namespace) -> None:
    import random

    weights = load_weights(args.weights, planner=args.planner)
    classic = args.planner in ("classic", "classic_internal")
    rng = random.Random(args.seed)
    names = list(PIECES.keys())
    board = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)
    lines = 0
    frames: list[np.ndarray] = []
    for pieces in range(1, args.max_pieces + 1):
        name = rng.choice(names)
        best = None
        for rot, col, after, got_lines, top, eroded in enumerate_model_placements(board, name):
            if classic:
                feats = classic_features(after, got_lines, N_ROWS - top, eroded)
            else:
                feats = model_features(after, got_lines, top, eroded)
            value = float(np.dot(weights, feats))
            if best is None or value > best[0]:
                best = (value, after, got_lines)
        if best is None:
            break
        board = best[1]
        lines += int(best[2])
        score = estimate_atari_score(lines)
        if args.out and (pieces == 1 or pieces % args.frame_every == 0 or score >= args.min_score):
            frames.append(draw_board(board, score, pieces, args.seed))
        if score >= args.min_score:
            break
    result = {"seed": args.seed, "score": estimate_atari_score(lines), "lines": lines, "pieces": pieces}
    print(json.dumps(result))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            frames.append(draw_board(board, result["score"], pieces, args.seed))
        imageio.mimsave(out, frames, fps=args.fps)
        out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out}")


def default_weights(planner: str) -> np.ndarray:
    if planner in ("classic", "classic_internal", "classic_calibrated", "lookahead_classic"):
        return CLASSIC_DEFAULT_WEIGHTS.copy()
    return (
        MODEL_DEFAULT_WEIGHTS.copy()
        if planner in ("model", "legacy_model", "legacy_calibrated", "model_calibrated", "lookahead_model")
        else DEFAULT_WEIGHTS.copy()
    )


def feature_names(planner: str) -> tuple[str, ...]:
    if planner in ("classic", "classic_internal", "classic_calibrated", "lookahead_classic"):
        return CLASSIC_FEATURE_NAMES
    return MODEL_FEATURE_NAMES if planner in ("model", "legacy_model", "legacy_calibrated", "model_calibrated", "lookahead_model") else FEATURE_NAMES


def load_weights(path: str | None, *, planner: str = "clone") -> np.ndarray:
    default = default_weights(planner)
    if path:
        weights = np.load(path)
        if len(weights) < len(default):
            padded = default.copy()
            padded[: len(weights)] = weights
            return padded
        return weights
    return default


def evaluate(args: argparse.Namespace) -> None:
    weights = load_weights(args.weights, planner=args.planner)
    rows = []
    for i in range(args.episodes):
        stats = play_episode(
            weights,
            seed=args.seed + i,
            sticky=args.sticky,
            max_pieces=args.max_pieces,
            depth=args.depth,
            beam=args.beam,
            top_k=args.top_k,
            planner=args.planner,
        )
        rows.append(stats)
        print({k: v for k, v in stats.items() if k != "frames"})
    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    lines = np.array([r["lines"] for r in rows], dtype=np.float64)
    decisions = np.array([r["decisions"] for r in rows], dtype=np.float64)
    summary = {
        "planner": args.planner,
        "weights": args.weights,
        "episodes": args.episodes,
        "seed_start": args.seed,
        "max_pieces": args.max_pieces,
        "sticky": args.sticky,
        "depth": args.depth,
        "beam": args.beam,
        "top_k": args.top_k,
        "mean_score": float(scores.mean()),
        "max_score": float(scores.max()),
        "mean_lines": float(lines.mean()),
        "max_lines": int(lines.max()),
        "mean_decisions": float(decisions.mean()),
        "rows": [{k: v for k, v in row.items() if k != "frames"} for row in rows],
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    print(f"mean_score={scores.mean():.1f} max_score={scores.max():.1f} mean_lines={lines.mean():.1f}")


def smoke(_: argparse.Namespace) -> None:
    env = make_env(render_mode="rgb_array")
    print("env:", ENV_ID)
    print("obs:", env.observation_space)
    print("actions:", env.unwrapped.get_action_meanings())
    obs, info = env.reset(seed=0)
    print("reset:", obs.shape, info)
    print("decoded cells:", int(decode_board(obs).sum()))
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")

    tp = sub.add_parser("train")
    tp.add_argument("--outdir", default="runs/cem")
    tp.add_argument("--generations", type=int, default=30)
    tp.add_argument("--population", type=int, default=24)
    tp.add_argument("--elite-frac", type=float, default=0.25)
    tp.add_argument("--init-std", type=float, default=1.0)
    tp.add_argument("--noise-floor", type=float, default=0.05)
    tp.add_argument("--rollouts", type=int, default=1)
    tp.add_argument("--max-pieces", type=int, default=180)
    tp.add_argument("--sticky", type=float, default=0.0)
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--warm-start", default=None)
    tp.add_argument("--top-k", type=int, default=48)
    planner_choices = (
        "clone",
        "model",
        "legacy_model",
        "legacy_calibrated",
        "classic",
        "classic_internal",
        "model_calibrated",
        "classic_calibrated",
        "lookahead_model",
        "lookahead_classic",
    )
    tp.add_argument("--planner", choices=planner_choices, default="model")

    ep = sub.add_parser("evaluate")
    ep.add_argument("--weights", default=None)
    ep.add_argument("--episodes", type=int, default=3)
    ep.add_argument("--max-pieces", type=int, default=500)
    ep.add_argument("--sticky", type=float, default=0.0)
    ep.add_argument("--seed", type=int, default=100)
    ep.add_argument("--depth", type=int, default=2)
    ep.add_argument("--beam", type=int, default=6)
    ep.add_argument("--top-k", type=int, default=48)
    ep.add_argument("--planner", choices=planner_choices, default="model")
    ep.add_argument("--out", default=None)

    rp = sub.add_parser("render")
    rp.add_argument("--weights", default=None)
    rp.add_argument("--out", default="runs/videos/agent.mp4")
    rp.add_argument("--min-score", type=int, default=2000)
    rp.add_argument("--min-lines", type=int, default=None)
    rp.add_argument("--tries", type=int, default=20)
    rp.add_argument("--max-pieces", type=int, default=1000)
    rp.add_argument("--max-frames", type=int, default=120000)
    rp.add_argument("--sticky", type=float, default=0.0)
    rp.add_argument("--seed", type=int, default=1000)
    rp.add_argument("--fps", type=int, default=30)
    rp.add_argument("--depth", type=int, default=2)
    rp.add_argument("--beam", type=int, default=6)
    rp.add_argument("--top-k", type=int, default=48)
    rp.add_argument("--planner", choices=planner_choices, default="model")

    sp = sub.add_parser("simulate")
    sp.add_argument("--weights", default=None)
    sp.add_argument("--planner", choices=("model", "classic", "classic_internal"), default="classic")
    sp.add_argument("--max-pieces", type=int, default=10000)
    sp.add_argument("--min-score", type=int, default=200000)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default=None)
    sp.add_argument("--frame-every", type=int, default=50)
    sp.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    if args.cmd == "smoke":
        smoke(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "evaluate":
        evaluate(args)
    elif args.cmd == "render":
        render(args)
    elif args.cmd == "simulate":
        simulate(args)
    print(f"elapsed={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
