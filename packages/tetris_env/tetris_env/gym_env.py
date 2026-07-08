from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .engine import Action, COLS, HIDDEN_ROWS, PIECES, ROWS, VISIBLE_ROWS, TetrisGame


PIECE_TO_INDEX = {name: idx for idx, name in enumerate(PIECES)}


REWARD_MODES = ("score", "lines")


class TetrisScoreEnv(gym.Env):
    """Custom Tetris environment with selectable reward shaping.

    Reward modes:

    - ``score``: raw engine score deltas (drop points, line scores, -1000 on
      top-out). Kept for backward compatibility with older checkpoints. Known
      issue for pure RL: per-cell drop points reward fast dropping far more
      densely than line clears, so PPO converges to dropping without clearing.
    - ``lines``: ``line_reward * cleared**2`` per lock, plus ``piece_reward``
      per locked piece, minus ``top_out_penalty`` on termination. No drop
      points. Squared line count rewards multi-line clears; the per-piece term
      is aligned with survival because the board tops out unless lines clear.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        max_pieces: int = 5000,
        seed: int | None = None,
        reward_mode: str = "score",
        line_reward: float = 10.0,
        piece_reward: float = 0.25,
        top_out_penalty: float = 10.0,
    ):
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        self.max_pieces = max_pieces
        self.reward_mode = reward_mode
        self.line_reward = float(line_reward)
        self.piece_reward = float(piece_reward)
        self.top_out_penalty = float(top_out_penalty)
        self._seed = seed
        self.game = TetrisGame(seed=seed)
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(0, 1, shape=(VISIBLE_ROWS, COLS), dtype=np.uint8),
                "active": spaces.Box(0, 1, shape=(VISIBLE_ROWS, COLS), dtype=np.uint8),
                "current_piece": spaces.Discrete(len(PIECES)),
                "piece_state": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
                "next_piece": spaces.Discrete(len(PIECES)),
            }
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.game = TetrisGame(seed=self._seed if seed is None else seed)
        return self._obs(), self._info()

    def step(self, action):
        lines_before = self.game.lines
        pieces_before = self.game.pieces
        result = self.game.step(action)
        terminated = result.terminated
        truncated = self.game.pieces >= self.max_pieces
        if self.reward_mode == "score":
            reward = result.reward
        else:
            cleared = self.game.lines - lines_before
            placed = self.game.pieces - pieces_before
            reward = self.line_reward * float(cleared * cleared) + self.piece_reward * float(placed)
            if terminated:
                reward -= self.top_out_penalty
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self):
        rows = []
        active = set(self.game.cells())
        for row in range(2, 22):
            cells = []
            for col in range(10):
                cells.append("@" if (row, col) in active else ("#" if self.game.board[row, col] else "."))
            rows.append("".join(cells))
        preview = f"next: {self.game.next_piece}"
        return preview + "\n" + "\n".join(rows)

    def _obs(self):
        current_name = self.game.current.name if self.game.current is not None else self.game.next_piece
        return {
            "board": self.game.visible_board,
            "active": self._active_board(),
            "current_piece": PIECE_TO_INDEX[current_name],
            "piece_state": self._piece_state(),
            "next_piece": PIECE_TO_INDEX[self.game.next_piece],
        }

    def _info(self):
        current_name = self.game.current.name if self.game.current is not None else None
        return {
            "score": self.game.score,
            "lines": self.game.lines,
            "pieces": self.game.pieces,
            "level": self.game.level,
            "current_piece": current_name,
            "current_row": None if self.game.current is None else self.game.current.row,
            "current_col": None if self.game.current is None else self.game.current.col,
            "current_rotation": None if self.game.current is None else self.game.current.rotation,
            "next_piece": self.game.next_piece,
        }

    def _active_board(self) -> np.ndarray:
        active = np.zeros((VISIBLE_ROWS, COLS), dtype=np.uint8)
        for row, col in self.game.cells():
            if HIDDEN_ROWS <= row < ROWS and 0 <= col < COLS:
                active[row - HIDDEN_ROWS, col] = 1
        return active

    def _piece_state(self) -> np.ndarray:
        if self.game.current is None:
            return np.zeros(3, dtype=np.float32)
        return np.array(
            [
                self.game.current.row / ROWS,
                self.game.current.col / COLS,
                self.game.current.rotation / 3.0,
            ],
            dtype=np.float32,
        )
