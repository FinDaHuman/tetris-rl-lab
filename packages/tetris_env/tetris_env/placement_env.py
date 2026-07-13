"""Afterstate (placement-level) Tetris environment — Track 5.

This exists to answer the one question the four-track study cannot: Track 4 beats
Track 3 by ~190x while changing *four* things at once (action space, hand-authored
features, queue lookahead, optimizer), so the project cannot say which one carries
it. This env changes exactly one of them.

It is deliberately austere. Relative to ``TetrisScoreEnv`` (Track 3) it changes
**only the action space**: one action places the current piece, instead of ~9
primitive nudges under gravity. Everything else is held fixed — same engine, same
``lines`` reward, and the observation is still the raw board plus piece identities.

What it must NOT have, or the comparison is worthless:

- **No hand-authored features.** No holes, heights, bumpiness, transitions. The
  agent sees the raw board and has to learn what a bad board looks like, exactly
  as Track 3 does. (Track 4 is handed all of that.)
- **No lookahead / search.** The agent gets the same one-piece preview Track 3
  gets. It may not evaluate future placements. (Track 4 searches 2 ply.)

So: Track 3 : primitive actions :: Track 5 : placement actions, all else equal.
The gap between them is the value of the action abstraction, in isolation.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .engine import COLS, PIECES, VISIBLE_ROWS, TetrisGame, enumerate_placements


PIECE_TO_INDEX = {name: idx for idx, name in enumerate(PIECES)}
PIECE_COUNT = len(PIECES)

N_ROTATIONS = 4
# Action = (rotation, column). Not every pair is legal (rotations are deduped for
# symmetric pieces, and columns depend on the piece's width), so the pair is
# resolved to the nearest legal placement rather than masked -- see _resolve().
ACTION_COUNT = N_ROTATIONS * COLS

OBS_SIZE = VISIBLE_ROWS * COLS + PIECE_COUNT * 2  # 200 board + 7 current + 7 next


class PlacementTetrisEnv(gym.Env):
    """One step = one piece placed. Reward is identical to TetrisScoreEnv's `lines` mode."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        max_pieces: int = 500,
        seed: int | None = None,
        line_reward: float = 10.0,
        piece_reward: float = 0.25,
        top_out_penalty: float = 10.0,
    ):
        self.max_pieces = int(max_pieces)
        self.line_reward = float(line_reward)
        self.piece_reward = float(piece_reward)
        self.top_out_penalty = float(top_out_penalty)
        self._seed = seed
        self.game = TetrisGame(seed=seed)
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is None and self._seed is not None:
            seed = self._seed
            self._seed = None
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.game = TetrisGame(seed=game_seed)
        return self._obs(), self._info()

    def step(self, action: int):
        placements = enumerate_placements(self.game)
        if not placements or self.game.game_over:
            # No legal placement: the stack has reached the spawn row.
            return self._obs(), -self.top_out_penalty, True, False, self._info()

        placement = self._resolve(int(action), placements)

        # Commit the placement. This is the same board transition the engine would
        # reach by rotating, shifting and hard-dropping into that slot; it is
        # verified against the primitive-action path in tests (placement_actions).
        self.game.board = placement.board.copy()
        self.game.lines += placement.lines
        self.game.score += placement.score_delta
        self.game.level = self.game.lines // 10
        self.game.pieces += 1
        self.game.spawn()

        terminated = self.game.game_over
        truncated = self.game.pieces >= self.max_pieces
        reward = self.line_reward * float(placement.lines**2) + self.piece_reward
        if terminated:
            reward -= self.top_out_penalty
        return self._obs(), reward, terminated, truncated, self._info()

    def _resolve(self, action: int, placements):
        """Map a (rotation, column) action onto a legal placement.

        Symmetric pieces have fewer than four distinct rotations and wide pieces
        cannot reach every column, so roughly half the 40 actions are illegal in any
        given state. Rather than depend on action masking (which SB3's PPO does not
        support), an illegal pair snaps to the legal placement closest to it —
        matching the requested rotation first, then the requested column. The
        mapping is deterministic, so the policy can learn it.
        """
        want_rot, want_col = divmod(action, COLS)
        same_rot = [p for p in placements if p.rotation == want_rot]
        pool = same_rot if same_rot else placements
        return min(pool, key=lambda p: (abs(p.col - want_col), abs(p.rotation - want_rot)))

    def _obs(self) -> np.ndarray:
        current = self.game.current.name if self.game.current is not None else self.game.next_piece
        current_piece = np.zeros(PIECE_COUNT, dtype=np.float32)
        next_piece = np.zeros(PIECE_COUNT, dtype=np.float32)
        current_piece[PIECE_TO_INDEX[current]] = 1.0
        next_piece[PIECE_TO_INDEX[self.game.next_piece]] = 1.0
        return np.concatenate(
            (
                self.game.visible_board.astype(np.float32).reshape(-1),
                current_piece,
                next_piece,
            )
        )

    def _info(self):
        return {
            "score": self.game.score,
            "lines": self.game.lines,
            "pieces": self.game.pieces,
            "level": self.game.level,
        }

    def render(self):
        return self.game.board[2:]
