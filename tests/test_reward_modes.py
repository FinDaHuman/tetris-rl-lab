from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

from tetris_env.engine import Action, HIDDEN_ROWS, PieceState, ROWS
from tetris_env.gym_env import TetrisScoreEnv


def test_invalid_reward_mode_rejected() -> None:
    with pytest.raises(ValueError):
        TetrisScoreEnv(reward_mode="bogus")


def test_score_mode_still_pays_drop_points() -> None:
    env = TetrisScoreEnv(seed=0, reward_mode="score")
    env.reset(seed=0)
    _, reward, _, _, info = env.step(Action.HARD_DROP)
    # Hard drop with no line clear pays 2 points per dropped cell in score mode.
    assert reward >= 2.0
    assert info["lines"] == 0


def test_lines_mode_pays_no_drop_points() -> None:
    env = TetrisScoreEnv(seed=0, reward_mode="lines", piece_reward=0.25)
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step(Action.HARD_DROP)
    # One piece locked, zero lines cleared: only the per-piece bonus.
    assert not terminated
    assert info["lines"] == 0
    assert reward == pytest.approx(0.25)


def test_lines_mode_rewards_line_clears_quadratically() -> None:
    env = TetrisScoreEnv(seed=0, reward_mode="lines", line_reward=10.0, piece_reward=0.25)
    env.reset(seed=0)
    # Bottom row full except where a forced O piece (columns 4-5) will land.
    env.game.board[ROWS - 1, :] = 1
    env.game.board[ROWS - 1, 4] = 0
    env.game.board[ROWS - 1, 5] = 0
    env.game.current = PieceState("O", 0, 3, 0)
    _, reward, _, _, info = env.step(Action.HARD_DROP)
    assert info["lines"] == 1
    assert reward == pytest.approx(10.0 * 1 + 0.25)


def test_lines_mode_penalizes_top_out() -> None:
    env = TetrisScoreEnv(seed=0, reward_mode="lines", piece_reward=0.25, top_out_penalty=10.0)
    env.reset(seed=0)
    # Fill everything below the hidden rows except column 0 so no line clears,
    # then lock a piece at the top; the next spawn must collide.
    env.game.board[HIDDEN_ROWS:, 1:] = 1
    env.game.current = PieceState("O", 0, 3, 0)
    _, reward, terminated, _, info = env.step(Action.HARD_DROP)
    assert terminated
    assert info["lines"] == 0
    assert reward == pytest.approx(0.25 - 10.0)
