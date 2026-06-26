from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "tetris_env"))

from agents.custom.tetris_custom_agent import play_episode
from agents.custom.pure_rl_custom_agent import make_env as make_pure_rl_custom_env
from tetris_env import DEFAULT_WEIGHTS
from tetris_env.engine import PIECES, TetrisGame
from tetris_env.gym_env import TetrisScoreEnv


def test_next_piece_is_visible_to_env_and_viewer() -> None:
    env = TetrisScoreEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs["current_piece"] in range(len(PIECES))
    assert obs["next_piece"] in range(len(PIECES))
    assert obs["active"].shape == obs["board"].shape
    assert obs["piece_state"].shape == (3,)
    assert info["current_piece"] in PIECES
    assert info["current_rotation"] in range(4)
    assert info["next_piece"] in PIECES
    assert "next:" in env.render()


def test_game_has_one_piece_preview() -> None:
    game = TetrisGame(seed=0)
    assert game.next_piece in PIECES
    assert game.current is not None
    assert game.current.name != ""


def test_custom_policy_runs_short_score_episode() -> None:
    stats = play_episode(weights=DEFAULT_WEIGHTS, seed=0, max_pieces=25)
    assert stats["pieces"] > 0
    assert stats["score"] >= 0


def test_pure_rl_custom_observation_is_flat_and_includes_preview() -> None:
    env = make_pure_rl_custom_env(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (20 * 10 * 2 + 3 + len(PIECES) * 2,)
    assert obs.dtype.name == "float32"
