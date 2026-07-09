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


def _spawn_pair(env: TetrisScoreEnv) -> tuple[str, str]:
    return env.game.current.name, env.game.next_piece


def test_unseeded_resets_give_new_piece_sequences() -> None:
    env = TetrisScoreEnv(seed=0)
    pairs = []
    for _ in range(10):
        env.reset()
        pairs.append(_spawn_pair(env))
    assert len(set(pairs)) > 1


def test_env_seed_makes_episode_stream_reproducible() -> None:
    env_a = TetrisScoreEnv(seed=123)
    env_b = TetrisScoreEnv(seed=123)
    for _ in range(5):
        env_a.reset()
        env_b.reset()
        assert _spawn_pair(env_a) == _spawn_pair(env_b)


def test_explicit_reset_seed_is_deterministic() -> None:
    env = TetrisScoreEnv()
    env.reset(seed=42)
    first = _spawn_pair(env)
    env.reset()  # advance the stream
    env.reset(seed=42)
    assert _spawn_pair(env) == first


def test_forced_lock_bounds_steps_per_piece() -> None:
    env = TetrisScoreEnv(seed=0, reward_mode="lines", max_steps_per_piece=3)
    env.reset(seed=0)
    rewards = []
    for _ in range(3):
        _, reward, terminated, truncated, info = env.step(0)  # NOOP
        rewards.append(reward)
        assert not (terminated or truncated)
    # A fresh piece needs ~20 gravity steps to lock naturally, so with a
    # 3-step cap the third step must have force-locked it.
    assert info["pieces"] == 1
    assert rewards[-1] >= env.piece_reward


def test_every_constant_action_policy_terminates_within_bound() -> None:
    max_pieces = 30
    for action in range(TetrisScoreEnv(seed=0).action_space.n):
        env = TetrisScoreEnv(seed=0, reward_mode="lines", max_pieces=max_pieces)
        env.reset(seed=action)
        bound = max_pieces * (env.max_steps_per_piece + 1)
        for step in range(bound):
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        assert terminated or truncated, f"action {action} never ended within {bound} steps"
