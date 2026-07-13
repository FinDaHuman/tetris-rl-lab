import numpy as np
import pytest

from tetris_env.engine import COLS, VISIBLE_ROWS, enumerate_placements
from tetris_env.placement_env import ACTION_COUNT, OBS_SIZE, PlacementTetrisEnv


def test_spaces():
    env = PlacementTetrisEnv(seed=0)
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == ACTION_COUNT == 40
    assert obs.shape == (OBS_SIZE,) == (214,)
    assert env.observation_space.contains(obs)


def test_observation_is_raw_board_only_no_hand_features():
    """The comparison against Track 3 is void if the agent is fed engineered features.

    The observation must be exactly: visible board (200) + current one-hot (7) +
    next one-hot (7). Nothing derived -- no holes, heights, bumpiness, transitions.
    """
    env = PlacementTetrisEnv(seed=3)
    obs, _ = env.reset(seed=3)
    for _ in range(20):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            break
    board = obs[: VISIBLE_ROWS * COLS].reshape(VISIBLE_ROWS, COLS)
    assert np.array_equal(board, env.game.visible_board.astype(np.float32))
    one_hots = obs[VISIBLE_ROWS * COLS :]
    assert one_hots.sum() == 2.0  # exactly one current + one next piece, nothing else
    assert obs.size == VISIBLE_ROWS * COLS + 14


@pytest.mark.parametrize("seed", [0, 1, 5])
def test_every_action_resolves_to_a_legal_placement(seed):
    """No action masking, so all 40 (rotation, column) pairs must map onto a real placement."""
    env = PlacementTetrisEnv(seed=seed)
    env.reset(seed=seed)
    for _ in range(40):
        placements = enumerate_placements(env.game)
        if not placements:
            break
        for action in range(ACTION_COUNT):
            assert env._resolve(action, placements) in placements
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            break


def test_one_step_is_exactly_one_piece():
    env = PlacementTetrisEnv(seed=2)
    env.reset(seed=2)
    for expected in range(1, 15):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        if term:
            break
        assert info["pieces"] == expected
        if trunc:
            break


def test_reward_matches_track3_lines_mode():
    """Reward must be identical to TetrisScoreEnv's `lines` mode or the tracks aren't comparable."""
    env = PlacementTetrisEnv(seed=11, line_reward=10.0, piece_reward=0.25, top_out_penalty=10.0)
    env.reset(seed=11)
    for _ in range(60):
        lines_before = env.game.lines
        _, reward, term, trunc, _ = env.step(env.action_space.sample())
        cleared = env.game.lines - lines_before
        expected = 10.0 * float(cleared**2) + 0.25
        if term:
            expected -= 10.0
        assert reward == pytest.approx(expected)
        if term or trunc:
            break


def test_top_out_terminates_with_penalty():
    env = PlacementTetrisEnv(seed=4, max_pieces=10_000)
    env.reset(seed=4)
    terminated = False
    for _ in range(5_000):
        _, reward, terminated, truncated, _ = env.step(0)  # always rotation 0, column 0: fills a corner
        if terminated:
            assert reward < 0
            break
        if truncated:
            break
    assert terminated, "hammering one column should top out"


class _FixedPolicy:
    """Stands in for a PPO model: cycles actions deterministically, no torch needed."""

    def __init__(self, actions):
        self._actions = list(actions)
        self._i = 0

    def predict(self, obs, deterministic=True):
        action = self._actions[self._i % len(self._actions)]
        self._i += 1
        return np.array([action]), None


class _NullWriter:
    """Accepts frames like VideoWriter but keeps none of them."""

    def __init__(self):
        self.frames = 0

    def append(self, frame, hold=1):
        self.frames += hold


def test_render_does_not_change_reported_stats():
    """Capturing a video must not alter the episode's score, lines, or pieces.

    The rendering path animates each placement with real engine actions, and
    ``TetrisGame.step(HARD_DROP)`` credits drop points to ``game.score`` -- while
    the evaluation path teleports the board and counts line clears only. If the two
    diverge, the published video and manifest report a score no evaluation can
    reproduce (this happened: 1846 vs 900 on seed 2).
    """
    from agents.custom.afterstate_custom_agent import play_and_render

    policy = _FixedPolicy([5, 12, 19, 26, 33, 2, 9, 16])
    plain = play_and_render(policy, None, seed=11, max_pieces=60)

    policy._i = 0  # same action sequence, now with capture
    captured = play_and_render(policy, None, seed=11, max_pieces=60, writer=_NullWriter())

    for key in ("score", "lines", "pieces"):
        assert plain[key] == captured[key], f"{key} differs with rendering: {plain} vs {captured}"
    assert captured["frames"] > 0, "writer should have received frames"
