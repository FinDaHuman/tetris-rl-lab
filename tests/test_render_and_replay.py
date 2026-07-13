import numpy as np
import pytest

from tetris_env import TetrisGame, enumerate_placements
from tetris_env.engine import Action, COLS, HIDDEN_ROWS, ROWS
from tetris_env.render import HEIGHT, WIDTH, ghost_state, render_frame
from tetris_env.replay import placement_actions, placement_drop_states


def test_render_frame_shape_on_fresh_game():
    game = TetrisGame(seed=0)
    frame = render_frame(game, title="t", seed=0)
    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert frame.dtype == np.uint8


def test_render_frame_after_locks_and_game_over():
    game = TetrisGame(seed=3)
    for _ in range(30):
        game.step(Action.HARD_DROP)
    frame = render_frame(game, score=123, flash=True)
    assert frame.shape == (HEIGHT, WIDTH, 3)
    # A stack exists, so the frame is not a uniform background.
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 2


def test_ghost_is_at_or_below_the_live_piece_and_rests_on_something():
    game = TetrisGame(seed=1)
    game.step(Action.HARD_DROP)
    ghost = ghost_state(game)
    assert ghost is not None
    assert ghost.row >= game.current.row
    assert ghost.col == game.current.col and ghost.rotation == game.current.rotation
    # One row lower must collide, otherwise it is not a landing position.
    from tetris_env.engine import PieceState

    below = PieceState(ghost.name, ghost.row + 1, ghost.col, ghost.rotation)
    assert game.collides(below)


@pytest.mark.parametrize("seed", [0, 1, 2, 5, 11])
def test_placement_actions_reproduce_the_planner_board(seed):
    """Every action sequence we return must land on exactly the planner's board.

    placement_actions is allowed to return None (per-step gravity or an SRS kick
    can make a placement unreachable), but it must never return a sequence that
    produces a different board -- that would silently desync the video from the
    result it claims to show.
    """
    game = TetrisGame(seed=seed)
    checked = 0
    reachable = 0
    for _ in range(40):
        placements = enumerate_placements(game)
        if not placements or game.game_over:
            break
        for placement in placements:
            actions = placement_actions(game, placement)
            checked += 1
            if actions is None:
                continue
            reachable += 1
            sim = game.clone()
            for action in actions:
                sim.step(action)
            assert np.array_equal(sim.board, placement.board)
            assert sim.pieces == game.pieces + 1
        # Advance the episode the way the planner does, to reach deeper boards.
        chosen = placements[len(placements) // 2]
        game.board = chosen.board.copy()
        game.lines += chosen.lines
        game.level = game.lines // 10
        game.pieces += 1
        game.spawn()
    assert checked > 0
    # On a shallow board almost everything should be reachable; if this collapses,
    # the video would be mostly synthetic fly-ins rather than real replays.
    assert reachable / checked > 0.5


def test_placement_drop_states_end_at_the_landing_slot():
    game = TetrisGame(seed=7)
    placement = enumerate_placements(game)[0]
    states = placement_drop_states(game, placement)
    assert states
    last = states[-1]
    assert last.col == placement.col
    assert last.rotation == placement.rotation
    for row, col in game.cells(last):
        assert 0 <= col < COLS
        assert row < ROWS
