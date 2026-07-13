"""Turn a planner's ``Placement`` back into primitive engine actions.

The tool-assisted agents pick a final ``Placement`` and commit it by overwriting
the board, which is correct but unwatchable: pieces teleport. To animate a
placement we need the rotate/shift/drop sequence that reaches it.

Two things make this non-trivial and force a verified replay rather than a
naive one: ``_try_rotate`` applies SRS kicks (so the column after rotating is
not the column before), and ``TetrisGame.step`` applies gravity after every
non-drop action (so a piece descends while it is still being manoeuvred, and
can even lock early against a tall stack). ``placement_actions`` therefore
simulates the candidate sequence on a clone and only returns it if the
resulting board is bit-identical to the placement the planner asked for.
"""

from __future__ import annotations

import numpy as np

from .engine import Action, PIECES, Placement, PieceState, ROWS, TetrisGame


def placement_actions(game: TetrisGame, placement: Placement) -> list[Action] | None:
    """Primitive actions reaching ``placement``, or None if gravity/kicks block it.

    A None return is not a bug: from a tall stack there may be no rotate-then-
    shift path that survives per-step gravity. Callers fall back to committing
    the placement directly.
    """
    if game.current is None or game.game_over:
        return None

    sim = game.clone()
    actions: list[Action] = []
    pieces_before = game.pieces

    def advance(action: Action) -> bool:
        sim.step(action)
        actions.append(action)
        # Gravity may have locked the piece mid-manoeuvre; the sequence is dead.
        return not (sim.game_over or sim.current is None or sim.pieces != pieces_before)

    turns = (placement.rotation - game.current.rotation) % 4
    rotations = [Action.ROTATE_CCW] if turns == 3 else [Action.ROTATE_CW] * turns
    for action in rotations:
        if not advance(action):
            return None
    if sim.current.rotation != placement.rotation % 4 and placement.piece != "O":
        return None

    delta = placement.col - sim.current.col
    shift = Action.RIGHT if delta > 0 else Action.LEFT
    for _ in range(abs(delta)):
        col_before = sim.current.col
        if not advance(shift):
            return None
        if sim.current.col == col_before:
            return None  # wall or stack blocked the shift

    sim.step(Action.HARD_DROP)
    actions.append(Action.HARD_DROP)
    if not np.array_equal(sim.board, placement.board):
        return None
    return actions


def placement_drop_states(game: TetrisGame, placement: Placement) -> list[PieceState]:
    """Synthetic fly-in path for a placement, used when ``placement_actions`` fails.

    Purely visual: shows the piece rotating, sliding across and dropping into the
    target slot. The caller still commits the planner's placement, so the game
    state stays exactly what the planner (and the recorded evaluation) produced.
    """
    if game.current is None:
        return []
    name = placement.piece
    start_row = game.current.row
    start_col = game.current.col
    rot = placement.rotation

    states: list[PieceState] = []
    step = 1 if placement.col >= start_col else -1
    for col in range(start_col, placement.col + step, step):
        states.append(PieceState(name, start_row, col, rot))

    shape = PIECES[name][rot % 4]
    row = start_row
    while True:
        nxt = PieceState(name, row + 1, placement.col, rot)
        cells = ((nxt.row + r, nxt.col + c) for r, c in shape)
        if any(rr >= ROWS or (rr >= 0 and game.board[rr, cc]) for rr, cc in cells):
            break
        row += 1
        states.append(nxt)
    return states
