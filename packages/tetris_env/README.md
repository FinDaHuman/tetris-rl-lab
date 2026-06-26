# `tetris_env`

Reusable score-mode Tetris environment for fast local agent training.

The engine models the pieces, a 7-bag generator, one-piece preview, two hidden
spawn rows, SRS-style wall kicks, hard and soft drops, line clears, and
guideline-style line scoring.

It exposes two surfaces:

- An action-step `gymnasium` environment for pure RL. Observations include the
  locked board, active falling-piece mask, current piece state, current piece
  identity, and exactly one next piece.
- A placement enumerator for tool-assisted high-score experiments that still use
  the same engine rules.
