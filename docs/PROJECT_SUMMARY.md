# Project Summary

TetrisGPT is a Python workspace for comparing pure reinforcement learning and
tool-assisted Tetris agents across two environments: real Atari Tetris through
ALE and a local custom score-mode Tetris engine.

## Goal

The project keeps four agent tracks separate:

1. Pure RL on ALE.
2. Tool-assisted high score on ALE.
3. Pure RL on the custom environment with one-piece preview.
4. Tool-assisted high score on the custom environment.

This separation matters because the tracks answer different questions. Pure RL
tracks test what a policy can learn from environment interaction. Tool-assisted
tracks test how high the project can score when planning, search, and engineered
features are allowed.

## Repository Layout

```text
agents/
  ale/
    pure_rl_ale_agent.py       Pure RL ALE PPO track.
    ale_tetris_agent.py        Tool-assisted ALE high-score track.
  custom/
    pure_rl_custom_agent.py    Pure RL custom PPO track.
    tetris_custom_agent.py     Tool-assisted custom high-score track.
    render_custom_episode.py   Custom-env video renderer.
packages/
  tetris_env/
    tetris_env/engine.py       Custom Tetris rules and physics.
    tetris_env/gym_env.py      Gymnasium API for pure RL.
    tetris_env/features.py     Tool-assisted placement features.
artifacts/
  ...                          Saved model outputs by track.
tests/
  test_custom_env.py           Custom env and policy smoke tests.
```

## Environment Boundaries

ALE tracks use `ALE/Tetris-v5` through Gymnasium/ALE. The pure RL ALE agent sees
only raw RGB frames. The tool-assisted ALE agent can decode frames and use a
model of the board to choose actions.

Custom tracks use the local engine in `packages/tetris_env`. The pure RL custom
agent uses the Gymnasium API and receives a one-piece preview. The tool-assisted
custom agent uses the same engine rules but can enumerate placements and score
candidate boards.

## Current State

- The custom engine implements a 10x20 visible board, two hidden spawn rows,
  seven tetrominoes, a 7-bag generator, SRS-style kicks, hard drop, soft drop,
  line clears, levels, and score-mode line scoring.
- The pure RL custom observation includes locked board, active piece mask,
  current piece state, current piece identity, and exactly one next piece.
- The ALE high-score archive contains a verified 37-line model.
- The custom high-score track has an initial smoke-trained model and renderer
  output under `artifacts/custom_best`.

## How To Work In This Repo

Use the model track first, then choose the matching files:

- For pure ALE RL, edit `agents/ale/pure_rl_ale_agent.py`.
- For ALE high-score planning, edit `agents/ale/ale_tetris_agent.py`.
- For pure custom RL, edit `agents/custom/pure_rl_custom_agent.py` or the
  Gymnasium surface in `packages/tetris_env/tetris_env/gym_env.py`.
- For custom high-score planning, edit `agents/custom/tetris_custom_agent.py`,
  `packages/tetris_env/tetris_env/features.py`, or the engine if rules need to
  change.

The custom engine is shared by both custom tracks. Changes to engine behavior
can affect pure RL, tool-assisted training, rendering, and tests, so those
changes should be verified with `python -m pytest`.

## Baseline Checks

```bash
python -m pytest
python agents/ale/pure_rl_ale_agent.py smoke
python agents/custom/pure_rl_custom_agent.py smoke
python ale_tetris_agent.py smoke
python agents/custom/tetris_custom_agent.py evaluate --weights artifacts/custom_best/best_weights.npy --episodes 1
```
