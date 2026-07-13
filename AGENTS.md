# Agent Track Boundaries

This repository intentionally has two academically constrained pure-RL tracks
and two unconstrained tool-assisted tracks.

## Pure RL Tracks

The pure RL tracks should remain suitable for university reinforcement learning
courses that use ALE or Gym-style environments. Do not add pretrained policies,
expert demonstrations, imitation learning, offline datasets, board decoders,
placement enumeration, game-state cloning for action selection, model-based
planning, hand-authored Tetris heuristics, direct board edits, or search to
these tracks.

- `agents/ale/pure_rl_ale_agent.py`
- `agents/custom/pure_rl_custom_agent.py`

These agents should learn only from environment observations, environment
rewards, and environment actions. Using a standard RL implementation such as
Stable-Baselines3 PPO is acceptable as long as training starts from scratch and
the course allows external RL libraries.

## Tool-Assisted Tracks

The non-pure-RL tracks are allowed to use any useful technique, including
planning, search, heuristic features, board/frame decoding, engine cloning,
CEM-style weight optimization, saved optimized weights, and other tool-assisted
methods.

- `agents/ale/ale_tetris_agent.py`
- `agents/custom/tetris_custom_agent.py`
- `agents/custom/render_custom_episode.py`

Keep artifacts under the matching track directories documented in
`docs/MODELS.md`.

## Shared Visualization Code (all four tracks)

Rendering is not assistance — every track may draw itself. These files are shared
and carry no track allegiance:

- `packages/tetris_env/tetris_env/render.py` — draws a game state as an RGB frame.
- `agents/video.py` — streaming mp4 writer.
- `tools/render_best_plays.py`, `artifacts/best_plays/live_play.py` — orchestration
  and the live viewer.

One file needs care: **`packages/tetris_env/tetris_env/replay.py`** converts a
planner's `Placement` back into primitive actions so the tool-assisted tracks can
be animated instead of teleporting pieces. It is planning-adjacent code sitting in
the shared engine package, so:

**A pure-RL track must never import `replay.py`, and must never call
`enumerate_placements`, `TetrisGame.clone`, or touch `game.board` directly — in
training, in evaluation, or in rendering.** A pure-RL agent renders by drawing the
states it reached through the Gymnasium step API, nothing more. If you find
yourself wanting placement search to make a pure-RL video look better, the video is
telling you the truth about the agent and should be left alone.

## Reporting Notes

This project will need a later report covering progress, what was built, how it
was built, and how each agent track performed. Future agents should preserve
run evidence, evaluation JSON, terminal transcripts, and session notes for that
report.

- Report planning notes: `docs/REPORTING_NOTES.md`
- Latest detailed session log: `docs/SESSION_2026-07-07.md`
- Immediate next commands: `NEXT_STEPS.md`
