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
