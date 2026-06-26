# Project Status

Active objective: keep four clear Tetris agent tracks in this repo.

No longer active: the real `ALE/Tetris-v5` 2000-line target.

Track 1: pure RL on ALE:

- CLI: `agents/ale/pure_rl_ale_agent.py`
- Default output: `artifacts/ale_pure_rl/ppo_ale_pure.zip`
- Policy input: raw Atari RGB frames only
- Default evaluation condition: sticky actions with `--sticky 0.25`

Track 2: tool-assisted high score on ALE:

- Checkpoint: `artifacts/ale_37_line/best_weights.npy`
- Verified result: 37 lines, score 3700, 259 pieces on seed 0
- Original implementation: `agents/ale/ale_tetris_agent.py`

Stable ALE copy:

- Checkpoint: `artifacts/ale_stable_high_score/best_weights.npy`
- Planner: `legacy_model`
- Verified seeds: 0 through 9
- Result on every checked seed: 37 lines, score 3700, 259 pieces
- Manifest: `artifacts/ale_stable_high_score/evaluation.json`

Track 3: pure RL on custom env:

- CLI: `agents/custom/pure_rl_custom_agent.py`
- Default output: `artifacts/custom_pure_rl/ppo_custom_pure.zip`
- Policy input: locked board, active falling-piece mask, current piece state,
  current piece identity, and one next piece
- Uses only the Gymnasium step API

Track 4: tool-assisted high score on custom env:

- Engine package: `packages/tetris_env`
- Agent CLI: `agents/custom/tetris_custom_agent.py`
- Viewer CLI: `agents/custom/render_custom_episode.py`
- Default best-model output: `artifacts/custom_best/best_weights.npy`
- Initial smoke-trained custom model: mean score 31900.0 over three 200-piece
  episodes seeded at 0, 1, and 2

Important environment behavior:

- Pure RL can see the current piece state and exactly one next piece.
- The viewer renders the same one-piece preview in a `NEXT` panel.
- The tool-assisted custom policy uses one-piece lookahead when scoring
  placements.
- The custom engine remains the source of standard Tetris rules and physics:
  hidden spawn rows, 7-bag pieces, SRS-style kicks, hard/soft drops, line clears,
  levels, and score-mode line scoring.
