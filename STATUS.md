# Project Status

Active objective: keep the Tetris agent tracks in this repo clearly separated —
four tracks in the 2x2 (env x method), plus Track 5, the controlled experiment
that isolates the action abstraction.

No longer active: the real `ALE/Tetris-v5` 2000-line target.

Track 1: pure RL on ALE:

- CLI: `agents/ale/pure_rl_ale_agent.py`
- Default output: `artifacts/ale_pure_rl/ppo_ale_pure.zip`
- Policy input: Atari frames only, with standard Atari preprocessing enabled by
  default for training
- Default evaluation condition: sticky actions with `--sticky 0.25`
- Final result (frozen 2026-07-13): 10M-frame PPO, 0 lines — mean/max
  native reward 0.0 over 25 episodes
  (`artifacts/ale_pure_rl/evaluation.json`)

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
- Current checkpoint: 100M PPO timesteps on the lines reward with
  VecNormalize (Night 2, 2026-07-09)
- Current eval: mean 1.04 lines (max 3, 21/25 episodes with a clear),
  mean score 361, over 25 deterministic 500-piece-cap episodes seeded
  1000-1024 — the project's first pure-RL line clears
- Policy input: locked board, active falling-piece mask, current piece state,
  current piece identity, and one next piece
- Uses only the Gymnasium step API

Track 4: tool-assisted high score on custom env:

- Engine package: `packages/tetris_env`
- Agent CLI: `agents/custom/tetris_custom_agent.py`
- Viewer CLI: `agents/custom/render_custom_episode.py`
- Default best-model output: `artifacts/custom_best/best_weights.npy`
- Method: placement enumeration + Dellacherie-style features + depth-2 queue
  lookahead, weights optimized CEM-style on a held-out seed set
- Final result (frozen 2026-07-13): mean score 215,530, mean lines 198.1 of a
  200-line ceiling over 10 episodes at the 500-piece cap, seeds 0-9
  (`artifacts/custom_best/evaluation_500.json`)
- That ceiling is the piece cap, not the agent (measured 2026-07-13): uncapped it
  does not top out — 10,000 pieces / 3,997 lines, still alive, ~0.4 lines/piece.
  At a 2,000-piece cap it clears 798 lines.

Track 5: afterstate RL on custom env (added 2026-07-13, after the freeze):

- CLI: `agents/custom/afterstate_custom_agent.py`
- Env: `packages/tetris_env/tetris_env/placement_env.py` (`Discrete(40)` =
  4 rotations x 10 columns; obs = raw board + current + next one-hot, 214 floats)
- Default output: `artifacts/custom_afterstate/ppo_custom_afterstate.zip`
- **This is the project's controlled experiment**, not a high-score attempt: it is
  Track 3's PPO with *only* the action space changed. No hand features, no
  lookahead, no hyperparameter changes. Boundary enforced in `AGENTS.md` and by
  `tests/test_afterstate_env.py`.
- Result (12M steps, 2.0 h): mean 5.60 lines (max 9, min 3, 25/25 episodes >= 1
  line), mean 47.6 pieces, over 25 deterministic episodes at the 500-piece cap
  (`artifacts/custom_afterstate/evaluation.json`)
- Conclusion: the action abstraction is worth 5.4x over Track 3 at matched
  experience (~11-12M pieces both), but closes only 2.3% of the Track 3 -> Track 4
  gap. Hand-authored features + lookahead + CEM carry the rest. This *refuted* the
  report's earlier claim that the abstraction was the dominant variable.
- Caveat: the run had not converged (ep_rew_mean still climbing 60 -> 75 over the
  last 4M steps), so 5.60 is a lower bound. Next run: 50-100M steps.

Playback (added 2026-07-13, after the freeze):

- Best episode of each track as mp4: `artifacts/best_plays/` (mp4s gitignored;
  regenerate with `python tools/render_best_plays.py`)
- Live viewer: `python artifacts/best_plays/live_play.py` (track 4, endless) or
  `--track 3` (restarts on top-out)
- No agent, model, or frozen result was changed by this work.

Important environment behavior:

- Pure RL can see the current piece state and exactly one next piece.
- The viewer renders the same one-piece preview in a `NEXT` panel.
- The tool-assisted custom policy uses one-piece lookahead when scoring
  placements.
- The custom engine remains the source of standard Tetris rules and physics:
  hidden spawn rows, 7-bag pieces, SRS-style kicks, hard/soft drops, line clears,
  levels, and score-mode line scoring.
