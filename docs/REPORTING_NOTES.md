# Reporting Notes

This project will later need a detailed report covering:

- What was built in the Tetris RL lab.
- Why the project is split into four tracks.
- How each agent was implemented or trained.
- What techniques were allowed or disallowed per track.
- How each agent performed.
- What experiments were run and what evidence supports the results.
- What limitations remain, especially on a low-end machine.

Future AI agents should read this file, `docs/MODELS.md`, `STATUS.md`,
`NEXT_STEPS.md`, and the latest `docs/SESSION_*.md` file before writing the
report.

## Current Track Framing

Track 1: pure RL on ALE

- Entry point: `agents/ale/pure_rl_ale_agent.py`
- Environment: `ALE/Tetris-v5`
- Algorithm: Stable-Baselines3 PPO with `CnnPolicy`
- Constraint: no board decoding, no planning, no search, no hand-authored Tetris
  heuristics.

Track 2: tool-assisted high score on ALE

- Entry point: `agents/ale/ale_tetris_agent.py`
- Root wrapper: `ale_tetris_agent.py`
- Current stable result: 37 lines, score 3700, 259 decisions on seeds 0-9 using
  `artifacts/ale_stable_high_score/best_weights.npy`.
- Allowed: frame decoding, board reconstruction, placement modeling, search,
  calibration, CEM-style weight optimization.

Track 3: pure RL on custom environment

- Entry point: `agents/custom/pure_rl_custom_agent.py`
- Environment: `packages/tetris_env/tetris_env/gym_env.py`
- Algorithm: Stable-Baselines3 PPO with `MlpPolicy`
- Observation: locked board, active falling-piece mask, current piece state,
  current piece identity, and exactly one next piece.
- Constraint: only Gymnasium observations, rewards, and actions.

Track 4: tool-assisted high score on custom environment

- Entry point: `agents/custom/tetris_custom_agent.py`
- Engine: `packages/tetris_env`
- Current method: placement enumeration, weighted placement features, queue
  lookahead, and CEM-style optimization.
- Allowed: engine cloning, search, heuristic features, optimized weights.

## Evidence To Preserve

Keep these artifacts for report writing:

- `Windows PowerShell.txt`: overnight Track 1 ALE pure-RL training/eval
  transcript.
- `runs/overnight_lines_20260707/custom_tool_queue/evaluation_200.json`: strong
  Track 4 result at 200 pieces.
- `runs/overnight_lines_20260707/custom_tool_queue/history.json`: optimization
  history for the strong Track 4 run.
- `runs/overnight_lines_20260707/custom_tool_queue/meta.json`: parameters and
  learned weights for the strong Track 4 run.
- `artifacts/ale_stable_high_score/evaluation.json`: stable Track 2 ALE
  baseline.
- `artifacts/custom_pure_rl/evaluation.json`: current Track 3 baseline.
- `artifacts/custom_best/custom_episode.json`: older Track 4 baseline/render
  sidecar.

## Report Structure Draft

1. Project objective and track separation.
2. Custom Tetris engine and ALE environment overview.
3. Pure-RL methods and constraints.
4. Tool-assisted methods and why they are separate.
5. Experiment timeline.
6. Results table by track.
7. Analysis of why Track 1/3 pure RL are hard.
8. Analysis of why Track 4 improved strongly with queue lookahead.
9. Hardware/runtime limitations.
10. Next work.

## Important Caveats

- Fixed on 2026-07-08: the Track 1 ALE pure-RL `evaluate --out` JSON crash
  (NumPy scalars inside ALE episode info). Eval JSON can be saved again;
  transcripts from before the fix remain valid evidence.
- On 2026-07-08 the Track 4 planner was optimized (~5x faster) and duplicate
  symmetric rotations were removed from placement enumeration. Evaluations of
  the same weights can differ mildly from pre-change numbers (spot check moved
  results slightly upward). Do not mix pre- and post-change evaluations in the
  same results table; prefer post-change numbers.
- Track 3 evaluations before 2026-07-08 used the score-based reward; newer
  runs default to the `lines` reward mode. The `reward` field in evaluation
  JSON is mode-dependent, but `score`/`lines`/`pieces` remain comparable.
- Track 4 at 200 pieces is nearly saturated (ceiling 80 lines), so progress
  moved to the 500-piece cap (ceiling 200 lines).
