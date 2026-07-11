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
report. `docs/EXPECTED_PERFORMANCE.md` holds the literature-grounded
expectations per track (with citations) and the human-level comparison —
use it for the report's analysis sections (items 7–9 below).

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
- `artifacts/custom_pure_rl/evaluation.json`: current Track 3 baseline —
  the Night 2 (2026-07-09) 100M-step lines-reward model, mean_lines 1.04
  over 25 episodes: the project's first pure-RL line clears. Full training
  evidence: `runs/plan_20260708/track3_lines_100m*` (log.txt, progress.csv,
  eval history, checkpoints every 5M).
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
  moved to the 500-piece cap (ceiling 200 lines). As of the Night 1 run
  (2026-07-08), Track 4 is ~198 mean lines at 500 pieces — saturated there
  too — and the promoted result was decided on mean score (215,530 vs
  213,780), not lines.
- Fixed on 2026-07-08 (evening): `TetrisScoreEnv` replayed the same piece
  sequence every episode for a given env seed. Training/eval-callback data
  from before the fix (including the Night 1 Track 3 pilot) used only
  `n_envs` distinct piece sequences; standalone `evaluate` results were
  always per-episode seeded and stay valid. Piece sequences for a given seed
  changed with the fix, so do not compare same-seed episodes across it.
- Fixed on 2026-07-09: `TetrisScoreEnv` episodes could be infinite — upward
  rotation kicks can cancel gravity, so a deterministic policy could hover a
  piece forever (this hung Night 2 attempt 1 at 4.0M/100M steps inside the
  eval callback, ~15 h at full CPU with no progress). The env now force-locks
  a piece after 50 non-locking steps (`max_steps_per_piece`), like real
  Tetris' move-limit lock delay. This slightly changes env dynamics for
  degenerate hovering play only; normal play is unaffected. Attempt 1's
  partial training data (≤4M steps) predates the fix and should not be
  merged with the restarted run's curves.
- Track 3 negative result (2026-07-10 → 11, Night 3): lr 1e-4 +
  top-out penalty 25 at 200M steps scored **0.44** mean lines (vs the
  promoted 1.04) and was not promoted. Two report-worthy findings: the
  lower lr fixed PPO's hot updates (approx_kl 0.15→~0.06, clip_fraction
  0.42→~0.24) but the run stayed flat, and the 2.5× top-out penalty
  induced *hovering* (episode steps ~240 → ~330–370 at unchanged ~28
  pieces; scores 361 → 96 from lost drop points) — a clean example of
  penalty-shaping creating a degenerate incentive, bounded only by the
  50-step force-lock. Reward numbers from this run are on a different
  scale (penalty 25) and must not sit in the same table as penalty-10
  runs without a note. Attribution is confounded: both changes landed in
  one run (deadline-driven bundling).
- Track 2 baseline re-confirmed 2026-07-11 (post-planner-optimization
  code path): 37 lines / 3700 / 259 decisions on seeds 0–2 —
  `runs/plan_20260708/track2_confirm.json`.
- Deadline triage 2026-07-11: the optional Track 1 non-sticky attempt was
  dropped; Track 1 evidence remains the earlier overnight transcript
  (`Windows PowerShell.txt`) plus `docs/EXPECTED_PERFORMANCE.md`.
- Track 3 milestone (2026-07-09, Night 2): first pure-RL line clears —
  mean_lines 1.04 / max 3 over 25 deterministic episodes at 100M steps.
  The promoted `artifacts/custom_pure_rl/` model replaced the 2026-07-03
  score-mode model (0 lines); do not mix their numbers — reward modes and
  env fixes differ. For the report's analysis section: the run plateaued at
  ~36M steps with PPO update stats hot the whole way (approx_kl ~0.15–0.18,
  clip_fraction ~0.41–0.44), and the trained agent tops out after ~28
  pieces — survival, not line-finding, is the current limiter.
