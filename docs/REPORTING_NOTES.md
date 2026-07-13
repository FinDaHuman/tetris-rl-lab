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

Scores are **not comparable across tracks**: Tracks 1–2 report an estimated Atari
score (100/line), Track 3 the engine score (which includes soft/hard-drop points),
Track 4 line-clear score only. Compare lines, and only at equal piece caps.

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

**The report is written: `docs/REPORT.md` (2026-07-13).** This file
remains the guide to which numbers may be compared.

Keep these artifacts for report writing:

- `artifacts/ale_pure_rl/evaluation.json`: Track 1 25-episode manifest
  (generated 2026-07-13; 0 lines, mean native reward 0.0) — the clean
  replacement for the transcript-only evidence.
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
- `artifacts/best_plays/manifest.json`: the seed, lines, score and frame count
  behind each track's best-play video (2026-07-13). The `.mp4` files themselves
  are gitignored — regenerate with `python tools/render_best_plays.py`.
- `artifacts/custom_best/custom_episode.json` / `.mp4`: **superseded.** An old
  Track 4 render (200 pieces / 74 lines) that predates the promoted 500-piece
  weights, and it sampled one frame per 25 pieces so it is a slideshow, not an
  animation. Do not cite it; use `artifacts/best_plays/track4_custom_tool.mp4`.

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
- **Track 4 line counts are only meaningful next to their piece cap.** Measured
  2026-07-13: the agent does not top out. Uncapped it reached 10,000 pieces /
  3,997 lines and was still alive when the probe was stopped by hand, holding
  ~0.4 lines/piece (the theoretical maximum) throughout. So its result is
  ≈ `0.4 × cap` — 198 lines at 500 pieces, 798 at 2,000 — and any Track 4 number
  quoted without its cap is meaningless. Never compare Track 4 line counts across
  different caps, and never describe it as "plateauing" at a ceiling: the ceiling
  is ours, not its. Scores are line-clear score only (no drop points), matching
  `play_episode` in `agents/custom/tetris_custom_agent.py`.
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
- ⚠ **The ALE seed does not change ALE/Tetris-v5's piece sequence** (verified
  2026-07-13: NOOP-only rollouts from seeds 0/1/2/42 are bit-identical). The ROM's
  piece generator is not seeded by ALE; the seed only drives sticky-action noise and
  no-op starts. Consequences for every ALE number in this repo:
  - "37 lines on all 10 seeds" is **not** evidence of robustness across games. It is
    the *same game* ten times — effective sample size for game-to-game variation is
    **1**, and the zero variance is an artifact. Do not report it as consistency, and
    do not compute a std over ALE seeds and present it as meaningful.
  - What *is* genuinely shown: Track 2 is robust to **action-execution noise** — it
    still scores exactly 37 lines under sticky 0.25 (re-checked 2026-07-13), because
    the planner is closed-loop and re-reads the board every piece.
  - Tracks 3/4 (custom engine, 7-bag seeded per episode) **do** have real seed
    diversity, so their variance numbers are meaningful. ALE and custom variance
    figures are therefore not comparable.
  - Condition mismatch to disclose: Track 1 trains/evaluates at `--sticky 0.25`;
    every Track 2 command defaults to `--sticky 0.0`. It does not change Track 2's
    result, but the two ALE tracks were not evaluated under identical conditions.
- ⚠ **Frames vs. agent steps.** Track 1's `timesteps: 10000000` counts *agent steps*;
  under frame-skip 4 that is ~40M emulator frames (~20% of the canonical 200M-frame
  Atari budget, which is 50M agent steps). Earlier drafts compared agent steps to
  frames directly and understated the budget as "1.5–3%". Use one unit and say which.
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
