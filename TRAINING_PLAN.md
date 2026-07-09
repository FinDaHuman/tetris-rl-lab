# Training Plan — 2026-07-08

All code changes for this plan are already implemented and tested (`python -m
pytest` → 14 passed). This file contains only the commands left to run, in
order. Nothing here retrains anything by surprise; every step writes its
outputs under `runs/plan_20260708/` and every night is wrapped in a PowerShell
transcript so the full console log is preserved for the report.

What changed in the code (summary):

- Track 4 planning is ~5x faster: `enumerate_placements` was rewritten as a
  numpy drop simulation (with an exact cell-wise fallback for covered pockets),
  symmetric duplicate rotations are no longer scored, and the placement
  features were vectorized. A 500-piece queue-lookahead episode now takes ~15s
  instead of ~78s. An oracle test (`tests/test_engine_placements.py`) proves
  the new enumeration produces the same placement set as the old code.
- Track 4 CEM training now promotes `best_weights.npy` using a fixed held-out
  seed set instead of the per-generation training seeds, so the saved best is
  no longer a lucky-seed artifact. `history.json` now records both train and
  holdout fitness.
- Track 3 has a new `--reward-mode lines` (now the default): reward is
  `10 * lines_cleared^2` per lock `+ 0.25` per placed piece `- 10` on top-out,
  and no drop points. The old score-based reward paid ~30+ points per piece
  just for dropping fast, which is why 200k steps learned zero line clears.
  The old behavior is still available with `--reward-mode score`.
- Track 1 `evaluate --out` no longer crashes on NumPy scalars in ALE episode
  info; eval JSON can be saved again.

Expected line ceilings for context: at a 500-piece cap the maximum possible is
200 lines (4 cells per piece, 10 per row). Default (untrained) weights already
reach ~192 lines at 500 pieces with queue lookahead, so Track 4 gains will be
in the last few lines and in mean consistency.

---

## Night 1 — Track 4 at 500 pieces + Track 3 timing pilot (~4.5 h total)

Run from the repo root in PowerShell, top to bottom. The transcript captures
every console line; the `--out` files capture machine-readable results.

```powershell
New-Item -ItemType Directory -Force -Path runs\plan_20260708
Start-Transcript -Path runs\plan_20260708\night1_transcript.txt -Append

# 1.1  Sanity: full test suite (about 1 minute). Abort the night if this fails.
python -m pytest

# 1.2  Validate the current Track 4 best at 500 pieces (about 2 minutes).
python agents/custom/tetris_custom_agent.py evaluate --weights runs/overnight_lines_20260707/custom_tool_queue/best_weights.npy --episodes 5 --max-pieces 500 --lookahead-depth 2 --lookahead-candidates 4 --future-source queue --out runs/plan_20260708/track4_eval500_current.json

# 1.3  Track 4: warm-start CEM training at 500 pieces (about 4 hours).
#      75 episodes per generation (24 pop x 3 rollouts + 3 holdout) x ~15 s.
python agents/custom/tetris_custom_agent.py train --outdir runs/plan_20260708/track4_queue_500 --warm-start runs/overnight_lines_20260707/custom_tool_queue/best_weights.npy --generations 12 --population 24 --rollouts 3 --holdout-rollouts 3 --max-pieces 500 --seed 17 --lookahead-depth 2 --lookahead-candidates 4 --future-source queue

# 1.4  Evaluate the new Track 4 weights at 500 pieces (about 3 minutes).
python agents/custom/tetris_custom_agent.py evaluate --weights runs/plan_20260708/track4_queue_500/best_weights.npy --episodes 10 --max-pieces 500 --lookahead-depth 2 --lookahead-candidates 4 --future-source queue --out runs/plan_20260708/track4_queue_500/evaluation_500.json

# 1.5  Track 3: 100k-step timing pilot with the new lines reward (15-30 min).
#      Purpose: read the "fps" value from the console log to size Night 2.
python agents/custom/pure_rl_custom_agent.py train --outdir runs/plan_20260708/track3_lines_pilot --logdir runs/plan_20260708/track3_lines_pilot_logs --timesteps 100000 --n-envs 8 --max-pieces 500 --reward-mode lines --seed 7 --eval-freq 50000 --checkpoint-freq 0

Stop-Transcript
```

**Decision gate after Night 1:**

- Track 4 promotion: promote only if `evaluation_500.json` (step 1.4) beats
  `track4_eval500_current.json` (step 1.2) on mean lines. To promote:

```powershell
Copy-Item runs\plan_20260708\track4_queue_500\best_weights.npy artifacts\custom_best\best_weights.npy
Copy-Item runs\plan_20260708\track4_queue_500\evaluation_500.json artifacts\custom_best\evaluation_500.json
Copy-Item runs\plan_20260708\track4_queue_500\history.json artifacts\custom_best\history.json
Copy-Item runs\plan_20260708\track4_queue_500\meta.json artifacts\custom_best\meta.json
```

- Track 3 sizing: note the steady-state `fps` printed by the pilot. Night 2
  timesteps = fps x 3600 x hours you can spare. Example: 250 fps x 8 h ≈ 7M.
  Use 5,000,000 if fps ≥ ~180, otherwise cut to 3,000,000.

---

## Night 1 outcome (recorded 2026-07-08, evening)

- Tests: 14 passed. Full console log preserved as
  `runs/plan_20260708/night1_console_full.txt` (the PowerShell transcript
  missed Python stdout, so the log was saved separately — do the same for
  future nights or just keep the console-window save).
- Track 4: **promoted.** New weights vs old at 10 episodes, seeds 0-9,
  500 pieces: mean score 215,530 vs 213,780 (better on 6/10 seeds), mean
  lines 198.1 vs 198.3 (a tie — both pressed against the 200-line ceiling).
  Holdout fitness rose 412,150 → 420,117 over 12 generations. Since lines
  are saturated and Track 4's objective is score, promotion was decided on
  mean score; old-weights 10-episode eval is
  `runs/plan_20260708/track4_eval500_current_10ep.json`. Track 4 is now
  effectively done — remaining headroom is ~1% score noise.
- Track 3 pilot: **~3,900 fps steady** — 100k steps took 38 seconds, so the
  original 5M-step Night 2 would finish in ~22 minutes. Night 2 below has
  been resized to 100M (~7 h). Learning signal is healthy: episode length
  grew 65 → 198 steps and reward -6.77 → -4.46 within 100k steps (still 0
  lines, expected this early).
- Bug found via the pilot's eval logs (5 identical eval episodes): the env
  replayed the *same piece sequence every episode*. Fixed on 2026-07-08 in
  `packages/tetris_env/tetris_env/gym_env.py` — each reset now draws a fresh
  game seed from the env's RNG stream (reproducible per env seed). Three
  regression tests added; suite is now 17 passed. The pilot's timing and
  trend remain valid; Night 2 trains on the fixed env with proper episode
  diversity.

## Night 2 — Track 3 main run with the lines reward (~7 h)

> **Attempt 1 (2026-07-08 → 09) hung — kill it and restart.** The run froze
> at exactly 4.0M steps (~10:40 PM, iteration 976) inside the eval callback:
> a py-spy stack dump showed the main thread spinning in
> `EvalCallback → evaluate_policy → model.predict` for 14+ hours at full CPU
> with zero new output anywhere. Root cause: upward SRS rotation kicks can
> offset gravity one-for-one, so a *deterministic* policy (which is what the
> eval callback uses) can hover one piece in a perfect state cycle forever,
> and the env only ended episodes when pieces locked. Fixed 2026-07-09 in
> `gym_env.py`: `max_steps_per_piece` (default 50) force-locks a piece after
> 50 non-locking steps, like real Tetris' move-limit lock delay, so every
> episode is bounded. Restart procedure:
>
> 1. `Ctrl+C` the stuck run (or `Stop-Process -Id <pid>`). Nothing further is
>    lost — no checkpoint existed yet (first was due at 5M) and the process
>    cannot save its in-memory model.
> 2. Keep attempt 1's outputs as evidence:
>    `Rename-Item runs\plan_20260708\track3_lines_100m track3_lines_100m_attempt1_hung`
>    `Rename-Item runs\plan_20260708\track3_lines_100m_logs track3_lines_100m_attempt1_hung_logs`
> 3. Re-run step 2.1 below unchanged (the fix is in the env, not the command).
>
> Encouraging sign from attempt 1: the 2M and 3M eval histories each contain
> an episode with positive reward (+6.0, +7.25), i.e. real line clears — the
> first ever for Track 3.

Sized to 100M steps using the measured ~3,900 fps (100M / 3900 ≈ 7.1 h).
Checkpoints every 5M mean an early Ctrl+C still leaves usable models under
the logdir's `checkpoints/` directory, so there is no harm in stopping it in
the morning if it is still running.

**Logging is now handled by the trainer itself** (added 2026-07-08 evening):
the training tables are written to `<logdir>/log.txt` and machine-readable
`<logdir>/progress.csv` as the run goes, so nothing is lost to PowerShell's
transcript gaps or the console screen buffer (which holds only a few
thousand lines — far less than a 7-hour run prints). The Start-Transcript
wrapper below is kept only as a record of which commands ran and when; the
real evidence is `log.txt` + `progress.csv` + the eval/tensorboard files.

```powershell
Start-Transcript -Path runs\plan_20260708\night2_transcript.txt -Append

# 2.1  Track 3: main PPO run on the lines reward, 100M steps (~7 h).
python agents/custom/pure_rl_custom_agent.py train --outdir runs/plan_20260708/track3_lines_100m --logdir runs/plan_20260708/track3_lines_100m_logs --timesteps 100000000 --n-envs 8 --max-pieces 500 --reward-mode lines --seed 7 --eval-freq 1000000 --checkpoint-freq 5000000

# 2.2  Evaluate the final model (about 10-20 minutes for 25 episodes).
python agents/custom/pure_rl_custom_agent.py evaluate --model runs/plan_20260708/track3_lines_100m/ppo_custom_pure.zip --vec-normalize runs/plan_20260708/track3_lines_100m/vec_normalize.pkl --episodes 25 --max-pieces 500 --deterministic --out runs/plan_20260708/track3_lines_100m/evaluation.json

# 2.3  Also evaluate the eval-callback best model. Note: this pairs the
#      callback-best policy with the final VecNormalize stats, which is the
#      standard approximation; matching per-checkpoint stats live under the
#      checkpoints directory if you want an exact pairing.
python agents/custom/pure_rl_custom_agent.py evaluate --model runs/plan_20260708/track3_lines_100m/best/best_model.zip --vec-normalize runs/plan_20260708/track3_lines_100m/vec_normalize.pkl --episodes 25 --max-pieces 500 --deterministic --out runs/plan_20260708/track3_lines_100m/evaluation_best.json

Stop-Transcript
```

**Decision gate after Night 2:**

- Any nonzero `mean_lines` is the first pure-RL line-clearing result in the
  project — promote whichever of 2.2/2.3 is better into
  `artifacts/custom_pure_rl/` (model zip, `vec_normalize.pkl`, evaluation
  JSON, `meta.json`).
- If still 0 lines: do not add more timesteps. The next levers, in order, are
  a higher `--line-reward` (e.g. 50), longer training `--max-pieces`, and an
  entropy schedule — one change at a time. (Also worth knowing: the pilot
  showed `clip_fraction` ~0.4 and `approx_kl` ~0.05, which is on the hot
  side for PPO — a lower learning rate, e.g. 1e-4, is another single-change
  lever if training looks unstable — attempt 1 reached `clip_fraction` ~0.47
  and `approx_kl` ~0.15 by 4M steps while reward was still improving, so
  watch these columns in `progress.csv`.)

## Night 3 (optional) — Track 1 final documented attempt

Only if you want a "non-sticky also failed / succeeded" data point for the
report. Expectation is 0 lines; that is a valid negative result. About 6-12 h
for 3M frames on this machine — check fps after the first 10 minutes and abort
if the projected finish is unacceptable. The Track 1 trainer writes
`<logdir>/log.txt` and `progress.csv` too (same fix as Track 3), so the
console log is not load-bearing here either.

```powershell
Start-Transcript -Path runs\plan_20260708\night3_transcript.txt -Append

# 3.1  Track 1: 3M non-sticky PPO (dummy vec env: safest for RAM).
python agents/ale/pure_rl_ale_agent.py train --outdir runs/plan_20260708/track1_nosticky_3m --logdir runs/plan_20260708/track1_nosticky_logs --timesteps 3000000 --n-envs 4 --vec-env dummy --sticky 0.0 --eval-freq 250000 --checkpoint-freq 1000000

# 3.2  Evaluate final and callback-best models (JSON --out works now).
python agents/ale/pure_rl_ale_agent.py evaluate --model runs/plan_20260708/track1_nosticky_3m/ppo_ale_pure.zip --episodes 25 --sticky 0.0 --out runs/plan_20260708/track1_nosticky_3m/evaluation.json
python agents/ale/pure_rl_ale_agent.py evaluate --model runs/plan_20260708/track1_nosticky_3m/best/best_model.zip --episodes 25 --sticky 0.0 --out runs/plan_20260708/track1_nosticky_3m/evaluation_best.json

# 3.3  Track 2: cheap baseline confirmation (unchanged code path, ~10 min).
python ale_tetris_agent.py evaluate --planner legacy_model --weights artifacts/ale_stable_high_score/best_weights.npy --episodes 3 --max-pieces 400 --seed 0 --out runs/plan_20260708/track2_confirm.json

Stop-Transcript
```

## Evidence produced by this plan (preserve for the report)

- `runs/plan_20260708/night*_transcript.txt` — full console logs.
- `runs/plan_20260708/track4_eval500_current.json` vs
  `runs/plan_20260708/track4_queue_500/evaluation_500.json` — Track 4
  before/after at 500 pieces.
- `runs/plan_20260708/track4_queue_500/history.json` — CEM train vs holdout
  fitness per generation (new format).
- `runs/plan_20260708/night1_console_full.txt` — full Night 1 console log
  (the transcript file missed Python stdout).
- `runs/plan_20260708/track4_eval500_current_10ep.json` — old Track 4
  weights at 10 episodes, the apples-to-apples baseline for the promotion.
- `runs/plan_20260708/track3_lines_100m_logs/log.txt` and `progress.csv` —
  full Night 2 training log written by the trainer itself.
- `runs/plan_20260708/track3_lines_100m/evaluation*.json` — first Track 3
  result under the lines reward.
- `runs/plan_20260708/track1_nosticky_3m/evaluation*.json` — Track 1
  non-sticky outcome (optional night).

## Caution on comparing to older numbers

The duplicate-rotation fix means Track 4 beam search now considers slightly
more distinct candidates, so evaluations of the *same weights* can differ
mildly from pre-change numbers (spot check: 2 episodes x 60 pieces went from
mean score 3450/21.0 lines to 3550/21.5 lines — equal or better). Cite
post-change evaluations for the report rather than mixing eras.
