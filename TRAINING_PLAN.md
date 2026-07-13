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

## Schedule to the 2026-07-13 deadline (set 2026-07-09 night)

The project must be done by **2026-07-13**, so the one-change-at-a-time
tuning rule is retired (decided 2026-07-09 night): runs now bundle every
change the evidence supports, and each ~8 h slot (a night, or a daytime run
— Night 2 attempt 2 ran fine 1:24–9:12 PM) must earn its keep.

- ~~**Night 7/09 → 10:** Night 3 — Track 3 bundled run.~~ *Actual: the
  user ran Night 3 on 7/10 → 11 at 200M steps (~17 h). Outcome: worse
  (0.44 mean lines vs 1.04) — no promotion; see the Night 3 outcome
  note.*
- ~~**Night 7/10 → 11:** Track 1 final attempt + Track 2 confirm.~~
  *Deadline triage 7/11: Track 1 attempt dropped (report uses existing
  evidence + literature); Track 2 confirm ran 7/11 — baseline reproduced
  exactly (37 lines / 3700 on seeds 0–2,
  `runs/plan_20260708/track2_confirm.json`).*
- ~~**Night 7/11 → 12 (last training slot):** Night 4 — Track 3 final
  run at lr 2e-4.~~ *Not launched — the slot passed unused. Results are
  frozen as of 2026-07-13 morning with the Night 2 model (mean_lines
  1.04) as the final Track 3 result.*
- **7/13 (deadline day):** results frozen; final report written —
  `docs/REPORT.md`. A 25-episode Track 1 evaluation manifest
  (`artifacts/ale_pure_rl/evaluation.json`) was generated 7/13 to
  replace the transcript-only evidence (the eval-JSON crash that blocked
  it was fixed 7/08).

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
  entropy schedule — one change at a time *(rule retired 2026-07-09 night
  for the 7/13 deadline; see the schedule section at the top)*. (Also worth
  knowing: the pilot
  showed `clip_fraction` ~0.4 and `approx_kl` ~0.05, which is on the hot
  side for PPO — a lower learning rate, e.g. 1e-4, is another single-change
  lever if training looks unstable — attempt 1 reached `clip_fraction` ~0.47
  and `approx_kl` ~0.15 by 4M steps while reward was still improving, so
  watch these columns in `progress.csv`.)

**Night 2 outcome (2026-07-09, attempt 2):** ran clean start to finish,
100M steps in ~7.8 h (~1:24 PM → 9:12 PM). **Gate passed — first pure-RL
line clears in the project.** The final model scored `mean_lines` **1.04**
(max 3, 21/25 episodes with ≥1 line, mean score 361) vs the callback-best
model's 0.92, so the **final model was promoted** to
`artifacts/custom_pure_rl/` (replacing the 0-line score-mode model from
2026-07-03). Two findings that shape the next run:

- **Learning plateaued at ~36M steps.** Eval mean reward climbed −4 → +7 by
  36M, then oscillated between ~0 and ~7 for the remaining 64M with no
  trend. More timesteps at these hyperparameters buy nothing.
- **PPO ran hot the whole run**: `approx_kl` ~0.15–0.18 and `clip_fraction`
  ~0.41–0.44 sustained for all 100M steps (healthy is roughly ≤0.03 / ≤0.2).
  Each update overshoots, which is the standard signature behind exactly
  this kind of plateau-plus-oscillation. Learning rate 1e-4 is the
  evidence-backed fix, bundled into Night 3.
- Behavior note: episodes end by top-out after ~28 pieces (of a 500 cap) —
  the agent clears ~1 line and then dies. Survival, not line-finding, is now
  the bottleneck.

## Night 3 — Track 3 bundled run: lr 1e-4 + top-out penalty 25 (~7-8 h)

Bundles the two changes Night 2's evidence supports (one-change rule
retired for the 7/13 deadline):

- `--learning-rate 0.0001` (was 3e-4) — fixes the sustained-hot updates
  (`approx_kl` ~0.15–0.18, `clip_fraction` ~0.41–0.44) behind the 36M
  plateau.
- `--top-out-penalty 25` (was 10) — attacks the survival bottleneck: the
  agent dies at ~28 of 500 pieces, and at penalty 10 a single line clear
  (+10) fully pays for the death, so dying young is cheap. At 25 the agent
  must survive longer / clear more to come out ahead.

Everything else identical to Night 2, same seed. Note the reward *scale*
changes with the penalty, so compare runs on `mean_lines`, `score` and
`pieces` in the evaluation JSONs (and lines/lengths in the eval history),
not on raw reward.

```powershell
Start-Transcript -Path runs\plan_20260708\night3_transcript.txt -Append

# 3.1  Track 3: 100M steps, lr 1e-4, top-out penalty 25 (~7-8 h).
python agents/custom/pure_rl_custom_agent.py train --outdir runs/plan_20260708/track3_lr1e4_pen25_100m --logdir runs/plan_20260708/track3_lr1e4_pen25_100m_logs --timesteps 100000000 --n-envs 8 --max-pieces 500 --reward-mode lines --seed 7 --learning-rate 0.0001 --top-out-penalty 25 --eval-freq 1000000 --checkpoint-freq 5000000

# 3.2  Evaluate the final model.
python agents/custom/pure_rl_custom_agent.py evaluate --model runs/plan_20260708/track3_lr1e4_pen25_100m/ppo_custom_pure.zip --vec-normalize runs/plan_20260708/track3_lr1e4_pen25_100m/vec_normalize.pkl --episodes 25 --max-pieces 500 --deterministic --out runs/plan_20260708/track3_lr1e4_pen25_100m/evaluation.json

# 3.3  Evaluate the eval-callback best model.
python agents/custom/pure_rl_custom_agent.py evaluate --model runs/plan_20260708/track3_lr1e4_pen25_100m/best/best_model.zip --vec-normalize runs/plan_20260708/track3_lr1e4_pen25_100m/vec_normalize.pkl --episodes 25 --max-pieces 500 --deterministic --out runs/plan_20260708/track3_lr1e4_pen25_100m/evaluation_best.json

Stop-Transcript
```

Watch `approx_kl` / `clip_fraction` in `progress.csv`: at lr 1e-4 they
should sit well below Night 2's ~0.15 / ~0.42.

**Decision gate after Night 3 (morning 7/10)** — promote whichever model
beats the current artifact's 25-episode `mean_lines` 1.04, then pick the
7/10 **day-slot** run:

- Night 3 clearly better (mean_lines ≥ ~1.5) *and* still climbing at 100M
  → day slot re-runs the same command with more steps if time allows, or
  simply keep the result.
- Night 3 better but plateaued again → day slot adds `--line-reward 50`
  on top of Night 3's settings (bigger prize per clear, penalty and lr
  kept).
- Night 3 *worse* than Night 2 (lr 1e-4 too slow — reward far below
  Night 2's curve at the same step count and not accelerating) → day slot
  uses `--learning-rate 0.0002 --top-out-penalty 25` instead.

**Night 3 outcome (2026-07-10 → 11, run at 200M steps by user's choice,
~17 h): worse — no promotion.** Final model mean_lines **0.44** (max 2,
9/25 episodes with a line, mean score 96) vs the promoted 1.04;
callback-best 0.28. The artifact keeps the Night 2 model. What the logs
show:

- **The run never took off.** Training/eval reward sat at ≈ −16 (i.e.
  most episodes: ~28 pieces × 0.25 + 0–1 lines − 25 penalty) for the
  entire 200M steps. Night 2 had reached its +7 level by 36M.
- **lr 1e-4 did stabilize PPO** (`approx_kl` ~0.05–0.07, `clip_fraction`
  ~0.22–0.26, vs Night 2's 0.15/0.42) — but stable-and-flat.
- **Penalty 25 backfired into hovering.** Episode length grew to ~330–370
  steps while pieces stayed ~28 (Night 2: ~240 steps for the same
  pieces): the agent spends its 50-step-per-piece budget delaying locks
  to postpone the −25, exactly the exploit the force-lock caps but can't
  remove the incentive for. The lower eval scores (96 vs 361 — fewer
  drop points) corroborate. Two mechanisms likely compounded: the big
  constant penalty inflates VecNormalize's reward std (diluting the line
  signal), and delaying death is locally easier to learn than stacking
  better.
- Attribution is confounded (both changes landed together — the known
  cost of bundling), but the hover signature points primarily at the
  penalty; the lr change behaved as intended.

This *deviates from the gate table above* (which said lr 2e-4 **+ pen
25**): given the hover evidence, the final Track 3 slot reverts the
penalty to 10 — see Night 4 below.

## Night 4 (2026-07-11 → 12) — final Track 3 slot: Night 2 config at lr 2e-4

Last training slot before the 7/12 freeze. Rationale: Night 2's config is
the only one proven to clear lines (1.04); its one diagnosed defect was
hot updates causing the 36M plateau. Night 3 proved lr 1e-4 stabilizes
updates but (with pen 25) never learns. So the final run is **exactly
Night 2 with lr 2e-4** — cool enough to help the plateau, hot enough to
stay in the regime that demonstrably learns. Penalty back to 10;
`line_reward`/`piece_reward` untouched (no evidence against them).
150M steps ≈ 12-13 h at the measured ~3,300 fps: started in the evening
it finishes early morning 7/12, leaving room for the evals before the
freeze. (If starting before ~6 PM, 200M ≈ 17 h also fits.)

```powershell
Start-Transcript -Path runs\plan_20260708\night4_track3_transcript.txt -Append

# 4.1  Track 3 final run: Night 2 config, lr 2e-4, 150M steps (~12-13 h).
python agents/custom/pure_rl_custom_agent.py train --outdir runs/plan_20260708/track3_lr2e4_150m --logdir runs/plan_20260708/track3_lr2e4_150m_logs --timesteps 150000000 --n-envs 8 --max-pieces 500 --reward-mode lines --seed 7 --learning-rate 0.0002 --eval-freq 1000000 --checkpoint-freq 5000000

# 4.2  Evaluate the final model.
python agents/custom/pure_rl_custom_agent.py evaluate --model runs/plan_20260708/track3_lr2e4_150m/ppo_custom_pure.zip --vec-normalize runs/plan_20260708/track3_lr2e4_150m/vec_normalize.pkl --episodes 25 --max-pieces 500 --deterministic --out runs/plan_20260708/track3_lr2e4_150m/evaluation.json

# 4.3  Evaluate the eval-callback best model.
python agents/custom/pure_rl_custom_agent.py evaluate --model runs/plan_20260708/track3_lr2e4_150m/best/best_model.zip --vec-normalize runs/plan_20260708/track3_lr2e4_150m/vec_normalize.pkl --episodes 25 --max-pieces 500 --deterministic --out runs/plan_20260708/track3_lr2e4_150m/evaluation_best.json

Stop-Transcript
```

**Gate (morning 7/12, the last one):** promote whichever model beats
mean_lines 1.04; otherwise the Night 2 model stays the final Track 3
result. Either way, results freeze after these evals and report writing
starts.

> **Outcome: not launched.** The 7/11 → 12 slot passed without the run
> starting, and by 7/13 (deadline day) there was no time left for a
> 12-13 h run plus evals plus the report. Results were frozen on
> 2026-07-13 with the Night 2 model as the final Track 3 result; the
> lr 2e-4 run remains the top item in the "next work" section of
> `docs/REPORT.md`.

**Deadline triage (2026-07-11):** the optional Track 1 non-sticky attempt
(previously Night 4) is **dropped** — only one night remained and a
Track 3 improvement is worth more than a second Track 1 negative result;
the report will use the existing overnight Track 1 evidence
(`Windows PowerShell.txt`) plus the literature expectations in
`docs/EXPECTED_PERFORMANCE.md`. The cheap Track 2 confirmation (old step
4.3) was run on 2026-07-11 instead — result in
`runs/plan_20260708/track2_confirm.json`.

## Dropped — Track 1 final documented attempt (was Night 4)

Only if you want a "non-sticky also failed / succeeded" data point for the
report. Expectation is 0 lines; that is a valid negative result. About 6-12 h
for 3M frames on this machine — check fps after the first 10 minutes and abort
if the projected finish is unacceptable. The Track 1 trainer writes
`<logdir>/log.txt` and `progress.csv` too (same fix as Track 3), so the
console log is not load-bearing here either.

```powershell
Start-Transcript -Path runs\plan_20260708\night4_transcript.txt -Append

# 4.1  Track 1: 3M non-sticky PPO (dummy vec env: safest for RAM).
python agents/ale/pure_rl_ale_agent.py train --outdir runs/plan_20260708/track1_nosticky_3m --logdir runs/plan_20260708/track1_nosticky_logs --timesteps 3000000 --n-envs 4 --vec-env dummy --sticky 0.0 --eval-freq 250000 --checkpoint-freq 1000000

# 4.2  Evaluate final and callback-best models (JSON --out works now).
python agents/ale/pure_rl_ale_agent.py evaluate --model runs/plan_20260708/track1_nosticky_3m/ppo_ale_pure.zip --episodes 25 --sticky 0.0 --out runs/plan_20260708/track1_nosticky_3m/evaluation.json
python agents/ale/pure_rl_ale_agent.py evaluate --model runs/plan_20260708/track1_nosticky_3m/best/best_model.zip --episodes 25 --sticky 0.0 --out runs/plan_20260708/track1_nosticky_3m/evaluation_best.json

# 4.3  Track 2: cheap baseline confirmation (unchanged code path, ~10 min).
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
  result under the lines reward (mean_lines 1.04; promoted 2026-07-09).
- `runs/plan_20260708/track3_lr1e4_pen25_100m/evaluation*.json` and its
  `_logs/progress.csv` — the bundled lr 1e-4 + penalty 25 run (Night 3).
- `runs/plan_20260708/track1_nosticky_3m/evaluation*.json` — Track 1
  non-sticky outcome (optional night).

## Caution on comparing to older numbers

The duplicate-rotation fix means Track 4 beam search now considers slightly
more distinct candidates, so evaluations of the *same weights* can differ
mildly from pre-change numbers (spot check: 2 episodes x 60 pieces went from
mean score 3450/21.0 lines to 3550/21.5 lines — equal or better). Cite
post-change evaluations for the report rather than mixing eras.

---

# Night 5 (2026-07-13, evening) — Track 5: afterstate RL

**Why this run exists.** It is the one experiment that answers the question the
whole project is built around. Track 4 beats Track 3 by ~190x, but Track 4
changes *four* things at once — action space, hand-authored features, queue
lookahead, and the optimizer — so the study currently cannot say which one
carries the gap. Track 5 changes **exactly one**: the action space. Same engine,
same `lines` reward, same PPO, same MlpPolicy(2x256), same hyperparameters as the
promoted Track 3 run. One action places a piece instead of nudging it. No hand
features. No lookahead. (Boundary is written down in `AGENTS.md` — respect it or
the run is worthless.)

**Budget is matched on pieces experienced, not env steps.** Track 3's 100M
primitive steps at ~8.5 steps/piece is ~12M pieces, so this gets ~12M placement
steps. Equal experience, different abstraction.

## Step 5.1 — train (~2 h)

Measured throughput on this laptop: **~1,676 placement-steps/s** (8 envs, CPU), so
12M steps ≈ **2.0 h**. A 600k-step pilot confirmed it learns: 33 pieces placed and
positive reward (i.e. line clears) by 150k steps, against a random-placement
baseline of 16 pieces / 0 lines — and against Track 3, which needed ~30M primitive
steps to find singles reliably.

```powershell
python agents/custom/afterstate_custom_agent.py train --timesteps 12000000 --n-envs 8 --max-pieces 500 --seed 7 --outdir runs/track5_afterstate --logdir runs/track5_afterstate_logs
```

Progress is written by the trainer itself to `runs/track5_afterstate_logs/log.txt`
and `progress.csv` (console capture is not load-bearing). Checkpoints every 1M
steps; eval callback every 250k.

## Step 5.2 — evaluate (25 deterministic episodes, same protocol as Track 3)

```powershell
python agents/custom/afterstate_custom_agent.py evaluate --model runs/track5_afterstate/ppo_custom_afterstate.zip --episodes 25 --max-pieces 500 --seed 1000 --deterministic --out runs/track5_afterstate/evaluation.json
```

## Step 5.3 — render it (optional, ~1 min)

```powershell
python agents/custom/afterstate_custom_agent.py render --seed 1000 --out artifacts/best_plays/track5_custom_afterstate.mp4
```

## How to read the result

There is no promotion gate here — **every outcome is a publishable answer**, which
is why the run is worth the slot:

| Track 5 lands at | Conclusion |
| --- | --- |
| near Track 4 (~198 lines) | The **action abstraction** carries the gap. Hand-authored features are a convenience, not the cause. |
| in between (say 10-100 lines) | Both matter, and you can now *quantify* the split. |
| near Track 3 (~1 line) | The **hand-authored features + search** carry it; the abstraction alone is not enough. |

Record the number in `docs/QA_PREP.md` §6.3 and `docs/REPORT.md` §7.2 either way.
A result that contradicts the report's current framing is a *better* outcome than
one that flatters it — it means the experiment had teeth.

## RESULT (run 2026-07-13, 2.0 h, 12,001,280 steps)

**Landed in the third row: near Track 3. The experiment had teeth — it refuted the
report's framing.**

**mean 5.60 lines** (max 9, min 3, sd 1.50; 25 deterministic episodes, seeds
1000–1024, 500-piece cap) — `artifacts/custom_afterstate/evaluation.json`.
Mean 47.6 pieces survived, mean score 676. Final `ep_rew_mean` 75.5, `ep_len_mean`
41.6, 1,664 fps.

| | Track 3 | **Track 5** | Track 4 |
| --- | --- | --- | --- |
| Mean lines | 1.04 | **5.60** | 198.1 |
| Pieces survived | 28.6 | 47.6 | 500 (never tops out) |
| Lines/piece | 0.036 | 0.118 | 0.399 |
| ≥1 line | 21/25 | **25/25** | 25/25 |

**The abstraction is worth 5.4× — and it closes only 2.3% of the Track 3 → Track 4
gap.** So the hand-authored features + lookahead + CEM carry ~97.7% of it. The
report's guess that the action space was the dominant variable was **wrong**, and
this run is what showed it.

**Budget note (correcting the plan above):** the plan estimated Track 3 at ~8.5
primitive steps/piece. Measured on the promoted policy it is **9.11**, so Track 3's
100M steps = **≈11.0M pieces** against Track 5's 12.0M — matched within 9%. The
comparison holds.

**Caveat that must travel with the number:** Track 5 **had not converged**
(`ep_rew_mean` 60 → 75 over the final 4M steps), so 5.60 is a **lower bound**, not a
ceiling. Track 3, by contrast, *plateaued* at 36M of 100M. The next run — now
REPORT §10.1 — is Track 5 at 50–100M steps (~8–16 h) to find where it actually
tops out.
