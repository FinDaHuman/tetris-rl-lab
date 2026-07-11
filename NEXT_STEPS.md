# Next Steps

Status on 2026-07-11 (afternoon — Night 3 worse, no promotion; final slot planned):

- Night 3 (lr 1e-4 + top-out penalty 25, run at 200M steps, ~17 h) came
  back **worse**: mean_lines 0.44 vs the promoted 1.04. The run never took
  off — reward flat at ≈ −16 for all 200M steps. lr 1e-4 did stabilize PPO
  (kl ~0.05–0.07, clip ~0.22–0.26), but penalty 25 taught the agent to
  *hover* (episode steps grew ~240 → ~330–370 at the same ~28 pieces) —
  delaying the −25 is easier to learn than stacking better. **Artifact
  unchanged** (Night 2 model, 1.04 lines, stays promoted).
- Track 2 baseline re-confirmed 2026-07-11: 37 lines / 3700 / 259
  decisions on seeds 0–2 (`runs/plan_20260708/track2_confirm.json`).
- Deadline triage: the optional Track 1 non-sticky attempt is dropped.
- **Next action: run Night 4 in [TRAINING_PLAN.md](TRAINING_PLAN.md)** —
  the final Track 3 slot: exactly Night 2's command with
  `--learning-rate 0.0002` (penalty back to 10), 150M steps (~12–13 h).
  Morning 7/12: evals, last gate (promote only above 1.04), then freeze
  and write the report (7/12–13).

Status on 2026-07-09 (night — Night 2 complete, gate passed, model promoted):

- Night 2 attempt 2 ran clean: 100M steps in ~7.8 h. **First pure-RL line
  clears in the project** — final model `mean_lines` 1.04 (max 3, 21/25
  episodes ≥1 line) vs callback-best 0.92. Final model **promoted to
  `artifacts/custom_pure_rl/`** (old 0-line score-mode artifact replaced;
  previous version remains in git history).
- Diagnostics: learning plateaued at ~36M steps (eval reward oscillated
  0–7 for the last 64M), and PPO ran hot the whole run (`approx_kl`
  ~0.15–0.18, `clip_fraction` ~0.41–0.44). Behavior bottleneck is survival:
  episodes top out after ~28 pieces of the 500 cap.
- **Deadline set: the project must be done by 2026-07-13**, so the
  one-change-at-a-time tuning rule is retired — runs now bundle every
  evidence-backed change. Full day-by-day schedule is at the top of
  [TRAINING_PLAN.md](TRAINING_PLAN.md).
- **Next action: run Night 3 in [TRAINING_PLAN.md](TRAINING_PLAN.md)** —
  Night 2's command with `--learning-rate 0.0001` (fixes the hot updates)
  **and** `--top-out-penalty 25` (attacks the survival bottleneck).
  Gate: promote only if 25-episode mean_lines beats 1.04; the gate table
  also picks the 7/10 day-slot run.
- The optional Track 1 final attempt is now Night 4 in the plan
  (night 7/10 → 11); report writing is reserved for 7/12–13.

Status on 2026-07-09 (afternoon — Night 2 attempt 1 hung, fix pushed):

- Night 2 attempt 1 froze at 4.0M/100M steps: the eval callback's
  deterministic policy hovered a piece forever via upward rotation kicks
  canceling gravity, and episodes only ended when pieces locked. Fixed in
  `gym_env.py` with `max_steps_per_piece=50` (force-lock after 50
  non-locking steps); 2 regression tests added — suite is 19 passed.
  Full diagnosis in `docs/SESSION_2026-07-09.md`.
- **Next action: kill the stuck process, rename the attempt-1 run dirs,
  re-run Night 2 step 2.1 unchanged** — procedure at the top of the Night 2
  section in [TRAINING_PLAN.md](TRAINING_PLAN.md).
- Attempt 1 upside: first Track 3 line clears ever seen (eval episodes at
  2M/3M with positive reward).

Status on 2026-07-08 (evening — Night 1 complete):

- Night 1 ran and its gate decisions are made (details in
  [TRAINING_PLAN.md](TRAINING_PLAN.md) "Night 1 outcome"):
  - Track 4 **promoted** to `artifacts/custom_best`: mean score 215,530 vs
    213,780 at 500 pieces (10 episodes, seeds 0-9); lines tied at ~198 of a
    200 ceiling. Track 4 is effectively done.
  - Track 3 pilot measured ~3,900 fps, so Night 2 was resized from 5M to
    **100M steps (~7 h)**.
- Episode-seeding bug found and fixed the same evening: the custom gym env
  replayed one fixed piece sequence per env every episode. Fixed in
  `gym_env.py` (fresh game seed per reset, reproducible stream); 3 regression
  tests added — suite is now 17 passed. Run Night 2 only with this fix in
  place.

**Next action: run Night 2 in [TRAINING_PLAN.md](TRAINING_PLAN.md)** (Track 3
lines-reward 100M run + evals), then optionally Night 3 (Track 1 final attempt
+ Track 2 confirm).

Remaining promotion rules:

- Track 3: the mean-lines > 0 gate was met and promoted on 2026-07-09;
  future runs promote to `artifacts/custom_pure_rl` only if their
  25-episode `mean_lines` beats the current 1.04.
- Keep all experiment outputs under `runs/` until promotion.
- Logging: both PPO trainers now write `<logdir>/log.txt` and `progress.csv`
  themselves, so console/transcript capture is no longer load-bearing.
  (PowerShell transcripts miss Python stdout and the console buffer holds
  only a few thousand lines — Night 1's console save,
  `runs/plan_20260708/night1_console_full.txt`, only worked because the run
  was short.)

Older 200-piece Track 4 context (superseded, kept for reference): the
2026-07-07 queue run reached mean 78.3 lines at the 200-piece cap, which is
near that cap's 80-line ceiling — hence the move to 500 pieces.
