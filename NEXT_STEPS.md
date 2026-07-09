# Next Steps

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
- **Next action: run Night 3 in [TRAINING_PLAN.md](TRAINING_PLAN.md)** —
  same command at `--learning-rate 0.0001` (the one evidence-backed change).
  Gate: promote only if 25-episode mean_lines beats 1.04.
- The optional Track 1 final attempt is now Night 4 in the plan.

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
