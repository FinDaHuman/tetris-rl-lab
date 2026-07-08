# Next Steps

Status on 2026-07-08 (evening — Night 1 complete):

- Night 1 ran and its gate decisions are made (details in
  [TRAINING_PLAN.md](TRAINING_PLAN.md) "Night 1 outcome"):
  - Track 4 **promoted** to `artifacts/custom_best`: mean score 215,530 vs
    213,780 at 500 pieces (10 episodes, seeds 0-9); lines tied at ~198 of a
    200 ceiling. Track 4 is effectively done.
  - Track 3 pilot measured ~3,900 fps, so Night 2 was resized from 5M to
    **50M steps (~4 h)**.
- Episode-seeding bug found and fixed the same evening: the custom gym env
  replayed one fixed piece sequence per env every episode. Fixed in
  `gym_env.py` (fresh game seed per reset, reproducible stream); 3 regression
  tests added — suite is now 17 passed. Run Night 2 only with this fix in
  place.

**Next action: run Night 2 in [TRAINING_PLAN.md](TRAINING_PLAN.md)** (Track 3
lines-reward 50M run + evals), then optionally Night 3 (Track 1 final attempt
+ Track 2 confirm).

Remaining promotion rules:

- Track 3: promote to `artifacts/custom_pure_rl` if mean lines > 0.
- Keep all experiment outputs under `runs/` until promotion.
- Tip: PowerShell transcripts miss Python stdout — also save the console
  window contents (as done for Night 1:
  `runs/plan_20260708/night1_console_full.txt`).

Older 200-piece Track 4 context (superseded, kept for reference): the
2026-07-07 queue run reached mean 78.3 lines at the 200-piece cap, which is
near that cap's 80-line ceiling — hence the move to 500 pieces.
