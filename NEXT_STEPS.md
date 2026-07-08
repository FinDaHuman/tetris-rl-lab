# Next Steps

Status on 2026-07-08:

- Code improvements landed and tested (14 tests pass):
  - Track 4 planning ~5x faster (vectorized features + numpy placement
    enumeration + symmetric-rotation dedupe).
  - Track 4 CEM now promotes best weights on a fixed held-out seed set.
  - Track 3 gained `--reward-mode lines` (now the train default) to fix the
    drop-point reward hack that kept pure RL at 0 lines.
  - Track 1 `evaluate --out` JSON crash fixed.
- No new training has been run since these changes.

**The complete, ordered training commands (with logging) are in
[TRAINING_PLAN.md](TRAINING_PLAN.md).** Run Night 1 first (Track 4 at 500
pieces + Track 3 timing pilot), then Night 2 (Track 3 lines-reward main run),
then optionally Night 3 (Track 1 final attempt + Track 2 confirm).

Promotion rules:

- Track 4: promote to `artifacts/custom_best` only if the new 500-piece eval
  beats `runs/plan_20260708/track4_eval500_current.json` on mean lines.
- Track 3: promote to `artifacts/custom_pure_rl` if mean lines > 0.
- Keep all experiment outputs under `runs/` until promotion.

Older 200-piece Track 4 context (superseded, kept for reference): the
2026-07-07 queue run reached mean 78.3 lines at the 200-piece cap, which is
near that cap's 80-line ceiling — hence the move to 500 pieces.
