# TetrisGPT

Clean monorepo for five Tetris agent tracks:

- `agents/ale/pure_rl_ale_agent.py` trains a pure RL model on real
  `ALE/Tetris-v5` frames and actions.
- `agents/ale/ale_tetris_agent.py` keeps the tool-assisted real ALE high-score
  work as a showcase.
- `agents/custom/pure_rl_custom_agent.py` trains a pure RL model on the custom
  Gymnasium environment with one-piece preview.
- `agents/custom/tetris_custom_agent.py` trains the tool-assisted custom
  high-score model with placement enumeration and one-piece lookahead.
- `agents/custom/afterstate_custom_agent.py` trains a pure RL model on the custom
  engine that acts at **placement level** instead of keypress level — the
  controlled experiment (Track 5).

**Tracks 1 and 2 run on ALE. Tracks 3, 4 and 5 run on the custom engine.**

Tracks 1–4 form a 2×2: the environment (real Atari vs a custom engine) × the
method (pure RL vs tool-assisted planning). They were built to isolate one
question: how much of "playing Tetris well" comes from *learning*, and how much
from the action abstraction and search that tools provide?

**Track 5 answers it.** It is Track 3 with exactly one variable changed — the
action space — so the 2×2 no longer has to guess. See Results.

The previous real-ALE 2000-line target is intentionally no longer active. The
best verified real-ALE 37-line model is set aside under `artifacts/ale_37_line`.

## Results

Frozen 2026-07-13. Full write-up and evidence: **[docs/REPORT.md](docs/REPORT.md)**.

| Track | Env | Method | Result |
| --- | --- | --- | --- |
| 1 | ALE | Pure RL (PPO, pixels) | **0 lines** after 10M frames |
| 2 | ALE | Tool-assisted | **37 lines / 3,700** — identical on every seed |
| 3 | Custom | Pure RL (PPO, primitive actions) | **mean 1.04 lines** (max 3) |
| 4 | Custom | Tool-assisted (search + CEM) | **mean 198.1 lines** at a 500-piece cap — and it never tops out: 10,000 pieces → 3,997 lines, still alive |
| 5 | Custom | Pure RL (PPO, **placement actions**, no features) | **mean 5.60 lines** (max 9) — the controlled experiment |

The headline: on the same engine and the same laptop, placement-level search
(Track 4) clears ~190× the lines of primitive-action RL (Track 3). **Track 5 is the
experiment that says why.** It is Track 3 with exactly one variable changed — PPO
picks a *placement* instead of a keypress, still with no hand-authored features and
no lookahead — and it reaches 5.60 lines on ~11–12M pieces of experience for both.

So the action abstraction is worth **5.4×**, and it is **not** what carries Tetris:
it closes only **2.3%** of the Track 3 → Track 4 gap. The remaining ~97.7% belongs to
the hand-authored Dellacherie features, the queue lookahead, and CEM. An earlier
version of this README claimed the opposite; Track 5 refuted it. Caveat: Track 5's
learning curve had not converged at 12M steps, so 5.60 is a **lower bound**, not a
ceiling ([REPORT §7.2](docs/REPORT.md)).

Watch any of it: `python artifacts/best_plays/live_play.py`.

## Docs

- **[Final report](docs/REPORT.md)** — the project deliverable
- [Code walkthrough](docs/QA_CODE_WALKTHROUGH.md) — how the engine (custom **and**
  ALE), the observation spaces, PPO and CEM actually work, line by line
- [Q&A prep](docs/QA_PREP.md) — the defense: why five tracks, what each number
  means, and where this work is genuinely weak
- [Model catalog](docs/MODELS.md) — per-track entry points, artifacts, commands
- [Project summary](docs/PROJECT_SUMMARY.md) — layout and orientation
- [Expected performance](docs/EXPECTED_PERFORMANCE.md) — literature-grounded
  expectations and the human-level comparison
- [Reporting notes](docs/REPORTING_NOTES.md) — which numbers may be compared
- [Track boundaries](AGENTS.md) — what each track is and isn't allowed to do
- [Status](STATUS.md) · [Next steps](NEXT_STEPS.md)

## Layout

```text
agents/
  ale/            Track 1 (pure RL) + Track 2 (tool-assisted) on ALE.
  custom/         Track 3 (pure RL), Track 4 (tool-assisted), Track 5 (afterstate RL).
  video.py        Streaming mp4 writer shared by every renderer.
packages/
  tetris_env/     Reusable Tetris engine, Gymnasium envs, features, frame renderer.
                  engine.py        the game itself (Tracks 3/4/5)
                  gym_env.py       primitive-action env (Track 3)
                  placement_env.py placement-action env (Track 5)
                  features.py      Dellacherie-style board features (Track 4)
tools/
  render_best_plays.py  Renders the best episode of each track to mp4.
tests/                  37 tests: engine, features, envs, replay, track boundaries.
docs/                   Report, Q&A prep, code walkthrough, model catalog.
artifacts/
  ale_pure_rl/          Track 1 PPO checkpoints.
  ale_37_line/          Saved real-ALE 37-line model metadata.
  ale_stable_high_score/ Track 2 stable weights + evaluation manifest.
  custom_pure_rl/       Track 3 PPO checkpoints (+ vec_normalize.pkl).
  custom_best/          Track 4 trained weights, history, evaluations.
  custom_afterstate/    Track 5 PPO checkpoint (+ vec_normalize.pkl), evaluation.
  best_plays/           Best episode per track as mp4, plus the live viewer.
runs/                   All experiment output (gitignored).
```

## Track 1: Pure RL on ALE

This track uses only Atari frames and ALE actions. It does not decode the board,
enumerate placements, use a model of Tetris pieces, or search future states.
Default training uses standard Atari preprocessing: grayscale 84x84 frames,
frame skip, no-op reset, and frame stacking. Sticky actions are enabled by
default with `--sticky 0.25`.

Smoke the ALE env:

```bash
python agents/ale/pure_rl_ale_agent.py smoke
```

Train a pure PPO model:

```bash
python agents/ale/pure_rl_ale_agent.py train --timesteps 100000 --n-envs 4 --sticky 0.25
python agents/ale/pure_rl_ale_agent.py train --timesteps 10000000 --n-envs 8 --vec-env subproc --sticky 0.25
```

Evaluate across many slippery seeds:

```bash
python agents/ale/pure_rl_ale_agent.py evaluate --model artifacts/ale_pure_rl/ppo_ale_pure.zip --episodes 25 --sticky 0.25 --out artifacts/ale_pure_rl/evaluation.json
```

## Track 2: Tool-Assisted High Score on ALE

This is the existing real ALE high-score path. It uses frame decoding, model
placement logic, search/calibration options, and CEM-style weight optimization.
The preserved 37-line model belongs to this track.

Smoke the old ALE integration:

```bash
python ale_tetris_agent.py smoke
```

Evaluate the preserved stable copy:

```bash
python ale_tetris_agent.py evaluate --planner legacy_model --weights artifacts/ale_stable_high_score/best_weights.npy --episodes 10 --max-pieces 400 --seed 0 --out artifacts/ale_stable_high_score/evaluation.json
```

For continued attempts beyond the 37-line plateau, keep `legacy_model` as the
proven reproduction path and run a larger warm-started search:

```bash
python ale_tetris_agent.py train --planner legacy_model --warm-start artifacts/ale_stable_high_score/best_weights.npy --generations 50 --population 48 --rollouts 4 --max-pieces 500
```

There is also an experimental emulator-validated legacy planner. It tests top
placements in cloned ALE state before committing, but it is intended as a
diagnostic/search variant rather than a replacement for the archived planner:

```bash
python ale_tetris_agent.py train --planner legacy_calibrated --warm-start artifacts/ale_stable_high_score/best_weights.npy --generations 20 --population 24 --rollouts 2 --max-pieces 500 --top-k 32
```

## Custom Tetris Environment

Shared by Tracks 3, 4 and 5. The engine (`packages/tetris_env/tetris_env/engine.py`)
models:

- 10x20 visible board with two hidden spawn rows.
- Seven tetrominoes with a 7-bag generator.
- Current piece state and one-piece preview exposed to pure RL code.
- One-piece preview exposed to the tool-assisted policy and viewer.
- SRS-style wall kicks.
- Hard drop, soft drop, line clears, levels, and score-mode line scoring.

Two Gymnasium environments sit on top of the same engine, and the difference
between them **is** the Track 3 / Track 5 experiment:

| Env | Action space | Used by |
| --- | --- | --- |
| `gym_env.py` (`TetrisScoreEnv`) | `Discrete(7)` — one keypress (left/right/rotate/drop) | Track 3 |
| `placement_env.py` (`PlacementTetrisEnv`) | `Discrete(40)` — one whole piece placed | Track 5 |

Both use the same `lines` reward: `10 × cleared² + 0.25 per piece − 10 on top-out`.

*(ALE Tetris is a different game with different mechanics — 5 actions, no hard drop,
and reward = lines cleared only. See [the code walkthrough §1B](docs/QA_CODE_WALKTHROUGH.md).)*

## Track 3: Pure RL on Custom Env

This track uses only the Gymnasium step API. The observation is a flat vector
containing the locked board, active falling-piece mask, current piece state,
current piece identity, and exactly one next piece.

Smoke the custom pure-RL env:

```bash
python agents/custom/pure_rl_custom_agent.py smoke
```

Train a pure PPO model:

```bash
python agents/custom/pure_rl_custom_agent.py train --timesteps 5000000 --n-envs 4
```

Evaluate it:

```bash
python agents/custom/pure_rl_custom_agent.py evaluate --model artifacts/custom_pure_rl/ppo_custom_pure.zip --episodes 10 --deterministic --out artifacts/custom_pure_rl/evaluation.json
```

## Track 4: Tool-Assisted High Score on Custom Env

This track uses the standard custom Tetris engine while allowing planning tools:
placement enumeration, weighted features, and internal-queue beam lookahead. The
game rules remain the engine rules: 10x20 board, hidden
spawn rows, 7-bag pieces, SRS-style kicks, hard/soft drops, line clears, levels,
and score-mode line scoring.

Train a custom score-mode model:

```bash
python agents/custom/tetris_custom_agent.py train --generations 20 --population 32 --rollouts 4 --lookahead-depth 2 --lookahead-candidates 4 --future-source queue
```

Evaluate the current custom best:

```bash
python agents/custom/tetris_custom_agent.py evaluate --weights artifacts/custom_best/best_weights.npy --episodes 5 --lookahead-depth 2 --lookahead-candidates 4 --future-source queue
```

Render an episode as video. Pieces are animated into place by replaying each
chosen placement as real engine actions, so the video shows the same board the
planner produced:

```bash
python agents/custom/render_custom_episode.py --weights artifacts/custom_best/best_weights.npy --max-pieces 2000 --out artifacts/best_plays/track4_custom_tool.mp4
```

Note that `--max-pieces` alone decides the result: this agent does not top out
(10,000 pieces → 3,997 lines, still alive), so it clears roughly `0.4 × max-pieces`
lines. Quote a Track 4 line count only alongside its cap.

## Track 5: Afterstate RL on Custom Env (the controlled experiment)

**Not a high-score attempt.** Tracks 3 and 4 differ in four ways at once (action
space, hand-authored features, lookahead, optimizer), so the 2×2 alone cannot say
which one carries the ~190× gap. Track 5 changes **exactly one**: PPO emits a
*placement* (`Discrete(40)` = 4 rotations × 10 columns) instead of a keypress.

Everything else is held identical to the promoted Track 3 run — same PPO, same
MlpPolicy 2×256, same reward, same hyperparameters, same seed. **No hand-authored
features** (the observation is the raw board + current + next piece, 214 floats) and
**no lookahead**. The boundary is written down in [AGENTS.md](AGENTS.md) and enforced
by `tests/test_afterstate_env.py`; adding a feature or a lookahead voids the
experiment.

Train it (12M steps ≈ 2.0 h on a CPU laptop):

```bash
python agents/custom/afterstate_custom_agent.py train --timesteps 12000000 --n-envs 8 --max-pieces 500 --seed 7 --outdir runs/track5_afterstate --logdir runs/track5_afterstate_logs
```

Evaluate it (same protocol as Track 3):

```bash
python agents/custom/afterstate_custom_agent.py evaluate --model artifacts/custom_afterstate/ppo_custom_afterstate.zip --episodes 25 --max-pieces 500 --seed 1000 --deterministic
```

**Result: mean 5.60 lines** vs Track 3's 1.04 — so the action abstraction is worth
**5.4×**, at matched experience (Track 3's 100M primitive steps ≈ 11.0M pieces at a
measured 9.11 steps/piece; Track 5 saw 12.0M). But it closes only **2.3%** of the
Track 3 → Track 4 gap, so the hand-authored features and lookahead carry the rest.
Caveat: the run had not converged, so 5.60 is a lower bound. Full analysis in
[REPORT §7.2](docs/REPORT.md).

## Best Plays (videos)

`artifacts/best_plays/` holds the best episode of each track as a watchable mp4.
Each track plays a batch of seeded episodes, and the best one (ranked by lines,
then score) is replayed with frame capture:

```bash
python tools/render_best_plays.py --tracks 1,2,3,4,5
```

The mp4 files are gitignored; `artifacts/best_plays/README.md` and `manifest.json`
record what each video shows.

To watch an agent play live in a window instead of a recording (runs until you
close it — the Track 4 planner never tops out):

```bash
python artifacts/best_plays/live_play.py            # track 4, endless
python artifacts/best_plays/live_play.py --track 3  # track 3, restarts on top-out
python artifacts/best_plays/live_play.py --track 5  # track 5, placement actions
```

Running **tracks 3 and 5 side by side** is the controlled experiment made visible:
same PPO, same network, same reward — only the action space differs.

Individual tracks can also be rendered directly:

```bash
python agents/ale/pure_rl_ale_agent.py render --seed 0            # track 1
python ale_tetris_agent.py render --planner legacy_model --weights artifacts/ale_stable_high_score/best_weights.npy  # track 2
python agents/custom/pure_rl_custom_agent.py render --seed 0      # track 3
python agents/custom/render_custom_episode.py --seed 0            # track 4
python agents/custom/afterstate_custom_agent.py render --seed 0   # track 5
```

## Archived ALE Baseline

The real ALE checkpoint preserved from the earlier work is:

```text
artifacts/ale_37_line/best_weights.npy
```

It reproduces the seed-0 result from the old `legacy_model` planner:

```text
lines=37
score=3700
pieces=259
```

For the legacy ALE planner, `pieces` is a receding-horizon decision count rather
than a strict locked-piece count.

There is also a separate stable-score copy:

```text
artifacts/ale_stable_high_score/best_weights.npy
```

Verified command:

```bash
python ale_tetris_agent.py evaluate --planner legacy_model --weights artifacts/ale_stable_high_score/best_weights.npy --episodes 10 --max-pieces 400 --seed 0 --out artifacts/ale_stable_high_score/evaluation.json
```

Seeds 0 through 9 all produced `37` lines, score `3700`, and `259` legacy
decisions.
