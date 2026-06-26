# TetrisGPT

Clean monorepo for four Tetris agent tracks:

- `agents/ale/pure_rl_ale_agent.py` trains a pure RL model on real
  `ALE/Tetris-v5` frames and actions.
- `agents/ale/ale_tetris_agent.py` keeps the tool-assisted real ALE high-score
  work as a showcase.
- `agents/custom/pure_rl_custom_agent.py` trains a pure RL model on the custom
  Gymnasium environment with one-piece preview.
- `agents/custom/tetris_custom_agent.py` trains the tool-assisted custom
  high-score model with placement enumeration and one-piece lookahead.

The previous real-ALE 2000-line target is intentionally no longer active. The
best verified real-ALE 37-line model is set aside under `artifacts/ale_37_line`.

## Docs

- [Model catalog](docs/MODELS.md)
- [Project summary](docs/PROJECT_SUMMARY.md)

## Layout

```text
agents/
  ale/       Pure-RL ALE agent plus tool-assisted ALE showcase.
  custom/    Pure-RL custom agent plus tool-assisted custom high-score agent.
packages/
  tetris_env/ Reusable Tetris engine and Gymnasium environment.
artifacts/
  ale_pure_rl/  Default output for pure-RL ALE PPO checkpoints.
  ale_37_line/  Saved real-ALE 37-line model metadata.
  custom_pure_rl/ Default output for pure-RL custom PPO checkpoints.
  custom_best/  Default output for custom-env trained weights and videos.
```

## Track 1: Pure RL on ALE

This track uses only raw Atari frames and ALE actions. It does not decode the
board, enumerate placements, use a model of Tetris pieces, or search future
states. Sticky actions are enabled by default with `--sticky 0.25`.

Smoke the ALE env:

```bash
python agents/ale/pure_rl_ale_agent.py smoke
```

Train a pure PPO model:

```bash
python agents/ale/pure_rl_ale_agent.py train --timesteps 100000 --n-envs 4 --sticky 0.25
```

Evaluate across many slippery seeds:

```bash
python agents/ale/pure_rl_ale_agent.py evaluate --model artifacts/ale_pure_rl/ppo_ale_pure.zip --episodes 25 --sticky 0.25
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
python ale_tetris_agent.py evaluate --planner legacy_model --weights artifacts/ale_stable_high_score/best_weights.npy --episodes 10 --max-pieces 400 --seed 0
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

The custom environment models:

- 10x20 visible board with two hidden spawn rows.
- Seven tetrominoes with a 7-bag generator.
- Current piece state and one-piece preview exposed to pure RL code.
- One-piece preview exposed to the tool-assisted policy and viewer.
- SRS-style wall kicks.
- Hard drop, soft drop, line clears, levels, and score-mode line scoring.

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
python agents/custom/pure_rl_custom_agent.py train --timesteps 100000 --n-envs 4
```

Evaluate it:

```bash
python agents/custom/pure_rl_custom_agent.py evaluate --model artifacts/custom_pure_rl/ppo_custom_pure.zip --episodes 10
```

## Track 4: Tool-Assisted High Score on Custom Env

This track uses the standard custom Tetris engine while allowing planning tools:
placement enumeration, weighted features, and one-piece lookahead over the
visible next shape. The game rules remain the engine rules: 10x20 board, hidden
spawn rows, 7-bag pieces, SRS-style kicks, hard/soft drops, line clears, levels,
and score-mode line scoring.

Train a custom score-mode model:

```bash
python agents/custom/tetris_custom_agent.py train --generations 20 --population 32 --rollouts 4
```

Evaluate the current custom best:

```bash
python agents/custom/tetris_custom_agent.py evaluate --weights artifacts/custom_best/best_weights.npy --episodes 5
```

Render a custom-env episode with the next-piece preview visible:

```bash
python agents/custom/render_custom_episode.py --weights artifacts/custom_best/best_weights.npy --out artifacts/custom_best/custom_episode.mp4
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

There is also a separate stable-score copy:

```text
artifacts/ale_stable_high_score/best_weights.npy
```

Verified command:

```bash
python ale_tetris_agent.py evaluate --planner legacy_model --weights artifacts/ale_stable_high_score/best_weights.npy --episodes 10 --max-pieces 400 --seed 0
```

Seeds 0 through 9 all produced `37` lines, score `3700`, and `259` pieces.
