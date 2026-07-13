# Model Catalog

This repo is organized around four separable Tetris agent tracks. Each track has
its own entry point, expected artifact path, environment boundary, and allowed
level of assistance.

## Separation Rules

- Pure RL tracks must learn through environment observations and environment
  actions only.
- Tool-assisted tracks may use planning, board decoding, placement enumeration,
  search, heuristic features, and offline weight optimization.
- ALE tracks use the real `ALE/Tetris-v5` environment.
- Custom tracks use the local engine in `packages/tetris_env`.
- Artifacts should stay under the matching `artifacts/<track>/` directory.

## Track 1: Pure RL on ALE

Purpose: train a model that plays real Atari Tetris through pure reinforcement
learning. It should work across many seeds and slippery/sticky action settings.
Score can be low; the important property is that the model is pure RL.

- Entry point: `agents/ale/pure_rl_ale_agent.py`
- Shared environment helper: `agents/ale/env.py`
- Environment: `ALE/Tetris-v5`
- Default artifact: `artifacts/ale_pure_rl/ppo_ale_pure.zip`
- Current algorithm: PPO from Stable-Baselines3
- Policy input: Atari frames only; default training uses standard Atari
  preprocessing to grayscale 84x84 frames plus frame stacking
- Policy output: ALE discrete actions
- Default slippery setting: `--sticky 0.25`
- Disallowed assistance: board decoding, piece detection, placement search,
  hand-authored Tetris model, direct board edits

Commands:

```bash
python agents/ale/pure_rl_ale_agent.py smoke
python agents/ale/pure_rl_ale_agent.py train --timesteps 100000 --n-envs 4 --sticky 0.25
python agents/ale/pure_rl_ale_agent.py train --timesteps 10000000 --n-envs 8 --vec-env subproc --sticky 0.25
python agents/ale/pure_rl_ale_agent.py evaluate --model artifacts/ale_pure_rl/ppo_ale_pure.zip --episodes 25 --sticky 0.25 --out artifacts/ale_pure_rl/evaluation.json
python agents/ale/pure_rl_ale_agent.py render --seed 0 --out artifacts/best_plays/track1_ale_pure_rl.mp4
```

The `render` subcommand captures the native ALE screen (not the 84x84 grayscale
observation the policy sees) and must be given the same `--sticky` / `--frame-stack`
/ preprocessing settings the model was trained with, or the policy is being fed an
observation it never saw. The defaults match `artifacts/ale_pure_rl/meta.json`.

## Track 2: Tool-Assisted High Score on ALE

Purpose: score as high as possible on the real ALE environment while allowing
all available tools. The existing 37-line model belongs to this track.

- Entry point: `agents/ale/ale_tetris_agent.py`
- Root wrapper: `ale_tetris_agent.py`
- Environment: `ALE/Tetris-v5`
- Preserved artifact: `artifacts/ale_37_line/best_weights.npy`
- Stable copy: `artifacts/ale_stable_high_score/best_weights.npy`
- Current verified planner: `legacy_model`
- Current verified result: 37 lines, score 3700, 259 pieces on checked seeds
- Experimental next-attempt planner: `legacy_calibrated`
- Allowed assistance: frame decoding, board reconstruction, placement modeling,
  search, calibration, hand-authored features, CEM-style weight optimization

Note: in the legacy ALE planner, the reported `pieces` value is best treated as
a receding-horizon decision count. The policy can replan while the same falling
piece is still active.

Commands:

```bash
python ale_tetris_agent.py smoke
python ale_tetris_agent.py evaluate --planner legacy_model --weights artifacts/ale_stable_high_score/best_weights.npy --episodes 10 --max-pieces 400 --seed 0 --out artifacts/ale_stable_high_score/evaluation.json
python ale_tetris_agent.py train --planner legacy_model --warm-start artifacts/ale_stable_high_score/best_weights.npy --generations 50 --population 48 --rollouts 4 --max-pieces 500
python ale_tetris_agent.py train --planner legacy_calibrated --warm-start artifacts/ale_stable_high_score/best_weights.npy --generations 20 --population 24 --rollouts 2 --max-pieces 500 --top-k 32
```

## Track 3: Pure RL on Custom Env

Purpose: train a pure RL model on the local custom Tetris environment. The model
can see the current piece state and exactly one next shape, but it must interact
only through the Gymnasium step API.

- Entry point: `agents/custom/pure_rl_custom_agent.py`
- Environment: `packages/tetris_env/tetris_env/gym_env.py`
- Default artifact: `artifacts/custom_pure_rl/ppo_custom_pure.zip`
- Current algorithm: PPO from Stable-Baselines3
- Policy input: flat vector of locked board, active piece mask, normalized
  piece row/col/rotation, current-piece one-hot, next-piece one-hot
- Current observation size: 417 floats
- Policy output: custom env discrete actions
- Training setup: VecNormalize for observations/rewards, checkpoint callbacks,
  deterministic evaluation JSON, configurable MLP size
- Reward modes (`--reward-mode`, default `lines`):
  - `lines`: `line_reward * cleared^2` per lock + `piece_reward` per placed
    piece - `top_out_penalty` on termination; no drop points. This is the
    recommended mode: the old score reward paid dense per-cell drop points
    that dominated the sparse line reward and taught drop-fast behavior.
  - `score`: raw engine score deltas (legacy behavior, kept for older
    checkpoints).
- Disallowed assistance: placement enumeration, direct board edits, lookahead,
  search, access to future pieces beyond one next piece
- Current promoted model (2026-07-09, Night 2): 100M steps PPO on the lines
  reward — mean 1.04 lines over 25 deterministic 500-piece episodes (max 3;
  21/25 episodes clear at least one line). First pure-RL line clears in the
  project. Evaluation manifests: `artifacts/custom_pure_rl/evaluation.json`
  (final model) and `evaluation_best.json` (eval-callback best, 0.92 lines).

Commands:

```bash
python agents/custom/pure_rl_custom_agent.py smoke
python agents/custom/pure_rl_custom_agent.py train --timesteps 5000000 --n-envs 8 --reward-mode lines
python agents/custom/pure_rl_custom_agent.py evaluate --model artifacts/custom_pure_rl/ppo_custom_pure.zip --episodes 10 --deterministic --out artifacts/custom_pure_rl/evaluation.json
python agents/custom/pure_rl_custom_agent.py render --seed 0 --out artifacts/best_plays/track3_custom_pure_rl.mp4
```

`artifacts/custom_pure_rl/vec_normalize.pkl` is **required** alongside the model:
it was trained on normalized observations, so loading the `.zip` without the stats
produces garbage actions rather than an error. `evaluate` and `render` both load it
automatically from the model's directory (`--vec-normalize` overrides the path).

## Track 4: Tool-Assisted High Score on Custom Env

Purpose: score as high as possible on the local custom engine while keeping the
game rules standard. This is the fast planning/optimization track.

- Entry point: `agents/custom/tetris_custom_agent.py`
- Viewer: `agents/custom/render_custom_episode.py`
- Environment and engine: `packages/tetris_env`
- Default artifact: `artifacts/custom_best/best_weights.npy`
- Current verified result (2026-07-08, 500-piece cap, 10 episodes, seeds
  0-9): mean score 215,530, mean lines 198.1 of a 200-line ceiling —
  `artifacts/custom_best/evaluation_500.json`
- **The 200-line ceiling is the piece cap, not the agent** (measured
  2026-07-13): uncapped it does not top out — 10,000 pieces / 3,997 lines and
  still alive when the probe was stopped, sustaining ~0.4 lines/piece. Read its
  line count as `0.4 × cap`. At a 2,000-piece cap it clears 798 lines.
- Current method: weighted placement features optimized by CEM-style training
  with queue-aware beam lookahead
- Best-weights promotion during training uses a fixed held-out seed set
  (`--holdout-rollouts`, default 3), not the per-generation training seeds;
  `history.json` records both train and holdout fitness
- Placement enumeration is a vectorized drop simulation that dedupes
  symmetric rotations (O, and the I/S/Z opposite pairs); a 500-piece
  queue-lookahead episode runs in roughly 15 seconds on the reference laptop
- Assistance used: placement enumeration, board features, direct engine cloning,
  internal queue lookahead
- Rule boundary: the agent may plan, but the engine remains standard Tetris
  logic and physics for this project

Commands:

```bash
python agents/custom/tetris_custom_agent.py train --generations 20 --population 32 --rollouts 4 --lookahead-depth 2 --lookahead-candidates 4 --future-source queue
python agents/custom/tetris_custom_agent.py evaluate --weights artifacts/custom_best/best_weights.npy --episodes 5 --lookahead-depth 2 --lookahead-candidates 4 --future-source queue
python agents/custom/render_custom_episode.py --weights artifacts/custom_best/best_weights.npy --max-pieces 2000 --out artifacts/best_plays/track4_custom_tool.mp4
```

## Playback

All four tracks render to mp4, and two can be watched live.

```bash
python tools/render_best_plays.py --tracks 1,2,3,4      # best episode per track -> artifacts/best_plays/
python artifacts/best_plays/live_play.py                # track 4 live, runs until closed
python artifacts/best_plays/live_play.py --track 3      # track 3 live, restarts on top-out
```

`render_best_plays.py` searches seeds per track, ranks by lines then score, and
re-renders the winner. Shared rendering code lives in the engine package
(`tetris_env/render.py` draws a frame, `tetris_env/replay.py` converts a planner
`Placement` back into primitive actions so Track 4 animates instead of teleporting)
and `agents/video.py` (streaming mp4 writer — do not accumulate frames in a list;
a 2,000-piece episode is ~10k frames).

**Boundary:** `replay.py` exists only to animate the tool-assisted tracks. It must
never be imported by a pure-RL training or evaluation path (see `AGENTS.md`).

## Artifact Ownership

```text
artifacts/
  ale_pure_rl/              Track 1 PPO checkpoints and metadata.
  ale_37_line/              Track 2 archived ALE 37-line checkpoint.
  ale_stable_high_score/    Track 2 stable ALE copy and evaluation manifest.
  custom_pure_rl/           Track 3 PPO checkpoints, metadata, VecNormalize stats.
  custom_best/              Track 4 optimized weights, history, and evaluations.
  best_plays/               Best episode per track as mp4 + live viewer.
                            README.md/manifest.json/live_play.py are committed;
                            the .mp4 files are gitignored (regenerate them).
```

Temporary smoke or experiment outputs should go under `runs/`, which is ignored.
