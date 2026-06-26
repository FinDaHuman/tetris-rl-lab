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
- Environment: `ALE/Tetris-v5`
- Default artifact: `artifacts/ale_pure_rl/ppo_ale_pure.zip`
- Current algorithm: PPO from Stable-Baselines3
- Policy input: raw Atari RGB frames only
- Policy output: ALE discrete actions
- Default slippery setting: `--sticky 0.25`
- Disallowed assistance: board decoding, piece detection, placement search,
  hand-authored Tetris model, direct board edits

Commands:

```bash
python agents/ale/pure_rl_ale_agent.py smoke
python agents/ale/pure_rl_ale_agent.py train --timesteps 100000 --n-envs 4 --sticky 0.25
python agents/ale/pure_rl_ale_agent.py evaluate --model artifacts/ale_pure_rl/ppo_ale_pure.zip --episodes 25 --sticky 0.25
```

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
python ale_tetris_agent.py evaluate --planner legacy_model --weights artifacts/ale_stable_high_score/best_weights.npy --episodes 10 --max-pieces 400 --seed 0
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
- Disallowed assistance: placement enumeration, direct board edits, lookahead,
  search, access to future pieces beyond one next piece

Commands:

```bash
python agents/custom/pure_rl_custom_agent.py smoke
python agents/custom/pure_rl_custom_agent.py train --timesteps 100000 --n-envs 4
python agents/custom/pure_rl_custom_agent.py evaluate --model artifacts/custom_pure_rl/ppo_custom_pure.zip --episodes 10
```

## Track 4: Tool-Assisted High Score on Custom Env

Purpose: score as high as possible on the local custom engine while keeping the
game rules standard. This is the fast planning/optimization track.

- Entry point: `agents/custom/tetris_custom_agent.py`
- Viewer: `agents/custom/render_custom_episode.py`
- Environment and engine: `packages/tetris_env`
- Default artifact: `artifacts/custom_best/best_weights.npy`
- Current method: weighted placement features optimized by CEM-style training
- Assistance used: placement enumeration, board features, direct engine cloning,
  one-piece lookahead
- Rule boundary: the agent may plan, but the engine remains standard Tetris
  logic and physics for this project

Commands:

```bash
python agents/custom/tetris_custom_agent.py train --generations 20 --population 32 --rollouts 4
python agents/custom/tetris_custom_agent.py evaluate --weights artifacts/custom_best/best_weights.npy --episodes 5
python agents/custom/render_custom_episode.py --weights artifacts/custom_best/best_weights.npy --out artifacts/custom_best/custom_episode.mp4
```

## Artifact Ownership

```text
artifacts/
  ale_pure_rl/              Track 1 PPO checkpoints and metadata.
  ale_37_line/              Track 2 archived ALE 37-line checkpoint.
  ale_stable_high_score/    Track 2 stable ALE copy and evaluation manifest.
  custom_pure_rl/           Track 3 PPO checkpoints and metadata.
  custom_best/              Track 4 optimized weights, history, and renders.
```

Temporary smoke or experiment outputs should go under `runs/`, which is ignored.
