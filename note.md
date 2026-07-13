# Overnight Clear-Line Plan

> **Superseded (2026-07-07 plan, kept for history).** Do not run these commands.
> They were written before the seeding fix, the force-lock fix, the `lines` reward
> default and the planner rewrite, and the "known issue" below (the Track 1 eval
> JSON crash) was fixed on 2026-07-08. The runbook that replaced this is
> `TRAINING_PLAN.md`; the final results are in `docs/REPORT.md`.

Goal: increase clear lines across all four tracks while respecting the project
track boundaries in `AGENTS.md`.

Current read from the last run:

- Track 1, ALE pure RL: 10M PPO timesteps finished, but eval printed 0 lines in
  all 25 episodes. Repeating the exact same command is not the best next use of
  time.
- Track 2, ALE tool-assisted: stable at 37 lines on checked seeds. Treat this
  as the current low-end-machine plateau unless you want a high-risk search.
- Track 3, custom pure RL: only 200k timesteps so far and 0 lines. This is the
  cheapest pure-RL track to improve.
- Track 4, custom tool-assisted: strongest custom result so far. Queue
  lookahead is expensive but gives the best line count.

Important known issue:

- `agents/ale/pure_rl_ale_agent.py evaluate --out ...` currently crashes while
  writing JSON because ALE episode info contains NumPy scalar values. Until that
  is fixed, run ALE pure-RL eval without `--out` and save the terminal transcript.

## Best Overnight Queue

Run this from the repo root in PowerShell. It writes all console output into one
transcript under `runs/overnight_lines_20260707/`.

```powershell
New-Item -ItemType Directory -Force -Path runs\overnight_lines_20260707
Start-Transcript -Path runs\overnight_lines_20260707\powershell_output.txt

# Track 3: pure RL on custom env.
# Best first use of compute: this track is undertrained and fast.
python agents/custom/pure_rl_custom_agent.py train --outdir runs/overnight_lines_20260707/custom_pure_rl_20m --logdir runs/overnight_lines_20260707/custom_pure_rl_logs --timesteps 20000000 --n-envs 4 --max-pieces 1000 --seed 7 --eval-freq 500000 --checkpoint-freq 1000000
python agents/custom/pure_rl_custom_agent.py evaluate --model runs/overnight_lines_20260707/custom_pure_rl_20m/ppo_custom_pure.zip --vec-normalize runs/overnight_lines_20260707/custom_pure_rl_20m/vec_normalize.pkl --episodes 25 --max-pieces 1000 --deterministic --out runs/overnight_lines_20260707/custom_pure_rl_20m/evaluation.json

# Track 4: tool-assisted custom env.
# Warm-start the current best and optimize directly under queue lookahead.
# This is expensive, but it is the best proper approach for higher clear lines.
python agents/custom/tetris_custom_agent.py train --outdir runs/overnight_lines_20260707/custom_tool_queue --warm-start artifacts/custom_best/best_weights.npy --generations 12 --population 24 --rollouts 3 --max-pieces 200 --seed 7 --lookahead-depth 2 --lookahead-candidates 4 --future-source queue
python agents/custom/tetris_custom_agent.py evaluate --weights runs/overnight_lines_20260707/custom_tool_queue/best_weights.npy --episodes 10 --max-pieces 200 --lookahead-depth 2 --lookahead-candidates 4 --future-source queue --out runs/overnight_lines_20260707/custom_tool_queue/evaluation_200.json

# Track 1: pure RL on ALE.
# The last sticky-action 10M run failed, so this is an easier pure-RL line-clear
# attempt: non-sticky ALE, more timesteps, and SubprocVecEnv for throughput.
# If RAM fails, use the fallback command below instead.
python agents/ale/pure_rl_ale_agent.py train --outdir runs/overnight_lines_20260707/ale_pure_rl_nosticky_25m --logdir runs/overnight_lines_20260707/ale_pure_rl_nosticky_logs --timesteps 25000000 --n-envs 6 --vec-env subproc --sticky 0.0 --eval-freq 500000 --checkpoint-freq 1000000
python agents/ale/pure_rl_ale_agent.py evaluate --model runs/overnight_lines_20260707/ale_pure_rl_nosticky_25m/ppo_ale_pure.zip --episodes 25 --sticky 0.0
python agents/ale/pure_rl_ale_agent.py evaluate --model runs/overnight_lines_20260707/ale_pure_rl_nosticky_25m/best/best_model.zip --episodes 25 --sticky 0.0

# Track 2: tool-assisted ALE validation.
# Do not spend the main overnight budget here; the current result is already
# stable at 37 lines. This just confirms the baseline after the run.
python ale_tetris_agent.py evaluate --planner legacy_model --weights artifacts/ale_stable_high_score/best_weights.npy --episodes 10 --max-pieces 400 --seed 0 --out runs/overnight_lines_20260707/ale_track2_stable_eval.json

Stop-Transcript
```

## If Track 1 Runs Out Of RAM

Use this safer Track 1 command instead. It is slower but avoids `SubprocVecEnv`.

```powershell
python agents/ale/pure_rl_ale_agent.py train --outdir runs/overnight_lines_20260707/ale_pure_rl_nosticky_20m_dummy --logdir runs/overnight_lines_20260707/ale_pure_rl_nosticky_dummy_logs --timesteps 20000000 --n-envs 4 --vec-env dummy --sticky 0.0 --eval-freq 500000 --checkpoint-freq 1000000
python agents/ale/pure_rl_ale_agent.py evaluate --model runs/overnight_lines_20260707/ale_pure_rl_nosticky_20m_dummy/ppo_ale_pure.zip --episodes 25 --sticky 0.0
```

## Optional Track 2 Gamble

Only run this if you specifically want to spend time trying to beat the 37-line
ALE tool-assisted plateau. It is less likely to pay off than Tracks 3 and 4.

```powershell
python ale_tetris_agent.py train --outdir runs/overnight_lines_20260707/ale_track2_legacy_calibrated --planner legacy_calibrated --warm-start artifacts/ale_stable_high_score/best_weights.npy --generations 12 --population 16 --rollouts 2 --max-pieces 500 --top-k 32 --seed 7
python ale_tetris_agent.py evaluate --planner legacy_calibrated --weights runs/overnight_lines_20260707/ale_track2_legacy_calibrated/best_weights.npy --episodes 10 --max-pieces 500 --top-k 32 --seed 0 --out runs/overnight_lines_20260707/ale_track2_legacy_calibrated/evaluation.json
```

## How To Pick Winners Tomorrow

- Track 1 winner: highest `estimated_lines` printed by eval. If both final and
  callback-best are 0 again, stop spending ALE pure-RL time without changing the
  training approach.
- Track 2 winner: any result above 37 lines is meaningful. Otherwise keep
  `artifacts/ale_stable_high_score/best_weights.npy`.
- Track 3 winner: highest `max_lines` and `mean_lines` in
  `custom_pure_rl_20m/evaluation.json`.
- Track 4 winner: highest `mean_lines` and `max_lines` in
  `custom_tool_queue/evaluation_200.json`. If it beats the current artifact,
  promote it after a longer eval.

Recommended next code change if the overnight run still gives 0 lines on Track
3: add a line-focused reward mode to the custom Gym environment. More PPO on the
current score/drop reward may keep teaching fast survival without teaching line
clears.
